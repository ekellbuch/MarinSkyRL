"""Per-token learner/rollout statistics handed to trainer callbacks.

The policy loss reduces every per-token quantity to a scalar mean before it
reaches the trainer. When ``trainer.token_stats.enabled`` is set, the policy
worker instead keeps, for every loss-masked response token of an optimizer
step, the scalars a token-level diagnostic needs: the learner and rollout
log-probabilities, the learner's token entropy, its top-1 logit margin, the
advantage, and the token identity. The trainer concatenates the per-rank
selections in data-parallel order, attaches the batch's sample ids, and stores
the result on ``trainer.last_token_stats`` for ``on_step_end`` callbacks.
Full-vocabulary logits never leave the worker.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional, Sequence, Tuple

import torch

TOKEN_STATS_METADATA_KEY = "token_stats"
_TENSOR_FIELDS = (
    "sample_index",
    "position",
    "token_id",
    "learner_logprob",
    "rollout_logprob",
    "entropy",
    "top1_margin",
    "sampled_is_top1",
    "advantage",
)


@dataclass(frozen=True)
class TokenStatsBatch:
    """Selected per-token scalars for one optimizer step, on CPU.

    Every tensor is one-dimensional with one entry per selected token.
    ``sample_index`` indexes ``sample_ids`` once the trainer has attached them;
    inside a worker it is the rank-local index into that rank's batch shard.
    ``rollout_logprob`` is ``None`` when the generator did not return logprobs.
    """

    global_step: int
    sample_ids: Tuple[str, ...]
    sample_index: torch.Tensor
    position: torch.Tensor
    token_id: torch.Tensor
    learner_logprob: torch.Tensor
    rollout_logprob: Optional[torch.Tensor]
    entropy: torch.Tensor
    top1_margin: torch.Tensor
    sampled_is_top1: torch.Tensor
    advantage: torch.Tensor

    def __len__(self) -> int:
        return int(self.sample_index.numel())


def top1_margin_from_logits(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(top1 - runner-up logit, top1 token id)`` along the vocabulary axis.

    The margin is computed in float32 from the two largest logits only, so no
    full-vocabulary copy of ``logits`` is made.
    """
    with torch.no_grad():
        top_values, top_tokens = torch.topk(logits, 2, dim=-1)
        margin = (top_values[..., 0].float() - top_values[..., 1].float()).contiguous()
        return margin, top_tokens[..., 0].contiguous()


def select_micro_batch_tokens(
    *,
    global_step: int,
    sequences: torch.Tensor,
    num_actions: int,
    loss_mask: Optional[torch.Tensor],
    learner_logprobs: torch.Tensor,
    rollout_logprobs: Optional[torch.Tensor],
    entropy: torch.Tensor,
    top1_margin: torch.Tensor,
    top1_token: torch.Tensor,
    advantages: torch.Tensor,
    sample_offset: int,
) -> TokenStatsBatch:
    """Keep the loss-masked response tokens of one micro-batch.

    All ``[batch, num_actions]`` inputs are the action-aligned slices the
    worker already holds; ``sequences`` is the full ``[batch, seq_len]`` token
    tensor whose last ``num_actions`` columns are the response. ``sample_offset``
    is the rank-local index of the micro-batch's first row.
    """
    response_ids = sequences[:, -num_actions:]
    if loss_mask is None:
        selected = torch.ones_like(response_ids, dtype=torch.bool)
    else:
        selected = loss_mask.bool()
    if selected.shape != response_ids.shape:
        raise ValueError(f"loss_mask shape {tuple(selected.shape)} does not match response {tuple(response_ids.shape)}")
    row_index, position = torch.nonzero(selected, as_tuple=True)

    def take(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[row_index, position].detach().to("cpu")

    return TokenStatsBatch(
        global_step=global_step,
        sample_ids=(),
        sample_index=(row_index + sample_offset).to("cpu", torch.int64),
        position=position.to("cpu", torch.int64),
        token_id=take(response_ids).to(torch.int64),
        learner_logprob=take(learner_logprobs).float(),
        rollout_logprob=None if rollout_logprobs is None else take(rollout_logprobs).float(),
        entropy=take(entropy).float(),
        top1_margin=take(top1_margin).float(),
        sampled_is_top1=take(top1_token) == take(response_ids),
        advantage=take(advantages).float(),
    )


def _empty_like(global_step: int, with_rollout: bool) -> TokenStatsBatch:
    long_empty = torch.zeros(0, dtype=torch.int64)
    float_empty = torch.zeros(0, dtype=torch.float32)
    return TokenStatsBatch(
        global_step=global_step,
        sample_ids=(),
        sample_index=long_empty,
        position=long_empty.clone(),
        token_id=long_empty.clone(),
        learner_logprob=float_empty,
        rollout_logprob=float_empty.clone() if with_rollout else None,
        entropy=float_empty.clone(),
        top1_margin=float_empty.clone(),
        sampled_is_top1=torch.zeros(0, dtype=torch.bool),
        advantage=float_empty.clone(),
    )


def concat_token_stats(
    shards: Sequence[TokenStatsBatch],
    *,
    global_step: int,
    sample_offsets: Optional[Sequence[int]] = None,
) -> TokenStatsBatch:
    """Concatenate shards, adding ``sample_offsets[i]`` to shard ``i``'s sample indices."""
    if sample_offsets is None:
        sample_offsets = [0] * len(shards)
    if len(sample_offsets) != len(shards):
        raise ValueError("one sample offset is required per shard")
    if not shards:
        return _empty_like(global_step, with_rollout=False)
    with_rollout = all(shard.rollout_logprob is not None for shard in shards)
    if with_rollout != any(shard.rollout_logprob is not None for shard in shards):
        raise ValueError("token stats shards disagree on whether rollout logprobs are present")
    columns = {}
    for name in _TENSOR_FIELDS:
        if name == "rollout_logprob" and not with_rollout:
            columns[name] = None
            continue
        if name == "sample_index":
            parts = [shard.sample_index + offset for shard, offset in zip(shards, sample_offsets)]
        else:
            parts = [getattr(shard, name) for shard in shards]
        columns[name] = torch.cat(parts)
    return TokenStatsBatch(global_step=global_step, sample_ids=(), **columns)


def gather_token_stats(
    actor_infos: Sequence[Any],
    outputs: Sequence[Any],
    *,
    sample_ids: Sequence[str],
    global_step: int,
) -> TokenStatsBatch:
    """Assemble one batch from the per-rank ``ppo_train`` outputs.

    ``outputs[i]`` is the ``TrainingOutputBatch`` returned by ``actor_infos[i]``.
    Mesh dispatch gives each data-parallel rank a contiguous chunk of the batch
    in rank order, so rank ``dp`` starts at ``dp * len(sample_ids) // dp_size``.
    Only one actor per data-parallel group carries the selection.
    """
    if len(actor_infos) != len(outputs):
        raise ValueError("one output is required per actor")
    collection = [
        (info.rank.dp, output.metadata[TOKEN_STATS_METADATA_KEY])
        for info, output in zip(actor_infos, outputs)
        if info.rank.is_collection_dp_rank()
    ]
    if not collection:
        raise ValueError("no collection rank returned token stats")
    dp_size = actor_infos[0].rank.dp_size
    if len(collection) != dp_size or sorted(dp for dp, _ in collection) != list(range(dp_size)):
        raise ValueError(f"expected one token stats shard per data-parallel rank ({dp_size}), got {len(collection)}")
    if len(sample_ids) % dp_size != 0:
        raise ValueError(f"{len(sample_ids)} sample ids are not divisible by dp_size={dp_size}")
    chunk = len(sample_ids) // dp_size
    collection.sort(key=lambda item: item[0])
    batch = concat_token_stats(
        [shard for _, shard in collection],
        global_step=global_step,
        sample_offsets=[dp * chunk for dp, _ in collection],
    )
    if len(batch) and int(batch.sample_index.max()) >= len(sample_ids):
        raise ValueError("token stats sample index exceeds the number of sample ids")
    return TokenStatsBatch(**{**_as_dict(batch), "sample_ids": tuple(sample_ids)})


def _as_dict(batch: TokenStatsBatch) -> dict:
    return {field.name: getattr(batch, field.name) for field in fields(batch)}
