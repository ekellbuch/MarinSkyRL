"""CPU tests for the per-token statistics handed to trainer callbacks.

uv run --isolated --group dev --extra cpu -- pytest skyrl-train/tests/cpu/utils/test_token_stats.py
"""

from dataclasses import dataclass

import pytest
import torch

from skyrl_train.training_batch import TrainingOutputBatch
from skyrl_train.utils.token_stats import (
    TOKEN_STATS_METADATA_KEY,
    concat_token_stats,
    gather_token_stats,
    select_micro_batch_tokens,
    top1_margin_from_logits,
)


def test_top1_margin_from_logits_matches_sorted_reference():
    logits = torch.tensor(
        [
            [[1.0, 4.0, 2.5, 0.0], [0.0, 0.0, 3.0, 3.0]],
            [[-1.0, -2.0, -3.0, -4.0], [5.0, 1.0, 1.0, 4.5]],
        ],
        dtype=torch.bfloat16,
    )

    margin, top1 = top1_margin_from_logits(logits)

    ordered = torch.sort(logits.float(), dim=-1, descending=True).values
    assert margin.dtype == torch.float32
    assert torch.equal(margin, ordered[..., 0] - ordered[..., 1])
    assert torch.equal(top1, logits.float().argmax(dim=-1))


def _micro_batch_inputs():
    # Two rows, three response tokens each; the loss mask keeps four positions.
    sequences = torch.tensor([[9, 9, 11, 12, 13], [9, 9, 21, 22, 23]])
    loss_mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    learner = torch.tensor([[-0.1, -0.2, -0.3], [-1.1, -1.2, -1.3]])
    rollout = torch.tensor([[-0.15, -0.2, -0.9], [-1.0, -1.25, -1.3]])
    entropy = torch.tensor([[0.5, 0.6, 0.7], [1.5, 1.6, 1.7]])
    margin = torch.tensor([[2.0, 0.0, 4.0], [0.5, 3.0, 1.0]])
    top1_token = torch.tensor([[11, 99, 13], [21, 22, 99]])
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    return dict(
        global_step=7,
        sequences=sequences,
        num_actions=3,
        loss_mask=loss_mask,
        learner_logprobs=learner,
        rollout_logprobs=rollout,
        entropy=entropy,
        top1_margin=margin,
        top1_token=top1_token,
        advantages=advantages,
        sample_offset=4,
    )


def test_select_micro_batch_tokens_keeps_loss_masked_positions_in_row_major_order():
    batch = select_micro_batch_tokens(**_micro_batch_inputs())

    assert len(batch) == 4
    assert batch.global_step == 7
    assert batch.sample_index.tolist() == [4, 4, 5, 5]
    assert batch.position.tolist() == [0, 1, 1, 2]
    assert batch.token_id.tolist() == [11, 12, 22, 23]
    assert batch.learner_logprob.tolist() == pytest.approx([-0.1, -0.2, -1.2, -1.3])
    assert batch.rollout_logprob.tolist() == pytest.approx([-0.15, -0.2, -1.25, -1.3])
    assert batch.entropy.tolist() == pytest.approx([0.5, 0.6, 1.6, 1.7])
    assert batch.top1_margin.tolist() == pytest.approx([2.0, 0.0, 3.0, 1.0])
    assert batch.sampled_is_top1.tolist() == [True, False, True, False]
    assert batch.advantage.tolist() == [1.0, 1.0, -1.0, -1.0]


def test_select_micro_batch_tokens_without_rollout_logprobs_or_mask():
    inputs = _micro_batch_inputs()
    inputs["rollout_logprobs"] = None
    inputs["loss_mask"] = None

    batch = select_micro_batch_tokens(**inputs)

    assert batch.rollout_logprob is None
    assert batch.position.tolist() == [0, 1, 2, 0, 1, 2]
    assert batch.token_id.tolist() == [11, 12, 13, 21, 22, 23]


def test_select_micro_batch_tokens_rejects_mismatched_mask():
    inputs = _micro_batch_inputs()
    inputs["loss_mask"] = torch.ones(2, 2)
    with pytest.raises(ValueError, match="loss_mask shape"):
        select_micro_batch_tokens(**inputs)


@dataclass
class _Rank:
    dp: int
    tp: int
    dp_size: int

    def is_collection_dp_rank(self) -> bool:
        return self.tp == 0


@dataclass
class _ActorInfo:
    rank: _Rank


def _output_with(shard) -> TrainingOutputBatch:
    output = TrainingOutputBatch()
    output.metadata = {"train_status": {}, TOKEN_STATS_METADATA_KEY: shard}
    return output


def test_gather_token_stats_orders_data_parallel_ranks_and_attaches_sample_ids():
    inputs = _micro_batch_inputs()
    inputs["sample_offset"] = 0
    shard_dp0 = select_micro_batch_tokens(**inputs)
    shard_dp1 = select_micro_batch_tokens(**{**inputs, "global_step": 7})
    # Actor order is deliberately not dp order, and dp rank 1 has a tensor-parallel replica.
    actor_infos = [
        _ActorInfo(_Rank(dp=1, tp=0, dp_size=2)),
        _ActorInfo(_Rank(dp=1, tp=1, dp_size=2)),
        _ActorInfo(_Rank(dp=0, tp=0, dp_size=2)),
    ]
    outputs = [_output_with(shard_dp1), _output_with(shard_dp1), _output_with(shard_dp0)]
    sample_ids = ["task-a", "task-a", "task-b", "task-b"]

    batch = gather_token_stats(actor_infos, outputs, sample_ids=sample_ids, global_step=7)

    assert batch.sample_ids == tuple(sample_ids)
    assert batch.sample_index.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [batch.sample_ids[i] for i in batch.sample_index.tolist()] == (["task-a"] * 4 + ["task-b"] * 4)
    assert batch.token_id.tolist() == [11, 12, 22, 23] * 2


def test_gather_token_stats_rejects_missing_data_parallel_shard():
    shard = select_micro_batch_tokens(**{**_micro_batch_inputs(), "sample_offset": 0})
    actor_infos = [_ActorInfo(_Rank(dp=0, tp=0, dp_size=2))]
    with pytest.raises(ValueError, match="one token stats shard per data-parallel rank"):
        gather_token_stats(actor_infos, [_output_with(shard)], sample_ids=["a", "b"], global_step=1)


def test_concat_token_stats_rejects_shards_that_disagree_on_rollout_logprobs():
    with_rollout = select_micro_batch_tokens(**_micro_batch_inputs())
    without = select_micro_batch_tokens(**{**_micro_batch_inputs(), "rollout_logprobs": None})
    with pytest.raises(ValueError, match="rollout logprobs"):
        concat_token_stats([with_rollout, without], global_step=1)


def test_concat_token_stats_of_no_shards_is_empty():
    batch = concat_token_stats([], global_step=3)
    assert len(batch) == 0
    assert batch.global_step == 3
