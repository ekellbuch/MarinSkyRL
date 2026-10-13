"""Run the Megatron output projection in FP32 and return per-token statistics instead of logits.

With ``trainer.policy.model.lm_head_compute_dtype=float32`` the policy model's final projection runs through
``GPTModel``'s ``output_processor`` hook. The hook casts the final hidden states and the projection weight to
FP32 and reduces one sequence chunk at a time to the label log-probability, the entropy and the top-1 margin,
so FP32 logits for the whole sequence are never materialized.

The projection parameter itself stays in the model dtype, and the distributed optimizer keeps its FP32 master
copy. Weight sync therefore ships the projection in the model dtype that the vLLM receiver checks, and both
sides compute with the same weights upcast to FP32. With tied embeddings the FP32 projection is a per-forward
copy of the shared embedding weight whose gradient flows back into the shared parameter.
"""

from dataclasses import dataclass
from typing import Any

import torch
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
)
from skyrl_train.distributed.megatron.model_utils import fp32_projection_token_stats
from skyrl_train.models.lm_head_precision import FP32_DTYPE_NAME


@dataclass(frozen=True)
class FP32ProjectionRequest:
    """What the output processor needs for one micro-batch.

    ``targets`` is ``[batch, local_seq]``: the next-token id at every position of the model's token layout on
    this context-parallel rank (packed or left-unpadded), as ``next_token_targets`` or
    ``packed_next_token_targets`` produce it.
    """

    targets: torch.Tensor
    vocabulary_size: int
    temperature: float
    chunk_size: int | None
    entropy_requires_grad: bool
    inference_only: bool
    compute_top1: bool


def uses_fp32_output_projection(dtype_name: str | None) -> bool:
    """Whether ``trainer.policy.model.lm_head_compute_dtype`` selects the FP32 learner projection."""
    if dtype_name is None:
        return False
    if dtype_name != FP32_DTYPE_NAME:
        raise ValueError(f"Unsupported lm_head_compute_dtype: {dtype_name}")
    return True


def fp32_projection_output_processor(
    *,
    hidden_states: torch.Tensor,
    output_layer: Any,
    output_weight: torch.Tensor | None,
    context: FP32ProjectionRequest,
    config: Any,
    **kwargs,
) -> torch.Tensor:
    """``GPTModel`` output processor returning ``[batch, local_seq, 4]`` FP32 statistics.

    Columns follow ``PROJECTION_STAT_FIELDS``. ``output_weight`` is the shared embedding weight when input and
    output embeddings are tied, otherwise ``None`` and the output layer's own weight is used.
    """
    if config.use_mup:
        raise ValueError("lm_head_compute_dtype=float32 does not support MuP logit scaling")
    if config.defer_embedding_wgrad_compute:
        raise ValueError("lm_head_compute_dtype=float32 does not support defer_embedding_wgrad_compute")
    if output_layer.bias is not None:
        raise ValueError("lm_head_compute_dtype=float32 requires a bias-free output layer")
    weight = output_layer.weight if output_weight is None else output_weight
    group = output_layer.tp_group
    # Every tensor-parallel rank needs the full sequence for its vocabulary shard. Each rank's input gradient
    # covers only its shard, so both regions sum the partial gradients in backward.
    if output_layer.sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=True, group=group
        )
    elif group.size() > 1:
        hidden_states = copy_to_tensor_model_parallel_region(hidden_states, group=group)
    stats = fp32_projection_token_stats(
        hidden_states,
        weight,
        context.targets.transpose(0, 1),
        vocab_start_index=group.rank() * weight.shape[0],
        vocabulary_size=context.vocabulary_size,
        temperature=context.temperature,
        chunk_size=context.chunk_size,
        group=group,
        entropy_requires_grad=context.entropy_requires_grad,
        inference_only=context.inference_only,
        compute_top1=context.compute_top1,
    )
    return stats.transpose(0, 1)
