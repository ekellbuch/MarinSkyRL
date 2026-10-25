# Ported from volcengine/verl#6660 (verl/models/transformers/qwen3_5.py, a106816b).
# The original copyright is reproduced below.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gated DeltaNet under Ulysses sequence parallelism.

`monkey_patch.py` makes Ulysses correct for full-attention layers by gathering
the sequence inside `_flash_attention_forward`. Qwen3.5's Gated DeltaNet layers
never reach that seam: each rank ran the recurrence and the causal conv over its
own sequence shard from a zero state, so every `sequence_parallel_size > 1`
geometry computed a different forward pass from sp1 (6-13 nats on the
selected-token logprobs of Qwen3.5-4B).

The fix is FLA's context-parallel path for the delta rule: each rank runs
`chunk_gated_delta_rule` on its shard with a `cp_context` that carries the
recurrent state across ranks, and the causal conv receives the previous rank's
last `kernel_size - 1` tokens as a prefix. transformers' own forward carries no
sequence boundaries (`seq_idx=None`), so a packed micro-batch is one sequence to
it; the sequence-parallel forward keeps that contract on the global sequence.

Requires `flash-linear-attention >= 0.5` (`fla.ops.cp`).
"""

from inspect import signature

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from skyrl_train.distributed.ulysses.utils import (
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
)

PATCHED_CLASSES = ("Qwen3_5GatedDeltaNet", "Qwen3_5MoeGatedDeltaNet")
_ORIGINAL_FORWARD = "_ulysses_original_forward"


class _ConvPrefixExchange(torch.autograd.Function):
    """Hand each rank the previous rank's last conv taps; rank 0 gets zeros.

    The exchange is an all-gather so every rank takes part in one collective.
    Its transpose sends the gradient of the received prefix back to the rank
    whose tail produced it.
    """

    @staticmethod
    def forward(ctx, tails: torch.Tensor, group: dist.ProcessGroup):
        ctx.group = group
        return _shift_from_previous_rank(tails.contiguous(), group)

    @staticmethod
    def backward(ctx, grad_prefix: torch.Tensor):
        return _shift_from_next_rank(grad_prefix.contiguous(), ctx.group), None


def _shift_from_previous_rank(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    rank = dist.get_rank(group)
    gathered = _all_gather(x, group)
    return torch.zeros_like(x) if rank == 0 else gathered[rank - 1]


def _shift_from_next_rank(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    gathered = _all_gather(x, group)
    return torch.zeros_like(x) if rank == world_size - 1 else gathered[rank + 1]


def _all_gather(x: torch.Tensor, group: dist.ProcessGroup) -> list[torch.Tensor]:
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, x, group=group)
    return gathered


def prepend_conv_prefix(
    mixed_qkv: torch.Tensor, kernel_size: int, group: dist.ProcessGroup
) -> tuple[torch.Tensor, int]:
    """Prefix `mixed_qkv` ([batch, channels, local_len]) with the previous rank's tail.

    Returns the prefixed tensor and the prefix length, which the caller drops
    from the conv output.
    """
    prefix_len = max(kernel_size - 1, 0)
    if prefix_len == 0:
        return mixed_qkv, 0
    if mixed_qkv.shape[-1] >= prefix_len:
        tails = mixed_qkv[..., -prefix_len:]
    else:
        tails = F.pad(mixed_qkv, (prefix_len - mixed_qkv.shape[-1], 0))
    prefix = _ConvPrefixExchange.apply(tails, group)
    return torch.cat((prefix, mixed_qkv), dim=-1), prefix_len


def build_cp_context(local_len: int, kernel_size: int, device: torch.device, group: dist.ProcessGroup):
    """FLA's context for one global sequence sharded evenly across the group."""
    from fla.ops.cp.context import build_cp_context as fla_build_cp_context

    total = local_len * dist.get_world_size(group)
    cu_seqlens = torch.tensor([0, total], dtype=torch.long, device=device)
    return fla_build_cp_context(
        cu_seqlens, group=group, conv1d_kernel_size=kernel_size, cu_seqlens_cpu=cu_seqlens.cpu()
    )


def gated_delta_net_forward(self, hidden_states: torch.Tensor, cache_params=None, attention_mask=None):
    """`Qwen3_5GatedDeltaNet.forward` with the recurrence carried across SP ranks.

    Outside a sequence-parallel group this is the original forward, so sp1 is
    untouched. Inside one, `hidden_states` is this rank's shard of the packed
    sequence: the conv borrows the previous rank's tail and the delta rule runs
    with FLA's `cp_context`.
    """
    group = get_ulysses_sequence_parallel_group()
    if group is None or get_ulysses_sequence_parallel_world_size(group) <= 1:
        return getattr(type(self), _ORIGINAL_FORWARD)(self, hidden_states, cache_params, attention_mask)
    if cache_params is not None:
        raise NotImplementedError("Gated DeltaNet under Ulysses sequence parallelism has no cached forward.")

    module = _modeling_module(self)
    hidden_states = module.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    if batch_size != 1:
        raise ValueError(
            f"Gated DeltaNet under Ulysses sequence parallelism expects one packed sequence, got batch {batch_size}."
        )
    cp_context = build_cp_context(seq_len, self.conv_kernel_size, hidden_states.device, group)

    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    conv_input, prefix_len = prepend_conv_prefix(mixed_qkv, self.conv_kernel_size, group)
    if self.causal_conv1d_fn is not None:
        mixed_qkv = self.causal_conv1d_fn(
            x=conv_input,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            seq_idx=None,
        )[..., prefix_len:]
    else:
        mixed_qkv = F.silu(self.conv1d(conv_input)[:, :, prefix_len : prefix_len + seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    core_attn_out, _ = self.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cp_context=cp_context,
    )

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z).reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


def _modeling_module(layer):
    import sys

    return sys.modules[type(layer).__module__]


def _accepts_cp_context(fn) -> bool:
    try:
        params = signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "cp_context" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def gated_delta_net_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    return [m for m in model.modules() if type(m).__name__.endswith("GatedDeltaNet")]


def apply_gated_delta_net_patch(model: torch.nn.Module, ulysses_sp_size: int) -> int:
    """Install the sequence-parallel forward on the model's Gated DeltaNet classes.

    Returns the number of layers covered. A no-op at sp1 and on models without
    Gated DeltaNet layers. Refuses, rather than computing a different model,
    when the installed kernel cannot carry state across ranks.
    """
    layers = gated_delta_net_layers(model)
    if ulysses_sp_size <= 1 or not layers:
        return 0
    unknown = sorted({type(m).__name__ for m in layers} - set(PATCHED_CLASSES))
    if unknown:
        raise NotImplementedError(
            f"sequence_parallel_size={ulysses_sp_size} with {unknown}: only {list(PATCHED_CLASSES)} carry the "
            "Gated DeltaNet recurrence across Ulysses ranks; other linear-attention layers would run each shard "
            "from a zero state."
        )
    for layer in layers:
        if not _accepts_cp_context(layer.chunk_gated_delta_rule):
            raise RuntimeError(
                f"sequence_parallel_size={ulysses_sp_size} needs a chunk_gated_delta_rule that accepts cp_context "
                f"(flash-linear-attention >= 0.5); {type(layer).__name__} is bound to {layer.chunk_gated_delta_rule}."
            )
    for cls in {type(m) for m in layers}:
        if not hasattr(cls, _ORIGINAL_FORWARD):
            setattr(cls, _ORIGINAL_FORWARD, cls.forward)
            cls.forward = gated_delta_net_forward
    logger.info(f"Monkey patch Gated DeltaNet forward for Ulysses sp={ulysses_sp_size} on {len(layers)} layers")
    return len(layers)
