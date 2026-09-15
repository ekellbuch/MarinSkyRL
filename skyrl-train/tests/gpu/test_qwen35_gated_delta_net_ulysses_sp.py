"""Qwen3.5 Gated DeltaNet under Ulysses SP matches the full-sequence forward.

Ported from volcengine/verl#6660
(tests/special_distributed/test_qwen35_linear_attention_ulysses_sp.py). Needs
`flash-linear-attention >= 0.5` for `fla.ops.cp`; opt-in, outside gpu_ci.

Run::

    torchrun --nproc_per_node=4 -m pytest -svv tests/gpu/test_qwen35_gated_delta_net_ulysses_sp.py
"""

import os

import pytest
import torch
import torch.distributed as dist
from skyrl_train.distributed.ulysses.gated_delta_net import _ORIGINAL_FORWARD, apply_gated_delta_net_patch
from skyrl_train.distributed.ulysses.utils import set_ulysses_sequence_parallel_group

pytest.importorskip("fla.ops.cp")
qwen35_config_mod = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")
qwen35_modeling_mod = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

pytestmark = pytest.mark.skipif("LOCAL_RANK" not in os.environ, reason="run with torchrun")

HIDDEN = 512
HEAD_DIM = 128


def _err_ratio(ref: torch.Tensor, actual: torch.Tensor) -> float:
    err = (ref.detach() - actual.detach()).flatten().float().square().mean().sqrt().item()
    base = ref.detach().flatten().float().square().mean().sqrt().item()
    return err / (base + 1e-8)


def _make_layer(device: torch.device):
    config = qwen35_config_mod.Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        linear_key_head_dim=HEAD_DIM,
        linear_value_head_dim=HEAD_DIM,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"],
        dtype=torch.bfloat16,
    )
    torch.manual_seed(1234)
    layer = qwen35_modeling_mod.Qwen3_5DecoderLayer(config, 0).to(device=device, dtype=torch.bfloat16)
    layer.eval()
    for param in layer.parameters():
        dist.broadcast(param.data, src=0)
    return layer


def _all_gather_seq(x: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, x.contiguous())
    return torch.cat(gathered, dim=1)


def _run_case(total_tokens: int, layer, device: torch.device):
    """One packed sequence of `total_tokens`, the shape model_wrapper hands the model."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert total_tokens % world_size == 0
    position_embeddings = (torch.empty(0, device=device), torch.empty(0, device=device))

    torch.manual_seed(5678 + total_tokens)
    full_hidden = torch.randn(1, total_tokens, HIDDEN, device=device, dtype=torch.bfloat16)
    dist.broadcast(full_hidden, src=0)

    set_ulysses_sequence_parallel_group(None)
    full_ref = full_hidden.detach().clone().requires_grad_(True)
    ref_out = layer(full_ref, position_embeddings=position_embeddings)
    ref_out.sum().backward()
    ref_grad = full_ref.grad.detach()
    layer.zero_grad(set_to_none=True)

    set_ulysses_sequence_parallel_group(dist.group.WORLD)
    local_len = total_tokens // world_size
    local_hidden = full_hidden[:, rank * local_len : (rank + 1) * local_len].detach().clone().requires_grad_(True)
    sp_out_local = layer(local_hidden, position_embeddings=position_embeddings)
    sp_out_local.sum().backward()
    set_ulysses_sequence_parallel_group(None)

    sp_out = _all_gather_seq(sp_out_local.detach())
    sp_grad = _all_gather_seq(local_hidden.grad.detach())
    if rank == 0:
        print(
            f"T={total_tokens}: max_out_diff={(sp_out - ref_out).abs().max().item():.6f} "
            f"max_grad_diff={(sp_grad - ref_grad).abs().max().item():.6f} "
            f"out_err_ratio={_err_ratio(ref_out, sp_out):.6f} grad_err_ratio={_err_ratio(ref_grad, sp_grad):.6f}"
        )
    torch.testing.assert_close(sp_out, ref_out.detach(), atol=2e-2, rtol=2e-2)
    assert _err_ratio(ref_grad, sp_grad) < 2e-3


@pytest.mark.parametrize("use_causal_conv1d_fn", [True, False], ids=["causal_conv1d_fn", "torch_conv1d_fallback"])
def test_gated_delta_net_matches_full_forward_under_ulysses_sp(use_causal_conv1d_fn: bool):
    assert torch.cuda.is_available(), "CUDA is required"
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    world_size = dist.get_world_size()

    layer = _make_layer(device)
    gdn_cls = type(layer.linear_attn)
    if not use_causal_conv1d_fn:
        layer.linear_attn.causal_conv1d_fn = None
    assert apply_gated_delta_net_patch(layer, world_size) == 1
    try:
        # Chunk-aligned, chunk-unaligned, and shorter than one 64-token chunk per rank.
        for total_tokens in (128 * world_size, 100 * world_size, 7 * world_size):
            _run_case(total_tokens, layer, device)
    finally:
        set_ulysses_sequence_parallel_group(None)
        gdn_cls.forward = getattr(gdn_cls, _ORIGINAL_FORWARD)
        delattr(gdn_cls, _ORIGINAL_FORWARD)
