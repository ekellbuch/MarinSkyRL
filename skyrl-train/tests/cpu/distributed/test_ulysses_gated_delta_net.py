"""Gated DeltaNet under Ulysses SP, on CPU with gloo.

The kernel that carries the recurrent state across ranks is FLA's and needs a
GPU (`tests/gpu/test_qwen35_gated_delta_net_ulysses_sp.py`). Everything around
it runs here: the conv-prefix exchange and its transpose, the shard bookkeeping,
and the whole layer forward with a stand-in delta rule that gathers the shards
and runs transformers' torch reference on the full sequence.

Run::

    python -m pytest tests/cpu/distributed/test_ulysses_gated_delta_net.py -v
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from skyrl_train.distributed.ulysses import gated_delta_net as gdn
from skyrl_train.distributed.ulysses.utils import set_ulysses_sequence_parallel_group
from tests.cpu.util import gloo_process_group

qwen35_config_mod = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")
qwen35_modeling_mod = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

HIDDEN = 64
HEAD_DIM = 16
KERNEL = 4
WORLD_SIZE = 2
PORT = 29571


def _make_layer():
    config = qwen35_config_mod.Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        linear_key_head_dim=HEAD_DIM,
        linear_value_head_dim=HEAD_DIM,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=KERNEL,
        layer_types=["linear_attention"],
        dtype=torch.float32,
    )
    torch.manual_seed(1234)
    layer = qwen35_modeling_mod.Qwen3_5GatedDeltaNet(config, 0).float()
    # The torch reference path everywhere: this test has no fla and no GPU.
    layer.causal_conv1d_fn = None
    layer.norm = qwen35_modeling_mod.Qwen3_5RMSNormGated(HEAD_DIM, eps=config.rms_norm_eps)
    layer.chunk_gated_delta_rule = qwen35_modeling_mod.torch_chunk_gated_delta_rule
    return layer


class _GatheringDeltaRule(torch.autograd.Function):
    """Stand-in for FLA's CP kernel: gather every shard, run the reference on the
    full sequence, keep this rank's slice. The backward gathers every rank's
    output gradient so the cross-rank state dependence is differentiated too."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, group):
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)

        def gather(x):
            parts = [torch.empty_like(x) for _ in range(world_size)]
            dist.all_gather(parts, x.contiguous(), group=group)
            return torch.cat(parts, dim=1).requires_grad_(True)

        inputs = [gather(t) for t in (q, k, v, g, beta)]
        with torch.enable_grad():
            out, _ = qwen35_modeling_mod.torch_chunk_gated_delta_rule(
                *inputs[:3],
                g=inputs[3],
                beta=inputs[4],
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
        ctx.inputs, ctx.out, ctx.group = inputs, out, group
        ctx.local = slice(rank * q.shape[1], (rank + 1) * q.shape[1])
        return out[:, ctx.local].detach()

    @staticmethod
    def backward(ctx, grad_local):
        parts = [torch.empty_like(grad_local) for _ in range(dist.get_world_size(ctx.group))]
        dist.all_gather(parts, grad_local.contiguous(), group=ctx.group)
        grads = torch.autograd.grad(ctx.out, ctx.inputs, torch.cat(parts, dim=1))
        return tuple(grad[:, ctx.local] for grad in grads) + (None,)


def _gathering_delta_rule(q, k, v, g, beta, cp_context=None, **kwargs):
    if cp_context is None:
        return qwen35_modeling_mod.torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, **kwargs)
    return _GatheringDeltaRule.apply(q, k, v, g, beta, cp_context.group), None


def _fake_cp_context(local_len, kernel_size, device, group):
    total = local_len * dist.get_world_size(group)
    return SimpleNamespace(group=group, cu_seqlens=torch.tensor([0, total]), conv1d_kernel_size=kernel_size)


def _conv_prefix_worker(rank, world_size):
    with gloo_process_group(rank, world_size, PORT):
        torch.manual_seed(7)
        channels, local_len = 5, 6
        full = torch.randn(1, channels, local_len * world_size)
        weight = torch.randn(channels, 1, KERNEL)
        conv = torch.nn.Conv1d(channels, channels, KERNEL, groups=channels, bias=False, padding=KERNEL - 1)
        conv.weight.data.copy_(weight)

        expected = conv(full)[:, :, : full.shape[-1]]
        shard = full[:, :, rank * local_len : (rank + 1) * local_len].clone().requires_grad_(True)
        prefixed, prefix_len = gdn.prepend_conv_prefix(shard, KERNEL, dist.group.WORLD)
        assert prefix_len == KERNEL - 1
        out = conv(prefixed)[:, :, prefix_len : prefix_len + local_len]
        torch.testing.assert_close(out, expected[:, :, rank * local_len : (rank + 1) * local_len])

        # The transpose of the exchange: the gradient of the full conv w.r.t. the
        # full input, restricted to this shard, has to come back through the prefix.
        full_ref = full.clone().requires_grad_(True)
        (conv(full_ref)[:, :, : full.shape[-1]] * 1.5).sum().backward()
        (out * 1.5).sum().backward()
        torch.testing.assert_close(shard.grad, full_ref.grad[:, :, rank * local_len : (rank + 1) * local_len])


def _layer_worker(rank, world_size):
    with gloo_process_group(rank, world_size, PORT + 1):
        layer = _make_layer()
        for p in layer.parameters():
            dist.broadcast(p.data, src=0)
        local_len = 12
        torch.manual_seed(99)
        full = torch.randn(1, local_len * world_size, HIDDEN)
        dist.broadcast(full, src=0)

        set_ulysses_sequence_parallel_group(None)
        full_ref = full.clone().requires_grad_(True)
        ref = layer(full_ref)
        ref.sum().backward()
        ref_grad = full_ref.grad.detach().clone()
        layer.zero_grad(set_to_none=True)

        layer.chunk_gated_delta_rule = _gathering_delta_rule
        set_ulysses_sequence_parallel_group(dist.group.WORLD)
        try:
            assert gdn.apply_gated_delta_net_patch(layer, world_size) == 1
            shard = full[:, rank * local_len : (rank + 1) * local_len].clone().requires_grad_(True)
            with patch.object(gdn, "build_cp_context", _fake_cp_context):
                out = layer(shard)
                out.sum().backward()
        finally:
            set_ulysses_sequence_parallel_group(None)
            type(layer).forward = getattr(type(layer), gdn._ORIGINAL_FORWARD)
            delattr(type(layer), gdn._ORIGINAL_FORWARD)

        torch.testing.assert_close(out, ref[:, rank * local_len : (rank + 1) * local_len], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            shard.grad, ref_grad[:, rank * local_len : (rank + 1) * local_len], atol=1e-5, rtol=1e-5
        )


def test_conv_prefix_matches_full_causal_conv():
    mp.spawn(_conv_prefix_worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)


def test_layer_forward_and_backward_match_full_sequence():
    mp.spawn(_layer_worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)


def test_patch_is_a_no_op_at_sp1_and_without_gated_delta_net():
    layer = _make_layer()
    assert gdn.apply_gated_delta_net_patch(layer, 1) == 0
    assert gdn.apply_gated_delta_net_patch(torch.nn.Linear(2, 2), 4) == 0
    assert not hasattr(type(layer), gdn._ORIGINAL_FORWARD)


def test_patch_refuses_a_kernel_without_cp_context():
    layer = _make_layer()  # bound to transformers' torch_chunk_gated_delta_rule
    with pytest.raises(RuntimeError, match="cp_context"):
        gdn.apply_gated_delta_net_patch(layer, 4)


def test_patch_refuses_unknown_gated_delta_net_classes():
    class OtherGatedDeltaNet(torch.nn.Module):
        pass

    with pytest.raises(NotImplementedError, match="OtherGatedDeltaNet"):
        gdn.apply_gated_delta_net_patch(OtherGatedDeltaNet(), 2)


def test_forward_outside_a_group_is_the_original():
    layer = _make_layer()
    layer.chunk_gated_delta_rule = _gathering_delta_rule  # accepts cp_context
    set_ulysses_sequence_parallel_group(None)
    x = torch.randn(2, 5, HIDDEN)
    expected = layer(x)
    assert gdn.apply_gated_delta_net_patch(layer, 2) == 1
    try:
        torch.testing.assert_close(layer(x), expected)
    finally:
        type(layer).forward = getattr(type(layer), gdn._ORIGINAL_FORWARD)
        delattr(type(layer), gdn._ORIGINAL_FORWARD)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    pytest.main([__file__, "-v"])
