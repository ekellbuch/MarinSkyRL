"""FP32 output projection reduced to per-token statistics one sequence chunk at a time.

``fp32_projection_token_stats`` replaces the Megatron output layer when
``trainer.policy.model.lm_head_compute_dtype=float32``. These tests pin it against an
unchunked FP32 reference built from full logits: values and gradients for chunk sizes that
do and do not divide the sequence, vocabulary padding, and a two-rank vocabulary split
simulated with Gloo, where each rank holds half of the projection rows.
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.distributed.megatron.model_utils import (  # noqa: E402
    PROJECTION_STAT_FIELDS,
    fp32_projection_token_stats,
)

SEQ, BATCH, HIDDEN, VOCAB = 13, 2, 8, 24


def _inputs(dtype=torch.float32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(SEQ, BATCH, HIDDEN, generator=generator).to(dtype)
    weight = torch.randn(VOCAB, HIDDEN, generator=generator).to(dtype)
    targets = torch.randint(VOCAB, (SEQ, BATCH), generator=generator)
    coefficients = torch.randn(2, SEQ, BATCH, generator=generator)
    return hidden, weight, targets, coefficients


def _reference(hidden, weight, targets, *, temperature=1.0, vocabulary_size=VOCAB):
    raw = torch.matmul(hidden.float(), weight.float().t())
    log_probs = (raw / temperature).log_softmax(dim=-1)
    logprob = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    top = raw[..., :vocabulary_size].topk(2, dim=-1)
    return logprob, entropy, top.values[..., 0] - top.values[..., 1], top.indices[..., 0]


def _stats(hidden, weight, targets, *, chunk_size, group=None, vocab_start_index=0, **overrides):
    kwargs = dict(
        vocab_start_index=vocab_start_index,
        vocabulary_size=VOCAB,
        temperature=1.0,
        chunk_size=chunk_size,
        group=group,
        entropy_requires_grad=True,
        inference_only=False,
        compute_top1=True,
    )
    kwargs.update(overrides)
    return fp32_projection_token_stats(hidden, weight, targets, **kwargs)


def _loss(logprob, entropy, coefficients):
    return (coefficients[0] * logprob).sum() + (coefficients[1] * entropy).sum()


@pytest.mark.parametrize("chunk_size", [None, 1, 4, 5, 13, 100])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_chunked_stats_match_unchunked_fp32_reference(chunk_size, temperature):
    hidden, weight, targets, coefficients = _inputs()
    hidden.requires_grad_(True)
    weight.requires_grad_(True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)

    stats = _stats(hidden, weight, targets, chunk_size=chunk_size, temperature=temperature)
    logprob, entropy, margin, top1 = _reference(reference_hidden, reference_weight, targets, temperature=temperature)

    assert stats.shape == (SEQ, BATCH, len(PROJECTION_STAT_FIELDS))
    assert stats.dtype is torch.float32
    torch.testing.assert_close(stats[..., 0], logprob, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(stats[..., 1], entropy, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(stats[..., 2], margin, rtol=1e-5, atol=1e-5)
    assert torch.equal(stats[..., 3].long(), top1)

    _loss(stats[..., 0], stats[..., 1], coefficients).backward()
    _loss(logprob, entropy, coefficients).backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=1e-5, atol=1e-5)


def test_bf16_parameters_compute_in_fp32_and_receive_model_dtype_gradients():
    hidden, weight, targets, coefficients = _inputs(torch.bfloat16, seed=1)
    hidden.requires_grad_(True)
    weight.requires_grad_(True)

    stats = _stats(hidden, weight, targets, chunk_size=4)
    logprob, entropy, _, _ = _reference(hidden.detach(), weight.detach(), targets)
    bf16_logprob = torch.matmul(hidden.detach(), weight.detach().t()).float().log_softmax(-1)
    bf16_logprob = bf16_logprob.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(stats[..., 0], logprob, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(stats[..., 1], entropy, rtol=1e-5, atol=1e-5)
    # The point of the FP32 head: the result is not the bf16 matmul's.
    assert not torch.allclose(stats[..., 0], bf16_logprob, rtol=1e-5, atol=1e-5)

    _loss(stats[..., 0], stats[..., 1], coefficients).backward()
    assert hidden.grad.dtype is torch.bfloat16
    assert weight.grad.dtype is torch.bfloat16


def test_entropy_without_grad_leaves_logprob_gradient_unchanged():
    hidden, weight, targets, coefficients = _inputs(seed=2)
    hidden.requires_grad_(True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)

    stats = _stats(hidden, weight, targets, chunk_size=3, entropy_requires_grad=False)
    (coefficients[0] * stats[..., 0]).sum().backward()
    logprob, _, _, _ = _reference(reference_hidden, weight, targets)
    (coefficients[0] * logprob).sum().backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=1e-5, atol=1e-5)


def test_inference_only_builds_no_graph():
    hidden, weight, targets, _ = _inputs(seed=3)
    hidden.requires_grad_(True)
    stats = _stats(hidden, weight, targets, chunk_size=4, inference_only=True)
    assert not stats.requires_grad


def test_padded_vocabulary_never_wins_top1():
    hidden, weight, targets, _ = _inputs(seed=4)
    vocabulary_size = VOCAB - 3
    weight[vocabulary_size:] = 100.0 * hidden.reshape(-1, HIDDEN).mean(0)
    targets = targets.clamp(max=vocabulary_size - 1)

    stats = _stats(hidden, weight, targets, chunk_size=4, vocabulary_size=vocabulary_size, inference_only=True)
    _, _, margin, top1 = _reference(hidden, weight, targets, vocabulary_size=vocabulary_size)

    assert int(stats[..., 3].max()) < vocabulary_size
    assert torch.equal(stats[..., 3].long(), top1)
    torch.testing.assert_close(stats[..., 2], margin, rtol=1e-5, atol=1e-5)


def test_top1_columns_are_zero_when_not_requested():
    hidden, weight, targets, _ = _inputs(seed=5)
    stats = _stats(hidden, weight, targets, chunk_size=4, inference_only=True, compute_top1=False)
    assert torch.equal(stats[..., 2:], torch.zeros_like(stats[..., 2:]))


def test_rejects_non_positive_chunk_size():
    hidden, weight, targets, _ = _inputs()
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        _stats(hidden, weight, targets, chunk_size=0)


def _vocab_parallel_rank(rank: int, world_size: int, store_path: str, chunk_size: int, result_path: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=world_size)
    try:
        hidden, weight, targets, coefficients = _inputs(seed=6)
        shard = VOCAB // world_size
        hidden = hidden.requires_grad_(True)
        weight_shard = weight[rank * shard : (rank + 1) * shard].clone().requires_grad_(True)
        stats = _stats(
            hidden,
            weight_shard,
            targets,
            chunk_size=chunk_size,
            group=dist.group.WORLD,
            vocab_start_index=rank * shard,
            temperature=0.9,
        )
        _loss(stats[..., 0], stats[..., 1], coefficients).backward()
        # Megatron's copy-to-tensor-parallel-region backward sums each rank's partial input gradient.
        hidden_grad = hidden.grad.clone()
        dist.all_reduce(hidden_grad)
        weight_grads = [torch.empty_like(weight_shard.grad) for _ in range(world_size)]
        dist.all_gather(weight_grads, weight_shard.grad)
        if rank == 0:
            torch.save(
                {"stats": stats.detach(), "hidden_grad": hidden_grad, "weight_grad": torch.cat(weight_grads)},
                result_path,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("chunk_size", [4, None])
def test_vocab_parallel_split_matches_unsplit_reference(tmp_path, chunk_size):
    world_size = 2
    result_path = str(tmp_path / "result.pt")
    mp.spawn(
        _vocab_parallel_rank,
        args=(world_size, str(tmp_path / "store"), chunk_size, result_path),
        nprocs=world_size,
        join=True,
    )
    result = torch.load(result_path)

    hidden, weight, targets, coefficients = _inputs(seed=6)
    hidden.requires_grad_(True)
    weight.requires_grad_(True)
    logprob, entropy, margin, top1 = _reference(hidden, weight, targets, temperature=0.9)
    _loss(logprob, entropy, coefficients).backward()

    torch.testing.assert_close(result["stats"][..., 0], logprob, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result["stats"][..., 1], entropy, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result["stats"][..., 2], margin, rtol=1e-5, atol=1e-5)
    assert torch.equal(result["stats"][..., 3].long(), top1)
    torch.testing.assert_close(result["hidden_grad"], hidden.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result["weight_grad"], weight.grad, rtol=1e-5, atol=1e-5)
