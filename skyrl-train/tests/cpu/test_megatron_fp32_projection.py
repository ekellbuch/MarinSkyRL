"""The Megatron policy wrapper with ``lm_head_compute_dtype=float32``.

A fake GPT chunk calls the wrapper's output processor the way ``GPTModel`` does, so the
target layout, the restoration of left-padded rows, the token statistics, and the
gradients reaching the projection weight and hidden states run through real wrapper code. The reference is the
same wrapper consuming full FP32 logits on the logits path.
"""

import pytest
import torch
from omegaconf import OmegaConf

from skyrl_train.objective.losses import TokenLoss
from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.workers.megatron import fp32_output_projection as projection_module  # noqa: E402
from skyrl_train.workers.megatron import megatron_model_wrapper as mmw  # noqa: E402

BATCH, SEQ, HIDDEN, VOCAB, NUM_ACTIONS = 2, 7, 6, 11, 4


def _cfg():
    return OmegaConf.create(
        {
            "trainer": {
                "use_sample_packing": False,
                "algorithm": {
                    "use_entropy_loss": False,
                    "entropy_loss_coef": 0.0,
                    "use_kl_loss": False,
                    "kl_estimator_type": "k1",
                    "kl_loss_coef": 0.0,
                    "loss_reduction": "token_mean",
                    "think_token_weight": 1.0,
                    "policy_loss_type": "regular",
                    "eps_clip_low": 0.2,
                    "eps_clip_high": 0.2,
                    "max_seq_len": SEQ,
                },
            },
        }
    )


def _logprob_loss(inputs, config):
    return TokenLoss(-inputs.log_probs, {})


class _OutputLayer:
    def __init__(self, weight, group):
        self.weight = weight
        self.tp_group = group
        self.sequence_parallel = False
        self.bias = None


class _FakeGPTChunk:
    """Embeds tokens, then either projects to logits or hands hidden states to an output processor."""

    def __init__(self, embedding, output_layer):
        self.embedding = embedding
        self.output_layer = output_layer
        self.training = True
        self.config = OmegaConf.create({"use_mup": False, "defer_embedding_wgrad_compute": False})

    def __call__(
        self,
        sequences,
        position_ids,
        attention_mask,
        packed_seq_params=None,
        fp32_output=False,
        output_processor=None,
        output_processor_context=None,
    ):
        hidden = self.embedding[sequences.long()].transpose(0, 1)  # [seq, batch, hidden]
        if output_processor is not None:
            return output_processor(
                hidden_states=hidden,
                output_layer=self.output_layer,
                output_weight=None,
                context=output_processor_context,
                config=self.config,
            )
        logits = torch.matmul(hidden.float(), self.output_layer.weight.float().t())
        return logits.transpose(0, 1).contiguous()


def _forward_backward_func(forward_step_func, data_iterator, model, num_microbatches, **kwargs):
    metrics_list = []
    for _ in range(num_microbatches):
        outputs, closure = forward_step_func(data_iterator, model[0])
        loss, metrics = closure(outputs)
        if not kwargs["forward_only"]:
            loss.backward()
        metrics_list.append(metrics)
    return metrics_list


@pytest.fixture
def megatron_mesh(single_rank_group, monkeypatch):
    mpu = mmw.mpu
    for name, value in {
        "is_pipeline_first_stage": lambda **kwargs: True,
        "is_pipeline_last_stage": lambda **kwargs: True,
        "get_context_parallel_world_size": lambda: 1,
        "get_tensor_model_parallel_world_size": lambda: 1,
        "get_tensor_model_parallel_rank": lambda: 0,
        "get_tensor_model_parallel_group": lambda: single_rank_group,
        "get_pipeline_model_parallel_last_rank": lambda: 0,
        "get_pipeline_model_parallel_group": lambda: single_rank_group,
        "get_data_parallel_group": lambda **kwargs: single_rank_group,
    }.items():
        monkeypatch.setattr(mpu, name, value, raising=False)
    monkeypatch.setattr(mmw, "get_forward_backward_func", lambda: _forward_backward_func)
    return single_rank_group


def _model_and_batch(group, seed=0):
    generator = torch.Generator().manual_seed(seed)
    embedding = torch.randn(VOCAB, HIDDEN, generator=generator).to(torch.bfloat16).requires_grad_(True)
    weight = torch.randn(VOCAB, HIDDEN, generator=generator).to(torch.bfloat16).requires_grad_(True)
    sequences = torch.randint(VOCAB, (BATCH, SEQ), generator=generator)
    # Left padding on the second row exercises the compact layout and its restoration.
    attention_mask = torch.ones(BATCH, SEQ, dtype=torch.long)
    attention_mask[1, :2] = 0
    sequences[1, :2] = 0
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    micro_batch = mmw.MegatronPolicyMicroBatch(
        sequences=sequences,
        attention_mask=attention_mask,
        position_ids=position_ids,
        num_actions=NUM_ACTIONS,
        old_action_log_probs=torch.zeros(BATCH, NUM_ACTIONS),
        base_action_log_probs=None,
        advantages=torch.randn(BATCH, NUM_ACTIONS, generator=generator),
        loss_mask=torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
        rollout_action_logprobs=-torch.rand(BATCH, NUM_ACTIONS, generator=generator),
        response_span_tags=None,
        sample_offset=3,
    )
    chunk = _FakeGPTChunk(embedding, _OutputLayer(weight, group))
    return chunk, micro_batch


def _wrapper(chunk, *, fp32_output_projection, chunk_size):
    wrapper = mmw.MegatronModelWrapper.__new__(mmw.MegatronModelWrapper)
    wrapper.cfg = _cfg()
    wrapper.actor_module = [chunk]
    wrapper.actor_optimizer = None
    wrapper.policy_loss_fn = _logprob_loss
    wrapper.use_sample_packing = False
    wrapper._logprob_chunk_size = chunk_size
    wrapper.vocabulary_size = VOCAB
    wrapper.fp32_output_projection = fp32_output_projection
    return wrapper


def _train(fp32_output_projection, group, *, chunk_size=3, temperature=0.8):
    chunk, micro_batch = _model_and_batch(group)
    _wrapper(chunk, fp32_output_projection=fp32_output_projection, chunk_size=chunk_size).forward_backward_mini_batch(
        micro_batches=[micro_batch], seq_len=SEQ, micro_batch_size=BATCH, temperature=temperature
    )
    return chunk


def _score(chunk, micro_batch, *, fp32_output_projection, temperature=0.8):
    chunk.training = False
    forward_batch = mmw.MegatronForwardMicroBatch(
        sequences=micro_batch.sequences,
        attention_mask=micro_batch.attention_mask,
        position_ids=micro_batch.position_ids,
        num_actions=NUM_ACTIONS,
    )
    wrapper = _wrapper(chunk, fp32_output_projection=fp32_output_projection, chunk_size=2)
    return wrapper.forward([forward_batch], seq_len=SEQ, micro_batch_size=BATCH, temperature=temperature)


@pytest.mark.parametrize("chunk_size", [2, 3, None])
def test_fp32_projection_gradients_match_fp32_logits_path(megatron_mesh, chunk_size):
    projected = _train(True, megatron_mesh, chunk_size=chunk_size)
    reference = _train(False, megatron_mesh, chunk_size=chunk_size)

    assert projected.output_layer.weight.grad.dtype is torch.bfloat16
    torch.testing.assert_close(projected.output_layer.weight.grad.float(), reference.output_layer.weight.grad.float())
    torch.testing.assert_close(projected.embedding.grad.float(), reference.embedding.grad.float())


def _token_stats(fp32_output_projection, group):
    chunk, micro_batch = _model_and_batch(group)
    stats = []
    _wrapper(chunk, fp32_output_projection=fp32_output_projection, chunk_size=3).forward_backward_mini_batch(
        micro_batches=[micro_batch],
        seq_len=SEQ,
        micro_batch_size=BATCH,
        temperature=0.8,
        token_stats=stats,
        global_step=5,
    )
    (batch,) = stats
    return micro_batch, batch


def test_fp32_projection_token_stats_match_fp32_logits_path(megatron_mesh):
    micro_batch, projected = _token_stats(True, megatron_mesh)
    _, reference = _token_stats(False, megatron_mesh)

    assert projected.global_step == 5
    for name in ("sample_index", "position", "token_id", "sampled_is_top1"):
        assert torch.equal(getattr(projected, name), getattr(reference, name))
    for name in ("learner_logprob", "entropy", "top1_margin", "advantage", "rollout_logprob"):
        torch.testing.assert_close(getattr(projected, name), getattr(reference, name), rtol=1e-5, atol=1e-5)
    # sample_offset shifts the rank-local row index; only loss-masked tokens are kept.
    assert sorted(set(projected.sample_index.tolist())) == [3, 4]
    assert len(projected) == int(micro_batch.loss_mask.sum())


def test_fp32_projection_logprobs_match_direct_fp32_reference(megatron_mesh):
    temperature = 0.8
    chunk, micro_batch = _model_and_batch(megatron_mesh)
    scores = _score(chunk, micro_batch, fp32_output_projection=True, temperature=temperature)
    hidden = chunk.embedding.detach()[micro_batch.sequences].float()
    log_probs = (hidden @ chunk.output_layer.weight.detach().float().t() / temperature).log_softmax(-1)
    response = micro_batch.sequences[:, -NUM_ACTIONS:]
    expected = log_probs[:, -NUM_ACTIONS - 1 : -1].gather(-1, response.unsqueeze(-1)).squeeze(-1)

    assert not scores.requires_grad
    torch.testing.assert_close(scores, expected, rtol=1e-5, atol=1e-5)


def test_scoring_forward_matches_fp32_logits_path(megatron_mesh):
    chunk, micro_batch = _model_and_batch(megatron_mesh)
    torch.testing.assert_close(
        _score(chunk, micro_batch, fp32_output_projection=True),
        _score(chunk, micro_batch, fp32_output_projection=False),
        rtol=1e-5,
        atol=1e-5,
    )


def test_fp32_projection_rejects_ftpo_scoring(megatron_mesh):
    chunk, micro_batch = _model_and_batch(megatron_mesh)
    forward_batch = mmw.MegatronForwardMicroBatch(
        sequences=micro_batch.sequences,
        attention_mask=micro_batch.attention_mask,
        position_ids=micro_batch.position_ids,
        num_actions=NUM_ACTIONS,
        ftpo_chosen_mask=torch.ones(BATCH, NUM_ACTIONS, dtype=torch.bool),
    )
    wrapper = _wrapper(chunk, fp32_output_projection=True, chunk_size=2)
    with pytest.raises(ValueError, match="FTPO"):
        wrapper.forward([forward_batch], seq_len=SEQ, micro_batch_size=BATCH)


def test_tied_output_weight_takes_the_fp32_projection_gradient(megatron_mesh):
    generator = torch.Generator().manual_seed(1)
    shared = torch.randn(VOCAB, HIDDEN, generator=generator).to(torch.bfloat16).requires_grad_(True)
    hidden = torch.randn(SEQ, BATCH, HIDDEN, generator=generator).to(torch.bfloat16)
    request = projection_module.FP32ProjectionRequest(
        targets=torch.randint(VOCAB, (BATCH, SEQ), generator=generator),
        vocabulary_size=VOCAB,
        temperature=1.0,
        chunk_size=3,
        entropy_requires_grad=False,
        inference_only=False,
        compute_top1=False,
    )
    output_layer = _OutputLayer(None, megatron_mesh)

    stats = projection_module.fp32_projection_output_processor(
        hidden_states=hidden,
        output_layer=output_layer,
        output_weight=shared,
        context=request,
        config=OmegaConf.create({"use_mup": False, "defer_embedding_wgrad_compute": False}),
    )
    stats[..., 0].sum().backward()

    assert stats.shape == (BATCH, SEQ, 4)
    assert shared.grad is not None and shared.grad.dtype is torch.bfloat16


@pytest.mark.parametrize("dtype_name, expected", [(None, False), ("float32", True)])
def test_uses_fp32_output_projection(dtype_name, expected):
    assert projection_module.uses_fp32_output_projection(dtype_name) is expected


def test_uses_fp32_output_projection_rejects_other_dtypes():
    with pytest.raises(ValueError, match="lm_head_compute_dtype"):
        projection_module.uses_fp32_output_projection("bfloat16")
