"""The chunked projection returns what the full-logits path returns, values and gradients."""

import copy

import pytest
import torch
from skyrl_train.model_wrapper import HFModelWrapper
from skyrl_train.models.chunked_logprobs import ChunkedLogprobHead, unpack_per_token
from transformers import Qwen2Config, Qwen2ForCausalLM

VOCAB = 97
SEQ = 13
NUM_ACTIONS = 7


def _tiny_causal_lm() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    return Qwen2ForCausalLM(config)


def _wrap(model, **kwargs) -> HFModelWrapper:
    return HFModelWrapper(copy.deepcopy(model), bf16=False, training_strategy="fsdp2", **kwargs)


def _batch():
    torch.manual_seed(1)
    sequences = torch.randint(0, VOCAB, (2, SEQ))
    attention_mask = torch.ones_like(sequences)
    attention_mask[1, :3] = 0  # left padding on the second row
    return sequences, attention_mask


def _forward(wrapper: HFModelWrapper, **kwargs):
    sequences, attention_mask = _batch()
    return wrapper(
        sequences,
        NUM_ACTIONS,
        attention_mask=attention_mask,
        temperature=0.7,
        return_output=True,
        compute_entropy=True,
        compute_top1_margin=True,
        **kwargs,
    )


@pytest.mark.parametrize("chunk_size", [SEQ, 4, 5, 1])
def test_chunked_forward_matches_full_logits_path(chunk_size):
    model = _tiny_causal_lm()
    reference = _wrap(model)
    chunked = _wrap(model, logprob_chunk_size=chunk_size)

    with torch.no_grad():
        ref_logprobs, ref_output = _forward(reference)
        chunk_logprobs, chunk_output = _forward(chunked)

    torch.testing.assert_close(chunk_logprobs, ref_logprobs)
    torch.testing.assert_close(chunk_output["entropy"], ref_output["entropy"].to(chunk_output["entropy"].dtype))
    torch.testing.assert_close(chunk_output["top1_margin"], ref_output["top1_margin"])
    assert torch.equal(chunk_output["top1_token"], ref_output["top1_token"])
    assert chunk_output["logits"].shape == (2, SEQ, 4)


def test_chunked_forward_matches_gradients_of_a_dppo_style_loss():
    model = _tiny_causal_lm()
    reference = _wrap(model)
    chunked = _wrap(model, logprob_chunk_size=4)
    torch.manual_seed(2)
    advantages = torch.randn(2, NUM_ACTIONS)
    loss_mask = torch.ones(2, NUM_ACTIONS)
    loss_mask[0, -2:] = 0

    def loss_of(wrapper):
        wrapper.zero_grad(set_to_none=True)
        logprobs, output = _forward(wrapper, entropy_requires_grad=True)
        entropy = output["entropy"][:, -NUM_ACTIONS - 1 : -1]
        loss = -(logprobs * advantages * loss_mask).sum() + 0.1 * (entropy * loss_mask).sum()
        loss.backward()
        return loss.detach()

    torch.testing.assert_close(loss_of(chunked), loss_of(reference))
    for (name, ref_param), (_, chunk_param) in zip(reference.named_parameters(), chunked.named_parameters()):
        assert ref_param.grad is not None, name
        torch.testing.assert_close(chunk_param.grad, ref_param.grad, msg=name)


def test_chunked_head_goes_through_the_float32_projection():
    model = _tiny_causal_lm().to(torch.bfloat16)
    reference = _wrap(model, lm_head_compute_dtype="float32")
    chunked = _wrap(model, lm_head_compute_dtype="float32", logprob_chunk_size=5)
    lm_head = chunked.model.get_output_embeddings()
    assert lm_head._marinskyrl_compute_dtype == "float32"
    assert lm_head.weight.dtype == torch.bfloat16
    assert isinstance(lm_head.forward, ChunkedLogprobHead)

    with torch.no_grad():
        ref_logprobs, _ = _forward(reference)
        chunk_logprobs, _ = _forward(chunked)

    assert ref_logprobs.dtype == chunk_logprobs.dtype == torch.float32
    torch.testing.assert_close(chunk_logprobs, ref_logprobs)


def test_projection_returns_plain_logits_outside_a_request():
    model = _tiny_causal_lm()
    chunked = _wrap(model, logprob_chunk_size=4)
    hidden = torch.randn(1, 3, model.config.hidden_size)

    logits = chunked.model.get_output_embeddings()(hidden)

    assert logits.shape == (1, 3, VOCAB)
    torch.testing.assert_close(logits, model.get_output_embeddings()(hidden))


def test_entropy_without_grad_is_detached_and_logprob_keeps_grad():
    model = _tiny_causal_lm()
    chunked = _wrap(model, logprob_chunk_size=4)

    logprobs, output = _forward(chunked, entropy_requires_grad=False)

    assert logprobs.requires_grad
    assert not output["entropy"].requires_grad
    assert not output["top1_margin"].requires_grad


def test_unpack_rejects_other_shapes():
    with pytest.raises(ValueError, match="packed"):
        unpack_per_token(torch.zeros(2, 3, VOCAB))


def test_request_contexts_do_not_nest():
    head = ChunkedLogprobHead(lambda hidden: hidden, chunk_size=2)
    labels = torch.zeros(1, 2, dtype=torch.long)
    with head.request(labels):
        with pytest.raises(RuntimeError, match="nest"):
            with head.request(labels):
                pass


def test_head_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError, match="positive"):
        ChunkedLogprobHead(lambda hidden: hidden, chunk_size=0)
