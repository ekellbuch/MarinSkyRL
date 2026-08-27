import torch
from skyrl_train.models.lm_head_precision import (
    configure_hf_lm_head_compute_dtype,
    configure_vllm_model_instance_lm_head_compute_dtype,
    patch_vllm_model_class_lm_head_compute_dtype,
)
from torch import nn
from torch.nn import functional


class TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_head = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head


def test_hf_lm_head_float32_compute_preserves_bf16_parameters_and_backpropagates():
    model = TinyCausalLM()
    hidden_states = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.bfloat16, requires_grad=True)
    expected = functional.linear(hidden_states.float(), model.lm_head.weight.float())

    configure_hf_lm_head_compute_dtype(model, "float32")
    actual = model.lm_head(hidden_states)
    actual.sum().backward()

    assert actual.dtype == torch.float32
    assert model.lm_head.weight.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)
    assert model.lm_head.weight.grad is not None
    assert hidden_states.grad is not None


def test_vllm_lm_head_float32_compute_casts_parameters_and_activations():
    class TinyVLLMModel:
        def __init__(self) -> None:
            self.lm_head = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)

        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.lm_head(hidden_states)

    patch_vllm_model_class_lm_head_compute_dtype(TinyVLLMModel, "float32")
    model = TinyVLLMModel()
    output = model.compute_logits(torch.ones((1, 3), dtype=torch.bfloat16))

    assert model.lm_head.weight.dtype == torch.float32
    assert output.dtype == torch.float32


def test_vllm_lm_head_float32_compute_is_restored_after_tied_weight_load():
    class TinyVLLMModel:
        def __init__(self) -> None:
            self.embed_tokens = nn.Embedding(2, 3, dtype=torch.bfloat16)
            self.lm_head = self.embed_tokens

        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return functional.linear(hidden_states, self.lm_head.weight)

    language_model = TinyVLLMModel()
    shell = type("TinyShell", (), {"language_model": language_model})()

    configure_vllm_model_instance_lm_head_compute_dtype(shell, "float32")
    output = language_model.compute_logits(torch.ones((1, 3), dtype=torch.bfloat16))

    assert language_model.embed_tokens.weight.dtype == torch.float32
    assert language_model.lm_head.weight.dtype == torch.float32
    assert output.dtype == torch.float32
