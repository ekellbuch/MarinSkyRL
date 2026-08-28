import builtins
import inspect
from types import SimpleNamespace

import pytest
import torch
from skyrl_train.models.lm_head_precision import (
    _record_vllm_reload_metadata,
    configure_hf_lm_head_compute_dtype,
    configure_vllm_model_instance_lm_head_compute_dtype,
    patch_vllm_model_class_lm_head_compute_dtype,
    ensure_vllm_lm_head_compute_dtype,
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

    original_signature = inspect.signature(TinyVLLMModel.__init__)
    patch_vllm_model_class_lm_head_compute_dtype(TinyVLLMModel, "float32")
    model = TinyVLLMModel()
    output = model.compute_logits(torch.ones((1, 3), dtype=torch.bfloat16))

    assert model.lm_head.weight.dtype == torch.float32
    assert output.dtype == torch.float32
    assert inspect.signature(TinyVLLMModel.__init__) == original_signature
    patch_vllm_model_class_lm_head_compute_dtype(TinyVLLMModel, None)


@pytest.mark.parametrize("first_dtype,second_dtype", [("float32", None), (None, "float32")])
def test_vllm_class_precision_configuration_is_independent_of_construction_order(first_dtype, second_dtype):
    class TinyVLLMModel:
        def __init__(self) -> None:
            self.lm_head = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)

        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.lm_head(hidden_states)

    original_init = TinyVLLMModel.__init__
    original_compute_logits = TinyVLLMModel.compute_logits

    models = []
    for dtype_name in (first_dtype, second_dtype):
        patch_vllm_model_class_lm_head_compute_dtype(TinyVLLMModel, dtype_name)
        model = TinyVLLMModel()
        models.append((model, dtype_name))
        output = model.compute_logits(torch.ones((1, 3), dtype=torch.bfloat16))
        expected_dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16
        assert model.lm_head.weight.dtype == expected_dtype
        assert output.dtype == expected_dtype

    if second_dtype is None:
        assert TinyVLLMModel.__init__ is original_init
        assert TinyVLLMModel.compute_logits is original_compute_logits
    for model, dtype_name in models:
        output = model.compute_logits(torch.ones((1, 3), dtype=torch.bfloat16))
        expected_dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16
        assert output.dtype == expected_dtype
    patch_vllm_model_class_lm_head_compute_dtype(TinyVLLMModel, None)


def test_vllm_tied_lm_head_survives_two_complete_loads_without_stale_storage(monkeypatch):
    class TinyVLLMModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(tie_word_embeddings=True, vocab_size=3)
            self.embed_tokens = nn.Embedding(5, 4, dtype=torch.bfloat16)
            self.lm_head = self.embed_tokens

        def load_weights(self, weights) -> set[str]:
            loaded = set()
            for name, weight in weights:
                assert name == "model.embed_tokens.weight"
                self.embed_tokens.weight.data.copy_(weight)
                loaded.add(name)
            return loaded

        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return functional.linear(hidden_states, self.lm_head.weight)[:, : self.config.vocab_size]

    language_model = TinyVLLMModel()
    shell = type("TinyShell", (), {"language_model": language_model})()
    monkeypatch.setattr(
        "skyrl_train.models.lm_head_precision._record_vllm_reload_metadata",
        lambda module: None,
    )
    hidden_states = torch.tensor(
        [[0.5, -1.0, 1.5, 2.0], [1.0, 0.25, -0.5, 0.75]],
        dtype=torch.bfloat16,
    )
    selected_tokens = torch.tensor([0, 2])
    loads = (
        torch.arange(20, dtype=torch.float32).reshape(5, 4).to(torch.bfloat16) / 8,
        torch.arange(20, 40, dtype=torch.float32).reshape(5, 4).to(torch.bfloat16) / 16,
    )
    head_identity = None

    for update_index, weight in enumerate(loads):
        loaded = language_model.load_weights([("model.embed_tokens.weight", weight)])
        assert loaded == {"model.embed_tokens.weight"}
        configure_vllm_model_instance_lm_head_compute_dtype(shell, "float32")

        assert language_model.embed_tokens.weight.dtype == torch.bfloat16
        assert language_model.lm_head.weight.dtype == torch.float32
        assert language_model.lm_head is not language_model.embed_tokens
        assert language_model.lm_head.weight.shape == (5, 4)
        torch.testing.assert_close(language_model.embed_tokens.weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(language_model.lm_head.weight, weight.float(), rtol=0, atol=0)

        current_identity = (
            id(language_model.lm_head),
            id(language_model.lm_head.weight),
            language_model.lm_head.weight.data_ptr(),
        )
        if update_index == 0:
            head_identity = current_identity
        else:
            assert current_identity == head_identity

        reference_logits = functional.linear(hidden_states.float(), weight.float())[:, :3]
        actual_logits = language_model.compute_logits(hidden_states)
        assert actual_logits.dtype == torch.float32
        assert actual_logits.shape == (2, 3)
        torch.testing.assert_close(actual_logits, reference_logits, rtol=0, atol=0)
        assert torch.equal(actual_logits.argmax(dim=-1), reference_logits.argmax(dim=-1))

        actual_logprobs = actual_logits.log_softmax(dim=-1)
        reference_logprobs = reference_logits.log_softmax(dim=-1)
        rows = torch.arange(selected_tokens.numel())
        torch.testing.assert_close(
            actual_logprobs[rows, selected_tokens],
            reference_logprobs[rows, selected_tokens],
            rtol=1e-5,
            atol=1e-5,
        )


def test_vllm_tied_lm_head_rejects_missing_reload_metadata_support(monkeypatch):
    original_import = builtins.__import__

    def reject_reload_module(name, *args, **kwargs):
        if name == "vllm.model_executor.model_loader.reload":
            raise ImportError("reload module unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_reload_module)

    with pytest.raises(RuntimeError, match="require vLLM layerwise reload metadata"):
        _record_vllm_reload_metadata(nn.Linear(2, 2))


def test_vllm_lm_head_restore_waits_for_layerwise_reload_to_materialize_weights():
    class IncompleteEmbedding(nn.Module):
        pass

    language_model = type(
        "IncompleteVLLMModel",
        (),
        {
            "_marinskyrl_lm_head_compute_dtype": "float32",
            "_marinskyrl_tied_projection": True,
        },
    )()
    language_model.embed_tokens = IncompleteEmbedding()
    language_model.lm_head = IncompleteEmbedding()
    shell = type("IncompleteShell", (), {"language_model": language_model})()

    assert not ensure_vllm_lm_head_compute_dtype(shell, "float32")


def test_vllm_tied_lm_head_registers_two_layerwise_reloads():
    reload_module = pytest.importorskip("vllm.model_executor.model_loader.reload")

    class TinyReloadableModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=True, vocab_size=3)
            self.embed_tokens = nn.Embedding(5, 4, dtype=torch.bfloat16)
            self.lm_head = self.embed_tokens

        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return functional.linear(hidden_states, self.lm_head.weight)[:, : self.config.vocab_size]

    def load_parameter(module: nn.Module, value: torch.Tensor) -> None:
        parameter = module.weight
        loader = getattr(parameter, "weight_loader", None)
        if loader is None:
            parameter.data.copy_(value)
        else:
            loader(parameter, value)

    model = TinyReloadableModel()
    reload_module.record_metadata_for_reloading(model)
    configure_vllm_model_instance_lm_head_compute_dtype(model, "float32")
    head_identity = (id(model.lm_head), id(model.lm_head.weight), model.lm_head.weight.data_ptr())
    loads = (
        torch.arange(20, dtype=torch.float32).reshape(5, 4).to(torch.bfloat16) / 8,
        torch.arange(20, 40, dtype=torch.float32).reshape(5, 4).to(torch.bfloat16) / 16,
    )

    for weight in loads:
        reload_module.initialize_layerwise_reload(model)
        load_parameter(model.embed_tokens, weight)
        load_parameter(model.lm_head, weight.float())
        reload_module.finalize_layerwise_reload(model, SimpleNamespace(dtype=torch.bfloat16))
        configure_vllm_model_instance_lm_head_compute_dtype(model, "float32")

        assert (id(model.lm_head), id(model.lm_head.weight), model.lm_head.weight.data_ptr()) == head_identity
        torch.testing.assert_close(model.embed_tokens.weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(model.lm_head.weight, weight.float(), rtol=0, atol=0)
