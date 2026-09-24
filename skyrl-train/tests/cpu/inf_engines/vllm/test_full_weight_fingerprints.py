import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from skyrl_train.inference_engines.vllm.vllm_engine import WorkerWrap
from skyrl_train.utils.tensor_fingerprint import canonical_tensor_fingerprint


def _parameter(values):
    return nn.Parameter(torch.tensor(values, dtype=torch.bfloat16))


def _weight_module(parameter):
    module = nn.Module()
    module.register_parameter("weight", parameter)
    return module


def test_fingerprint_named_weights_reconstructs_qwen35_stacked_parameters():
    q = _parameter([[1, 2, 3], [4, 5, 6]])
    k = _parameter([[7, 8, 9]])
    v = _parameter([[10, 11, 12]])
    gate = _parameter([[13, 14, 15], [16, 17, 18]])
    up = _parameter([[19, 20, 21], [22, 23, 24]])
    in_proj_qkv = _parameter([[25, 26, 27], [28, 29, 30]])
    in_proj_z = _parameter([[31, 32, 33]])
    in_proj_b = _parameter([[34, 35, 36]])
    in_proj_a = _parameter([[37, 38, 39]])

    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.qkv_proj = _weight_module(_parameter(torch.cat((q, k, v)).tolist()))
    layer.mlp = nn.Module()
    layer.mlp.gate_up_proj = _weight_module(_parameter(torch.cat((gate, up)).tolist()))
    layer.linear_attn = nn.Module()
    layer.linear_attn.in_proj_qkvz = _weight_module(_parameter(torch.cat((in_proj_qkv, in_proj_z)).tolist()))
    layer.linear_attn.in_proj_ba = _weight_module(_parameter(torch.cat((in_proj_b, in_proj_a)).tolist()))
    text_model = nn.Module()
    text_model.layers = nn.ModuleList([layer])
    language_model = nn.Module()
    language_model.model = text_model
    language_model.config = SimpleNamespace(tie_word_embeddings=False)
    model = nn.Module()
    model.language_model = language_model

    wrapper = object.__new__(WorkerWrap)
    wrapper.model_runner = SimpleNamespace(model=model)
    expected = {
        "model.language_model.layers.0.self_attn.q_proj.weight": q,
        "model.language_model.layers.0.self_attn.k_proj.weight": k,
        "model.language_model.layers.0.self_attn.v_proj.weight": v,
        "model.language_model.layers.0.mlp.gate_proj.weight": gate,
        "model.language_model.layers.0.mlp.up_proj.weight": up,
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": in_proj_qkv,
        "model.language_model.layers.0.linear_attn.in_proj_z.weight": in_proj_z,
        "model.language_model.layers.0.linear_attn.in_proj_b.weight": in_proj_b,
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": in_proj_a,
    }
    expected_shapes = {name: list(tensor.shape) for name, tensor in expected.items()}

    actual = wrapper.fingerprint_named_weights(list(expected), expected_shapes)

    for name, tensor in expected.items():
        assert actual[name]["found"]
        assert actual[name]["mode"] == "stacked"
        assert actual[name]["fingerprint"] == canonical_tensor_fingerprint(tensor)
        assert "tensor" not in actual[name]


def test_fingerprint_named_weights_restores_equivalent_direct_shape():
    convolution = _parameter(torch.arange(8).reshape(4, 2).tolist())
    layer = nn.Module()
    layer.linear_attn = nn.Module()
    layer.linear_attn.conv1d = _weight_module(convolution)
    text_model = nn.Module()
    text_model.layers = nn.ModuleList([layer])
    language_model = nn.Module()
    language_model.model = text_model
    language_model.config = SimpleNamespace(tie_word_embeddings=False)
    model = nn.Module()
    model.language_model = language_model

    wrapper = object.__new__(WorkerWrap)
    wrapper.model_runner = SimpleNamespace(model=model)
    name = "model.language_model.layers.0.linear_attn.conv1d.weight"
    expected_shape = [4, 1, 2]

    actual = wrapper.fingerprint_named_weights([name], {name: expected_shape})[name]

    assert actual["found"]
    assert actual["actual_shape"] == expected_shape
    assert actual["fingerprint"] == canonical_tensor_fingerprint(convolution.reshape(expected_shape))
    assert "tensor" not in actual


class _LogitsModel(nn.Module):
    def compute_logits(self, hidden_states):
        projection = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
            device=hidden_states.device,
        )
        return hidden_states.float() @ projection


def test_head_input_capture_restores_class_descriptor():
    model = _LogitsModel()
    wrapper = object.__new__(WorkerWrap)
    wrapper.model_runner = SimpleNamespace(model=model)
    hidden_states = torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16)

    assert "compute_logits" not in model.__dict__
    wrapper.begin_head_input_capture(selected_token=1)
    expected_logits = model.compute_logits(hidden_states)
    captures = wrapper.end_head_input_capture()

    assert "compute_logits" not in model.__dict__
    assert model.compute_logits(hidden_states).equal(expected_logits)
    assert captures == [
        {
            "head_input": [2.0, 3.0],
            "head_input_dtype": "torch.bfloat16",
            "head_input_shape": [2],
            "compute_logits_input_shape": [1, 2],
            "logits_dtype": "torch.float32",
            "logits_shape": [1, 3],
            "selected_logit": 19.0,
            "logsumexp": expected_logits[0].logsumexp(dim=-1).item(),
            "selected_logprob": (expected_logits[0, 1] - expected_logits[0].logsumexp(dim=-1)).item(),
            "layer_trace": [],
        }
    ]


def test_head_input_capture_restores_instance_callable():
    model = _LogitsModel()
    original_compute_logits = model.compute_logits

    def instance_compute_logits(hidden_states):
        return original_compute_logits(hidden_states) + 1.0

    model.compute_logits = instance_compute_logits
    wrapper = object.__new__(WorkerWrap)
    wrapper.model_runner = SimpleNamespace(model=model)

    wrapper.begin_head_input_capture(selected_token=0)
    model.compute_logits(torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16))
    wrapper.end_head_input_capture()

    assert model.__dict__["compute_logits"] is instance_compute_logits


def test_report_runtime_installation_matches_live_module(capsys):
    from skyrl_train.inference_engines.vllm import vllm_engine

    expected_sha256 = hashlib.sha256(Path(vllm_engine.__file__).read_bytes()).hexdigest()
    wrapper = object.__new__(WorkerWrap)

    payload = wrapper.report_runtime_installation(expected_sha256)

    assert payload["matches_checkout"]
    assert payload["vllm_engine_file"] == str(Path(vllm_engine.__file__).resolve())
    assert "SKYRL_ENGINECORE_RUNTIME" in capsys.readouterr().out


def test_report_runtime_installation_rejects_stale_module():
    wrapper = object.__new__(WorkerWrap)

    with pytest.raises(RuntimeError, match="EngineCore MarinSkyRL source mismatch"):
        wrapper.report_runtime_installation("0" * 64)
