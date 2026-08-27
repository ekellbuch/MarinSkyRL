"""Explicit precision control for final language-model projections."""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional

FP32_DTYPE_NAME = "float32"
VLLM_LM_HEAD_COMPUTE_DTYPE_ENV = "SKYRL_VLLM_LM_HEAD_COMPUTE_DTYPE"


def configure_hf_lm_head_compute_dtype(model: nn.Module, dtype_name: str | None) -> bool:
    """Run a Hugging Face causal LM's final projection in the requested dtype."""
    if dtype_name is None:
        return False
    if dtype_name != FP32_DTYPE_NAME:
        raise ValueError(f"Unsupported lm_head_compute_dtype: {dtype_name}")

    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if not isinstance(lm_head, nn.Linear):
        raise TypeError("lm_head_compute_dtype requires a torch.nn.Linear output embedding")
    if getattr(lm_head, "_marinskyrl_compute_dtype", None) == dtype_name:
        return False

    def fp32_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        bias = lm_head.bias.float() if lm_head.bias is not None else None
        return functional.linear(hidden_states.float(), lm_head.weight.float(), bias)

    lm_head.forward = fp32_forward
    lm_head._marinskyrl_compute_dtype = dtype_name
    return True


def restore_vllm_lm_head_compute_dtype(model: Any, dtype_name: str | None = None) -> bool:
    """Restore the configured vLLM projection dtype after a weight update."""
    if dtype_name is not None and dtype_name != FP32_DTYPE_NAME:
        raise ValueError(f"Unsupported lm_head_compute_dtype: {dtype_name}")
    candidates = (model, getattr(model, "language_model", None))
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_dtype = dtype_name or getattr(type(candidate), "_marinskyrl_lm_head_compute_dtype", None)
        if candidate_dtype is None:
            continue
        if candidate_dtype != FP32_DTYPE_NAME:
            raise ValueError(f"Unsupported lm_head_compute_dtype: {candidate_dtype}")
        lm_head = getattr(candidate, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "float"):
            lm_head.float()
            return True
    return False


def patch_vllm_model_class_lm_head_compute_dtype(model_class: type, dtype_name: str) -> bool:
    """Run a vLLM model class's final projection in the requested dtype."""
    if dtype_name != FP32_DTYPE_NAME:
        raise ValueError(f"Unsupported lm_head_compute_dtype: {dtype_name}")
    if getattr(model_class, "_marinskyrl_lm_head_compute_dtype", None) == dtype_name:
        return False
    if not hasattr(model_class, "compute_logits"):
        return False

    original_init: Callable[..., None] = model_class.__init__
    original_compute_logits: Callable[..., Any] = model_class.compute_logits

    def fp32_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        if not restore_vllm_lm_head_compute_dtype(self):
            raise TypeError("lm_head_compute_dtype requires a vLLM model with a floating-point lm_head")

    def fp32_compute_logits(self, hidden_states, *args, **kwargs):
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.float()
        return original_compute_logits(self, hidden_states, *args, **kwargs)

    model_class.__init__ = fp32_init
    model_class.compute_logits = fp32_compute_logits
    model_class._marinskyrl_lm_head_compute_dtype = dtype_name
    return True


def configure_vllm_model_instance_lm_head_compute_dtype(model: Any, dtype_name: str) -> None:
    """Configure an already-created vLLM model inside its EngineCore process."""
    candidate = getattr(model, "language_model", None) or model
    patch_vllm_model_class_lm_head_compute_dtype(type(candidate), dtype_name)
    if not restore_vllm_lm_head_compute_dtype(model, dtype_name):
        raise TypeError("lm_head_compute_dtype requires a vLLM model with a floating-point lm_head")


def configure_vllm_qwen3_5_lm_head_compute_dtype(dtype_name: str | None) -> tuple[str, ...]:
    """Configure the Qwen3.5 vLLM implementations before engine construction."""
    if dtype_name is None:
        return ()
    if dtype_name != FP32_DTYPE_NAME:
        raise ValueError(f"Unsupported lm_head_compute_dtype: {dtype_name}")

    patched_classes = []
    for module_name in ("vllm.model_executor.models.qwen3_5", "vllm.model_executor.models.qwen3_5_mtp"):
        try:
            module = __import__(module_name, fromlist=[""])
        except ImportError:
            continue
        for attribute_name in dir(module):
            if not attribute_name.startswith("Qwen3_5"):
                continue
            model_class = getattr(module, attribute_name)
            if isinstance(model_class, type) and patch_vllm_model_class_lm_head_compute_dtype(model_class, dtype_name):
                patched_classes.append(f"{module_name}.{attribute_name}")
    if not patched_classes:
        raise RuntimeError("No Qwen3.5 vLLM model class accepted lm_head_compute_dtype=float32")
    return tuple(sorted(patched_classes))
