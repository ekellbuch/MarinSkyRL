"""Explicit precision control for final language-model projections."""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional

FP32_DTYPE_NAME = "float32"


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

    is_zero3_lm_head = any(hasattr(parameter, "ds_id") for parameter in lm_head.parameters(recurse=False))
    if not is_zero3_lm_head:
        lm_head.float()
    original_forward = lm_head.forward

    def fp32_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        if is_zero3_lm_head:
            bias = lm_head.bias.float() if lm_head.bias is not None else None
            return functional.linear(hidden_states.float(), lm_head.weight.float(), bias)
        return original_forward(hidden_states.float())

    lm_head.forward = fp32_forward
    lm_head._marinskyrl_compute_dtype = dtype_name
    return True


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
        lm_head = getattr(self, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "float"):
            raise TypeError("lm_head_compute_dtype requires a vLLM model with a floating-point lm_head")
        lm_head.float()

    def fp32_compute_logits(self, hidden_states, *args, **kwargs):
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.float()
        return original_compute_logits(self, hidden_states, *args, **kwargs)

    model_class.__init__ = fp32_init
    model_class.compute_logits = fp32_compute_logits
    model_class._marinskyrl_lm_head_compute_dtype = dtype_name
    return True


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
