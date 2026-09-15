"""Compact, exact tensor fingerprints for distributed diagnostics."""

import hashlib

import torch


def canonical_tensor_fingerprint(tensor: torch.Tensor, chunk_elements: int = 1 << 20) -> dict[str, object]:
    """Hash a tensor's values as canonical contiguous FP32 without returning it.

    The fixed-size host chunks bound temporary CPU memory while preserving the
    exact comparison semantics used by the weight-sync tests.
    """
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")

    flat = tensor.detach().contiguous().view(-1)
    digest = hashlib.sha256()
    for offset in range(0, flat.numel(), chunk_elements):
        chunk = flat[offset : offset + chunk_elements].to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(chunk.numpy().tobytes())

    return {
        "canonical_dtype": "float32",
        "numel": flat.numel(),
        "shape": list(tensor.shape),
        "sha256": digest.hexdigest(),
    }
