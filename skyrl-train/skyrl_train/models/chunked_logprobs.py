"""Per-token statistics computed inside the output projection, one sequence chunk at a time.

The HF causal LM forward ends with ``lm_head(hidden_states)`` and hands back
``[B, S, V]`` logits. The wrapper only ever reduces those logits to per-token
scalars (the label log-probability, the entropy, the top-1 margin), but the
whole tensor is alive between the projection and the reduction: at Qwen3.5's
248,320-entry vocabulary and an FP32 head that is 4 bytes per entry, 32.5 GB
for one 32,768-token sample and 83 GiB at the paper's 90,112-token window.

``ChunkedLogprobHead`` replaces the projection's ``forward`` for the duration
of one wrapper forward. It receives the final hidden states, applies the
original projection to one sequence chunk at a time, reduces that chunk to
the requested scalars, and returns a ``[B, S, 4]`` float32 tensor in place of
the logits. Only one chunk's full-vocabulary logits exist at any time; with
autograd enabled each chunk is checkpointed so backward recomputes it. The
original projection is called unchanged, so ``lm_head_compute_dtype``
semantics (``lm_head_precision``) are preserved exactly.

Working inside the projection rather than after the model call keeps FSDP,
context parallelism, Ulysses sequence parallelism and sample packing
untouched: the hidden states and the labels are whatever the wrapper already
aligned for that rank, and the packed output has the same ``[B, S]`` layout as
the logits it replaces.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from skyrl_train.utils.token_stats import top1_margin_from_logits
from skyrl_train.utils.torch_utils import _EntropyFromLogits, logprobs_from_logits

# Channels of the packed ``[B, S, PACKED_CHANNELS]`` output.
LOGPROB = 0
ENTROPY = 1
TOP1_MARGIN = 2
TOP1_TOKEN = 3
PACKED_CHANNELS = 4


@dataclass(frozen=True)
class _Request:
    labels: torch.Tensor
    temperature: float
    compute_entropy: bool
    entropy_requires_grad: bool
    compute_top1_margin: bool


class ChunkedLogprobHead:
    """Stand-in ``forward`` for an output projection that returns per-token scalars.

    Outside a :meth:`request` context the projection behaves exactly as before,
    so model loading, weight export and any other caller see plain logits.
    """

    def __init__(self, projection: Callable[[torch.Tensor], torch.Tensor], chunk_size: int) -> None:
        if chunk_size <= 0:
            raise ValueError(f"logprob_chunk_size must be a positive integer, got {chunk_size}")
        self._projection = projection
        self.chunk_size = int(chunk_size)
        self._request: Optional[_Request] = None

    @classmethod
    def install(cls, model: nn.Module, chunk_size: int) -> "ChunkedLogprobHead":
        """Replace ``model``'s output projection ``forward`` and return the head."""
        lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if lm_head is None:
            raise TypeError("logprob_chunk_size requires a model with an output embedding")
        head = cls(lm_head.forward, chunk_size)
        lm_head.forward = head
        lm_head._marinskyrl_chunked_logprob_head = head
        return head

    @contextlib.contextmanager
    def request(
        self,
        labels: torch.Tensor,
        *,
        temperature: float = 1.0,
        compute_entropy: bool = False,
        entropy_requires_grad: bool = True,
        compute_top1_margin: bool = False,
    ) -> Iterator[None]:
        """Make the next projection call return packed per-token scalars for ``labels``."""
        if self._request is not None:
            raise RuntimeError("ChunkedLogprobHead request contexts do not nest")
        self._request = _Request(
            labels=labels,
            temperature=float(temperature),
            compute_entropy=bool(compute_entropy),
            entropy_requires_grad=bool(entropy_requires_grad),
            compute_top1_margin=bool(compute_top1_margin),
        )
        try:
            yield
        finally:
            self._request = None

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        request = self._request
        if request is None:
            return self._projection(hidden_states)
        if hidden_states.dim() != 3:
            raise ValueError(f"expected [batch, seq, hidden] hidden states, got {tuple(hidden_states.shape)}")
        labels = request.labels
        if labels.shape != hidden_states.shape[:2]:
            raise ValueError(
                f"labels {tuple(labels.shape)} do not match hidden states {tuple(hidden_states.shape[:2])}"
            )
        use_checkpoint = torch.is_grad_enabled() and hidden_states.requires_grad
        chunks = []
        for start in range(0, hidden_states.size(1), self.chunk_size):
            end = min(start + self.chunk_size, hidden_states.size(1))
            hidden_chunk = hidden_states[:, start:end]
            label_chunk = labels[:, start:end]
            # ``request`` travels as an argument: with checkpointing the chunk is
            # recomputed during backward, after the request context has exited.
            if use_checkpoint:
                packed = checkpoint(self._reduce_chunk, hidden_chunk, label_chunk, request, use_reentrant=False)
            else:
                packed = self._reduce_chunk(hidden_chunk, label_chunk, request)
            chunks.append(packed)
        return torch.cat(chunks, dim=1)

    def _reduce_chunk(self, hidden_chunk: torch.Tensor, label_chunk: torch.Tensor, request: _Request) -> torch.Tensor:
        logits = self._projection(hidden_chunk)
        if request.temperature != 1.0:
            logits = logits / request.temperature
        # ``inplace_backward`` would let the flash-attn cross entropy overwrite
        # ``logits`` during backward; the entropy backward reads the same tensor.
        log_probs = logprobs_from_logits(logits, label_chunk, inplace_backward=False).float()
        packed = [log_probs]
        if request.compute_entropy:
            if request.entropy_requires_grad:
                entropy = _EntropyFromLogits.apply(logits, None)
            else:
                with torch.no_grad():
                    entropy = _EntropyFromLogits.apply(logits.detach(), None)
            packed.append(entropy.float())
        else:
            packed.append(torch.zeros_like(log_probs))
        if request.compute_top1_margin:
            margin, top1 = top1_margin_from_logits(logits)
            packed.extend((margin, top1.float()))
        else:
            packed.extend((torch.zeros_like(log_probs), torch.zeros_like(log_probs)))
        return torch.stack(packed, dim=-1)


def unpack_per_token(
    packed: torch.Tensor, *, entropy_requires_grad: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a packed ``[B, S, 4]`` head output into logprob, entropy, margin, top-1 token.

    The packed tensor carries one autograd graph, so an entropy that was computed
    without grad is detached here to match the full-logits path.
    """
    if packed.dim() != 3 or packed.size(-1) != PACKED_CHANNELS:
        raise ValueError(f"expected a [batch, seq, {PACKED_CHANNELS}] packed tensor, got {tuple(packed.shape)}")
    entropy = packed[..., ENTROPY]
    if not entropy_requires_grad:
        entropy = entropy.detach()
    return (
        packed[..., LOGPROB],
        entropy,
        packed[..., TOP1_MARGIN].detach(),
        packed[..., TOP1_TOKEN].detach().to(torch.long),
    )
