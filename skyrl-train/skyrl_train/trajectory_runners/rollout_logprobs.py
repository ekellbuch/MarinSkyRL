"""Gather behavior-policy logprobs for a generated group without failing the run.

A behavior-referenced objective (DPPO, TIS, behavior_clip) needs the rollout
logprob of every trainable token. Harbor cannot always deliver them: an episode
rolled back after ``ContextLengthExceededError`` or a soft timeout keeps its
tokens but loses its rollout details. Before this module the runner raised on
such a trajectory when logprobs were required, and the fully-async generation
worker treats any exception as fatal, so one group ended the whole run.

The contract now: a trainable trajectory without logprobs is masked out of the
loss (``loss_eligible=False``, zero loss mask, zero-filled placeholder logprobs
so every list stays aligned with ``response_token_ids``). A group whose every
trajectory is masked reaches admission as ``FULLY_MASKED`` and is rejected
there; a group with some real logprobs trains on those trajectories only.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Optional, Sequence

from loguru import logger

MISSING_ROLLOUT_LOGPROBS_REASON = "missing_rollout_logprobs"


def mask_missing_rollout_logprobs(output: Any) -> None:
    """Exclude one trajectory from the loss because its rollout logprobs are absent."""
    output.loss_mask = [0] * len(output.evidence.response_token_ids)
    output.disposition = replace(
        output.disposition,
        loss_eligible=False,
        reason=MISSING_ROLLOUT_LOGPROBS_REASON,
    )


def gather_rollout_logprobs(
    outputs: Sequence[Any],
    *,
    required: bool,
    expect_logprobs: bool,
    group_label: str = "",
) -> Optional[List[List[float]]]:
    """Return per-trajectory rollout logprobs aligned with ``response_token_ids``.

    Args:
        outputs: generated trajectories; each exposes ``evidence.behavior_logprobs``,
            ``evidence.response_token_ids``, ``loss_mask`` and ``disposition``.
        required: the policy objective consumes rollout logprobs (DPPO/TIS).
            Trainable trajectories without them are masked, never raised on.
        expect_logprobs: Harbor was asked to collect rollout details, so a
            missing list is worth a warning even when not required.
        group_label: identifies the group in log lines.

    Returns:
        ``None`` when no trajectory carries logprobs and none are required
        (legacy behavior); otherwise one list per trajectory, zero-filled for
        trajectories without logprobs.
    """
    missing = [output for output in outputs if output.evidence.behavior_logprobs is None]
    has_any = len(missing) < len(outputs)

    if not has_any and not required:
        if missing and expect_logprobs:
            logger.error(
                f"Rollout-logprob mode: ALL {len(outputs)} trajectories missing logprobs{group_label}. "
                "This batch cannot use a behavior-referenced policy objective. "
                "Check if Harbor is collecting rollout_details (collect_rollout_details=true) "
                "and if context length errors are preventing logprob collection."
            )
        return None

    masked = 0
    gathered: List[List[float]] = []
    for output in outputs:
        if output.evidence.behavior_logprobs is not None:
            gathered.append(list(output.evidence.behavior_logprobs))
            continue
        if required and any(output.loss_mask):
            mask_missing_rollout_logprobs(output)
            masked += 1
        # Failed and masked trajectories carry no loss, so aligned placeholders
        # cannot affect the objective.
        gathered.append([0.0] * len(output.evidence.response_token_ids))

    if missing and (required or expect_logprobs):
        ids = ", ".join(str(getattr(output, "trajectory_id", "?")) for output in missing[:3])
        more = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
        logger.warning(
            f"Rollout-logprob mode: {len(missing)}/{len(outputs)} trajectories missing logprobs{group_label}; "
            f"{masked} trainable trajectories masked out of the loss "
            f"(reason={MISSING_ROLLOUT_LOGPROBS_REASON}); the rest were already untrainable. "
            f"Likely context-length rollbacks. Trajectories: {ids}{more}."
        )
    return gathered
