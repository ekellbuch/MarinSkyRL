"""Shared state records for fully asynchronous rollout generation."""

from dataclasses import dataclass, field
from typing import Any, List, Protocol

from skyrl_train.trajectory_runners.base import TrajectoryBatch


@dataclass
class GenerationAttempt:
    """Stable identity and callback-owned state for one rollout submission."""

    task_id: str
    selection_source: str
    optimizer_step_at_selection: int
    callback_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedOutputGroup:
    """One prompt's rollout samples and the metadata needed to retry them."""

    trajectory_batch: TrajectoryBatch
    uid: str
    earliest_model_step: int
    source_prompts: List[dict]
    generation_attempt: GenerationAttempt


@dataclass
class GenerationBufferState:
    """Completed rollout groups and pending source-prompt retries in a checkpoint."""

    completed_groups: List[GeneratedOutputGroup]
    retry_prompts: List[List[dict]]

    def pending_uids(self) -> set[str]:
        """Return dataset UIDs whose work survives in this checkpoint."""
        uids = set()
        for group in self.completed_groups:
            if not isinstance(group.uid, str):
                raise ValueError("completed generation group uid must be a string")
            uids.add(group.uid)
        for prompts in self.retry_prompts:
            for prompt in prompts:
                uid = prompt.get("uid")
                if not isinstance(uid, str):
                    raise ValueError("retry prompt uid must be a string")
                uids.add(uid)
        return uids


class GenerationQueuesProvider(Protocol):
    """Live generation queues that can provide checkpoint state."""

    def snapshot(self) -> GenerationBufferState: ...
