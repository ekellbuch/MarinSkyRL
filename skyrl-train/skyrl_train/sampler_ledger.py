"""Durable prompt-selection ledger and optional external-readiness gate."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

DEFAULT_READY_TIMEOUT_SECONDS = 1800.0
DEFAULT_READY_POLL_SECONDS = 0.5
DATASET_SELECTION_SOURCE = "dataset"
STALE_RETRY_SELECTION_SOURCE = "stale_retry"
SELECTION_SOURCES = frozenset({DATASET_SELECTION_SOURCE, STALE_RETRY_SELECTION_SOURCE})
SAMPLER_ATTEMPT_ID_KEY = "_sampler_attempt_id"


class SamplerLedger:
    """Record actual async sampler events before rollout submission.

    When ``ready_dir`` is configured, a selected prompt does not reach the
    trajectory runner until an external preflight has atomically published a
    matching ``<task_id>.json`` marker. This lets image preparation follow the
    real async sampler, including speculative candidates and stale retries,
    without warming the complete dataset.
    """

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        path: Path | None,
        *,
        ready_dir: Path | None = None,
        ready_timeout_seconds: float = DEFAULT_READY_TIMEOUT_SECONDS,
        ready_poll_seconds: float = DEFAULT_READY_POLL_SECONDS,
    ) -> None:
        if path is None and ready_dir is not None:
            raise ValueError("sampler readiness requires a sampler ledger path")
        if ready_timeout_seconds <= 0:
            raise ValueError("sampler ready_timeout_seconds must be positive")
        if ready_poll_seconds <= 0:
            raise ValueError("sampler ready_poll_seconds must be positive")
        if path is not None and not path.is_absolute():
            raise ValueError("sampler ledger path must be absolute")
        if ready_dir is not None and not ready_dir.is_absolute():
            raise ValueError("sampler ready_dir must be absolute")

        self.path = path
        self.ready_dir = ready_dir
        self.ready_timeout_seconds = ready_timeout_seconds
        self.ready_poll_seconds = ready_poll_seconds
        self._lock = Lock()
        self._next_attempt_id = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._next_attempt_id = self._recover_next_attempt_id()
        if self.ready_dir is not None:
            self.ready_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, cfg: DictConfig | None) -> "SamplerLedger":
        values = OmegaConf.to_container(cfg, resolve=True) if cfg is not None else {}
        assert isinstance(values, dict)
        path_value = values.get("path")
        ready_dir_value = values.get("ready_dir")
        return cls(
            Path(path_value) if path_value else None,
            ready_dir=Path(ready_dir_value) if ready_dir_value else None,
            ready_timeout_seconds=float(values.get("ready_timeout_seconds", DEFAULT_READY_TIMEOUT_SECONDS)),
            ready_poll_seconds=float(values.get("ready_poll_seconds", DEFAULT_READY_POLL_SECONDS)),
        )

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def _recover_next_attempt_id(self) -> int:
        assert self.path is not None
        if not self.path.exists():
            return 0
        attempt_ids = []
        for line_number, line in enumerate(self.path.read_text().splitlines(), start=1):
            if not line:
                continue
            record = json.loads(line)
            if record.get("schema_version") != self._SCHEMA_VERSION:
                raise ValueError(f"unsupported sampler ledger schema on line {line_number}")
            if record.get("event") == "selected":
                attempt_id = record.get("attempt_id")
                if not isinstance(attempt_id, int) or attempt_id < 0:
                    raise ValueError(f"invalid sampler attempt ID on line {line_number}")
                attempt_ids.append(attempt_id)
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("sampler ledger contains duplicate attempt IDs")
        return max(attempt_ids, default=-1) + 1

    def _append(self, record: Mapping[str, Any]) -> None:
        if self.path is None:
            return
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())

    def record_selection(
        self,
        prompts: Sequence[Mapping[str, Any]],
        *,
        optimizer_step: int,
        selection_source: str,
    ) -> int | None:
        """Append a selection and return its attempt ID, or ``None`` when disabled."""
        if not self.enabled:
            return None
        if len(prompts) != 1:
            raise ValueError("fully async sampler ledger expects one prompt per selection")
        task_id = prompts[0].get("uid")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("sampled prompt must have a non-empty uid")
        if selection_source not in SELECTION_SOURCES:
            raise ValueError(f"unknown sampler selection source: {selection_source}")
        with self._lock:
            attempt_id = self._next_attempt_id
            self._next_attempt_id += 1
        self._append(
            {
                "attempt_id": attempt_id,
                "event": "selected",
                "optimizer_step_at_selection": int(optimizer_step),
                "schema_version": self._SCHEMA_VERSION,
                "selection_source": selection_source,
                "task_id": task_id,
            }
        )
        return attempt_id

    def record_outcome(
        self,
        attempt_id: int | None,
        *,
        optimizer_step: int,
        outcome: str,
        task_id: str,
    ) -> None:
        if attempt_id is None:
            return
        self._append(
            {
                "attempt_id": attempt_id,
                "event": "outcome",
                "optimizer_step_at_outcome": int(optimizer_step),
                "outcome": outcome,
                "schema_version": self._SCHEMA_VERSION,
                "task_id": task_id,
            }
        )

    async def wait_until_ready(self, task_id: str) -> None:
        if self.ready_dir is None:
            return
        marker = self.ready_dir / f"{task_id}.json"
        deadline = time.monotonic() + self.ready_timeout_seconds
        while True:
            try:
                record = json.loads(marker.read_text())
            except FileNotFoundError:
                record = None
            if record is not None:
                if record.get("task_id") != task_id or record.get("status") != "ready":
                    raise ValueError(f"invalid sampler readiness marker: {marker}")
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for sampler readiness marker: {marker}")
            await asyncio.sleep(self.ready_poll_seconds)
