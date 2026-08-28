import asyncio
import json
from pathlib import Path

import pytest

from skyrl_train.sampler_ledger import SamplerLedger
from skyrl_train.trajectory_runners.harbor.dataset import TerminalBenchTaskDataset


def test_sampler_ledger_records_selection_and_outcome(tmp_path):
    ledger_path = tmp_path / "sampler.jsonl"
    ledger = SamplerLedger(ledger_path)

    attempt_id = ledger.record_selection([{"uid": "task-1"}], optimizer_step=3, selection_source="dataset")
    ledger.record_outcome(
        attempt_id,
        optimizer_step=4,
        outcome="insufficient_reward_spread",
        task_id="task-1",
    )

    records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert records == [
        {
            "attempt_id": 0,
            "event": "selected",
            "optimizer_step_at_selection": 3,
            "schema_version": 1,
            "selection_source": "dataset",
            "task_id": "task-1",
        },
        {
            "attempt_id": 0,
            "event": "outcome",
            "optimizer_step_at_outcome": 4,
            "outcome": "insufficient_reward_spread",
            "schema_version": 1,
            "task_id": "task-1",
        },
    ]


def test_sampler_ledger_resumes_attempt_ids(tmp_path):
    ledger_path = tmp_path / "sampler.jsonl"
    first = SamplerLedger(ledger_path)
    assert first.record_selection([{"uid": "a"}], optimizer_step=1, selection_source="dataset") == 0

    resumed = SamplerLedger(ledger_path)
    assert resumed.record_selection([{"uid": "b"}], optimizer_step=1, selection_source="stale_retry") == 1


def test_sampler_ledger_waits_for_valid_ready_marker(tmp_path):
    ledger = SamplerLedger(
        tmp_path / "sampler.jsonl",
        ready_dir=tmp_path / "ready",
        ready_timeout_seconds=0.01,
        ready_poll_seconds=0.001,
    )

    with pytest.raises(TimeoutError):
        asyncio.run(ledger.wait_until_ready("task-1"))

    (tmp_path / "ready" / "task-1.json").write_text(json.dumps({"status": "ready", "task_id": "task-1"}))
    asyncio.run(ledger.wait_until_ready("task-1"))


def test_sampler_ledger_rejects_malformed_ready_marker(tmp_path):
    ledger = SamplerLedger(
        tmp_path / "sampler.jsonl",
        ready_dir=tmp_path / "ready",
    )
    (tmp_path / "ready" / "task-1.json").write_text("{")

    with pytest.raises(json.JSONDecodeError):
        asyncio.run(ledger.wait_until_ready("task-1"))


def test_sampler_ledger_rejects_relative_paths():
    with pytest.raises(ValueError, match="absolute"):
        SamplerLedger(Path("relative/sampler.jsonl"))


def test_terminal_bench_dataset_sorts_task_directories(tmp_path):
    for name in ("task-z", "task-a", "task-m"):
        task_dir = tmp_path / name
        task_dir.mkdir()
        task_dir.joinpath("instruction.md").write_text(name)

    dataset = TerminalBenchTaskDataset([str(tmp_path)])

    assert [dataset[index]["uid"] for index in range(len(dataset))] == [
        "task-a",
        "task-m",
        "task-z",
    ]
