"""S5: opencode per-trial literal correlation in TerminalBenchGenerator.

opencode is a CLI agent that bypasses harbor Chat, so it returns EMPTY
rollout_details even behind a co-located RecordProxy (which writes ONE shared
worker-side log for all concurrent trials). The generator recovers each trial's
token_ids/logprobs from that shared log by the per-trial correlation id harbor
stamped (x-ot-trial-id), so TIS works AND concurrent GRPO same-seed trials never
bleed into each other.

Exercises the two helpers as unbound methods on a SimpleNamespace fake self (the
CPU-test pattern from test_preserve_logprobs_on_timeout.py) so no tokenizer /
engine / config is needed.
"""

import json
import os
import sys
import types

import pytest

_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

# The generator pulls in the harbor agentic-RL stack (absent under the CPU extra).
try:
    from terminal_bench.terminal_bench_generator import TerminalBenchGenerator  # noqa: E402
except ImportError:
    pytest.skip("harbor deps unavailable (agentic RL extra not installed)", allow_module_level=True)

# The correlation builder ships in harbor; skip if this harbor predates the bridge.
try:
    import harbor.literal.rollout_build  # noqa: F401
except ImportError:
    pytest.skip("harbor without the literal rollout_build bridge", allow_module_level=True)


_correlate = TerminalBenchGenerator._maybe_correlate_opencode_rollout_details
_load = TerminalBenchGenerator._load_literal_log_entries
_load_for_trial = TerminalBenchGenerator._literal_log_entries_for_trial
_ensure_loaded = TerminalBenchGenerator._ensure_literal_log_loaded


def _bind_log_methods(s):
    """Bind the incremental log reader + per-trial index accessor onto a fake self."""
    s._ensure_literal_log_loaded = types.MethodType(_ensure_loaded, s)
    s._load_literal_log_entries = types.MethodType(_load, s)
    s._literal_log_entries_for_trial = types.MethodType(_load_for_trial, s)
    return s


def _fake_self(collect=True, literal_log_path=None):
    s = types.SimpleNamespace(_collect_rollout_details=collect, _literal_log_path=literal_log_path)
    return _bind_log_methods(s)


def _result(trial_id, rollout_details=None):
    md = {"rollout_correlation_id": trial_id} if trial_id else None
    agent_result = types.SimpleNamespace(rollout_details=rollout_details, metadata=md)
    return types.SimpleNamespace(agent_result=agent_result)


def _entry(trial_id, ts, pids, cids, lps):
    return {
        "timestamp": ts,
        "status_code": 200,
        "trial_id": trial_id,
        "request": {"messages": [{"role": "user", "content": "same task"}]},
        "literal": {
            "prompt_token_ids": pids,
            "completion_token_ids": cids,
            "logprobs": lps,
        },
    }


def _write_log(tmp_path, entries):
    p = tmp_path / "literal.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return str(p)


def test_correlates_per_trial_no_bleed_for_identical_seed_group(tmp_path, monkeypatch):
    """n=2 rollouts of the IDENTICAL prompt, interleaved in one shared log: each
    trial gets ITS OWN turns; no cross-rollout bleed; result is mutated in place."""
    entries = [
        _entry("A", 1.0, [1], [10], [-0.1]),
        _entry("B", 1.1, [1], [20], [-0.2]),
        _entry("A", 2.0, [1, 10], [11], [-0.3]),
        _entry("B", 2.1, [1, 20], [21], [-0.4]),
    ]
    monkeypatch.setenv("OTAGENT_LITERAL_LOG_PATH", _write_log(tmp_path, entries))

    ra = _result("A")
    a = _correlate(_fake_self(), ra, None)
    assert a[0]["completion_token_ids"] == [[10], [11]]
    assert a[0]["logprobs"] == [[-0.1], [-0.3]]
    assert ra.agent_result.rollout_details == a  # persisted onto the result

    rb = _result("B")
    b = _correlate(_fake_self(), rb, None)
    assert b[0]["completion_token_ids"] == [[20], [21]]
    # No bleed.
    assert 20 not in [t for turn in a[0]["completion_token_ids"] for t in turn]
    assert 10 not in [t for turn in b[0]["completion_token_ids"] for t in turn]


def test_noop_when_flag_off(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "OTAGENT_LITERAL_LOG_PATH",
        _write_log(tmp_path, [_entry("A", 1.0, [1], [10], [-0.1])]),
    )
    assert _correlate(_fake_self(collect=False), _result("A"), None) is None


def test_noop_when_already_populated(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "OTAGENT_LITERAL_LOG_PATH",
        _write_log(tmp_path, [_entry("A", 1.0, [1], [10], [-0.1])]),
    )
    existing = [{"completion_token_ids": [[99]], "logprobs": [[-0.9]]}]
    assert _correlate(_fake_self(), _result("A"), existing) is existing


def test_noop_when_no_correlation_id(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "OTAGENT_LITERAL_LOG_PATH",
        _write_log(tmp_path, [_entry("A", 1.0, [1], [10], [-0.1])]),
    )
    assert _correlate(_fake_self(), _result(None), None) is None


def test_noop_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("OTAGENT_LITERAL_LOG_PATH", raising=False)
    assert _correlate(_fake_self(), _result("A"), None) is None


def test_noop_when_log_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("OTAGENT_LITERAL_LOG_PATH", str(tmp_path / "nope.jsonl"))
    assert _correlate(_fake_self(), _result("A"), None) is None


def test_noop_when_trial_absent_from_log(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "OTAGENT_LITERAL_LOG_PATH",
        _write_log(tmp_path, [_entry("A", 1.0, [1], [10], [-0.1])]),
    )
    assert _correlate(_fake_self(), _result("Z"), None) is None


def test_load_entries_incremental_appends_without_reparse(tmp_path):
    """The append-only log is read incrementally: a second call after new lines are
    appended returns ALL entries, and the byte offset advances (no whole-file re-parse)."""
    p = _write_log(tmp_path, [_entry("A", 1.0, [1], [10], [-0.1])])
    s = _fake_self()
    first = s._load_literal_log_entries(p)
    assert len(first) == 1
    off1 = s._literal_log_state["offset"]
    assert off1 == os.path.getsize(p)
    # append a new line, re-read: incremental pickup, offset advances.
    with open(p, "a") as fh:
        fh.write(json.dumps(_entry("A", 2.0, [1, 10], [11], [-0.3])) + "\n")
    second = s._load_literal_log_entries(p)
    assert len(second) == 2
    assert s._literal_log_state["offset"] == os.path.getsize(p) > off1
    # entries accumulate in place (same list object extended, not rebuilt).
    assert second[0]["literal"]["completion_token_ids"] == [10]
    assert second[1]["literal"]["completion_token_ids"] == [11]


def test_per_trial_index_matches_full_scan(tmp_path):
    """_literal_log_entries_for_trial returns exactly the trial-scoped subset that a
    full-scan filter on _load_literal_log_entries would (semantics preserved for #37/#38)."""
    entries = [
        _entry("A", 1.0, [1], [10], [-0.1]),
        _entry("B", 1.1, [1], [20], [-0.2]),
        _entry("A", 2.0, [1, 10], [11], [-0.3]),
        _entry("B", 2.1, [1, 20], [21], [-0.4]),
    ]
    p = _write_log(tmp_path, entries)
    s = _fake_self()
    all_entries = s._load_literal_log_entries(p)
    for tid in ("A", "B", "Z"):
        idx = s._literal_log_entries_for_trial(p, tid)
        full = [e for e in all_entries if e.get("trial_id") == tid]
        assert idx == full
    assert [e["literal"]["completion_token_ids"] for e in s._literal_log_entries_for_trial(p, "A")] == [[10], [11]]


def test_load_entries_handles_partial_trailing_line(tmp_path):
    """A line split across reads (append mid-line) is carried as a partial and parsed
    only once complete — never dropped, never double-counted."""
    p = tmp_path / "literal.jsonl"
    full = json.dumps(_entry("A", 1.0, [1], [10], [-0.1]))
    half = len(full) // 2
    p.write_text(full[:half])  # incomplete line, no newline
    s = _fake_self()
    assert s._load_literal_log_entries(p) == []  # nothing complete yet
    with open(p, "a") as fh:
        fh.write(full[half:] + "\n")  # complete the line
    out = s._load_literal_log_entries(p)
    assert len(out) == 1 and out[0]["trial_id"] == "A"


def test_rotation_resets_offset(tmp_path):
    """A shrink (preempt-resume writes a fresh token-suffixed log at the same path) resets
    the reader instead of returning stale/garbled state."""
    p = _write_log(tmp_path, [_entry("A", 1.0, [1], [10], [-0.1]), _entry("A", 2.0, [1], [11], [-0.2])])
    s = _fake_self()
    assert len(s._load_literal_log_entries(p)) == 2
    _write_log(tmp_path, [_entry("A", 9.0, [1], [99], [-0.9])])  # rewrite smaller
    out = s._load_literal_log_entries(p)
    assert len(out) == 1 and out[0]["literal"]["completion_token_ids"] == [99]


# --- opencode chat_history reconstruction (feeds _process_trial_result) ---------

_build_chat = TerminalBenchGenerator._maybe_build_opencode_chat_history


class _FakeTokenizer:
    """Minimal tokenizer: decode ids -> a marker string (content is opaque to the
    reconstruction; the exact-id alignment path consumes the raw ids, not this text)."""

    def decode(self, ids, skip_special_tokens=True):  # noqa: D401
        return "assistant-final:" + ",".join(str(i) for i in ids)


def _chat_self(collect=True, tokenizer=None, literal_log_path=None):
    s = types.SimpleNamespace(
        _collect_rollout_details=collect,
        tokenizer=tokenizer or _FakeTokenizer(),
        _literal_log_path=literal_log_path,
    )
    return _bind_log_methods(s)


def test_chat_history_resolves_log_from_cfg_path_when_env_unset(tmp_path, monkeypatch):
    """The Ray-boundary fix: the log path is threaded via cfg (self._literal_log_path);
    the helper must resolve it even when OTAGENT_LITERAL_LOG_PATH is absent from the
    worker env (which is exactly what broke fullgate14 → tis/skipped_fraction=1.0)."""
    monkeypatch.delenv("OTAGENT_LITERAL_LOG_PATH", raising=False)
    log = _write_log(tmp_path, [_entry_msgs("A", 1.0, [{"role": "user", "content": "task"}], [5, 6])])
    ch = _build_chat(_chat_self(literal_log_path=log), _result("A"))
    assert ch == [{"role": "user", "content": "task"}, {"role": "assistant", "content": "assistant-final:5,6"}]


def test_correlate_resolves_log_from_cfg_path_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("OTAGENT_LITERAL_LOG_PATH", raising=False)
    log = _write_log(tmp_path, [_entry("A", 1.0, [1], [10], [-0.1])])
    out = _correlate(_fake_self(literal_log_path=log), _result("A"), None)
    assert out[0]["completion_token_ids"] == [[10]]


def _entry_msgs(trial_id, ts, messages, cids):
    """A shared-log entry with an explicit request.messages + a completion turn."""
    return {
        "timestamp": ts,
        "status_code": 200,
        "trial_id": trial_id,
        "request": {"messages": messages},
        "literal": {"prompt_token_ids": [1], "completion_token_ids": cids, "logprobs": [-0.1] * len(cids)},
    }


def test_chat_history_uses_last_request_plus_decoded_final_completion(tmp_path, monkeypatch):
    # Two turns of a growing conversation; the LAST request carries the full prior
    # context, and we append the final turn's decoded completion as the last assistant.
    convo_t1 = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "solve X"},
    ]
    convo_t2 = convo_t1 + [
        {"role": "assistant", "content": "turn-1 answer"},
        {"role": "user", "content": "tool result"},
    ]
    entries = [
        _entry_msgs("A", 1.0, convo_t1, [10, 11]),
        _entry_msgs("A", 2.0, convo_t2, [20, 21]),
    ]
    monkeypatch.setenv("OTAGENT_LITERAL_LOG_PATH", _write_log(tmp_path, entries))
    ch = _build_chat(_chat_self(), _result("A"))
    # base = last request's messages (the full convo) + appended final assistant.
    assert ch[:-1] == convo_t2
    assert ch[-1] == {"role": "assistant", "content": "assistant-final:20,21"}


def test_chat_history_single_turn(tmp_path, monkeypatch):
    msgs = [{"role": "user", "content": "one-shot task"}]
    monkeypatch.setenv("OTAGENT_LITERAL_LOG_PATH", _write_log(tmp_path, [_entry_msgs("A", 1.0, msgs, [7, 8, 9])]))
    ch = _build_chat(_chat_self(), _result("A"))
    assert ch == msgs + [{"role": "assistant", "content": "assistant-final:7,8,9"}]


def test_chat_history_none_when_flag_off(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "OTAGENT_LITERAL_LOG_PATH",
        _write_log(tmp_path, [_entry_msgs("A", 1.0, [{"role": "user", "content": "t"}], [1])]),
    )
    assert _build_chat(_chat_self(collect=False), _result("A")) is None


def test_chat_history_none_when_no_correlation_id(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "OTAGENT_LITERAL_LOG_PATH",
        _write_log(tmp_path, [_entry_msgs("A", 1.0, [{"role": "user", "content": "t"}], [1])]),
    )
    assert _build_chat(_chat_self(), _result(None)) is None


def test_chat_history_none_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("OTAGENT_LITERAL_LOG_PATH", raising=False)
    assert _build_chat(_chat_self(), _result("A")) is None


def test_chat_history_none_when_trial_absent(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "OTAGENT_LITERAL_LOG_PATH",
        _write_log(tmp_path, [_entry_msgs("A", 1.0, [{"role": "user", "content": "t"}], [1])]),
    )
    assert _build_chat(_chat_self(), _result("Z")) is None


def test_chat_history_none_when_no_completion_bearing_record(tmp_path, monkeypatch):
    # status 200 but empty completion -> not a usable turn -> None (honest drop).
    e = _entry_msgs("A", 1.0, [{"role": "user", "content": "t"}], [])
    monkeypatch.setenv("OTAGENT_LITERAL_LOG_PATH", _write_log(tmp_path, [e]))
    assert _build_chat(_chat_self(), _result("A")) is None


def test_chat_history_none_when_request_has_no_messages(tmp_path, monkeypatch):
    e = _entry_msgs("A", 1.0, [{"role": "user", "content": "t"}], [1])
    e["request"] = {}  # malformed: no messages
    monkeypatch.setenv("OTAGENT_LITERAL_LOG_PATH", _write_log(tmp_path, [e]))
    assert _build_chat(_chat_self(), _result("A")) is None


def test_chat_history_picks_latest_timestamp_regardless_of_order(tmp_path, monkeypatch):
    # Entries out of order in the file; the LATEST timestamp must be the base.
    early = _entry_msgs("A", 1.0, [{"role": "user", "content": "early"}], [1])
    late = _entry_msgs("A", 9.0, [{"role": "user", "content": "late"}], [2])
    monkeypatch.setenv("OTAGENT_LITERAL_LOG_PATH", _write_log(tmp_path, [late, early]))
    ch = _build_chat(_chat_self(), _result("A"))
    assert ch[0] == {"role": "user", "content": "late"}
