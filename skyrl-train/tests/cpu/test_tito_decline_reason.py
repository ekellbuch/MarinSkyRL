"""Every full-TITO decline says which check rejected the trajectory.

Job 7161070 declined 42 of 42 assemblies and job 7157954 declined 160 of 160,
both emitting one warning with no discriminator. The assembler has seven
independent checks, so a 100% decline rate was not diagnosable from a log: it
could not even be established whether the same check fired each time.

These cases drive each check from a valid baseline, perturbing one thing at a
time. They exercise only the decline paths, which all return before the
tokenizer is used, so the fixtures are plain id lists.
"""

import pytest

from skyrl_train.trajectory_runners.trajectory_processing import _assemble_response_ids_tito_full

GEN_PROMPT = [9, 10]
# p1 == p0 + c0 + observation, which is the invariant the assembler checks.
P0 = [1, 2, 3] + GEN_PROMPT
C0 = [20, 21]
P1 = P0 + C0 + [30] + GEN_PROMPT
C1 = [40]


def _assemble(prompt_ids, token_ids, *, n_assistant=2, generation_prompt_ids=GEN_PROMPT):
    reason = []
    result = _assemble_response_ids_tito_full(
        [{"role": "assistant", "content": ""} for _ in range(n_assistant)],
        None,  # tokenizer: every decline returns before it is used
        generation_prompt_ids,
        None,
        token_ids,
        prompt_ids,
        None,
        None,
        None,
        None,
        decline_reason=reason,
    )
    return result, reason


def test_absent_streams_are_named() -> None:
    result, reason = _assemble(None, None)

    assert result is None
    assert reason == ["served id streams absent"]


def test_stream_count_mismatch_reports_both_counts() -> None:
    result, reason = _assemble([P0], [C0, C1])

    assert result is None
    assert "2 completion vs 1 prompt" in reason[0]


def test_assistant_message_count_mismatch_is_named() -> None:
    result, reason = _assemble([P0, P1], [C0, C1], n_assistant=3)

    assert result is None
    assert "3 assistant messages but 2 served turns" in reason[0]


@pytest.mark.parametrize("empty", [[], None])
def test_empty_per_turn_stream_names_its_turn(empty) -> None:
    result, reason = _assemble([P0, empty], [C0, C1])

    assert result is None
    assert reason[0].startswith("turn 1:")
    assert "empty or not a list" in reason[0]


def test_prefix_invariant_reports_the_first_differing_index() -> None:
    """The index is the diagnosis.

    Near zero means the streams are unrelated; near the end of the previous
    stream means a boundary or whitespace drift in a turn fed back as text.
    """
    broken = list(P1)
    broken[6] = 999  # inside the P0 + C0 prefix, at a known offset

    result, reason = _assemble([P0, broken], [C0, C1])

    assert result is None
    assert "turn 1: prefix invariant failed at index 6 of 7" in reason[0]


def test_turn_zero_without_the_generation_prompt_is_named() -> None:
    result, reason = _assemble([P0, P1], [C0, C1], generation_prompt_ids=[77, 78])

    assert result is None
    assert "does not end with the generation prompt" in reason[0]


# The completion-offset check has no test. It is unreachable once the prefix
# invariant passes: that invariant chains, so ``prompt_token_ids[-1]`` contains
# every earlier completion at exactly its own offset, and the last turn's region
# is true by construction of ``served_full``. Any fixture that breaks the offset
# breaks the prefix invariant first, which is what the check above reports. It is
# kept in the assembler as a defensive check and named like the others.


def test_reason_is_optional_so_existing_callers_are_unaffected() -> None:
    assert _assemble_response_ids_tito_full([], None, GEN_PROMPT, None, None, None, None, None, None, None) is None
