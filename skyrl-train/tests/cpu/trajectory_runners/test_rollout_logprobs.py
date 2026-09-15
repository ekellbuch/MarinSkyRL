"""Rollout-logprob gathering masks trajectories instead of failing the run."""

from types import SimpleNamespace

from skyrl_gym.verification import TrainingDisposition

from skyrl_train.trajectory_runners.projections import project_loss_mask
import pytest

from skyrl_train.trajectory_runners.rollout_logprobs import (
    MISSING_ROLLOUT_LOGPROBS_REASON,
    gather_rollout_logprobs,
    require_rollout_details_collection,
)


def _output(trajectory_id, response_ids, logprobs, loss_mask=None):
    return SimpleNamespace(
        trajectory_id=trajectory_id,
        evidence=SimpleNamespace(
            response_token_ids=tuple(response_ids),
            behavior_logprobs=None if logprobs is None else tuple(logprobs),
        ),
        loss_mask=[1] * len(response_ids) if loss_mask is None else loss_mask,
        disposition=TrainingDisposition(
            loss_eligible=True, baseline_eligible=True, reason="verified", exception_type=None
        ),
    )


def test_whole_group_without_logprobs_is_masked_not_raised():
    outputs = [_output(f"t{i}", [11, 12, 13], None) for i in range(3)]

    gathered = gather_rollout_logprobs(outputs, required=True, expect_logprobs=True, group_label=" for task-a")

    assert gathered == [[0.0, 0.0, 0.0]] * 3
    for output in outputs:
        assert output.loss_mask == [0, 0, 0]
        assert output.disposition.loss_eligible is False
        assert output.disposition.reason == MISSING_ROLLOUT_LOGPROBS_REASON
        assert project_loss_mask(output, list(output.evidence.response_token_ids)) == [0, 0, 0]


def test_partial_group_keeps_trajectories_that_have_logprobs():
    kept = _output("t0", [1, 2], [-0.1, -0.2])
    lost = _output("t1", [3, 4, 5], None)

    gathered = gather_rollout_logprobs([kept, lost], required=True, expect_logprobs=True)

    assert gathered == [[-0.1, -0.2], [0.0, 0.0, 0.0]]
    assert kept.loss_mask == [1, 1] and kept.disposition.loss_eligible is True
    assert lost.loss_mask == [0, 0, 0] and lost.disposition.loss_eligible is False


def test_already_untrainable_trajectory_is_left_alone():
    failed = _output("t0", [0], None, loss_mask=[0])
    failed.disposition = TrainingDisposition(
        loss_eligible=False, baseline_eligible=True, reason="trajectory failure", exception_type="AgentTimeoutError"
    )
    kept = _output("t1", [1], [-0.5])

    gathered = gather_rollout_logprobs([failed, kept], required=True, expect_logprobs=True)

    assert gathered == [[0.0], [-0.5]]
    assert failed.disposition.reason == "trajectory failure"
    assert failed.disposition.exception_type == "AgentTimeoutError"


def test_not_required_and_nothing_present_returns_none():
    outputs = [_output("t0", [1, 2], None)]

    assert gather_rollout_logprobs(outputs, required=False, expect_logprobs=False) is None
    assert outputs[0].loss_mask == [1, 1]
    assert outputs[0].disposition.loss_eligible is True


def test_not_required_partial_presence_zero_fills_without_masking():
    kept = _output("t0", [1], [-0.3])
    lost = _output("t1", [2, 3], None)

    gathered = gather_rollout_logprobs([kept, lost], required=False, expect_logprobs=True)

    assert gathered == [[-0.3], [0.0, 0.0]]
    assert lost.loss_mask == [1, 1] and lost.disposition.loss_eligible is True


def test_required_logprobs_without_collection_fails_at_startup():
    with pytest.raises(ValueError, match="collect_rollout_details=true"):
        require_rollout_details_collection(required=True, collect=False)


@pytest.mark.parametrize("required,collect", [(True, True), (False, False), (False, True)])
def test_collection_check_passes_when_consistent(required, collect):
    require_rollout_details_collection(required=required, collect=collect)
