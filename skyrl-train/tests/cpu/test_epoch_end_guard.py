"""The epoch epilogue must not fire on a stop that is not an epoch boundary.

`DataTrackingCallback.on_epoch_end_async` clears the epoch-scoped consumed-UID
set and advances the epoch counter; `_finalize_training` then writes the final
checkpoint from that tracker. So firing `on_epoch_end` after a `max_steps` stop
persists `consumed_uids_in_epoch=[]` beside a non-zero
`total_samples_consumed`, and that is the checkpoint `resume_mode=latest`
loads. `_AsyncDataloader.load_state_from_checkpoint` rewinds to row 0 and
relies entirely on that set to skip consumed rows, so the resumed run retrains
prompts the previous run already trained. `AdmissionRejection.DUPLICATE_UID`
does not catch it: `_partition_completed_groups` dedupes within one admission
batch and against occupied UIDs, with no memory of earlier steps.

Observed on a 10-step run over a dataset of at least 629 rows: `global_step_9`
carried all 72 of its UIDs, `global_step_10` carried none of its 80 and read
`epoch: 1`.

Run with:
  uv run --isolated --group dev --extra cpu pytest tests/cpu/test_epoch_end_guard.py
"""

import pytest

from skyrl_train.callbacks.builtin import DataTrackingCallback
from skyrl_train.fully_async_trainer import _epoch_completed
from skyrl_train.utils.data_tracker import DataConsumptionTracker


# global_step is post-increment: after the last step of epoch 0 of a 64-step
# epoch it is 65, and a run stopped by max_steps=10 leaves it at 11.
@pytest.mark.parametrize(
    "global_step, epoch, num_steps_per_epoch, completed",
    [
        (11, 0, 64, False),  # max_steps=10 stopped ten steps into the epoch
        (65, 0, 64, True),  # the epoch's own steps ran out
        (64, 0, 64, False),  # one step short of the boundary
        (11, 0, 10, True),  # max_steps happened to equal the epoch length
        (129, 1, 64, True),  # a later epoch's boundary
        (70, 1, 64, False),  # ...and a stop inside it
    ],
)
def test_only_a_finished_epoch_counts_as_completed(
    global_step: int, epoch: int, num_steps_per_epoch: int, completed: bool
) -> None:
    assert _epoch_completed(global_step, epoch, num_steps_per_epoch) is completed


@pytest.mark.asyncio
async def test_the_tracker_keeps_its_uids_when_the_epoch_did_not_end() -> None:
    """The consequence the guard exists for, at the seam that shows it.

    Two updates of eight prompts, then the stop. Without the guard the epilogue
    would clear the tracker here and the final checkpoint would be written from
    an empty set.
    """
    tracker = DataConsumptionTracker(mini_batch_size=8, num_steps_per_epoch=64)
    callback = DataTrackingCallback(tracker)
    await tracker.mark_consumed([f"task_{i:06d}" for i in range(8)])
    await tracker.mark_consumed([f"task_{i:06d}" for i in range(8, 16)])

    # global_step 3 after two steps of a 64-step epoch.
    if _epoch_completed(3, 0, 64):
        await callback.on_epoch_end_async(state=None, control=None)

    state = tracker.get_state()
    assert state.total_samples_consumed == 16
    assert len(state.consumed_uids_in_epoch) == 16
    assert state.epoch == 0


@pytest.mark.asyncio
async def test_a_real_epoch_boundary_still_clears() -> None:
    """The clear has to keep happening, or epoch two starts with epoch one's
    UIDs still in the skip set and the dataloader runs out of rows to hand out.
    """
    tracker = DataConsumptionTracker(mini_batch_size=8, num_steps_per_epoch=2)
    callback = DataTrackingCallback(tracker)
    await tracker.mark_consumed([f"task_{i:06d}" for i in range(16)])

    # global_step 3 after both steps of a 2-step epoch.
    assert _epoch_completed(3, 0, 2)
    await callback.on_epoch_end_async(state=None, control=None)

    state = tracker.get_state()
    assert state.consumed_uids_in_epoch == []
    assert state.epoch == 1
    assert state.total_samples_consumed == 16
