"""CPU tests for building callbacks from the trainer.callbacks YAML list."""

import asyncio

import pytest

from skyrl_train.callbacks.base import CallbackHandler, TrainerCallback, TrainerControl
from skyrl_train.callbacks.builtin import ProgressCallback, create_callback_from_config


def test_create_callback_from_registry_type_passes_parameters():
    callback = create_callback_from_config({"type": "progress", "log_interval": 4})
    assert isinstance(callback, ProgressCallback)
    assert callback.log_interval == 4


def test_create_callback_from_import_path_passes_parameters():
    callback = create_callback_from_config(
        {"type": "skyrl_train.callbacks.builtin:ProgressCallback", "log_interval": 9}
    )
    assert isinstance(callback, ProgressCallback)
    assert callback.log_interval == 9


def test_create_callback_from_unknown_type_lists_registry():
    with pytest.raises(ValueError, match="Unknown callback type 'nope'"):
        create_callback_from_config({"type": "nope"})


def test_create_callback_from_missing_import_path_raises_module_error():
    with pytest.raises(ModuleNotFoundError):
        create_callback_from_config({"type": "no_such_package.module:Callback"})


class _TaskNames:
    """An experiment-owned callback that is not a TrainerCallback, as import-path callbacks may be."""

    def __init__(self):
        self.task_names = {}

    def on_generation_selected(self, state, control, *, attempt, prompts, **kwargs):
        for prompt in prompts:
            self.task_names[prompt["uid"]] = prompt["prompt"].rsplit("/", 1)[-1]
        return control


def test_generation_selected_reaches_plain_and_base_callbacks():
    plain = _TaskNames()
    handler = CallbackHandler([TrainerCallback(), plain])
    prompts = [{"uid": "u0", "prompt": "tasks/fix-git"}, {"uid": "u1", "prompt": "tasks/sqlite"}]
    control = TrainerControl()

    result = asyncio.run(
        handler.call_event_async("on_generation_selected", None, control, attempt=(), prompts=prompts, trainer=None)
    )

    assert result is control
    assert plain.task_names == {"u0": "fix-git", "u1": "sqlite"}
