"""CPU tests for building callbacks from the trainer.callbacks YAML list."""

import pytest

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
