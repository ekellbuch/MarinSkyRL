import numpy as np
import pytest
import torch

from tests.grug_training_parity import (
    assert_close,
    load_grug_training_oracle,
)


def test_committed_grug_training_oracle_loads() -> None:
    oracle = load_grug_training_oracle()

    assert oracle.manifest["schema_version"] == 1
    assert oracle.manifest["padding"] == "none"
    assert oracle.observations["input_ids"].shape == (1, 6)
    assert oracle.observations["logits"].shape == (1, 6, 16)


def test_parity_tolerance_distinguishes_outputs_from_gradients() -> None:
    expected = np.zeros(1, dtype=np.float32)
    difference_between_policies = torch.tensor([3e-5])

    with pytest.raises(AssertionError):
        assert_close("output", difference_between_policies, expected)
    assert_close("gradient", difference_between_policies, expected, gradient=True)
