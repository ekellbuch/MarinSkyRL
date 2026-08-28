from types import SimpleNamespace

import pytest

from skyrl_train.models.qwen3_5_vlm import (
    qwen3_5_vllm_internal_weight_candidates,
    remove_vision_no_split_modules,
)


def test_remove_vision_no_split_modules_keeps_text_classes():
    model = SimpleNamespace(
        _no_split_modules={"Qwen3_5VisionBlock", "Qwen3_5DecoderLayer"},
    )

    remove_vision_no_split_modules(model)

    assert model._no_split_modules == ["Qwen3_5DecoderLayer"]


@pytest.mark.parametrize(
    ("name", "tied", "expected"),
    [
        (
            "model.language_model.layers.0.input_layernorm.weight",
            False,
            ("language_model.model.layers.0.input_layernorm.weight",),
        ),
        (
            "model.embed_tokens.weight",
            False,
            ("language_model.model.embed_tokens.weight",),
        ),
        (
            "lm_head.weight",
            False,
            ("language_model.lm_head.weight",),
        ),
        (
            "lm_head.weight",
            True,
            ("language_model.lm_head.weight", "language_model.model.embed_tokens.weight"),
        ),
        ("visual.patch_embed.weight", False, ("visual.patch_embed.weight",)),
    ],
)
def test_qwen3_5_vllm_internal_weight_candidates(name, tied, expected):
    assert qwen3_5_vllm_internal_weight_candidates(name, tied_word_embeddings=tied) == expected
