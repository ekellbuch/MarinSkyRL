from types import SimpleNamespace

from skyrl_train.models.qwen3_5_vlm import remove_vision_no_split_modules


def test_remove_vision_no_split_modules_keeps_text_classes():
    model = SimpleNamespace(
        _no_split_modules={"Qwen3_5VisionBlock", "Qwen3_5DecoderLayer"},
    )

    remove_vision_no_split_modules(model)

    assert model._no_split_modules == ["Qwen3_5DecoderLayer"]
