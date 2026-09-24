import asyncio
from types import SimpleNamespace

import pytest

from skyrl_train.inference_engines.vllm.vllm_engine import AsyncVLLMInferenceEngine


class _Queue:
    def __init__(self, output, events):
        self.output = output
        self.events = events

    def get_nowait(self):
        output, self.output = self.output, None
        if output is not None:
            self.events.append(("collect", output.value))
        return output

    async def get(self):
        raise AssertionError("the fake queue contains its final output")


class _LLM:
    def __init__(self):
        self.events = []
        self.paused = False
        self.fail_request_id = None

    async def pause_generation(self, *, mode, clear_cache):
        assert mode == "keep"
        assert clear_cache is False
        self.paused = True
        self.events.append(("pause",))

    async def add_request(self, *, request_id, prompt, params, lora_request):
        assert self.paused
        assert lora_request is None
        self.events.append(("add", request_id, prompt["prompt_token_ids"], params))
        if request_id == self.fail_request_id:
            raise RuntimeError("injected admission failure")
        output = SimpleNamespace(finished=True, value=request_id)
        return _Queue(output, self.events)

    async def resume_generation(self):
        assert self.paused
        self.paused = False
        self.events.append(("resume",))

    async def abort(self, request_ids):
        self.events.append(("abort", tuple(request_ids)))


@pytest.mark.asyncio
async def test_async_vllm_admits_a_logical_batch_before_collecting(monkeypatch):
    engine = object.__new__(AsyncVLLMInferenceEngine)
    engine.llm = _LLM()
    engine._is_lora = False
    engine._batch_admission_lock = asyncio.Lock()
    sampling_params = object()
    prompts = [[1, 2], [3, 4]]
    monkeypatch.setattr(engine, "_preprocess_prompts", lambda _: (prompts, sampling_params))
    monkeypatch.setattr(engine, "_postprocess_outputs", lambda outputs: outputs)

    outputs = await engine.generate({})

    assert [output.value for output in outputs] == [event[1] for event in engine.llm.events if event[0] == "add"]
    assert [event[0] for event in engine.llm.events] == [
        "pause",
        "add",
        "add",
        "resume",
        "collect",
        "collect",
    ]


@pytest.mark.asyncio
async def test_async_vllm_resumes_and_aborts_after_partial_admission(monkeypatch):
    engine = object.__new__(AsyncVLLMInferenceEngine)
    engine.llm = _LLM()
    engine._is_lora = False
    engine._batch_admission_lock = asyncio.Lock()
    request_ids = iter(["first", "second"])
    engine.llm.fail_request_id = "second"
    monkeypatch.setattr(
        "skyrl_train.inference_engines.vllm.vllm_engine.uuid4",
        lambda: SimpleNamespace(hex=next(request_ids)),
    )
    monkeypatch.setattr(engine, "_preprocess_prompts", lambda _: ([[1, 2], [3, 4]], object()))

    with pytest.raises(RuntimeError, match="injected admission failure"):
        await engine.generate({})

    assert [event[0] for event in engine.llm.events] == ["pause", "add", "add", "resume", "abort"]
    assert engine.llm.events[-1] == ("abort", ("first", "second"))
