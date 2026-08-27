"""
# Run only vllm tests (requires vllm extra):
uv run --isolated --group dev --extra vllm --extra deepspeed pytest tests/gpu/gpu_ci/test_policy_local_engines_e2e.py -m "vllm"

"""

import asyncio
import json
import math
import os
from pathlib import Path

import hydra
import pytest
import ray
import torch
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend

from tests.gpu.utils import get_test_prompts, init_inference_engines, init_worker_with_type, run_inference

MODEL = os.environ.get("SKYRL_GPU_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LM_HEAD_COMPUTE_DTYPE = os.environ.get("SKYRL_GPU_TEST_LM_HEAD_COMPUTE_DTYPE")
FLASH_ATTN = os.environ.get("SKYRL_GPU_TEST_FLASH_ATTN", "0") == "1"
VERIFY_PARITY = os.environ.get("SKYRL_GPU_TEST_VERIFY_PARITY", "0") == "1"
PARITY_OUTPUT = os.environ.get("SKYRL_GPU_TEST_PARITY_OUTPUT")
CALIBRATION_PROMPTS = (
    "Reply with four words about the sky.",
    "Name four common kitchen items.",
    "Give four words associated with winter.",
    "Reply with four words about music.",
    "Name four colors.",
    "Give four words associated with travel.",
    "Reply with four words about a library.",
    "Name four common animals.",
)
CALIBRATION_TOKENS_PER_PROMPT = 4


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _error_summary(values: list[dict]) -> dict:
    absolute_errors = [value["absolute_error"] for value in values]
    relative_errors = [value["relative_error"] for value in values]
    return {
        "comparisons": len(values),
        "top1_matches": sum(value["top1_match"] for value in values),
        "absolute_error": {
            "max": max(absolute_errors),
            "p50": _percentile(absolute_errors, 0.50),
            "p95": _percentile(absolute_errors, 0.95),
        },
        "relative_error": {
            "max": max(relative_errors),
            "p50": _percentile(relative_errors, 0.50),
            "p95": _percentile(relative_errors, 0.95),
        },
    }


def get_test_actor_config() -> DictConfig:
    """Get base config with test-specific overrides."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

        # Override specific parameters
        cfg.trainer.policy.model.path = MODEL
        cfg.trainer.policy.model.lm_head_compute_dtype = LM_HEAD_COMPUTE_DTYPE
        cfg.trainer.flash_attn = FLASH_ATTN
        cfg.trainer.critic.model.path = ""
        cfg.trainer.placement.policy_num_gpus_per_node = 2
        cfg.generator.async_engine = True
        cfg.generator.enable_ray_prometheus_stats = True
        cfg.generator.num_inference_engines = 1
        cfg.generator.run_engines_locally = True

        return cfg


@pytest.mark.parametrize(
    ("colocate_all", "weight_sync_backend", "strategy", "backend", "tp_size"),
    [
        pytest.param(False, "nccl", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(True, "nccl", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(False, "gloo", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(True, "gloo", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(False, "nccl", "deepspeed", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(True, "nccl", "deepspeed", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(False, "nccl", "fsdp2", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(True, "nccl", "fsdp2", "vllm", 2, marks=pytest.mark.vllm),
        pytest.param(False, "nccl", "fsdp2", "vllm", 1, marks=pytest.mark.vllm),
        # TODO(Charlie): add TP > 1 tests for sglang when we support it
        pytest.param(False, "nccl", "deepspeed", "sglang", 1, marks=pytest.mark.sglang),
        pytest.param(True, "nccl", "deepspeed", "sglang", 1, marks=pytest.mark.sglang),
        pytest.param(False, "nccl", "fsdp2", "sglang", 1, marks=pytest.mark.sglang),
        pytest.param(True, "nccl", "fsdp2", "sglang", 1, marks=pytest.mark.sglang),
        pytest.param(False, "gloo", "fsdp", "sglang", 1, marks=pytest.mark.sglang),
        pytest.param(True, "gloo", "fsdp", "sglang", 1, marks=pytest.mark.sglang),
    ],
    ids=[
        "no_colocate_nccl_fsdp_vllm",
        "colocate_nccl_fsdp_vllm",
        "no_colocate_gloo_fsdp_vllm",
        "colocate_gloo_fsdp_vllm",
        "no_colocate_nccl_deepspeed_vllm",
        "colocate_nccl_deepspeed_vllm",
        "no_colocate_nccl_fsdp2_vllm",
        "colocate_nccl_fsdp2_vllm",
        "no_colocate_nccl_fsdp2_vllm_tp1",
        "no_colocate_nccl_deepspeed_sglang",
        "colocate_nccl_deepspeed_sglang",
        "no_colocate_nccl_fsdp2_sglang",
        "colocate_nccl_fsdp2_sglang",
        "no_colocate_gloo_fsdp_sglang",
        "colocate_gloo_fsdp_sglang",
    ],
)
def test_policy_local_engines_e2e(ray_init_fixture, colocate_all, weight_sync_backend, strategy, backend, tp_size):
    """
    Tests initalizing the policy actor group and inference engine, syncing weights, and performing generation.
    """
    try:
        cfg = get_test_actor_config()
        cfg.trainer.placement.colocate_all = colocate_all
        cfg.generator.weight_sync_backend = weight_sync_backend
        cfg.trainer.strategy = strategy
        cfg.generator.backend = backend
        cfg.generator.inference_engine_tensor_parallel_size = tp_size
        if VERIFY_PARITY:
            cfg.generator.max_logprobs = 2

        # If colocate is True, this will load the engine, sleep, and wake up the engine
        client, pg = init_inference_engines(
            model=MODEL,
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=cfg.trainer.placement.colocate_all,
            backend=backend,
            sleep_level=2,  # since we explicitly sync weights
        )

        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=cfg.trainer.placement.colocate_all,
            num_gpus_per_node=cfg.generator.inference_engine_tensor_parallel_size,
            cfg=cfg,
        )
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        asyncio.run(client.reset_prefix_cache())
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        parity_values = []

        def write_parity_checkpoint():
            assert PARITY_OUTPUT, "SKYRL_GPU_TEST_PARITY_OUTPUT is required for the parity calibration"
            expected_per_load = len(CALIBRATION_PROMPTS) * CALIBRATION_TOKENS_PER_PROMPT
            completed_loads = [
                update_index
                for update_index in (1, 2)
                if sum(value["load"] == update_index for value in parity_values) == expected_per_load
            ]
            payload = {
                "schema_version": 1,
                "design": {
                    "prompts": list(CALIBRATION_PROMPTS),
                    "tokens_per_prompt": CALIBRATION_TOKENS_PER_PROMPT,
                    "loads": 2,
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
                "status": {
                    "completed_loads": completed_loads,
                    "comparisons": len(parity_values),
                    "expected_comparisons": 2 * expected_per_load,
                    "complete": completed_loads == [1, 2],
                },
                "loads": {
                    str(update_index): _error_summary(
                        [value for value in parity_values if value["load"] == update_index]
                    )
                    for update_index in completed_loads
                },
                "overall": _error_summary(parity_values),
                "pairs": parity_values,
            }
            output_path = Path(PARITY_OUTPUT)
            temporary_path = output_path.with_name(f".{output_path.name}.tmp")
            temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
            temporary_path.replace(output_path)

        def verify_synced_weights(update_index: int):
            weight_names = [
                "model.language_model.embed_tokens.weight",
                "model.language_model.layers.0.input_layernorm.weight",
                "lm_head.weight",
            ]
            policy_weights = {}
            per_rank = ray.get(policy.async_run_ray_method("pass_through", "read_post_step_weights", weight_names))
            for rank_weights in per_rank:
                if isinstance(rank_weights, dict):
                    policy_weights.update(
                        {name: tensor for name, tensor in rank_weights.items() if isinstance(tensor, torch.Tensor)}
                    )
            assert set(policy_weights) == set(weight_names), policy_weights.keys()

            engine_actor = client.engines[0].inference_engine_actor
            engine_per_rank = ray.get(engine_actor.read_engine_weights.remote(weight_names, False))
            if isinstance(engine_per_rank, dict):
                engine_per_rank = [engine_per_rank]
            assert len(engine_per_rank) == 1, len(engine_per_rank)
            for name in weight_names:
                entry = engine_per_rank[0][name]
                assert entry["found"], (name, entry)
                actual = entry["tensor"]
                expected = policy_weights[name].to(actual.dtype)
                if name in {"model.language_model.embed_tokens.weight", "lm_head.weight"}:
                    assert actual.shape[0] >= expected.shape[0], (name, actual.shape, expected.shape)
                    actual = actual[: expected.shape[0]]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert engine_per_rank[0]["model.language_model.embed_tokens.weight"]["dtype"] == "bfloat16"
            assert engine_per_rank[0]["lm_head.weight"]["dtype"] == "float32"
            assert (
                engine_per_rank[0]["lm_head.weight"]["internal_name"]
                != engine_per_rank[0]["model.language_model.embed_tokens.weight"]["internal_name"]
            )
            print(f"Verified exact learner/vLLM weights after complete load {update_index}: {weight_names}")
            return engine_per_rank[0]

        def generate_with_logprob_checks(update_index: int):
            sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
            sampling_params["logprobs"] = 1
            if VERIFY_PARITY:
                prompts = [[{"role": "user", "content": content}] for content in CALIBRATION_PROMPTS]
                sampling_params["max_tokens"] = CALIBRATION_TOKENS_PER_PROMPT
                sampling_params["temperature"] = 0.0
                sampling_params["ignore_eos"] = True
                prompt_token_ids = client.tokenizer.apply_chat_template(
                    prompts,
                    add_generation_prompt=True,
                    add_special_tokens=False,
                    return_dict=True,
                    tokenize=True,
                )["input_ids"]
                outputs = asyncio.run(
                    client.generate(
                        InferenceEngineInput(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
                    )
                )
            else:
                prompts = get_test_prompts(MODEL)
                outputs = asyncio.run(run_inference(client, prompts, sampling_params))

            assert len(outputs["responses"]) == len(outputs["response_ids"])
            response_logprobs = outputs["response_logprobs"]
            assert response_logprobs is not None
            assert len(response_logprobs) == len(outputs["response_ids"])
            assert all(len(ids) == len(logprobs) for ids, logprobs in zip(outputs["response_ids"], response_logprobs))
            assert all(math.isfinite(logprob) for logprobs in response_logprobs for logprob in logprobs)
            if VERIFY_PARITY:
                assert len(outputs["response_ids"]) == len(CALIBRATION_PROMPTS)
                assert all(len(tokens) == CALIBRATION_TOKENS_PER_PROMPT for tokens in outputs["response_ids"])
                scoring_params = dict(sampling_params)
                scoring_params.update({"max_tokens": 1, "logprobs": 0, "prompt_logprobs": 2})
                scored_outputs = asyncio.run(
                    client.generate(
                        InferenceEngineInput(
                            prompt_token_ids=[
                                prompt_ids + response_ids
                                for prompt_ids, response_ids in zip(
                                    prompt_token_ids, outputs["response_ids"], strict=True
                                )
                            ],
                            sampling_params=scoring_params,
                        )
                    )
                )
                prompt_logprobs = scored_outputs["prompt_logprobs"]
                assert prompt_logprobs is not None
                assert len(prompt_logprobs) == len(outputs["response_ids"])
                for prompt_index, (prompt_ids, response_ids, rollout_logprobs) in enumerate(
                    zip(prompt_token_ids, outputs["response_ids"], response_logprobs, strict=True)
                ):
                    for token_index, (selected_token, rollout_logprob) in enumerate(
                        zip(response_ids, rollout_logprobs, strict=True)
                    ):
                        rollout_candidates = prompt_logprobs[prompt_index][len(prompt_ids) + token_index]
                        assert rollout_candidates is not None
                        assert selected_token in rollout_candidates
                        ranked_rollout_candidates = sorted(
                            rollout_candidates.items(), key=lambda item: item[1], reverse=True
                        )
                        assert len(ranked_rollout_candidates) >= 2
                        rollout_top_candidates = [
                            {"token": int(token), "logprob": float(logprob)}
                            for token, logprob in ranked_rollout_candidates[:2]
                        ]
                        learner_results = ray.get(
                            policy.async_run_ray_method(
                                "pass_through",
                                "direct_next_token_parity_for_sync_test",
                                prompt_ids + response_ids[:token_index],
                                selected_token,
                            )
                        )
                        assert len(learner_results) == 1, learner_results
                        result = learner_results[0]
                        assert result["gdn_fast_path"], result
                        assert math.isfinite(result["selected_logprob"]), result
                        assert math.isfinite(result["selected_logit"]), result
                        assert math.isfinite(result["logsumexp"]), result
                        assert math.isclose(
                            result["selected_logprob"],
                            result["selected_logit"] - result["logsumexp"],
                            rel_tol=0.0,
                            abs_tol=1e-6,
                        ), result
                        assert len(result["top_candidates"]) == 2, result
                        absolute_error = abs(result["selected_logprob"] - rollout_logprob)
                        relative_error = absolute_error / max(abs(rollout_logprob), 1e-12)
                        parity_values.append(
                            {
                                "load": update_index,
                                "prompt_index": prompt_index,
                                "token_index": token_index,
                                "selected_token": selected_token,
                                "learner_top1": result["top1"],
                                "learner_top_candidates": result["top_candidates"],
                                "learner_top1_margin": result["top1_margin"],
                                "rollout_top1": rollout_top_candidates[0]["token"],
                                "rollout_top_candidates": rollout_top_candidates,
                                "rollout_top1_margin": (
                                    rollout_top_candidates[0]["logprob"] - rollout_top_candidates[1]["logprob"]
                                ),
                                "top1_match": result["top1"] == selected_token,
                                "learner_selected_logit": result["selected_logit"],
                                "learner_logsumexp": result["logsumexp"],
                                "learner_logprob": result["selected_logprob"],
                                "rollout_logprob": rollout_logprob,
                                "absolute_error": absolute_error,
                                "relative_error": relative_error,
                            }
                        )
                load_values = [value for value in parity_values if value["load"] == update_index]
                assert len(load_values) == len(CALIBRATION_PROMPTS) * CALIBRATION_TOKENS_PER_PROMPT
                print(
                    f"Measured learner/vLLM selected-token logprobs after complete load {update_index}: "
                    f"{json.dumps({'summary': _error_summary(load_values), 'pairs': load_values}, sort_keys=True)}"
                )
                write_parity_checkpoint()
            print(
                f"Verified rollout logprobs after complete load {update_index} "
                f"for {sum(map(len, response_logprobs))} tokens"
            )
            return outputs

        if VERIFY_PARITY:
            first_engine_weights = verify_synced_weights(1)
            generate_with_logprob_checks(1)

            mutation_results = ray.get(
                policy.async_run_ray_method(
                    # The learner is the unwrapped text tower (``model.*``); the
                    # extractor adds ``model.language_model.*`` for vLLM.
                    "pass_through",
                    "perturb_weight_for_sync_test",
                    "model.embed_tokens.weight",
                    0.5,
                )
            )
            assert mutation_results and all(result["changed"] for result in mutation_results), mutation_results
            print(f"Changed learner state before complete load 2: {mutation_results}")

            asyncio.run(client.reset_prefix_cache())
            ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
            second_engine_weights = verify_synced_weights(2)
            second_outputs = generate_with_logprob_checks(2)

            assert not torch.equal(
                first_engine_weights["lm_head.weight"]["tensor"],
                second_engine_weights["lm_head.weight"]["tensor"],
            )
            for identity_key in ("internal_name", "parameter_id", "data_ptr"):
                assert (
                    first_engine_weights["lm_head.weight"][identity_key]
                    == second_engine_weights["lm_head.weight"][identity_key]
                )
            print("Verified stable FP32 projection storage across two complete loads")
            assert len(parity_values) == 2 * len(CALIBRATION_PROMPTS) * CALIBRATION_TOKENS_PER_PROMPT
            assert all(value["top1_match"] for value in parity_values), parity_values
            outputs = second_outputs
        else:
            sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
            outputs = asyncio.run(run_inference(client, get_test_prompts(MODEL), sampling_params))
            assert len(outputs["responses"]) == len(outputs["response_ids"])
        print(f"Example output: {outputs['responses'][0]}, {outputs['stop_reasons'][0]}")
    finally:
        ray.shutdown()
