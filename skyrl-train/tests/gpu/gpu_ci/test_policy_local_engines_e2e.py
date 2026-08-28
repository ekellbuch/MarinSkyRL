"""
# Run only vllm tests (requires vllm extra):
uv run --isolated --group dev --extra vllm --extra deepspeed pytest tests/gpu/gpu_ci/test_policy_local_engines_e2e.py -m "vllm"

"""

import asyncio
import hashlib
import json
import math
import os
from types import MappingProxyType

import hydra
import pytest
import ray
from omegaconf import DictConfig, open_dict
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.callbacks import TrainerCallback
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.utils.tracking import Tracking

from tests.gpu.dppo_diagnostics import PolicyWorker as DPPOPolicyWorker
from tests.gpu.dppo_diagnostics import compact_layer_capture
from tests.gpu.utils import get_test_prompts, init_inference_engines, init_worker_with_type, run_inference

MODEL = os.environ.get("SKYRL_GPU_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LM_HEAD_COMPUTE_DTYPE = os.environ.get("SKYRL_GPU_TEST_LM_HEAD_COMPUTE_DTYPE")
FLASH_ATTN = os.environ.get("SKYRL_GPU_TEST_FLASH_ATTN", "0") == "1"
VERIFY_PARITY = os.environ.get("SKYRL_GPU_TEST_VERIFY_PARITY", "0") == "1"
VERIFY_DPPO_UPDATE = os.environ.get("SKYRL_GPU_TEST_VERIFY_DPPO_UPDATE", "0") == "1"
EXPECTED_VLLM_ENGINE_SHA256 = os.environ.get("SKYRL_GPU_TEST_VLLM_ENGINE_SHA256")
PARITY_PROMPTS = (
    "Reply with four words about the sky.",
    "Name four common kitchen items.",
    "Give four words associated with winter.",
    "Reply with four words about music.",
    "Name four colors.",
    "Give four words associated with travel.",
    "Reply with four words about a library.",
    "Name four common animals.",
)
PARITY_TOKENS_PER_PROMPT = 4
DPPO_DIVERGENCE_THRESHOLD = 0.1
# A logprob error epsilon changes one sampled-token probability by at most
# exp(epsilon) - 1. Keep the worst pair below 80% of the DPPO TV threshold and
# the p95 pair below 50%.
MAX_SELECTED_LOGPROB_ERROR = math.log1p(0.8 * DPPO_DIVERGENCE_THRESHOLD)
P95_SELECTED_LOGPROB_ERROR = math.log1p(0.5 * DPPO_DIVERGENCE_THRESHOLD)


def test_compact_layer_capture_preserves_causal_conv_payload() -> None:
    capture = {
        "comparisons": {"live_vs_released": {"exact": True}},
        "inputs": {"weight": {"fingerprint": {"sha256": "weight"}}},
    }
    layer_entry = {"layer": 0, "causal_conv": [capture]}

    compact_layer_capture(layer_entry, "causal_conv", "causal-convolution")

    assert layer_entry["causal_conv"] == capture
    assert layer_entry["causal_conv"]["inputs"]["weight"]["fingerprint"]["sha256"] == "weight"
    assert layer_entry["causal_conv"]["comparisons"]["live_vs_released"]["exact"] is True


@pytest.mark.parametrize("captures", [[], [{}, {}]])
def test_compact_layer_capture_rejects_non_singleton_causal_conv(captures: list[dict]) -> None:
    layer_entry = {"layer": 0, "causal_conv": captures}

    with pytest.raises(RuntimeError, match="Expected one learner causal-convolution capture"):
        compact_layer_capture(layer_entry, "causal_conv", "causal-convolution")


class _OnePromptDataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return (
            [{"role": "user", "content": PARITY_PROMPTS[0]}],
            None,
            {},
            "dppo-update",
        )

    @staticmethod
    def collate_fn(entries):
        return [
            {"prompt": prompt, "env_class": env_class, "env_extras": env_extras, "uid": uid}
            for prompt, env_class, env_extras, uid in entries
        ]


class _InferenceTrajectoryRunner(TrajectoryRunner):
    trajectory_runner_cfg = MappingProxyType({})

    def __init__(self, client):
        self.client = client
        self.last_batch = None

    async def _run(self, input_batch, disable_tqdm=False):
        del disable_tqdm
        prompt_token_ids = self.client.tokenizer.apply_chat_template(
            input_batch["prompts"],
            add_generation_prompt=True,
            add_special_tokens=False,
            return_dict=True,
            tokenize=True,
        )["input_ids"]
        outputs = await self.client.generate(
            InferenceEngineInput(
                prompt_token_ids=prompt_token_ids,
                sampling_params=input_batch["sampling_params"],
            )
        )
        response_ids = outputs["response_ids"]
        rollout_logprobs = outputs["response_logprobs"]
        assert rollout_logprobs is not None
        assert len(response_ids) == 2
        assert all(len(ids) == len(logprobs) for ids, logprobs in zip(response_ids, rollout_logprobs, strict=True))
        assert all(math.isfinite(value) for row in rollout_logprobs for value in row)
        batch = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": response_ids,
            "rewards": [0.0, 1.0],
            "loss_masks": [[1] * len(ids) for ids in response_ids],
            "stop_reasons": outputs["stop_reasons"],
            "rollout_metrics": {},
            "rollout_logprobs": rollout_logprobs,
        }
        self.last_batch = batch
        return batch


class _StepMetrics(TrainerCallback):
    error_behavior = "raise"

    def __init__(self):
        self.metrics = None

    def on_step_end(self, state, control, **kwargs):
        del control, kwargs
        self.metrics = dict(state.metrics)


class _PreinitializedWeightSyncTrainer(RayPPOTrainer):
    """Reuse the communicator and rollout weights established by this test."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reuse_preloaded_rollout_weights = True
        self.weight_sync_calls = 0

    def init_weight_sync_state(self):
        # The test initializes this state before load 1 so it can verify the
        # learner/vLLM handshake independently of the optimizer update.
        return None

    async def _sync_policy_for_rollouts(self):
        self.weight_sync_calls += 1
        if self._reuse_preloaded_rollout_weights:
            self._reuse_preloaded_rollout_weights = False
            return
        await super()._sync_policy_for_rollouts()


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


def _head_input_error_summary(learner_values: list[float], engine_values: list[float]) -> dict:
    assert len(learner_values) == len(engine_values)
    absolute_errors = [abs(learner - engine) for learner, engine in zip(learner_values, engine_values, strict=True)]
    first_mismatch = next((index for index, error in enumerate(absolute_errors) if error != 0.0), None)
    return {
        "elements": len(absolute_errors),
        "exact": first_mismatch is None,
        "first_mismatch": first_mismatch,
        "max_absolute_error": max(absolute_errors),
        "p50_absolute_error": _percentile(absolute_errors, 0.50),
        "p95_absolute_error": _percentile(absolute_errors, 0.95),
        "learner_at_first_mismatch": None if first_mismatch is None else learner_values[first_mismatch],
        "engine_at_first_mismatch": None if first_mismatch is None else engine_values[first_mismatch],
    }


def _token_fingerprint_summary(learner_fingerprints: list[dict], engine_fingerprints: list[dict]) -> dict:
    assert len(learner_fingerprints) == len(engine_fingerprints)
    per_token_exact = [
        learner == engine for learner, engine in zip(learner_fingerprints, engine_fingerprints, strict=True)
    ]
    return {
        "per_token_exact": per_token_exact,
        "first_mismatch_token": next((index for index, exact in enumerate(per_token_exact) if not exact), None),
    }


def _logprob_error_gate(summary: dict) -> dict:
    absolute_error = summary["absolute_error"]
    return {
        "max": {
            "value": absolute_error["max"],
            "threshold": MAX_SELECTED_LOGPROB_ERROR,
            "passed": absolute_error["max"] <= MAX_SELECTED_LOGPROB_ERROR,
        },
        "p95": {
            "value": absolute_error["p95"],
            "threshold": P95_SELECTED_LOGPROB_ERROR,
            "passed": absolute_error["p95"] <= P95_SELECTED_LOGPROB_ERROR,
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
            with open_dict(cfg.generator):
                cfg.generator.max_logprobs = 2
        if VERIFY_DPPO_UPDATE:
            assert VERIFY_PARITY, "The DPPO update gate includes both pre- and post-update parity checks"
            assert not colocate_all, "The DPPO update gate uses separate learner and inference GPUs"
            cfg.generator.n_samples_per_prompt = 2
            cfg.generator.sampling_params.max_generate_length = PARITY_TOKENS_PER_PROMPT
            cfg.generator.sampling_params.logprobs = 1
            cfg.trainer.algorithm.policy_loss_type = "dppo"
            cfg.trainer.algorithm.dppo_divergence_type = "tv"
            cfg.trainer.algorithm.dppo_divergence_threshold = 0.1
            cfg.trainer.algorithm.use_kl_loss = False
            cfg.trainer.algorithm.use_kl_in_reward = False
            cfg.trainer.algorithm.use_tis = False
            with open_dict(cfg.trainer.algorithm):
                cfg.trainer.algorithm.max_seq_len = (
                    cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length
                )
            cfg.trainer.train_batch_size = 1
            cfg.trainer.policy_mini_batch_size = 1
            cfg.trainer.micro_forward_batch_size_per_gpu = 1
            cfg.trainer.micro_train_batch_size_per_gpu = 1
            cfg.trainer.update_epochs_per_batch = 1
            cfg.trainer.epochs = 1
            cfg.trainer.max_steps = 1
            cfg.trainer.eval_before_train = False
            cfg.trainer.resume_mode = "none"
            cfg.trainer.logger = "console"

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

        if VERIFY_PARITY:
            assert EXPECTED_VLLM_ENGINE_SHA256 is not None, (
                "SKYRL_GPU_TEST_VLLM_ENGINE_SHA256 must identify the prepared checkout"
            )
            engine_actor = client.engines[0].inference_engine_actor
            runtime_installations = ray.get(
                engine_actor.report_runtime_installation.remote(EXPECTED_VLLM_ENGINE_SHA256)
            )
            if isinstance(runtime_installations, dict):
                runtime_installations = [runtime_installations]
            assert runtime_installations
            assert all(
                installation["matches_checkout"] and installation["vllm_engine_sha256"] == EXPECTED_VLLM_ENGINE_SHA256
                for installation in runtime_installations
            ), runtime_installations
            print(f"SKYRL_ENGINECORE_RUNTIME_RESULT {json.dumps(runtime_installations, sort_keys=True)}")

        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=cfg.trainer.placement.colocate_all,
            num_gpus_per_node=cfg.generator.inference_engine_tensor_parallel_size,
            cfg=cfg,
            worker_cls=DPPOPolicyWorker,
        )
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        asyncio.run(client.reset_prefix_cache())
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        parity_values = []
        verified_engine_weights = {}

        def emit_parity_result():
            expected_per_load = len(PARITY_PROMPTS) * PARITY_TOKENS_PER_PROMPT
            overall_summary = _error_summary(parity_values)
            completed_loads = [
                update_index
                for update_index in (1, 2)
                if sum(value["load"] == update_index for value in parity_values) == expected_per_load
            ]
            payload = {
                "schema_version": 1,
                "design": {
                    "prompts": list(PARITY_PROMPTS),
                    "tokens_per_prompt": PARITY_TOKENS_PER_PROMPT,
                    "loads": 2,
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
                "thresholds": {
                    "dppo_divergence": DPPO_DIVERGENCE_THRESHOLD,
                    "max_selected_logprob_absolute_error": MAX_SELECTED_LOGPROB_ERROR,
                    "p95_selected_logprob_absolute_error": P95_SELECTED_LOGPROB_ERROR,
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
                "overall": overall_summary,
                "error_gate": _logprob_error_gate(overall_summary),
                "pairs": parity_values,
            }
            print(f"SKYRL_DPPO_PARITY_RESULT {json.dumps(payload, sort_keys=True, allow_nan=False)}")

        def verify_synced_weights(update_index: int):
            representative_names = [
                "model.language_model.embed_tokens.weight",
                "model.language_model.layers.0.input_layernorm.weight",
                "lm_head.weight",
            ]
            policy_fingerprints = {}
            per_rank = ray.get(policy.async_run_ray_method("pass_through", "fingerprint_broadcast_weights"))
            for rank_fingerprints in per_rank:
                if isinstance(rank_fingerprints, dict):
                    policy_fingerprints.update(rank_fingerprints)
            assert all(name in policy_fingerprints for name in representative_names), policy_fingerprints.keys()
            weight_names = sorted(policy_fingerprints)

            engine_actor = client.engines[0].inference_engine_actor
            expected_shapes = {name: value["shape"] for name, value in policy_fingerprints.items()}
            engine_per_rank = ray.get(engine_actor.fingerprint_engine_weights.remote(weight_names, expected_shapes))
            if isinstance(engine_per_rank, dict):
                engine_per_rank = [engine_per_rank]
            assert len(engine_per_rank) == 1, len(engine_per_rank)
            mismatches = []
            for name in weight_names:
                entry = engine_per_rank[0][name]
                if not entry["found"]:
                    mismatches.append({"name": name, "reason": "not_found", "engine": entry})
                elif "tensor" in entry:
                    mismatches.append({"name": name, "reason": "returned_tensor"})
                elif entry.get("shape_mismatch"):
                    mismatches.append({"name": name, "reason": "shape_mismatch", "engine": entry})
                elif entry["fingerprint"] != policy_fingerprints[name]:
                    mismatches.append(
                        {
                            "name": name,
                            "reason": "fingerprint_mismatch",
                            "learner": policy_fingerprints[name],
                            "engine": entry,
                        }
                    )
            assert not mismatches, {
                "mismatch_count": len(mismatches),
                "sample": mismatches[:20],
                "weight_count": len(weight_names),
            }
            assert engine_per_rank[0]["model.language_model.embed_tokens.weight"]["dtype"] == "bfloat16"
            assert engine_per_rank[0]["lm_head.weight"]["dtype"] == "float32"
            assert (
                engine_per_rank[0]["lm_head.weight"]["internal_name"]
                != engine_per_rank[0]["model.language_model.embed_tokens.weight"]["internal_name"]
            )
            print(f"Verified {len(weight_names)} exact learner/vLLM weights after complete load {update_index}")
            verified_engine_weights[update_index] = engine_per_rank[0]
            return engine_per_rank[0]

        def generate_with_logprob_checks(update_index: int):
            sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
            sampling_params["logprobs"] = 1
            if VERIFY_PARITY:
                prompts = [[{"role": "user", "content": content}] for content in PARITY_PROMPTS]
                sampling_params["max_tokens"] = PARITY_TOKENS_PER_PROMPT
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
                assert len(outputs["response_ids"]) == len(PARITY_PROMPTS)
                assert all(len(tokens) == PARITY_TOKENS_PER_PROMPT for tokens in outputs["response_ids"])
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
                                "score_next_token",
                                prompt_ids + response_ids[:token_index],
                                selected_token,
                            )
                        )
                        assert len(learner_results) == 1, learner_results
                        result = learner_results[0]
                        assert math.isfinite(result["selected_logprob"]), result
                        assert math.isfinite(result["selected_logit"]), result
                        assert math.isfinite(result["logsumexp"]), result
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
                assert len(load_values) == len(PARITY_PROMPTS) * PARITY_TOKENS_PER_PROMPT
                print(
                    f"Measured learner/vLLM selected-token logprobs after complete load {update_index}: "
                    f"{json.dumps({'summary': _error_summary(load_values), 'pairs': load_values}, sort_keys=True)}"
                )
                emit_parity_result()
                load_gate = _logprob_error_gate(_error_summary(load_values))

                if not all(entry["passed"] for entry in load_gate.values()):
                    max_error_pair = max(load_values, key=lambda value: value["absolute_error"])
                    diagnostic_prompt_indices = {
                        value["prompt_index"] for value in load_values if not value["top1_match"]
                    }
                    diagnostic_prompt_indices.add(max_error_pair["prompt_index"])
                    diagnostic_pairs = [
                        value for value in load_values if value["prompt_index"] in diagnostic_prompt_indices
                    ]

                    async def collect_fast_path_diagnostics():
                        diagnostics = []
                        diagnostic_generation_params = dict(sampling_params)
                        diagnostic_generation_params["logprobs"] = 2
                        for prompt_index in dict.fromkeys(value["prompt_index"] for value in diagnostic_pairs):
                            prompt_pair_indices = [
                                value["token_index"]
                                for value in diagnostic_pairs
                                if value["prompt_index"] == prompt_index
                            ]
                            prompt_ids = prompt_token_ids[prompt_index]
                            original_response_ids = outputs["response_ids"][prompt_index]
                            repeats = []
                            for _ in range(2):
                                await client.reset_prefix_cache()
                                generated = await client.generate(
                                    InferenceEngineInput(
                                        prompt_token_ids=[prompt_ids],
                                        sampling_params=diagnostic_generation_params,
                                    )
                                )
                                await client.reset_prefix_cache()
                                rescored = await client.generate(
                                    InferenceEngineInput(
                                        prompt_token_ids=[prompt_ids + original_response_ids],
                                        sampling_params=scoring_params,
                                    )
                                )
                                repeated_prompt_logprobs = rescored["prompt_logprobs"]
                                assert repeated_prompt_logprobs is not None
                                selected_rescores = []
                                for token_index in prompt_pair_indices:
                                    selected_token = original_response_ids[token_index]
                                    candidates = repeated_prompt_logprobs[0][len(prompt_ids) + token_index]
                                    assert candidates is not None
                                    assert selected_token in candidates
                                    selected_rescores.append(
                                        {
                                            "token_index": token_index,
                                            "selected_token": selected_token,
                                            "selected_logprob": float(candidates[selected_token]),
                                            "top_candidates": [
                                                {"token": int(token), "logprob": float(logprob)}
                                                for token, logprob in sorted(
                                                    candidates.items(), key=lambda item: item[1], reverse=True
                                                )[:2]
                                            ],
                                        }
                                    )
                                repeats.append(
                                    {
                                        "generated_response_ids": generated["response_ids"][0],
                                        "generated_response_logprobs": generated["response_logprobs"][0],
                                        "selected_rescores": selected_rescores,
                                    }
                                )
                            await client.reset_prefix_cache()
                            duplicate_batch = await client.generate(
                                InferenceEngineInput(
                                    prompt_token_ids=[prompt_ids, prompt_ids],
                                    sampling_params=diagnostic_generation_params,
                                )
                            )
                            first_singleton_ids = repeats[0]["generated_response_ids"]
                            shared_prefix_length = 0
                            for original_token, repeated_token in zip(
                                original_response_ids, first_singleton_ids, strict=False
                            ):
                                if original_token != repeated_token:
                                    break
                                shared_prefix_length += 1
                            diagnostics.append(
                                {
                                    "prompt_index": prompt_index,
                                    "prompt_token_count": len(prompt_ids),
                                    "prompt_token_sha256": hashlib.sha256(
                                        json.dumps(prompt_ids, separators=(",", ":")).encode()
                                    ).hexdigest(),
                                    "original_response_ids": original_response_ids,
                                    "original_response_token_sha256": hashlib.sha256(
                                        json.dumps(original_response_ids, separators=(",", ":")).encode()
                                    ).hexdigest(),
                                    "original_response_logprobs": response_logprobs[prompt_index],
                                    "original_singleton_shared_prefix_length": shared_prefix_length,
                                    "singleton_repeats_identical": repeats[0] == repeats[1],
                                    "duplicate_batch": {
                                        "response_ids": duplicate_batch["response_ids"],
                                        "response_logprobs": duplicate_batch["response_logprobs"],
                                        "identical": (
                                            duplicate_batch["response_ids"][0] == duplicate_batch["response_ids"][1]
                                            and duplicate_batch["response_logprobs"][0]
                                            == duplicate_batch["response_logprobs"][1]
                                        ),
                                    },
                                    "repeats": repeats,
                                }
                            )
                        return diagnostics

                    weights_before_diagnostics = verified_engine_weights[update_index]
                    diagnostics = asyncio.run(collect_fast_path_diagnostics())
                    head_input_diagnostics = []
                    engine_actor = client.engines[0].inference_engine_actor
                    for pair in diagnostic_pairs:
                        prompt_index = pair["prompt_index"]
                        token_index = pair["token_index"]
                        selected_token = pair["selected_token"]
                        prefix_ids = (
                            prompt_token_ids[prompt_index] + outputs["response_ids"][prompt_index][:token_index]
                        )
                        learner_results = ray.get(
                            policy.async_run_ray_method(
                                "pass_through",
                                "score_next_token",
                                prefix_ids,
                                selected_token,
                                True,
                                len(prompt_token_ids[prompt_index]),
                            )
                        )
                        assert len(learner_results) == 1, learner_results
                        learner_result = learner_results[0]

                        async def capture_engine_head_input():
                            capture_started = False
                            try:
                                await asyncio.to_thread(
                                    ray.get,
                                    engine_actor.begin_head_input_capture.remote(selected_token),
                                )
                                capture_started = True
                                await client.reset_prefix_cache()
                                capture_params = dict(sampling_params)
                                capture_params.update({"max_tokens": 1, "logprobs": 2})
                                generated = await client.generate(
                                    InferenceEngineInput(
                                        prompt_token_ids=[prefix_ids],
                                        sampling_params=capture_params,
                                    )
                                )
                            finally:
                                if capture_started:
                                    captures = await asyncio.to_thread(
                                        ray.get, engine_actor.end_head_input_capture.remote()
                                    )
                            return generated, captures

                        captured_generation, engine_captures_per_rank = asyncio.run(capture_engine_head_input())
                        assert len(engine_captures_per_rank) == 1, engine_captures_per_rank
                        engine_captures = engine_captures_per_rank[0]
                        assert len(engine_captures) == 1, engine_captures
                        engine_capture = engine_captures[0]
                        assert engine_capture["compute_logits_input_shape"][0] == 1, engine_capture
                        assert learner_result["head_input_shape"] == engine_capture["head_input_shape"]
                        learner_layer_trace = learner_result.pop("layer_trace")
                        engine_layer_trace = engine_capture.pop("layer_trace")
                        assert len(learner_layer_trace) == len(engine_layer_trace)
                        layer_diagnostics = []
                        for learner_layer, engine_layer in zip(
                            learner_layer_trace,
                            engine_layer_trace,
                            strict=True,
                        ):
                            assert learner_layer["layer"] == engine_layer["layer"]
                            assert learner_layer["mixer"] == engine_layer["mixer"]
                            projection_diagnostics = {}
                            learner_projections = learner_layer.pop("projections", None)
                            engine_projections = engine_layer.pop("projections", None)
                            assert (learner_projections is None) == (engine_projections is None)
                            if learner_projections is not None:
                                assert engine_projections is not None
                                for mode, engine_mode in engine_projections.items():
                                    mode_diagnostics = []
                                    for projection_name, learner_projection in learner_projections.items():
                                        engine_projection = engine_mode[projection_name]
                                        assert learner_projection["shape"] == engine_projection["shape"]
                                        mode_diagnostics.append(
                                            {
                                                "projection": projection_name,
                                                "learner_dtype": learner_projection["dtype"],
                                                "engine_dtype": engine_projection["dtype"],
                                                "shape": learner_projection["shape"],
                                                "error": _head_input_error_summary(
                                                    learner_projection["values"],
                                                    engine_projection.pop("values"),
                                                ),
                                                "token_fingerprints": _token_fingerprint_summary(
                                                    learner_projection["token_fingerprints"],
                                                    engine_projection["token_fingerprints"],
                                                ),
                                            }
                                        )
                                    projection_diagnostics[mode] = mode_diagnostics
                                for projection in learner_projections.values():
                                    projection.pop("values")
                            stage_diagnostics = []
                            learner_stages = learner_layer.pop("mixer_stages", None)
                            engine_stages = engine_layer.pop("mixer_stages", None)
                            assert (learner_stages is None) == (engine_stages is None)
                            if learner_stages is not None:
                                assert engine_stages is not None
                                assert learner_stages.keys() == engine_stages.keys()
                                for stage_name, learner_stage in learner_stages.items():
                                    engine_stage = engine_stages[stage_name]
                                    assert learner_stage["shape"] == engine_stage["shape"]
                                    stage_diagnostic = {
                                        "stage": stage_name,
                                        "learner_dtype": learner_stage["dtype"],
                                        "engine_dtype": engine_stage["dtype"],
                                        "shape": learner_stage["shape"],
                                        "error": _head_input_error_summary(
                                            learner_stage.pop("values"),
                                            engine_stage.pop("values"),
                                        ),
                                    }
                                    learner_tokens = learner_stage.pop("token_fingerprints", None)
                                    engine_tokens = engine_stage.pop("token_fingerprints", None)
                                    assert (learner_tokens is None) == (engine_tokens is None)
                                    if learner_tokens is not None:
                                        assert engine_tokens is not None
                                        stage_diagnostic["token_fingerprints"] = _token_fingerprint_summary(
                                            learner_tokens,
                                            engine_tokens,
                                        )
                                    stage_diagnostics.append(stage_diagnostic)
                            mlp_stage_diagnostics = None
                            learner_mlp_stages = learner_layer.pop("mlp_stages", None)
                            engine_mlp_stages = engine_layer.pop("mlp_stages", None)
                            engine_mlp_replays = engine_layer.pop("mlp_replays", None)
                            assert (learner_mlp_stages is None) == (engine_mlp_stages is None)
                            assert (learner_mlp_stages is None) == (engine_mlp_replays is None)
                            if learner_mlp_stages is not None:
                                assert engine_mlp_stages is not None
                                assert engine_mlp_replays is not None
                                assert learner_mlp_stages.keys() == {
                                    "activation",
                                    "down",
                                    "gate",
                                    "product",
                                    "up",
                                }
                                assert engine_mlp_stages.keys() == {"down", "gate", "product", "up"}
                                runtime = []
                                for stage_name, engine_stage in engine_mlp_stages.items():
                                    learner_stage = learner_mlp_stages[stage_name]
                                    assert learner_stage["shape"] == engine_stage["shape"]
                                    runtime.append(
                                        {
                                            "stage": stage_name,
                                            "learner_dtype": learner_stage["dtype"],
                                            "engine_dtype": engine_stage["dtype"],
                                            "shape": learner_stage["shape"],
                                            "error": _head_input_error_summary(
                                                learner_stage["values"],
                                                engine_stage.pop("values"),
                                            ),
                                        }
                                    )
                                replay_targets = {
                                    "separate_gate": "gate",
                                    "separate_up": "up",
                                    "native_activation": "activation",
                                    "separate_native_activation": "activation",
                                    "native_product": "product",
                                    "separate_native_product": "product",
                                    "fused_separate_product": "product",
                                    "fused_separate_down": "down",
                                }
                                replays = []
                                for replay_name, learner_stage_name in replay_targets.items():
                                    replay = engine_mlp_replays[replay_name]
                                    learner_stage = learner_mlp_stages[learner_stage_name]
                                    assert learner_stage["shape"] == replay["shape"]
                                    replays.append(
                                        {
                                            "replay": replay_name,
                                            "learner_stage": learner_stage_name,
                                            "learner_dtype": learner_stage["dtype"],
                                            "engine_dtype": replay["dtype"],
                                            "shape": learner_stage["shape"],
                                            "error": _head_input_error_summary(
                                                learner_stage["values"],
                                                replay.pop("values"),
                                            ),
                                        }
                                    )
                                for learner_stage in learner_mlp_stages.values():
                                    learner_stage.pop("values")
                                mlp_stage_diagnostics = {
                                    "runtime": runtime,
                                    "engine_replays": replays,
                                }
                            attention_diagnostics = None
                            learner_attention_stages = learner_layer.pop("attention_stages", None)
                            learner_attention_replays = learner_layer.pop("attention_replays", None)
                            engine_attention_stages = engine_layer.pop("attention_stages", None)
                            engine_attention_replays = engine_layer.pop("attention_replays", None)
                            assert (learner_attention_stages is None) == (learner_attention_replays is None)
                            assert (learner_attention_stages is None) == (engine_attention_stages is None)
                            assert (learner_attention_stages is None) == (engine_attention_replays is None)
                            if learner_attention_stages is not None:
                                assert learner_attention_replays is not None
                                assert engine_attention_stages is not None
                                assert engine_attention_replays is not None

                                def compare_attention_payload(label, learner_payload, engine_payload):
                                    assert learner_payload["shape"] == engine_payload["shape"]
                                    comparison = {
                                        "comparison": label,
                                        "learner_dtype": learner_payload["dtype"],
                                        "engine_dtype": engine_payload["dtype"],
                                        "shape": learner_payload["shape"],
                                        "error": _head_input_error_summary(
                                            learner_payload["values"],
                                            engine_payload["values"],
                                        ),
                                    }
                                    learner_sequence = learner_payload.get("sequence_fingerprint")
                                    engine_sequence = engine_payload.get("sequence_fingerprint")
                                    if learner_sequence is not None or engine_sequence is not None:
                                        assert learner_payload["sequence_shape"] == engine_payload["sequence_shape"]
                                        comparison.update(
                                            {
                                                "sequence_shape": learner_payload["sequence_shape"],
                                                "sequence_fingerprint_exact": learner_sequence == engine_sequence,
                                                "token_fingerprints": _token_fingerprint_summary(
                                                    learner_payload["token_fingerprints"],
                                                    engine_payload["token_fingerprints"],
                                                ),
                                            }
                                        )
                                    return comparison

                                runtime_targets = {
                                    "q_raw": "q_raw",
                                    "gate": "gate",
                                    "k_raw": "k_raw",
                                    "v_raw": "v_raw",
                                    "post_gate": "post_gate",
                                    "out_proj": "out_proj",
                                }
                                runtime = [
                                    compare_attention_payload(
                                        f"runtime:{engine_name}",
                                        learner_attention_stages[learner_name],
                                        engine_attention_stages[engine_name],
                                    )
                                    for engine_name, learner_name in runtime_targets.items()
                                ]
                                processed_targets = {
                                    "q_rope": "q_rope",
                                    "k_rope": "k_rope",
                                    "v": "v",
                                    "attention_core": "attention_core",
                                }
                                processed = [
                                    compare_attention_payload(
                                        f"learner_replay_vs_engine_live:{engine_name}",
                                        learner_attention_replays[learner_name],
                                        engine_attention_stages[engine_name],
                                    )
                                    for engine_name, learner_name in processed_targets.items()
                                ]
                                processed.extend(
                                    (
                                        compare_attention_payload(
                                            "runtime_eager:q_norm",
                                            learner_attention_stages["q_norm"],
                                            engine_attention_replays["runtime_eager_q_norm"],
                                        ),
                                        compare_attention_payload(
                                            "runtime_eager:k_norm",
                                            learner_attention_stages["k_norm"],
                                            engine_attention_replays["runtime_eager_k_norm"],
                                        ),
                                    )
                                )
                                replay_targets = {
                                    "separate_q_raw": ("stages", "q_raw"),
                                    "separate_gate": ("stages", "gate"),
                                    "separate_k_raw": ("stages", "k_raw"),
                                    "separate_v_raw": ("stages", "v_raw"),
                                    "runtime_eager_q_rope": ("replays", "q_rope"),
                                    "runtime_eager_k_rope": ("replays", "k_rope"),
                                    "runtime_eager_v": ("replays", "v"),
                                    "runtime_eager_gate": ("stages", "gate"),
                                    "separate_eager_q_norm": ("stages", "q_norm"),
                                    "separate_eager_k_norm": ("stages", "k_norm"),
                                    "separate_eager_q_rope": ("replays", "q_rope"),
                                    "separate_eager_k_rope": ("replays", "k_rope"),
                                    "separate_eager_v": ("replays", "v"),
                                    "separate_eager_gate": ("stages", "gate"),
                                    "separate_fused_q_rope": ("replays", "q_rope"),
                                    "separate_fused_k_rope": ("replays", "k_rope"),
                                    "separate_fused_v": ("replays", "v"),
                                    "separate_fused_gate": ("stages", "gate"),
                                    "post_gate": ("replays", "post_gate"),
                                }
                                replay_comparisons = []
                                for replay_name, (learner_source, learner_name) in replay_targets.items():
                                    learner_payloads = (
                                        learner_attention_stages
                                        if learner_source == "stages"
                                        else learner_attention_replays
                                    )
                                    replay_comparisons.append(
                                        compare_attention_payload(
                                            f"engine_replay:{replay_name}",
                                            learner_payloads[learner_name],
                                            engine_attention_replays[replay_name],
                                        )
                                    )
                                same_side = {
                                    "learner_live_vs_learner_replay_post_gate": compare_attention_payload(
                                        "learner_live_vs_learner_replay:post_gate",
                                        learner_attention_stages["post_gate"],
                                        learner_attention_replays["post_gate"],
                                    ),
                                    "learner_live_vs_learner_replay_out_proj": compare_attention_payload(
                                        "learner_live_vs_learner_replay:out_proj",
                                        learner_attention_stages["out_proj"],
                                        learner_attention_replays["out_proj"],
                                    ),
                                    "engine_live_vs_engine_replay_post_gate": compare_attention_payload(
                                        "engine_live_vs_engine_replay:post_gate",
                                        engine_attention_stages["post_gate"],
                                        engine_attention_replays["post_gate"],
                                    ),
                                }
                                for payloads in (
                                    learner_attention_stages,
                                    learner_attention_replays,
                                    engine_attention_stages,
                                    engine_attention_replays,
                                ):
                                    for payload in payloads.values():
                                        payload.pop("values")
                                attention_diagnostics = {
                                    "runtime": runtime,
                                    "processed": processed,
                                    "engine_replays": replay_comparisons,
                                    "same_side_replays": same_side,
                                    "learner": {
                                        "stages": learner_attention_stages,
                                        "replays": learner_attention_replays,
                                    },
                                    "engine": {
                                        "stages": engine_attention_stages,
                                        "replays": engine_attention_replays,
                                    },
                                }
                            learner_fla_core = learner_layer.pop("fla_core", None)
                            engine_fla_core = engine_layer.pop("fla_core", None)
                            assert (learner_fla_core is None) == (engine_fla_core is None)
                            fla_core = None
                            if learner_fla_core is not None:
                                assert engine_fla_core is not None
                                input_comparisons = {}
                                for name in ("q", "k", "v", "g", "beta"):
                                    learner_input = learner_fla_core["inputs"][name]
                                    engine_input = engine_fla_core["inputs"][name]
                                    learner_effective_fingerprint = (
                                        learner_input["normalized_fingerprint"]
                                        if name in {"q", "k"}
                                        else learner_input["fingerprint"]
                                    )
                                    learner_effective_token_fingerprints = (
                                        learner_input["normalized_token_fingerprints"]
                                        if name in {"q", "k"}
                                        else learner_input["token_fingerprints"]
                                    )
                                    input_comparisons[name] = {
                                        "directly_comparable": name not in {"q", "k"},
                                        "fingerprint_exact": (
                                            learner_input["fingerprint"] == engine_input["fingerprint"]
                                            if name not in {"q", "k"}
                                            else None
                                        ),
                                        "learner_semantics": (
                                            "raw; normalized inside FLA" if name in {"q", "k"} else "FLA input"
                                        ),
                                        "engine_semantics": (
                                            "normalized before FLA" if name in {"q", "k"} else "FLA input"
                                        ),
                                        "effective_comparison": {
                                            "learner_semantics": (
                                                "normalized with released FLA l2norm_fwd"
                                                if name in {"q", "k"}
                                                else "live FLA input"
                                            ),
                                            "engine_semantics": "live FLA input",
                                            "fingerprint_exact": (
                                                learner_effective_fingerprint == engine_input["fingerprint"]
                                            ),
                                            "token_fingerprints": _token_fingerprint_summary(
                                                learner_effective_token_fingerprints,
                                                engine_input["token_fingerprints"],
                                            ),
                                        },
                                        "learner": learner_input,
                                        "engine": engine_input,
                                    }
                                engine_initial_state = engine_fla_core["inputs"]["initial_state"]
                                engine_causal_conv = engine_layer.pop("causal_conv")
                                learner_causal_conv = learner_layer.pop("causal_conv")
                                fla_core = {
                                    "cross_engine": {
                                        "causal_conv": {
                                            "inputs": {
                                                "weight_fingerprint_exact": (
                                                    learner_causal_conv["inputs"]["weight"]["fingerprint"]
                                                    == engine_causal_conv["inputs"]["weight"]["fingerprint"]
                                                ),
                                                "x_fingerprint_exact": (
                                                    learner_causal_conv["inputs"]["x"]["fingerprint"]
                                                    == engine_causal_conv["inputs"]["x"]["fingerprint"]
                                                ),
                                                "x_token_fingerprints": _token_fingerprint_summary(
                                                    learner_causal_conv["inputs"]["x"]["token_fingerprints"],
                                                    engine_causal_conv["inputs"]["x"]["token_fingerprints"],
                                                ),
                                            },
                                            "learner_live_vs_engine_scratch": {
                                                "fingerprint_exact": (
                                                    learner_causal_conv["live"]["fingerprint"]
                                                    == engine_causal_conv["scratch_replay"]["fingerprint"]
                                                ),
                                                "token_fingerprints": _token_fingerprint_summary(
                                                    learner_causal_conv["live"]["token_fingerprints"],
                                                    engine_causal_conv["scratch_replay"]["token_fingerprints"],
                                                ),
                                            },
                                            "learner_vs_released_replay": {
                                                "fingerprint_exact": (
                                                    learner_causal_conv["released_replay"]["fingerprint"]
                                                    == engine_causal_conv["released_replay"]["fingerprint"]
                                                ),
                                                "token_fingerprints": _token_fingerprint_summary(
                                                    learner_causal_conv["released_replay"]["token_fingerprints"],
                                                    engine_causal_conv["released_replay"]["token_fingerprints"],
                                                ),
                                            },
                                            "learner": learner_causal_conv,
                                            "engine": engine_causal_conv,
                                        },
                                        "inputs": input_comparisons,
                                        "output_fingerprint_exact": (
                                            learner_fla_core["live"]["output"]["fingerprint"]
                                            == engine_fla_core["live"]["output_fingerprint"]
                                        ),
                                        "initial_state": {
                                            "directly_comparable": False,
                                            "learner": "None; FLA initializes an implicit zero N,H,K,V state",
                                            "engine_layout_semantics": "N,H,V,K",
                                            "engine": engine_initial_state,
                                        },
                                    },
                                    "learner": learner_fla_core,
                                    "engine": engine_fla_core,
                                }
                            boundary_diagnostics = []
                            for boundary in ("mixer_input", "mixer_output", "mlp_input", "mlp_output"):
                                learner_boundary = learner_layer[boundary]
                                engine_boundary = engine_layer[boundary]
                                assert learner_boundary["shape"] == engine_boundary["shape"]
                                boundary_diagnostic = {
                                    "boundary": boundary,
                                    "learner_dtype": learner_boundary["dtype"],
                                    "engine_dtype": engine_boundary["dtype"],
                                    "shape": learner_boundary["shape"],
                                    "error": _head_input_error_summary(
                                        learner_boundary.pop("values"),
                                        engine_boundary.pop("values"),
                                    ),
                                }
                                learner_tokens = learner_boundary.pop("token_fingerprints", None)
                                engine_tokens = engine_boundary.pop("token_fingerprints", None)
                                assert (learner_tokens is None) == (engine_tokens is None)
                                if learner_tokens is not None:
                                    boundary_diagnostic["token_fingerprints"] = _token_fingerprint_summary(
                                        learner_tokens,
                                        engine_tokens,
                                    )
                                boundary_diagnostics.append(boundary_diagnostic)
                            layer_diagnostics.append(
                                {
                                    "layer": learner_layer["layer"],
                                    "mixer": learner_layer["mixer"],
                                    "boundaries": boundary_diagnostics,
                                    "fla_core": fla_core,
                                    "projections": projection_diagnostics,
                                    "mixer_stages": stage_diagnostics,
                                    "mlp_stages": mlp_stage_diagnostics,
                                    "attention": attention_diagnostics,
                                }
                            )
                        head_input_diagnostics.append(
                            {
                                "load": update_index,
                                "prompt_index": prompt_index,
                                "token_index": token_index,
                                "prefix_token_count": len(prefix_ids),
                                "prefix_token_sha256": hashlib.sha256(
                                    json.dumps(prefix_ids, separators=(",", ":")).encode()
                                ).hexdigest(),
                                "selected_token": selected_token,
                                "generated_token": captured_generation["response_ids"][0][0],
                                "learner_head_input_dtype": learner_result["head_input_dtype"],
                                "engine_head_input_dtype": engine_capture["head_input_dtype"],
                                "learner_head_input_shape": learner_result["head_input_shape"],
                                "learner_output_embedding_input_shape": learner_result["output_embedding_input_shape"],
                                "engine_head_input_shape": engine_capture["head_input_shape"],
                                "engine_compute_logits_input_shape": engine_capture["compute_logits_input_shape"],
                                "learner_logits_dtype": learner_result["logits_dtype"],
                                "engine_logits_dtype": engine_capture["logits_dtype"],
                                "learner_selected_logit": learner_result["selected_logit"],
                                "engine_selected_logit": engine_capture["selected_logit"],
                                "selected_logit_absolute_error": abs(
                                    learner_result["selected_logit"] - engine_capture["selected_logit"]
                                ),
                                "learner_logsumexp": learner_result["logsumexp"],
                                "engine_logsumexp": engine_capture["logsumexp"],
                                "logsumexp_absolute_error": abs(
                                    learner_result["logsumexp"] - engine_capture["logsumexp"]
                                ),
                                "learner_selected_logprob": learner_result["selected_logprob"],
                                "engine_selected_logprob": engine_capture["selected_logprob"],
                                "layers": layer_diagnostics,
                                "error": _head_input_error_summary(
                                    learner_result.pop("head_input"),
                                    engine_capture.pop("head_input"),
                                ),
                            }
                        )
                    weights_after_diagnostics = verify_synced_weights(update_index)
                    fingerprint_names = (
                        "model.language_model.embed_tokens.weight",
                        "model.language_model.layers.0.input_layernorm.weight",
                        "lm_head.weight",
                    )
                    diagnostic_payload = {
                        "load": update_index,
                        "resolved_sampling_params": {
                            name: sampling_params.get(name)
                            for name in (
                                "temperature",
                                "top_p",
                                "top_k",
                                "min_p",
                                "repetition_penalty",
                                "presence_penalty",
                                "frequency_penalty",
                                "seed",
                            )
                        },
                        "max_num_batched_tokens": cfg.generator.get("max_num_batched_tokens"),
                        "weights_unchanged": all(
                            weights_before_diagnostics[name]["fingerprint"]
                            == weights_after_diagnostics[name]["fingerprint"]
                            for name in fingerprint_names
                        ),
                        "head_inputs": head_input_diagnostics,
                        "prompts": diagnostics,
                    }
                    print(
                        "SKYRL_DPPO_FAST_PATH_DIAGNOSTIC "
                        f"{json.dumps(diagnostic_payload, sort_keys=True, allow_nan=False)}"
                    )
                assert load_gate["max"]["passed"], load_gate
                assert load_gate["p95"]["passed"], load_gate
            print(
                f"Verified rollout logprobs after complete load {update_index} "
                f"for {sum(map(len, response_logprobs))} tokens"
            )
            return outputs

        if VERIFY_PARITY:
            first_engine_weights = verify_synced_weights(1)
            generate_with_logprob_checks(1)
            dppo_update_evidence = None
            if VERIFY_DPPO_UPDATE:
                trajectory_runner = _InferenceTrajectoryRunner(client)
                step_metrics = _StepMetrics()
                trainer = _PreinitializedWeightSyncTrainer(
                    cfg=cfg,
                    tracker=Tracking("gpu-ci", "dppo-one-update", backends="console", config=cfg),
                    tokenizer=client.tokenizer,
                    train_dataset=_OnePromptDataset(),
                    inference_engine_client=client,
                    trajectory_runner=trajectory_runner,
                    colocate_pg=pg,
                    callbacks=[step_metrics],
                )
                trainer.policy_model = policy
                asyncio.run(trainer._train_loop())
                assert trainer.global_step == 1
                assert trainer.weight_sync_calls == 2
                assert step_metrics.metrics is not None
                assert trajectory_runner.last_batch is not None

                required_metrics = (
                    "policy/final_loss",
                    "policy/policy_loss",
                    "policy/raw_grad_norm",
                    "policy/policy_update_steps",
                    "policy/dppo/masked_fraction",
                    "policy/dppo/divergence_mean",
                    "policy/dppo/divergence_max",
                    "policy/dppo/max_retained_log_ratio",
                )
                update_metrics = {name: float(step_metrics.metrics[name]) for name in required_metrics}
                assert all(math.isfinite(value) for value in update_metrics.values()), update_metrics
                assert update_metrics["policy/raw_grad_norm"] > 0.0, update_metrics
                assert update_metrics["policy/policy_update_steps"] == 1.0, update_metrics
                assert update_metrics["policy/dppo/masked_fraction"] == 0.0, update_metrics

                batch = trajectory_runner.last_batch
                active_tokens = sum(sum(mask) for mask in batch["loss_masks"])
                rollout_tokens = sum(len(row) for row in batch["rollout_logprobs"])
                assert active_tokens > 0
                assert active_tokens == rollout_tokens
                assert all(
                    len(ids) == len(mask) == len(logprobs)
                    for ids, mask, logprobs in zip(
                        batch["response_ids"], batch["loss_masks"], batch["rollout_logprobs"], strict=True
                    )
                )
                dppo_update_evidence = {
                    "schema_version": 1,
                    "config": {
                        "policy_loss_type": cfg.trainer.algorithm.policy_loss_type,
                        "dppo_divergence_type": cfg.trainer.algorithm.dppo_divergence_type,
                        "dppo_divergence_threshold": cfg.trainer.algorithm.dppo_divergence_threshold,
                        "samples_per_prompt": cfg.generator.n_samples_per_prompt,
                        "optimizer_updates": 1,
                    },
                    "rollout": {
                        "trajectories": len(batch["response_ids"]),
                        "rollout_logprob_tokens": rollout_tokens,
                        "active_loss_mask_tokens": active_tokens,
                        "finite_rollout_logprobs": True,
                    },
                    "metrics": update_metrics,
                }
                print(f"Completed one real DPPO optimizer update: {json.dumps(dppo_update_evidence, sort_keys=True)}")
            else:
                mutation_results = ray.get(
                    policy.async_run_ray_method(
                        # The learner is the unwrapped text tower (``model.*``); the
                        # extractor adds ``model.language_model.*`` for vLLM.
                        "pass_through",
                        "perturb_weight",
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

            assert (
                first_engine_weights["lm_head.weight"]["fingerprint"]["sha256"]
                != second_engine_weights["lm_head.weight"]["fingerprint"]["sha256"]
            )
            for identity_key in ("internal_name", "parameter_id", "data_ptr"):
                assert (
                    first_engine_weights["lm_head.weight"][identity_key]
                    == second_engine_weights["lm_head.weight"][identity_key]
                )
            print("Verified stable FP32 projection storage across two complete loads")
            if dppo_update_evidence is not None:
                dppo_update_evidence["weight_sync"] = {
                    "updated_lm_head": True,
                    "exact_learner_engine_match": True,
                    "post_update_inference": True,
                }
                print(f"SKYRL_DPPO_UPDATE_RESULT {json.dumps(dppo_update_evidence, sort_keys=True, allow_nan=False)}")
            assert len(parity_values) == 2 * len(PARITY_PROMPTS) * PARITY_TOKENS_PER_PROMPT
            assert all(value["top1_match"] for value in parity_values), parity_values
            overall_gate = _logprob_error_gate(_error_summary(parity_values))
            assert overall_gate["max"]["passed"], overall_gate
            assert overall_gate["p95"]["passed"], overall_gate
            outputs = second_outputs
        else:
            sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
            outputs = asyncio.run(run_inference(client, get_test_prompts(MODEL), sampling_params))
            assert len(outputs["responses"]) == len(outputs["response_ids"])
        print(f"Example output: {outputs['responses'][0]}, {outputs['stop_reasons'][0]}")
    finally:
        ray.shutdown()
