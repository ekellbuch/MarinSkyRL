"""
# Run only vllm tests (requires vllm extra):
uv run --isolated --group dev --extra vllm --extra deepspeed pytest tests/gpu/gpu_ci/test_policy_local_engines_e2e.py -m "vllm"

"""

import asyncio
import math
import os

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
                prompts = [[{"role": "user", "content": "Reply with one word."}]]
                sampling_params["max_tokens"] = 1
                sampling_params["temperature"] = 0.0
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
                assert len(outputs["response_ids"]) == 1 and len(outputs["response_ids"][0]) == 1
                selected_token = outputs["response_ids"][0][0]
                learner_results = ray.get(
                    policy.async_run_ray_method(
                        "pass_through",
                        "direct_next_token_parity_for_sync_test",
                        prompt_token_ids[0],
                        selected_token,
                    )
                )
                assert learner_results, learner_results
                assert all(result["top1"] == selected_token for result in learner_results), learner_results
                rollout_logprob = response_logprobs[0][0]
                assert all(
                    abs(result["selected_logprob"] - rollout_logprob) <= 1e-5 for result in learner_results
                ), (learner_results, rollout_logprob)
                print(
                    f"Verified learner/vLLM FP32 top-1 and selected-token logprob after complete load "
                    f"{update_index}: prompt_token_ids={prompt_token_ids[0]}, learner={learner_results}, "
                    f"rollout_logprob={rollout_logprob}"
                )
            print(
                f"Verified rollout logprobs after complete load {update_index} "
                f"for {sum(map(len, response_logprobs))} tokens"
            )
            return outputs

        if VERIFY_PARITY:
            first_engine_weights = verify_synced_weights(1)
            generate_with_logprob_checks(1)

            mutation_results = ray.get(
                policy.async_run_ray_method("pass_through", "perturb_weight_for_sync_test", "lm_head.weight", 0.5)
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
            outputs = second_outputs
        else:
            sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
            outputs = asyncio.run(run_inference(client, get_test_prompts(MODEL), sampling_params))
            assert len(outputs["responses"]) == len(outputs["response_ids"])
        print(f"Example output: {outputs['responses'][0]}, {outputs['stop_reasons'][0]}")
    finally:
        ray.shutdown()
