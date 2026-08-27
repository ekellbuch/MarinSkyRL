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

        if VERIFY_PARITY:
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
            assert engine_per_rank[0]["lm_head.weight"]["dtype"] == "float32"
            print(f"Verified exact learner/vLLM weights: {weight_names}")

        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        if VERIFY_PARITY:
            sampling_params["logprobs"] = 1
        outputs = asyncio.run(run_inference(client, get_test_prompts(MODEL), sampling_params))

        assert len(outputs["responses"]) == len(outputs["response_ids"])
        if VERIFY_PARITY:
            response_logprobs = outputs["response_logprobs"]
            assert response_logprobs is not None
            assert len(response_logprobs) == len(outputs["response_ids"])
            assert all(len(ids) == len(logprobs) for ids, logprobs in zip(outputs["response_ids"], response_logprobs))
            assert all(math.isfinite(logprob) for logprobs in response_logprobs for logprob in logprobs)
            print(f"Verified rollout logprobs for {sum(map(len, response_logprobs))} tokens")
        print(f"Example output: {outputs['responses'][0]}, {outputs['stop_reasons'][0]}")
    finally:
        ray.shutdown()
