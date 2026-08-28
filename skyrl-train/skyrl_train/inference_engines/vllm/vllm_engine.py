import json
import os
import threading
from typing import List, Any, Dict, Optional, Tuple, Iterator, AsyncGenerator
from dataclasses import dataclass, fields as _dataclass_fields
from loguru import logger
from http import HTTPStatus
import ray
import torch
import asyncio
import vllm
from types import SimpleNamespace
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm.outputs import STREAM_FINISHED

from skyrl_train.numa_policy import NUMA_AFFINITY_ENV

# vLLM 0.16+ reorganized entrypoints into sub-packages.
# Try new paths first, fall back to old paths for backwards compatibility.
try:
    # vLLM >= 0.16
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
    )
    from vllm.entrypoints.openai.completion.protocol import (
        CompletionRequest,
        CompletionResponse,
    )
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse
except ImportError:
    # vLLM < 0.16 (old flat layout)
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
    from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        ErrorResponse,
        CompletionRequest,
        CompletionResponse,
    )

try:
    from vllm.v1.metrics.loggers import LoggingStatLogger
except ImportError:
    LoggingStatLogger = None  # Not available in all vLLM versions
from vllm.lora.request import LoRARequest
from torch.distributed import destroy_process_group
from skyrl_train.distributed.utils import init_custom_process_group
from uuid import uuid4
import warnings
from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from skyrl_train.weight_sync import WeightLoader
from skyrl_train.models.grug_moe import is_grug_router_bias
from skyrl_train.models.lm_head_precision import (
    VLLM_LM_HEAD_COMPUTE_DTYPE_ENV,
    configure_vllm_model_instance_lm_head_compute_dtype,
    configure_vllm_qwen3_5_lm_head_compute_dtype,
)
from skyrl_train.models.qwen3_5_vlm import qwen3_5_vllm_internal_weight_candidates
from skyrl_train.inference_engines.vllm.utils import (
    pop_openai_kwargs,
    ensure_token_ids_in_sse_chunk,
    PrefixCacheHitRateAccumulator,
)
from skyrl_train.utils import get_tcp_url, str_to_torch_dtype, torch_dtype_to_str
from skyrl_train.utils.tensor_fingerprint import canonical_tensor_fingerprint
import time
from packaging import version


def _parse_vllm_version() -> version.Version:
    """Parse vllm.__version__, treating 'dev' or other invalid strings as 999.0.0."""
    try:
        return version.Version(vllm.__version__)
    except version.InvalidVersion:
        return version.parse("999.0.0")


def _build_error_response(message: str, type_phrase: str, code: int) -> Dict[str, Any]:
    """Build an OpenAI-style ErrorResponse dict, robust to vLLM's ErrorInfo move.

    vLLM >= 0.10 wraps the error fields in a nested ``ErrorInfo``; older vLLM put
    them flat on ``ErrorResponse``. vLLM 0.16 ALSO relocated ``ErrorInfo`` out of
    the flat ``vllm.entrypoints.openai.protocol`` module (which no longer exists)
    into ``vllm.entrypoints.openai.engine.protocol`` — importing the old path
    raised ``ModuleNotFoundError`` inside the engine's request-error handler
    (vllm_engine.py:1591), turning every recoverable per-request error into an
    unhandled crash. Try the new sub-package path first, then the old flat path,
    then fall back to the flat-field ErrorResponse for pre-0.10 vLLM.
    """
    ErrorInfo = None
    try:  # vLLM >= 0.16 (sub-package layout, same module as ErrorResponse)
        from vllm.entrypoints.openai.engine.protocol import ErrorInfo  # type: ignore
    except ImportError:
        try:  # vLLM 0.10–0.15 (flat layout)
            from vllm.entrypoints.openai.protocol import ErrorInfo  # type: ignore
        except ImportError:
            ErrorInfo = None

    if ErrorInfo is not None:
        return ErrorResponse(
            error=ErrorInfo(message=message, type=type_phrase, code=code),
        ).model_dump()
    # pre-0.10 vLLM: flat fields directly on ErrorResponse.
    return ErrorResponse(message=message, type=type_phrase, code=code).model_dump()


# Guard so the fake/meta registration runs at most once per worker process.
_NORM_META_FAKES_REGISTERED = False


def ensure_norm_meta_fakes_registered() -> None:
    """Register Meta/fake kernels for vLLM's RMSNorm custom ops (idempotent).

    WHY: the layerwise weight-reload bracket (``skyrl_begin_weight_reload`` ->
    ``initialize_layerwise_reload``) restores every layer's params/buffers onto
    the **meta** device, then the CP>1 finalize path materializes those meta
    tensors and traces ``process_weights_after_loading`` over them. That trace
    dispatches ``torch.ops._C.rms_norm`` (and its sibling
    ``fused_add_rms_norm``) with Meta tensors, but the vLLM fork registers those
    C++ ops for ``torch::kCUDA`` ONLY (csrc/torch_bindings.cpp) — no Meta/fake
    kernel — so the dispatch dies with:
      ``NotImplementedError: _C::rms_norm: attempted to run this operator with
      Meta tensors, but there was no fake impl or Meta kernel registered``
    which killed all 3 inference engines during ``sync_weights`` on the CP4
    30B-A3B RL cell (agent_logs/2026-07-07_grid30bc_rmsnorm_meta_sync_weights.md).
    The CP1 sibling cell never traces these on meta, so it survived — hence this
    is CP>1-specific.

    Both ops are shape/dtype-preserving in-place normalizations whose C++ schema
    (torch_bindings.cpp) returns nothing:
      ``rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> ()``
      ``fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, float epsilon) -> ()``
    The correct fake for a mutating op that returns nothing is a no-op returning
    ``None`` (identical shape to vLLM's own ``_C::scaled_fp4_quant.out`` fake).
    This ONLY fires under meta-tensor tracing (the CP>1 sync path); real CUDA
    execution keeps using the registered CUDA kernel, so numerics/MoE routing are
    UNCHANGED. Ships via the /app source sync (no gpu-rl image rebuild).

    Called from the weight-reload bracket rather than at import time because the
    ``_C`` custom-op library is only guaranteed loaded once the vLLM model is
    built — by the first weight sync ``torch.ops._C.rms_norm`` exists.
    """
    global _NORM_META_FAKES_REGISTERED
    if _NORM_META_FAKES_REGISTERED:
        return

    _C = getattr(torch.ops, "_C", None)
    if _C is None:  # vLLM C-extension not loaded yet; try again on the next call.
        return

    from torch.library import register_fake

    # (name, arg-count) pairs. Fake matches the C++ schema exactly: a mutating op
    # that returns nothing => the fake returns None (no output tensors to fake).
    def _rms_norm_fake(result, input, weight, epsilon):  # -> ()
        return None

    def _fused_add_rms_norm_fake(input, residual, weight, epsilon):  # -> ()
        return None

    registrations = (
        ("rms_norm", "_C::rms_norm", _rms_norm_fake),
        ("fused_add_rms_norm", "_C::fused_add_rms_norm", _fused_add_rms_norm_fake),
    )
    for attr, qualname, fake in registrations:
        if not hasattr(_C, attr):
            continue  # op not present in this vLLM build; nothing to register.
        try:
            register_fake(qualname, fake)
            logger.info(f"Registered Meta/fake kernel for {qualname} (CP>1 weight-sync fix)")
        except RuntimeError as e:
            # A fake already exists (e.g. a future vLLM registers one, or a
            # sibling engine in-process already ran this) — that's fine, leave it.
            logger.debug(f"Skipping fake registration for {qualname}: {e}")

    _NORM_META_FAKES_REGISTERED = True


@dataclass
class Logprob:
    logprob: float
    rank: int
    token_id: str


def setup_envvars_for_vllm(kwargs, bundle_indices):
    noset_visible_devices = kwargs.pop("noset_visible_devices")
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # TODO(Charlie): may not be needed.

    # When custom all-reduce is disabled (e.g. for TP=2 on H100 where
    # SymmMemCommunicator rendezvous fails), also disable symmetric memory
    # via env var — the engine arg alone doesn't prevent SymmMemCommunicator
    # from being instantiated.
    if kwargs.get("disable_custom_all_reduce"):
        os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
        logger.info("setup_envvars_for_vllm: set VLLM_ALLREDUCE_USE_SYMM_MEM=0 (disable_custom_all_reduce=True)")
    if kwargs.get("distributed_executor_backend") == "ray":
        # a hack to make the script work.
        # stop ray from manipulating *_VISIBLE_DEVICES
        # at the top-level when the distributed_executor_backend is ray.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
    elif noset_visible_devices:
        # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
        # when the distributed_executor_backend is not rayargs and
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

    num_gpus = kwargs.pop("num_gpus")
    if bundle_indices is not None:
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
        logger.info(f"creating LLM with bundle_indices={bundle_indices}")

    # Set NUMA CPU affinity for single-GPU (TP=1) inference actors.
    # For TP>1, affinity is set per-worker via WorkerWrap.set_numa_affinity().
    #
    # When NUMA affinity is enabled, we also disable vLLM V1 multiprocessing.
    # vLLM's V1 engine spawns EngineCore as a separate subprocess using
    # multiprocessing with start_method="spawn" (forced when running inside a
    # Ray actor — see vllm.utils._maybe_force_spawn). Spawned processes do NOT
    # inherit the parent's CPU affinity, so NUMA binding set here would be lost.
    # Disabling V1 multiprocessing forces EngineCore to run in the same process,
    # where our affinity settings take effect.
    executor_backend = kwargs.get("distributed_executor_backend")
    logger.info(
        f"setup_envvars_for_vllm: distributed_executor_backend={executor_backend}, "
        f"{NUMA_AFFINITY_ENV}={os.environ.get(NUMA_AFFINITY_ENV, '<unset>')}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
        f"VLLM_ENABLE_V1_MULTIPROCESSING={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', '<unset>')}"
    )
    # NOTE: the `mp` executor backend (Qwen3-Next R3 capture path) MUST keep v1
    # multiprocessing ENABLED — it spawns its TP worker subprocesses via the v1 mp
    # path, and disabling it cancels the shm message queue at warm-up
    # ("RuntimeError: cancelled"). NUMA single-GPU pinning does not apply to the
    # multi-GPU mp engine anyway, so skip this branch for mp.
    if executor_backend not in ("ray", "mp"):
        try:
            from skyrl_train.utils.numa import is_numa_affinity_enabled, set_numa_affinity_for_gpu

            numa_enabled = is_numa_affinity_enabled()
            logger.info(f"setup_envvars_for_vllm: numa_enabled={numa_enabled}")
            if numa_enabled:
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
                logger.info("setup_envvars_for_vllm: set VLLM_ENABLE_V1_MULTIPROCESSING=0 for NUMA affinity")
                cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if cuda_devs:
                    gpu_ids = [int(x) for x in cuda_devs.split(",")]
                    if len(gpu_ids) == 1:
                        set_numa_affinity_for_gpu(gpu_ids[0])
        except Exception as e:
            logger.warning(f"setup_envvars_for_vllm: NUMA affinity setup failed: {e}")


def _refresh_vllm_lm_head_compute_dtype(model) -> None:
    dtype_name = os.environ.get(VLLM_LM_HEAD_COMPUTE_DTYPE_ENV)
    if dtype_name is not None:
        configure_vllm_model_instance_lm_head_compute_dtype(model, dtype_name)


class WorkerWrap:
    def set_numa_affinity(self):
        """Set CPU affinity to match this worker's GPU NUMA node.

        Called via collective_rpc for TP>1 configurations so each
        vLLM EngineCore process binds to its GPU's local CPUs.
        """
        try:
            from skyrl_train.utils.numa import set_numa_affinity_for_gpu

            gpu_id = self.device.index if self.device is not None else 0
            set_numa_affinity_for_gpu(gpu_id)
        except Exception:
            pass

    def test_rpc(self, *args, **kwargs):
        """Test RPC call to worker"""
        return args, kwargs

    def init_weight_update_communicator(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
        override_existing: bool = False,
    ):
        """Init torch process group for model weights update"""
        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"

        if getattr(self, "_model_update_group", None):
            if override_existing:
                logger.info("Destroying existing model update group")
                destroy_process_group(self._model_update_group)
                self._model_update_group = None
            else:
                warnings.warn(
                    "Detected an existing weights update group. For overriding, use `generator.override_existing_update_group=True`"
                )

        rank = torch.distributed.get_rank() + rank_offset
        logger.info(
            f"torch.distributed.get_rank(): {torch.distributed.get_rank()}, rank_offset: {rank_offset}, rank: {rank}, world_size: {world_size}, group_name: {group_name}"
        )

        self._model_update_group = init_custom_process_group(
            backend=backend,
            init_method=get_tcp_url(master_address, master_port),
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        logger.info(
            f"init_weight_update_communicator: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

        # Create receiver now that we have all the state
        self._weight_receiver = VLLMWeightTransferReceiver(
            model_update_group=self._model_update_group,
            model_config=self.model_config,
            device=self.device,
        )

    @staticmethod
    def _apply_fp8_weight_loader_patches(*, fuse_weights: bool = False):
        """Patch Fp8LinearMethod.process_weights_after_loading to preserve weight_loader.

        Following verl's approach: after FP8 processing creates new Parameter objects,
        copy custom attributes (weight_loader, output_dim, input_dim, subclass_type)
        from the original specialized parameter so weight sync can reload weights.
        """
        if not fuse_weights:
            return

        try:
            from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
        except ImportError:
            return

        original_process = Fp8LinearMethod.process_weights_after_loading

        def patched_process(self_method, layer, *args, **kwargs):
            # Save original param attributes before processing
            saved_attrs = {}
            for pname, param in layer.named_parameters():
                attrs = {}
                for attr in (
                    "weight_loader",
                    "output_dim",
                    "input_dim",
                    "_output_dim",
                    "_input_dim",
                    "packed_dim",
                    "packed_factor",
                    "tp_rank",
                    "tp_size",
                    "logical_widths",
                    "output_sizes",
                ):
                    if hasattr(param, attr):
                        attrs[attr] = getattr(param, attr)
                attrs["subclass_type"] = type(param)
                saved_attrs[pname] = attrs

            # Call original process_weights_after_loading
            result = original_process(layer, *args, **kwargs)

            # Restore attributes on new parameters
            for pname, param in layer.named_parameters():
                if pname in saved_attrs:
                    for attr, value in saved_attrs[pname].items():
                        try:
                            setattr(param, attr, value)
                        except (AttributeError, TypeError):
                            pass

            return result

        Fp8LinearMethod.process_weights_after_loading = patched_process

    def skyrl_begin_weight_reload(self) -> None:
        """RENAMED from ``start_weight_update`` AND NOW WIRED (the #1685 fix).

        WHY RENAMED: the baked vLLM ``gpu_worker.Worker`` now ALSO defines
        ``start_weight_update``/``finish_weight_update`` (vllm/v1/worker/gpu_worker.py),
        so a WorkerWrap method of the SAME name trips vLLM's ``init_worker_extension``
        shadow-assert ("Worker class already has an attribute finish_weight_update")
        -> EngineCore crash. (The base Worker versions also REQUIRE a configured
        ``weight_transfer_engine``, which SkyRL's NCCL-broadcast path does NOT set up, so
        they cannot be reused directly — hence this SkyRL-native bracket.)

        WHY NOW WIRED (2026-06-27, the disagg-engine diag root cause): on CoreWeave H100
        the unquantized MoE backend auto-selects FlashInfer CUTLASS, whose
        ``process_weights_after_loading`` applies ``swap_w13_to_w31`` ([gate;up]->[up;gate]
        kernel layout) at the INITIAL from-disk load. The RL disaggregated update path
        does immediate per-chunk ``model.load_weights`` with NO subsequent
        ``process_weights_after_loading`` (base_loader.py:80 runs it only on load_model),
        so the update OVERWRITES the [up;gate] kernel buffer with raw checkpoint [gate;up]
        and NEVER re-swaps -> the kernel reads transposed halves -> token-salad. PROVEN by
        the kernel-format discriminator: PRE(from-disk)=[up;gate], POST(RL-update)=[gate;up],
        both ep ranks, layers 0 & 24. The cure is to bracket the whole multi-chunk sync with
        vLLM's layerwise reload so ``finalize`` re-runs ``process_weights_after_loading``
        (the swap) EXACTLY once at the end. Triton clusters select no-swap backends, so
        finalize is swap-inert there -> byte-identical (reconciles Jupiter-OK/CoreWeave-salad).

        Bracket-open one weight sync with a SINGLE vLLM layerwise-reload init.

        Port of SkyRL #1685 + #1737 (the silent-MoE-corruption fix). vLLM's
        ``process_weights_after_loading`` permutes ``FusedMoE.w13_weight`` in place
        at engine init (``[w1;w3] -> [w3;w1]`` kernel layout) while our fork's
        ``replace_parameter`` preserves each param's ``weight_loader``. A naive
        SECOND ``model.load_weights`` (the RL sync) then re-invokes ``_load_w13`` and
        writes raw ``[w1;w3]`` into a ``[w3;w1]`` buffer -> silent value corruption ->
        token-salad. The cure is to run vLLM's layerwise reload ONCE around the whole
        (multi-chunk, streamed) sync: ``initialize_layerwise_reload`` restores layers
        to meta + wraps each weight_loader to DEFER processing; per-chunk raw
        ``model.load_weights`` then just buffers; ``finalize_layerwise_reload``
        (in ``finish_weight_update``) materializes + processes every layer EXACTLY
        ONCE. A per-chunk ``reload_weights`` would instead re-finalize on every call
        and restore layers absent from that chunk (#1737), so we bracket the whole
        sync rather than reload per chunk.

        Idempotent guard: a second ``start`` without a ``finish`` is a protocol error.
        """
        if getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError(
                "start_weight_update called while a weight update is already active. Call finish_weight_update first."
            )
        # Register the RMSNorm Meta/fake kernels BEFORE the reload restores layers
        # to the meta device — the CP>1 finalize path traces _C::rms_norm on those
        # meta tensors and would otherwise crash (see ensure_norm_meta_fakes_registered).
        ensure_norm_meta_fakes_registered()
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.reload import initialize_layerwise_reload

        model = self.model_runner.model
        with set_current_vllm_config(self.vllm_config), torch.device(self.device):
            initialize_layerwise_reload(model)
        self._skyrl_weight_update_active = True

    def skyrl_finish_weight_reload(self) -> None:
        """RENAMED from ``finish_weight_update`` + NOW WIRED — see
        ``skyrl_begin_weight_reload`` for the collision + root-cause rationale.

        Bracket-close the weight sync with a SINGLE layerwise-reload finalize.

        Materializes + runs ``process_weights_after_loading`` over the WHOLE weight
        set exactly once -> re-applies the FlashInfer-CUTLASS ``swap_w13_to_w31`` the
        per-chunk ``model.load_weights`` skips. Must be called after every chunk's
        ``load_weights`` (and after ``end_weight_update``'s fused flush).
        """
        if not getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError("skyrl_begin_weight_reload must be called before skyrl_finish_weight_reload.")
        # Idempotent no-op if begin already registered them; guards the case where
        # finalize is the first meta-materializing call in this worker.
        ensure_norm_meta_fakes_registered()
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

        model = self.model_runner.model
        with set_current_vllm_config(self.vllm_config), torch.device(self.device):
            finalize_layerwise_reload(model, self.model_config)
        self._skyrl_weight_update_active = False
        _refresh_vllm_lm_head_compute_dtype(model)

    def begin_weight_update(self) -> None:
        """Start accumulating weights for batched load_weights call.

        When fused loading is requested, weights are accumulated instead of loaded
        immediately. Call end_weight_update() to flush and apply them all at once
        via model.load_weights(), which handles packed module mapping (qkv_proj, gate_up_proj).
        Weights are stored on CPU to avoid GPU OOM during accumulation.
        """
        self._accumulated_weights = []

    def _is_fp8_model(self):
        """Check if the model uses FP8 quantization."""
        quant_config = getattr(self.model_runner.model, "quant_config", None)
        if quant_config is None:
            return False
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        return isinstance(quant_config, Fp8Config)

    def _quantize_weights_for_fp8(self, weights):
        """Quantize BF16 weights to FP8 before loading into FP8 model.

        Follows verl's approach: quantize each weight tensor to FP8 with
        per-tensor scale, then yield (name, fp8_tensor) and (name_scale, scale).
        Non-linear weights (layernorm, embedding) are passed through as-is.
        """
        import torch
        from vllm._custom_ops import scaled_fp8_quant

        model = self.model_runner.model
        # Build set of parameter names that are FP8 quantized
        # These are the linear layer weights (not biases, not layernorms, not embeddings)
        fp8_param_names = set()
        for name, module in model.named_modules():
            from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

            if hasattr(module, "quant_method") and isinstance(module.quant_method, Fp8LinearMethod):
                for pname, _ in module.named_parameters():
                    if "weight" in pname and "scale" not in pname:
                        full_name = f"{name}.{pname}" if name else pname
                        fp8_param_names.add(full_name)

        for name, tensor in weights:
            # Check if this weight maps to an FP8-quantized parameter
            # The name might be "layers.0.self_attn.q_proj.weight" but the
            # FP8 param is "layers.0.self_attn.qkv_proj.weight"
            # We need to check the ORIGINAL unfused name against the fused params
            is_fp8 = False
            packed_mapping = getattr(model, "packed_modules_mapping", {})
            # Reverse mapping: q_proj -> qkv_proj, gate_proj -> gate_up_proj
            reverse_map = {}
            for fused, originals in packed_mapping.items():
                for orig in originals:
                    reverse_map[orig] = fused

            # Try to find the FP8 param name
            check_name = name
            parts = name.rsplit(".", 2)
            if len(parts) >= 2:
                module_part = parts[-2]  # e.g. "q_proj"
                if module_part in reverse_map:
                    check_name = name.replace(module_part, reverse_map[module_part])

            if check_name in fp8_param_names or name in fp8_param_names:
                is_fp8 = True

            if is_fp8 and tensor.dtype != torch.float8_e4m3fn:
                # Move to GPU, quantize, move back to CPU
                gpu_tensor = tensor.to(device="cuda", dtype=torch.bfloat16)
                fp8_tensor, scale = scaled_fp8_quant(gpu_tensor)
                yield (name, fp8_tensor.cpu())
                # Yield the scale with the FUSED param name
                scale_name = check_name.replace(".weight", ".weight_scale")
                yield (scale_name, scale.cpu())
                del gpu_tensor, fp8_tensor
            else:
                yield (name, tensor)

    def _restore_param_subclasses(self, model):
        """Temporarily restore param __class__ to subclass_type for weight loading.

        After process_weights_after_loading, params are plain Parameter but have
        subclass_type saved. Restoring __class__ makes weight_loader dispatch work.
        Returns list of (param, original_class) for cleanup.
        """
        patched = []
        for name, param in model.named_parameters():
            subclass_type = getattr(param, "subclass_type", None)
            if subclass_type is not None and type(param) is not subclass_type:
                original_class = type(param)
                param.__class__ = subclass_type
                patched.append((param, original_class))
        return patched

    def _undo_param_subclasses(self, patched):
        """Undo the temporary __class__ patching."""
        for param, original_class in patched:
            param.__class__ = original_class

    def end_weight_update(self) -> None:
        """Flush accumulated weights via model.load_weights().

        For FP8 models: quantizes BF16 weights to FP8 before loading,
        following verl's approach. Also temporarily restores param subclass
        types so weight_loader dispatch works correctly with FP8 params.
        """
        import gc

        if hasattr(self, "_accumulated_weights") and self._accumulated_weights:
            model = self.model_runner.model
            if self._is_fp8_model():
                import torch
                import gc
                from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
                from vllm._custom_ops import scaled_fp8_quant

                # Receiver-side FP8 quantization: BF16 weights arrive via NCCL,
                # fuse stacked params, quantize to FP8, write directly to model.
                weight_index = {name: tensor for name, tensor in self._accumulated_weights}
                stacked = [
                    ("qkv_proj", "q_proj", "q"),
                    ("qkv_proj", "k_proj", "k"),
                    ("qkv_proj", "v_proj", "v"),
                    ("gate_up_proj", "gate_proj", 0),
                    ("gate_up_proj", "up_proj", 1),
                ]

                for mname, module in model.named_modules():
                    if not (hasattr(module, "quant_method") and isinstance(module.quant_method, Fp8LinearMethod)):
                        continue
                    param = module.weight
                    device = param.device
                    is_stacked = any(mname.endswith(pn) for pn, _, _ in stacked)

                    if is_stacked:
                        shard_list = []
                        for param_name, weight_name, shard_id in stacked:
                            if not mname.endswith(param_name):
                                continue
                            src_name = mname.replace(param_name, weight_name) + ".weight"
                            if src_name in weight_index:
                                shard_list.append(weight_index[src_name])
                        if shard_list:
                            full_bf16 = torch.cat(shard_list, dim=0).to(
                                device=device, dtype=torch.bfloat16, non_blocking=True
                            )
                            torch.cuda.current_stream().synchronize()
                            fp8_full, scale = scaled_fp8_quant(full_bf16)
                            param.data.copy_(fp8_full)
                            if hasattr(module, "weight_scale"):
                                module.weight_scale.data.copy_(scale.squeeze())
                            del full_bf16, fp8_full, scale, shard_list
                    else:
                        src_name = mname + ".weight"
                        if src_name in weight_index:
                            bf16_w = weight_index[src_name].to(device=device, dtype=torch.bfloat16, non_blocking=True)
                            torch.cuda.current_stream().synchronize()
                            fp8_w, scale = scaled_fp8_quant(bf16_w)
                            param.data.copy_(fp8_w)
                            if hasattr(module, "weight_scale"):
                                module.weight_scale.data.copy_(scale.squeeze())
                            del bf16_w, fp8_w, scale

                # Load non-FP8 params (layernorms, embeddings)
                params_dict = dict(model.named_parameters())
                for name, tensor in self._accumulated_weights:
                    if name in params_dict:
                        param = params_dict[name]
                        if param.dtype != torch.float8_e4m3fn:
                            param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))

                del weight_index

                gc.collect()
                torch.cuda.empty_cache()
            else:
                model.load_weights(weights=iter(self._accumulated_weights))
            if not getattr(self, "_skyrl_weight_update_active", False):
                _refresh_vllm_lm_head_compute_dtype(model)
            self._accumulated_weights.clear()
            del self._accumulated_weights
            gc.collect()
            import torch

            torch.cuda.empty_cache()

    def load_weights(self, request: NamedWeightsUpdateRequest) -> None:
        """Load weights using the receiver.

        This method is called via collective_rpc from VLLMWeightLoader.

        When fused loading is requested and begin_weight_update() was called,
        weights are accumulated on CPU instead of loaded immediately.

        Args:
            request: Weight update request with names, dtypes, shapes, etc.
        """
        weight_list = []
        for name, tensor in self._weight_receiver.receive_weights(request):
            weight_list.append((name, tensor))

        if hasattr(self, "_accumulated_weights"):
            # Batched mode: move to CPU and accumulate for later flush
            for name, tensor in weight_list:
                self._accumulated_weights.append((name, tensor.cpu()))
            del weight_list
        else:
            # Immediate mode (default): load right away
            self.model_runner.model.load_weights(weights=weight_list)
            if not getattr(self, "_skyrl_weight_update_active", False) and any(
                name == "lm_head.weight" or name.endswith("embed_tokens.weight") for name, _ in weight_list
            ):
                _refresh_vllm_lm_head_compute_dtype(self.model_runner.model)
            for weight in weight_list:
                del weight

    # TODO (sumanthrh): Add destroy process group RPC as a atexit handler to Trainer code.
    def destroy_weights_update_group(self):
        if not getattr(self, "_model_update_group", None):
            warnings.warn("No model update group to destroy")
            return
        destroy_process_group(self._model_update_group)

    def read_named_weights(
        self,
        hf_names,
        dump_inventory: bool = False,
        expected_shapes=None,
    ):
        """Read engine-side weights back
        from the live vLLM model, reconstructed under the HF parameter names the
        trainer broadcasts.

        This is the symmetric inverse of ``load_weights`` (vLLM consumes HF-named
        tensors in ``model.load_weights`` and maps them into its internal
        fused/sharded params; here we read those internal params back and rebuild
        the HF view). It returns CPU fp32 tensors for the established MoE
        diagnostics.

        Supported HF name forms (Qwen1.5-MoE / Qwen2MoE vLLM layout):
          * ``model.embed_tokens.weight``                       -> VocabParallelEmbedding (TP vocab-sharded)
          * ``model.layers.{i}.mlp.gate.weight`` (router)       -> ReplicatedLinear (full copy every rank)
          * ``model.layers.{i}.self_attn.o_proj.weight``        -> RowParallelLinear (TP input-sharded)
          * ``model.layers.{i}.mlp.experts.{j}.gate_proj.weight`` -> RoutedExperts w13_weight[local_e, :I]
          * ``...experts.{j}.up_proj.weight``                   -> RoutedExperts w13_weight[local_e, I:]
          * ``...experts.{j}.down_proj.weight``                 -> RoutedExperts w2_weight[local_e]

        Args:
            hf_names: list of HF parameter names to read back.
            dump_inventory: if True, also returns the full ``named_parameters()``
                name->shape inventory under key ``__inventory__`` (first run aid).
            expected_shapes: optional HF name-to-shape mapping used to reconstruct
                dense projections stored in fused vLLM parameters.
        The Qwen3.5 multimodal shell accepts the sender-side broadcast namespace
        ``model.language_model.*``. It is resolved to vLLM's internal
        ``language_model.model.*`` namespace before direct lookup.
        """
        import re
        import torch as _torch

        model = self.model_runner.model
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        all_params = {**params, **buffers}

        try:
            from vllm.distributed import parallel_state as _ps

            tp_rank = _ps.get_tensor_model_parallel_rank()
            tp_size = _ps.get_tensor_model_parallel_world_size()
        except Exception:
            tp_rank, tp_size = 0, 1
        try:
            ep_rank = _ps.get_ep_group().rank_in_group
            ep_size = _ps.get_ep_group().world_size
        except Exception:
            ep_rank, ep_size = 0, 1

        def _payload(tensor):
            return {"tensor": tensor.detach().to("cpu", dtype=_torch.float32).contiguous()}

        expected_shapes = expected_shapes or {}
        out = {}
        if dump_inventory:
            out["__inventory__"] = {n: list(p.shape) for n, p in all_params.items()}
        out["__ranks__"] = {"tp_rank": tp_rank, "tp_size": tp_size, "ep_rank": ep_rank, "ep_size": ep_size}

        expert_re = re.compile(r"^(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")

        for name in hf_names:
            entry = {"found": False}
            try:
                # 1. Direct (replicated) match: router gate, norms, etc.
                language_model = getattr(model, "language_model", None)
                language_config = getattr(language_model, "config", None)
                candidates = qwen3_5_vllm_internal_weight_candidates(
                    name,
                    tied_word_embeddings=bool(getattr(language_config, "tie_word_embeddings", False)),
                )
                direct_name = next((candidate for candidate in candidates if candidate in all_params), name)
                if direct_name in all_params:
                    tensor = all_params[direct_name]
                    entry = {
                        "data_ptr": tensor.data_ptr(),
                        "found": True,
                        "parameter_id": id(tensor),
                        "mode": "direct",
                        "internal_name": direct_name,
                        "dtype": torch_dtype_to_str(tensor.dtype),
                        **_payload(tensor),
                    }
                    out[name] = entry
                    continue

                # 2. Dense stacked projections. At TP=1, rebuild the exact HF
                # tensor view from vLLM's fused storage without transferring the
                # fused tensor through Ray. The real Qwen3.5 parity smoke uses
                # one engine rank; other TP layouts are reported as unsupported
                # rather than silently compared with the wrong shard.
                stacked_groups = (
                    (("q_proj", "k_proj", "v_proj"), "qkv_proj"),
                    (("gate_proj", "up_proj"), "gate_up_proj"),
                    (("in_proj_qkv", "in_proj_z"), "in_proj_qkvz"),
                    (("in_proj_b", "in_proj_a"), "in_proj_ba"),
                )
                stacked_match = None
                if ".experts." not in name:
                    for shard_names, fused_name in stacked_groups:
                        for shard_index, shard_name in enumerate(shard_names):
                            marker = f".{shard_name}."
                            if marker not in name:
                                continue
                            internal_name = next(
                                (
                                    candidate.replace(marker, f".{fused_name}.")
                                    for candidate in candidates
                                    if candidate.replace(marker, f".{fused_name}.") in all_params
                                ),
                                None,
                            )
                            if internal_name is not None:
                                stacked_match = (shard_names, shard_index, shard_name, internal_name)
                            break
                        if stacked_match is not None:
                            break
                if stacked_match is not None:
                    shard_names, shard_index, shard_name, internal_name = stacked_match
                    if tp_size != 1:
                        out[name] = {
                            "found": False,
                            "mode": "stacked",
                            "internal_name": internal_name,
                            "note": f"full stacked reconstruction requires tp_size=1, got {tp_size}",
                        }
                        continue
                    expected_shape = expected_shapes.get(name)
                    peer_shapes = [
                        expected_shapes.get(name.replace(f".{shard_name}.", f".{peer_name}."))
                        for peer_name in shard_names
                    ]
                    if expected_shape is None or any(shape is None for shape in peer_shapes):
                        out[name] = {
                            "found": False,
                            "mode": "stacked",
                            "internal_name": internal_name,
                            "note": "missing expected shapes for stacked projection",
                        }
                        continue
                    tensor = all_params[internal_name]
                    offset = sum(shape[0] for shape in peer_shapes[:shard_index])
                    length = expected_shape[0]
                    if tensor.ndim < 1 or tensor.shape[0] < offset + length:
                        out[name] = {
                            "found": False,
                            "mode": "stacked",
                            "internal_name": internal_name,
                            "actual_shape": list(tensor.shape),
                            "note": f"cannot select rows [{offset}:{offset + length}]",
                        }
                        continue
                    tensor = tensor.narrow(0, offset, length)
                    entry = {
                        "data_ptr": tensor.data_ptr(),
                        "found": True,
                        "parameter_id": id(all_params[internal_name]),
                        "mode": "stacked",
                        "internal_name": internal_name,
                        "shard_index": shard_index,
                        "dtype": torch_dtype_to_str(tensor.dtype),
                        **_payload(tensor),
                    }
                    out[name] = entry
                    continue

                # 3. Routed expert -> FusedMoE fused weights.
                m = expert_re.match(name)
                if m is not None:
                    prefix, gj, proj = m.group(1), int(m.group(2)), m.group(3)
                    # vLLM RoutedExperts stores w13_weight [n_local_experts, 2*I, H] and
                    # w2_weight [n_local_experts, H, I]. Local experts are a contiguous
                    # EP slice: global expert gj lives on ep_rank == gj // n_local.
                    w13 = all_params.get(f"{prefix}.experts.routed_experts.w13_weight")
                    w2 = all_params.get(f"{prefix}.experts.routed_experts.w2_weight")
                    if w13 is None or w2 is None:
                        # Fallback: scan for any experts.*weight tensor under this prefix.
                        cand = {
                            k: v
                            for k, v in all_params.items()
                            if k.startswith(f"{prefix}.experts.") and k.endswith("weight")
                        }
                        entry = {"found": False, "note": f"no w13/w2; candidates={list(cand.keys())}"}
                        out[name] = entry
                        continue
                    n_local = w13.shape[0]
                    owner_ep = gj // n_local
                    if owner_ep != ep_rank:
                        entry = {"found": False, "mode": "expert", "owner_ep": owner_ep, "skip": True}
                        out[name] = entry
                        continue
                    local_e = gj - owner_ep * n_local
                    if proj == "down_proj":
                        t = w2[local_e]
                    else:
                        inter = w13.shape[1] // 2
                        t = w13[local_e, :inter] if proj == "gate_proj" else w13[local_e, inter:]
                    entry = {
                        "found": True,
                        "mode": "expert",
                        "owner_ep": owner_ep,
                        "local_e": local_e,
                        "dtype": torch_dtype_to_str(t.dtype),
                        **_payload(t),
                    }
                    out[name] = entry
                    continue

                # 4. Unknown / unsupported name.
                entry = {"found": False, "note": "no mapping"}
                out[name] = entry
            except Exception as e:  # never crash the collective_rpc
                out[name] = {"found": False, "error": repr(e)}
        return out

    def fingerprint_named_weights(self, hf_names, expected_shapes):
        """Return compact exact fingerprints for requested engine weights."""
        fingerprints = {"__ranks__": None}
        for name in hf_names:
            # Read and hash one tensor at a time so the engine actor never holds
            # a second full-model CPU copy. Only the compact digest leaves the
            # actor.
            weights = self.read_named_weights([name], expected_shapes=expected_shapes)
            if fingerprints["__ranks__"] is None:
                fingerprints["__ranks__"] = weights["__ranks__"]
            entry = dict(weights[name])
            tensor = entry.pop("tensor", None)
            if tensor is None:
                fingerprints[name] = entry
                del weights
                continue

            actual_shape = list(tensor.shape)
            expected_shape = expected_shapes.get(name)
            compared_tensor = tensor
            if expected_shape is not None and actual_shape != expected_shape:
                expected_numel = 1
                for dimension in expected_shape:
                    expected_numel *= dimension
                if tensor.numel() == expected_numel:
                    compared_tensor = tensor.reshape(expected_shape)
                    actual_shape = list(compared_tensor.shape)
                can_trim_vocab_padding = (
                    len(actual_shape) == len(expected_shape)
                    and actual_shape[1:] == expected_shape[1:]
                    and actual_shape[0] >= expected_shape[0]
                )
                if actual_shape == expected_shape:
                    pass
                elif not can_trim_vocab_padding:
                    entry.update(
                        {
                            "actual_shape": actual_shape,
                            "expected_shape": expected_shape,
                            "shape_mismatch": True,
                        }
                    )
                    fingerprints[name] = entry
                    del tensor, weights
                    continue
                else:
                    compared_tensor = tensor[: expected_shape[0]]
            entry.update(
                {
                    "actual_shape": actual_shape,
                    "shape_mismatch": False,
                    "fingerprint": canonical_tensor_fingerprint(compared_tensor),
                }
            )
            fingerprints[name] = entry
            del tensor, weights
        return fingerprints

    def begin_head_input_capture(self, selected_token: int):
        """Capture bounded inputs to the live model's next logits computation."""
        if getattr(self, "_skyrl_head_input_capture_active", False):
            raise RuntimeError("A vLLM head-input capture is already active")

        model = self.model_runner.model
        had_instance_compute_logits = "compute_logits" in model.__dict__
        instance_compute_logits = model.__dict__.get("compute_logits")
        original_compute_logits = model.compute_logits
        captures = []
        layer_captures = []
        layer_hooks = []
        forward_core_patches = []

        def tensor_payload(tensor):
            if tensor.ndim == 3:
                if tensor.shape[0] != 1:
                    raise RuntimeError(f"Expected vLLM diagnostic batch size 1, got {tensor.shape}")
                tensor = tensor[0, -1]
            elif tensor.ndim == 2:
                tensor = tensor[-1]
            else:
                raise RuntimeError(f"Expected vLLM hidden states with rank 2 or 3, got {tensor.shape}")
            return {
                "values": tensor.detach().float().cpu().tolist(),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }

        def token_heads_payload(tensor, num_heads):
            if tensor.ndim != 2 or tensor.shape[0] < num_heads:
                raise RuntimeError(f"Expected flattened token heads [tokens * heads, head_dim], got {tensor.shape}")
            tensor = tensor[-num_heads:].reshape(-1)
            return {
                "values": tensor.detach().float().cpu().tolist(),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }

        def capture_input(_module, args, kwargs, *, destination, key):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None:
                hidden_states = args[0]
            destination.setdefault(key, []).append(tensor_payload(hidden_states))

        def capture_output(_module, args, kwargs, output, *, destination, key):
            del args
            hidden_states = kwargs.get("output")
            if hidden_states is None:
                hidden_states = output[0] if isinstance(output, tuple) else output
            destination.setdefault(key, []).append(tensor_payload(hidden_states))

        def capture_projection_output(_module, _args, output, *, destination, keys, sizes):
            projected = output[0] if isinstance(output, tuple) else output
            for key, tensor in zip(keys, projected.split(sizes, dim=-1), strict=True):
                payload = tensor_payload(tensor)
                payload["token_fingerprints"] = token_fingerprints(tensor)
                destination.setdefault(key, []).append(payload)

        def exact_error_summary(actual, expected):
            difference = (actual.float() - expected.float()).abs().reshape(-1)
            return {
                "exact": bool(torch.equal(actual, expected)),
                "max": float(difference.max().item()),
                "mismatch_count": int(torch.count_nonzero(difference).item()),
                "p95": float(torch.quantile(difference, 0.95).item()),
                "shape": list(actual.shape),
            }

        def tensor_layout(tensor):
            return {
                "contiguous": tensor.is_contiguous(),
                "device": str(tensor.device),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "storage_nbytes": tensor.untyped_storage().nbytes(),
                "storage_offset": tensor.storage_offset(),
                "stride": list(tensor.stride()),
            }

        def token_fingerprints(tensor):
            if tensor.ndim < 2:
                raise ValueError(f"Expected a token dimension, got {tensor.shape}")
            if tensor.ndim >= 3:
                if tensor.shape[0] != 1:
                    raise ValueError(f"Expected batch size one, got {tensor.shape}")
                return [canonical_tensor_fingerprint(tensor[:, index]) for index in range(tensor.shape[1])]
            return [canonical_tensor_fingerprint(tensor[index : index + 1]) for index in range(tensor.shape[0])]

        def alias_summary(tensor, destination):
            return {
                "same_data_pointer": tensor.data_ptr() == destination.data_ptr(),
                "same_storage": (tensor.untyped_storage().data_ptr() == destination.untyped_storage().data_ptr()),
                "same_storage_offset": tensor.storage_offset() == destination.storage_offset(),
            }

        language_model = getattr(model, "language_model", None)
        text_model = getattr(language_model, "model", None)
        layers = getattr(text_model, "layers", ())
        for layer_index, layer in enumerate(layers):
            mixer_name = "linear_attn" if layer.layer_type == "linear_attention" else "self_attn"
            mixer = getattr(layer, mixer_name)
            entry = {"layer": layer_index, "mixer": mixer_name}
            layer_captures.append(entry)
            if layer_index == 0 and mixer_name == "linear_attn":
                qkv_size = (mixer.key_dim * 2 + mixer.value_dim) // mixer.tp_size
                z_size = mixer.value_dim // mixer.tp_size
                ba_size = mixer.in_proj_ba.weight.shape[0] // 2
                projections = {"runtime": {}}
                entry["projections"] = projections
                stages = {}
                entry["mixer_stages"] = stages
                fla_inputs = []
                fla_outputs = []
                fla_values = []
                fla_captures = []
                entry["fla_core"] = fla_captures
                causal_conv_captures = []
                entry["causal_conv"] = causal_conv_captures
                if hasattr(mixer, "in_proj_qkv"):
                    layer_hooks.extend(
                        (
                            mixer.in_proj_qkv.register_forward_hook(
                                lambda module, args, output, destination=projections["runtime"]: (
                                    capture_projection_output(
                                        module,
                                        args,
                                        output,
                                        destination=destination,
                                        keys=("qkv",),
                                        sizes=(qkv_size,),
                                    )
                                )
                            ),
                            mixer.in_proj_z.register_forward_hook(
                                lambda module, args, output, destination=projections["runtime"]: (
                                    capture_projection_output(
                                        module,
                                        args,
                                        output,
                                        destination=destination,
                                        keys=("z",),
                                        sizes=(z_size,),
                                    )
                                )
                            ),
                        )
                    )
                else:
                    layer_hooks.append(
                        mixer.in_proj_qkvz.register_forward_hook(
                            lambda module, args, output, destination=projections["runtime"]: capture_projection_output(
                                module,
                                args,
                                output,
                                destination=destination,
                                keys=("qkv", "z"),
                                sizes=(qkv_size, z_size),
                            )
                        )
                    )
                layer_hooks.append(
                    mixer.in_proj_ba.register_forward_hook(
                        lambda module, args, output, destination=projections["runtime"]: capture_projection_output(
                            module,
                            args,
                            output,
                            destination=destination,
                            keys=("b", "a"),
                            sizes=(ba_size, ba_size),
                        )
                    )
                )

                def capture_fla_input(_module, args, kwargs, *, destination=fla_inputs):
                    argument_names = (
                        "q",
                        "k",
                        "v",
                        "g",
                        "beta",
                        "initial_state",
                        "output_final_state",
                        "cu_seqlens",
                        "chunk_indices",
                        "chunk_offsets",
                        "use_qk_l2norm_in_kernel",
                    )
                    values = {
                        name: kwargs[name] if name in kwargs else args[index]
                        for index, name in enumerate(argument_names)
                    }
                    if values["use_qk_l2norm_in_kernel"]:
                        raise RuntimeError("Expected pre-normalized Q/K in the Qwen3.5 prefill path")
                    values["core_attn_out"] = kwargs.get(
                        "core_attn_out",
                        args[len(argument_names)] if len(args) > len(argument_names) else None,
                    )
                    captured_values = {
                        name: value.detach().clone() if isinstance(value, torch.Tensor) else value
                        for name, value in values.items()
                        if name != "core_attn_out"
                    }
                    captured_values["core_attn_out"] = values["core_attn_out"]
                    captured_values["input_layouts"] = {
                        name: tensor_layout(value) for name, value in values.items() if isinstance(value, torch.Tensor)
                    }
                    destination.append(captured_values)

                def capture_fla_output(
                    _module,
                    _args,
                    _kwargs,
                    output,
                    *,
                    source=fla_inputs,
                    live_outputs=fla_outputs,
                    live_values=fla_values,
                    destination=fla_captures,
                    backend=mixer.gdn_prefill_backend,
                    backend_method=mixer.chunk_gated_delta_rule._forward_method,
                ):
                    if len(source) != 1:
                        raise RuntimeError(f"Expected one pending FLA input capture, got {len(source)}")
                    values = source.pop()
                    live_output, live_final_state = output
                    live_outputs.append(live_output.detach())
                    live_values.append({name: values[name].detach().clone() for name in ("q", "k", "v")})

                    from fla.ops.gated_delta_rule.chunk import (
                        chunk_gated_delta_rule_fwd as released_chunk_gated_delta_rule_fwd,
                    )
                    from vllm.model_executor.layers.fla.ops.chunk import (
                        chunk_gated_delta_rule_fwd as vllm_chunk_gated_delta_rule_fwd,
                    )

                    released_g, released_output, released_A, released_final_state, _ = (
                        released_chunk_gated_delta_rule_fwd(
                            q=values["q"].clone(),
                            k=values["k"].clone(),
                            v=values["v"].clone(),
                            g=values["g"].clone(),
                            beta=values["beta"].clone(),
                            scale=values["k"].shape[-1] ** -0.5,
                            initial_state=values["initial_state"].clone(),
                            output_final_state=values["output_final_state"],
                            cu_seqlens=values["cu_seqlens"],
                            chunk_indices=values["chunk_indices"],
                            transpose_state_layout=True,  # Match vLLM's [N, H, V, K] cache layout.
                        )
                    )
                    vllm_arguments = {
                        "q": values["q"].clone(),
                        "k": values["k"].clone(),
                        "v": values["v"].clone(),
                        "g": values["g"].clone(),
                        "beta": values["beta"].clone(),
                        "scale": values["k"].shape[-1] ** -0.5,
                        "initial_state": values["initial_state"].clone(),
                        "output_final_state": values["output_final_state"],
                        "cu_seqlens": values["cu_seqlens"],
                        "chunk_indices": values["chunk_indices"],
                        "chunk_offsets": values["chunk_offsets"],
                    }
                    (
                        vllm_g,
                        vllm_output,
                        vllm_A,
                        vllm_final_state,
                        _,
                        _,
                        _,
                    ) = vllm_chunk_gated_delta_rule_fwd(**vllm_arguments)

                    live_chunk_destination = values["core_attn_out"]
                    destination_template = live_chunk_destination
                    if destination_template is None:
                        destination_template = live_output.squeeze(0)
                    replay_destination = torch.empty_strided(
                        destination_template.shape,
                        destination_template.stride(),
                        dtype=destination_template.dtype,
                        device=destination_template.device,
                    )
                    buffered_arguments = {
                        **vllm_arguments,
                        "q": values["q"].clone(),
                        "k": values["k"].clone(),
                        "v": values["v"].clone(),
                        "g": values["g"].clone(),
                        "beta": values["beta"].clone(),
                        "initial_state": values["initial_state"].clone(),
                        "core_attn_out": replay_destination,
                    }
                    (
                        buffered_g,
                        buffered_output,
                        buffered_A,
                        buffered_final_state,
                        _,
                        _,
                        _,
                    ) = vllm_chunk_gated_delta_rule_fwd(**buffered_arguments)
                    replay_destination_view = replay_destination[: buffered_output.numel()].view_as(buffered_output)

                    live_chunk_destination_view = None
                    if live_chunk_destination is not None:
                        live_chunk_destination_view = live_chunk_destination[: live_output.numel()].view_as(live_output)
                    contract_output = buffered_output if live_chunk_destination is not None else vllm_output
                    contract_final_state = (
                        buffered_final_state if live_chunk_destination is not None else vllm_final_state
                    )
                    destination.append(
                        {
                            "backend": {
                                "method": backend_method.__name__,
                                "method_module": backend_method.__module__,
                                "selected": backend,
                            },
                            "inputs": {
                                name: {
                                    "fingerprint": canonical_tensor_fingerprint(values[name]),
                                    "layout": values["input_layouts"][name],
                                    **(
                                        {"token_fingerprints": token_fingerprints(values[name])}
                                        if name != "initial_state"
                                        else {}
                                    ),
                                }
                                for name in ("q", "k", "v", "g", "beta", "initial_state")
                            },
                            "live": {
                                "chunk_destination_fingerprint": (
                                    canonical_tensor_fingerprint(live_chunk_destination_view)
                                    if live_chunk_destination_view is not None
                                    else None
                                ),
                                "chunk_destination_layout": (
                                    tensor_layout(live_chunk_destination)
                                    if live_chunk_destination is not None
                                    else None
                                ),
                                "chunk_destination_supplied": live_chunk_destination is not None,
                                "final_state_fingerprint": canonical_tensor_fingerprint(live_final_state),
                                "final_state_layout": tensor_layout(live_final_state),
                                "output_fingerprint": canonical_tensor_fingerprint(live_output),
                                "output_layout": tensor_layout(live_output),
                                "output_vs_chunk_destination": (
                                    exact_error_summary(live_output, live_chunk_destination_view)
                                    if live_chunk_destination_view is not None
                                    else None
                                ),
                                "output_chunk_destination_aliasing": (
                                    alias_summary(live_output, live_chunk_destination_view)
                                    if live_chunk_destination_view is not None
                                    else None
                                ),
                            },
                            "metadata": {
                                name: (values[name].detach().cpu().tolist() if values[name] is not None else None)
                                for name in ("cu_seqlens", "chunk_indices", "chunk_offsets")
                            },
                            "metadata_layouts": {
                                name: values["input_layouts"].get(name)
                                for name in ("cu_seqlens", "chunk_indices", "chunk_offsets")
                            },
                            "options": {
                                "output_final_state": values["output_final_state"],
                                "state_layout": "N,H,V,K",
                                "use_qk_l2norm_in_kernel": values["use_qk_l2norm_in_kernel"],
                            },
                            "live_vs_vllm_replay": {
                                "output": exact_error_summary(live_output, contract_output),
                                "final_state": exact_error_summary(live_final_state, contract_final_state),
                            },
                            "vllm_buffer_contract": {
                                "buffered_destination_fingerprint": canonical_tensor_fingerprint(
                                    replay_destination_view
                                ),
                                "buffered_destination_layout": tensor_layout(replay_destination),
                                "buffered_destination_vs_returned": exact_error_summary(
                                    replay_destination_view, buffered_output
                                ),
                                "buffered_return_aliasing": alias_summary(buffered_output, replay_destination_view),
                                "buffered_vs_unbuffered": {
                                    "A": exact_error_summary(buffered_A, vllm_A),
                                    "final_state": exact_error_summary(buffered_final_state, vllm_final_state),
                                    "g": exact_error_summary(buffered_g, vllm_g),
                                    "output": exact_error_summary(buffered_output, vllm_output),
                                },
                            },
                            "vllm_vs_released": {
                                "g": exact_error_summary(vllm_g, released_g),
                                "A": exact_error_summary(vllm_A, released_A),
                                "output": exact_error_summary(contract_output, released_output),
                                "final_state": exact_error_summary(contract_final_state, released_final_state),
                                "released_final_state_fingerprint": canonical_tensor_fingerprint(released_final_state),
                                "released_final_state_layout": tensor_layout(released_final_state),
                                "vllm_final_state_fingerprint": canonical_tensor_fingerprint(contract_final_state),
                                "vllm_final_state_layout": tensor_layout(contract_final_state),
                            },
                        }
                    )

                layer_hooks.append(
                    mixer.chunk_gated_delta_rule.register_forward_pre_hook(capture_fla_input, with_kwargs=True)
                )
                layer_hooks.append(
                    mixer.chunk_gated_delta_rule.register_forward_hook(capture_fla_output, with_kwargs=True)
                )

                had_instance_forward_core = "_forward_core" in mixer.__dict__
                instance_forward_core = mixer.__dict__.get("_forward_core")
                original_forward_core = mixer._forward_core

                def capture_forward_core(*args, original=original_forward_core, layer_mixer=mixer, **kwargs):
                    mixed_qkv = kwargs.get("mixed_qkv")
                    if mixed_qkv is None:
                        mixed_qkv = args[0]
                    model_destination = kwargs.get("core_attn_out")
                    if model_destination is None:
                        model_destination = args[3]

                    from causal_conv1d import causal_conv1d_fn as released_causal_conv1d_fn
                    from vllm.forward_context import get_forward_context
                    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
                    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
                        causal_conv1d_fn as vllm_causal_conv1d_fn,
                    )
                    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

                    forward_context = get_forward_context()
                    attn_metadata_raw = forward_context.attn_metadata
                    if not isinstance(attn_metadata_raw, dict):
                        raise RuntimeError("Expected per-layer GDN attention metadata")
                    attn_metadata = attn_metadata_raw[layer_mixer.prefix]
                    if not isinstance(attn_metadata, GDNAttentionMetadata):
                        raise RuntimeError("Expected GDN attention metadata for the layer-zero diagnostic")
                    num_actual_tokens = attn_metadata.num_actual_tokens
                    query_start_loc = attn_metadata.non_spec_query_start_loc.detach().clone()
                    has_initial_state = attn_metadata.has_initial_state.detach().clone()
                    state_indices = attn_metadata.non_spec_state_indices_tensor.detach().clone()
                    captured_x = mixed_qkv[:num_actual_tokens].detach().clone()
                    if query_start_loc.cpu().tolist() != [0, captured_x.shape[0]]:
                        raise RuntimeError(
                            "Expected one complete prefill sequence in the causal-convolution diagnostic"
                        )
                    if torch.any(has_initial_state):
                        raise RuntimeError("Expected a cold zero-state causal-convolution prefill")
                    if (
                        attn_metadata.num_prefills != 1
                        or attn_metadata.num_decodes != 0
                        or attn_metadata.spec_sequence_masks is not None
                    ):
                        raise RuntimeError("Expected one non-speculative prefill and no decodes")

                    conv_weight = (
                        layer_mixer.conv1d.weight.view(
                            layer_mixer.conv1d.weight.shape[0], layer_mixer.conv1d.weight.shape[2]
                        )
                        .detach()
                        .clone()
                    )
                    conv_bias = layer_mixer.conv1d.bias
                    conv_bias = conv_bias.detach().clone() if conv_bias is not None else None
                    live_conv_state = (
                        layer_mixer.kv_cache[0]
                        if is_conv_state_dim_first()
                        else layer_mixer.kv_cache[0].transpose(-1, -2)
                    )

                    result = original(*args, **kwargs)
                    if len(fla_outputs) != 1 or len(fla_values) != 1 or len(fla_captures) != 1:
                        raise RuntimeError("Expected one live FLA output before the model destination capture")
                    live_output = fla_outputs.pop()
                    live_fla_values = fla_values.pop()

                    def run_vllm_conv(token_major_x, *, activation, dirty_state=False):
                        channel_last_x = token_major_x.contiguous().transpose(0, 1)
                        scratch_state = torch.zeros(
                            2,
                            channel_last_x.shape[0],
                            conv_weight.shape[1] - 1,
                            dtype=live_conv_state.dtype,
                            device=channel_last_x.device,
                        )
                        if dirty_state:
                            scratch_state.fill_(1)
                        scratch_query_start_loc = torch.tensor(
                            [0, token_major_x.shape[0]], dtype=torch.int32, device=channel_last_x.device
                        )
                        scratch_output = vllm_causal_conv1d_fn(
                            channel_last_x,
                            conv_weight,
                            conv_bias,
                            conv_states=scratch_state,
                            query_start_loc=scratch_query_start_loc,
                            cache_indices=torch.ones(1, dtype=torch.int32, device=channel_last_x.device),
                            has_initial_state=torch.zeros(1, dtype=torch.bool, device=channel_last_x.device),
                            activation=activation,
                            metadata=None,
                            validate_data=True,
                        )
                        return scratch_output.transpose(0, 1).unsqueeze(0), scratch_state[1:2]

                    def run_released_conv(token_major_x, *, activation):
                        channel_first_x = token_major_x.transpose(0, 1).unsqueeze(0)
                        return released_causal_conv1d_fn(
                            channel_first_x,
                            conv_weight,
                            bias=conv_bias,
                            seq_idx=None,
                            activation=activation,
                        ).transpose(1, 2)

                    scratch_output, scratch_state = run_vllm_conv(captured_x, activation=layer_mixer.activation)
                    scratch_raw_output, _ = run_vllm_conv(captured_x, activation=None)
                    dirty_output, _ = run_vllm_conv(captured_x, activation=layer_mixer.activation, dirty_state=True)
                    released_output = run_released_conv(captured_x, activation=layer_mixer.activation)
                    released_raw_output = run_released_conv(captured_x, activation=None)
                    scratch_q, scratch_k, scratch_v = layer_mixer.rearrange_mixed_qkv(scratch_output.squeeze(0))
                    del scratch_q, scratch_k
                    scratch_v = scratch_v.unsqueeze(0)
                    expected_state = torch.nn.functional.pad(
                        captured_x.to(live_conv_state.dtype).transpose(0, 1),
                        (conv_weight.shape[1] - 1, 0),
                    )[:, -(conv_weight.shape[1] - 1) :].unsqueeze(0)

                    token0 = captured_x[:1]
                    vllm_token0_raw, _ = run_vllm_conv(token0, activation=None)
                    vllm_token0_silu, _ = run_vllm_conv(token0, activation="silu")
                    released_token0_raw = run_released_conv(token0, activation=None)
                    released_token0_silu = run_released_conv(token0, activation="silu")
                    manual_token0_raw = token0.float() * conv_weight[:, -1].float()
                    if conv_bias is not None:
                        manual_token0_raw = manual_token0_raw + conv_bias.float()
                    manual_token0_raw = manual_token0_raw.to(vllm_token0_raw.dtype).unsqueeze(0)
                    manual_token0_silu = torch.nn.functional.silu(manual_token0_raw.float()).to(vllm_token0_silu.dtype)

                    def tensor_payload(tensor):
                        return {
                            "fingerprint": canonical_tensor_fingerprint(tensor),
                            "layout": tensor_layout(tensor),
                            "token_fingerprints": token_fingerprints(tensor),
                        }

                    causal_conv_captures.append(
                        {
                            "backend": {
                                "method": vllm_causal_conv1d_fn.__qualname__,
                                "method_module": vllm_causal_conv1d_fn.__module__,
                                "released_method": released_causal_conv1d_fn.__qualname__,
                                "released_method_module": released_causal_conv1d_fn.__module__,
                            },
                            "inputs": {
                                "x": tensor_payload(captured_x.unsqueeze(0)),
                                "weight": {
                                    "fingerprint": canonical_tensor_fingerprint(conv_weight),
                                    "layout": tensor_layout(conv_weight),
                                },
                                "bias": (
                                    {
                                        "fingerprint": canonical_tensor_fingerprint(conv_bias),
                                        "layout": tensor_layout(conv_bias),
                                    }
                                    if conv_bias is not None
                                    else None
                                ),
                                "live_conv_state_layout": tensor_layout(live_conv_state),
                            },
                            "live_post_conv_v": tensor_payload(live_fla_values["v"]),
                            "scratch_replay": tensor_payload(scratch_output),
                            "released_replay": tensor_payload(released_output),
                            "comparisons": {
                                "dirty_vs_zero_state": exact_error_summary(dirty_output, scratch_output),
                                "live_post_conv_v_vs_scratch": exact_error_summary(live_fla_values["v"], scratch_v),
                                "scratch_state_vs_expected_tail": exact_error_summary(scratch_state, expected_state),
                                "scratch_vs_released": exact_error_summary(scratch_output, released_output),
                                "token0_raw": {
                                    "full_vs_singleton": exact_error_summary(
                                        scratch_raw_output[:, :1], vllm_token0_raw
                                    ),
                                    "released_full_vs_singleton": exact_error_summary(
                                        released_raw_output[:, :1], released_token0_raw
                                    ),
                                    "scratch_vs_manual": exact_error_summary(vllm_token0_raw, manual_token0_raw),
                                    "scratch_vs_released": exact_error_summary(vllm_token0_raw, released_token0_raw),
                                },
                                "token0_silu": {
                                    "full_vs_singleton": exact_error_summary(scratch_output[:, :1], vllm_token0_silu),
                                    "released_full_vs_singleton": exact_error_summary(
                                        released_output[:, :1], released_token0_silu
                                    ),
                                    "scratch_vs_manual": exact_error_summary(vllm_token0_silu, manual_token0_silu),
                                    "scratch_vs_released": exact_error_summary(vllm_token0_silu, released_token0_silu),
                                },
                            },
                            "metadata": {
                                "has_initial_state": has_initial_state.cpu().tolist(),
                                "num_actual_tokens": num_actual_tokens,
                                "num_decode_tokens": attn_metadata.num_decode_tokens,
                                "num_decodes": attn_metadata.num_decodes,
                                "num_prefills": attn_metadata.num_prefills,
                                "query_start_loc": query_start_loc.cpu().tolist(),
                                "spec_sequence_masks": None,
                                "state_indices": state_indices.cpu().tolist(),
                            },
                            "options": {"activation": layer_mixer.activation, "state_layout": "N,C,K-1"},
                        }
                    )
                    if (
                        model_destination.ndim == live_output.ndim - 1
                        and model_destination.shape[1:] == live_output.shape[2:]
                    ):
                        token_offset = model_destination.shape[0] - live_output.shape[1]
                        if token_offset < 0:
                            raise RuntimeError("Model destination is shorter than the live FLA output")
                        model_destination_view = model_destination[token_offset:].unsqueeze(0)
                    else:
                        token_offset = None
                        model_destination_view = model_destination.reshape(-1)[: live_output.numel()].view_as(
                            live_output
                        )
                    fla_captures[0]["model_destination"] = {
                        "fingerprint": canonical_tensor_fingerprint(model_destination_view),
                        "layout": tensor_layout(model_destination),
                        "output_aliasing": alias_summary(live_output, model_destination_view),
                        "output_error": exact_error_summary(live_output, model_destination_view),
                        "token_offset": token_offset,
                    }
                    return result

                mixer._forward_core = capture_forward_core
                forward_core_patches.append((mixer, had_instance_forward_core, instance_forward_core))

                def capture_norm_input(
                    _module,
                    args,
                    kwargs,
                    *,
                    destination=stages,
                    heads=mixer.num_v_heads // mixer.tp_size,
                ):
                    core_attn_out = kwargs.get("x")
                    if core_attn_out is None:
                        core_attn_out = args[0]
                    gate = kwargs.get("residual")
                    if gate is None:
                        gate = args[1]
                    destination.setdefault("core_attn_out", []).append(token_heads_payload(core_attn_out, heads))
                    destination.setdefault("z", []).append(token_heads_payload(gate, heads))

                def capture_norm_output(
                    _module,
                    _args,
                    _kwargs,
                    output,
                    *,
                    destination=stages,
                    heads=mixer.num_v_heads // mixer.tp_size,
                ):
                    destination.setdefault("norm_output", []).append(token_heads_payload(output, heads))

                layer_hooks.append(mixer.norm.register_forward_pre_hook(capture_norm_input, with_kwargs=True))
                layer_hooks.append(mixer.norm.register_forward_hook(capture_norm_output, with_kwargs=True))
                layer_hooks.append(
                    mixer.out_proj.register_forward_pre_hook(
                        lambda module, args, kwargs, destination=stages: capture_input(
                            module,
                            args,
                            kwargs,
                            destination=destination,
                            key="out_proj_input",
                        ),
                        with_kwargs=True,
                    )
                )
                layer_hooks.append(
                    mixer.out_proj.register_forward_hook(
                        lambda module, args, kwargs, output, destination=stages: capture_output(
                            module,
                            args,
                            kwargs,
                            output,
                            destination=destination,
                            key="out_proj_output",
                        ),
                        with_kwargs=True,
                    )
                )
            if layer_index == 0:
                mlp_stages = {}
                mlp_replays = {}
                entry["mlp_stages"] = mlp_stages
                entry["mlp_replays"] = mlp_replays

                def capture_gate_up_output(
                    module,
                    args,
                    output,
                    *,
                    destination=mlp_stages,
                    replays=mlp_replays,
                ):
                    hidden_states = args[0]
                    projected = output[0] if isinstance(output, tuple) else output
                    gate, up = projected.chunk(2, dim=-1)
                    destination.setdefault("gate", []).append(tensor_payload(gate))
                    destination.setdefault("up", []).append(tensor_payload(up))

                    split_size = gate.shape[-1]
                    weight = module.weight
                    bias = getattr(module, "bias", None)
                    gate_bias = None if bias is None else bias[:split_size]
                    up_bias = None if bias is None else bias[split_size:]
                    separate_gate = torch.nn.functional.linear(
                        hidden_states,
                        weight[:split_size],
                        gate_bias,
                    )
                    separate_up = torch.nn.functional.linear(
                        hidden_states,
                        weight[split_size:],
                        up_bias,
                    )
                    native_activation = torch.nn.functional.silu(gate)
                    separate_native_activation = torch.nn.functional.silu(separate_gate)
                    replays.setdefault("separate_gate", []).append(tensor_payload(separate_gate))
                    replays.setdefault("separate_up", []).append(tensor_payload(separate_up))
                    replays.setdefault("native_activation", []).append(tensor_payload(native_activation))
                    replays.setdefault("separate_native_activation", []).append(
                        tensor_payload(separate_native_activation)
                    )
                    replays.setdefault("native_product", []).append(tensor_payload(native_activation * up))
                    replays.setdefault("separate_native_product", []).append(
                        tensor_payload(separate_native_activation * separate_up)
                    )

                layer_hooks.append(layer.mlp.gate_up_proj.register_forward_hook(capture_gate_up_output))
                layer_hooks.append(
                    layer.mlp.act_fn.register_forward_hook(
                        lambda module, args, output, destination=mlp_stages: capture_output(
                            module,
                            args,
                            {},
                            output,
                            destination=destination,
                            key="product",
                        )
                    )
                )
                layer_hooks.append(
                    layer.mlp.down_proj.register_forward_hook(
                        lambda module, args, output, destination=mlp_stages: capture_output(
                            module,
                            args,
                            {},
                            output,
                            destination=destination,
                            key="down",
                        )
                    )
                )
            layer_hooks.append(
                mixer.register_forward_pre_hook(
                    lambda module, args, kwargs, destination=entry: capture_input(
                        module,
                        args,
                        kwargs,
                        destination=destination,
                        key="mixer_input",
                    ),
                    with_kwargs=True,
                )
            )
            layer_hooks.append(
                mixer.register_forward_hook(
                    lambda module, args, kwargs, output, destination=entry: capture_output(
                        module,
                        args,
                        kwargs,
                        output,
                        destination=destination,
                        key="mixer_output",
                    ),
                    with_kwargs=True,
                )
            )
            layer_hooks.append(
                layer.mlp.register_forward_pre_hook(
                    lambda module, args, kwargs, destination=entry: capture_input(
                        module,
                        args,
                        kwargs,
                        destination=destination,
                        key="mlp_input",
                    ),
                    with_kwargs=True,
                )
            )
            layer_hooks.append(
                layer.mlp.register_forward_hook(
                    lambda module, args, kwargs, output, destination=entry: capture_output(
                        module,
                        args,
                        kwargs,
                        output,
                        destination=destination,
                        key="mlp_output",
                    ),
                    with_kwargs=True,
                )
            )

        def capture_compute_logits(hidden_states, *args, **kwargs):
            captured_hidden_state = None
            if len(captures) < 8:
                captured_hidden_state = hidden_states[0].detach().float().cpu().contiguous()
            logits = original_compute_logits(hidden_states, *args, **kwargs)
            if captured_hidden_state is not None:
                selected_logit = logits[0, selected_token]
                logsumexp = logits[0].logsumexp(dim=-1)
                captures.append(
                    {
                        "head_input": captured_hidden_state.tolist(),
                        "head_input_dtype": str(hidden_states.dtype),
                        "head_input_shape": list(captured_hidden_state.shape),
                        "compute_logits_input_shape": list(hidden_states.shape),
                        "logits_dtype": str(logits.dtype),
                        "logits_shape": list(logits.shape),
                        "selected_logit": float(selected_logit.item()),
                        "logsumexp": float(logsumexp.item()),
                        "selected_logprob": float((selected_logit - logsumexp).item()),
                    }
                )
            return logits

        self._skyrl_original_compute_logits = original_compute_logits
        self._skyrl_had_instance_compute_logits = had_instance_compute_logits
        self._skyrl_instance_compute_logits = instance_compute_logits
        self._skyrl_head_input_captures = captures
        self._skyrl_layer_captures = layer_captures
        self._skyrl_layer_capture_hooks = layer_hooks
        self._skyrl_forward_core_patches = forward_core_patches
        model.compute_logits = capture_compute_logits
        self._skyrl_head_input_capture_active = True
        return {"active": True}

    def end_head_input_capture(self):
        """Restore logits computation and return the bounded diagnostic capture."""
        if not getattr(self, "_skyrl_head_input_capture_active", False):
            raise RuntimeError("No vLLM head-input capture is active")

        model = self.model_runner.model
        if self._skyrl_had_instance_compute_logits:
            model.compute_logits = self._skyrl_instance_compute_logits
        else:
            del model.compute_logits
        for mixer, had_instance_forward_core, instance_forward_core in self._skyrl_forward_core_patches:
            if had_instance_forward_core:
                mixer._forward_core = instance_forward_core
            else:
                del mixer._forward_core
        for hook in self._skyrl_layer_capture_hooks:
            hook.remove()
        captures = self._skyrl_head_input_captures
        if len(captures) == 1:
            for layer_entry in self._skyrl_layer_captures:
                for key in ("mixer_input", "mixer_output", "mlp_input", "mlp_output"):
                    values = layer_entry[key]
                    if len(values) != 1:
                        raise RuntimeError(
                            f"Expected one vLLM {key} capture in layer {layer_entry['layer']}, got {len(values)}"
                        )
                    layer_entry[key] = values[0]
                for mode, projection_captures in layer_entry.get("projections", {}).items():
                    for key, values in projection_captures.items():
                        if len(values) != 1:
                            raise RuntimeError(
                                f"Expected one vLLM {mode} {key} projection capture in layer "
                                f"{layer_entry['layer']}, got {len(values)}"
                            )
                        projection_captures[key] = values[0]
                for key, values in layer_entry.get("mixer_stages", {}).items():
                    if len(values) != 1:
                        raise RuntimeError(
                            f"Expected one vLLM {key} stage capture in layer {layer_entry['layer']}, got {len(values)}"
                        )
                    layer_entry["mixer_stages"][key] = values[0]
                for capture_name in ("mlp_stages", "mlp_replays"):
                    for key, values in layer_entry.get(capture_name, {}).items():
                        if len(values) != 1:
                            raise RuntimeError(
                                f"Expected one vLLM {key} {capture_name} capture in layer "
                                f"{layer_entry['layer']}, got {len(values)}"
                            )
                        layer_entry[capture_name][key] = values[0]
                fla_core = layer_entry.get("fla_core")
                if fla_core is not None:
                    if len(fla_core) != 1:
                        raise RuntimeError(
                            f"Expected one vLLM FLA core capture in layer {layer_entry['layer']}, got {len(fla_core)}"
                        )
                    if "model_destination" not in fla_core[0]:
                        raise RuntimeError(f"Missing vLLM model destination capture in layer {layer_entry['layer']}")
                    layer_entry["fla_core"] = fla_core[0]
                causal_conv = layer_entry.get("causal_conv")
                if causal_conv is not None:
                    if len(causal_conv) != 1:
                        raise RuntimeError(
                            f"Expected one vLLM causal-convolution capture in layer "
                            f"{layer_entry['layer']}, got {len(causal_conv)}"
                        )
                    layer_entry["causal_conv"] = causal_conv[0]
            captures[0]["layer_trace"] = self._skyrl_layer_captures
        del self._skyrl_original_compute_logits
        del self._skyrl_had_instance_compute_logits
        del self._skyrl_instance_compute_logits
        del self._skyrl_head_input_captures
        del self._skyrl_layer_captures
        del self._skyrl_layer_capture_hooks
        del self._skyrl_forward_core_patches
        self._skyrl_head_input_capture_active = False
        return captures

    def read_expert_slots_raw(self, layer_idx: int):
        """TEST-ONLY (D1/D2 disaggregated-receive diag): return THIS engine worker's
        RAW per-local-slot FusedMoE expert weights + the engine's OWN expert_map for
        ``layer_idx``, with NO assumption about global<->local placement.

        Unlike ``read_named_weights`` (which maps a requested HF expert ``gj`` to a slot
        via ``gj // n_local`` — a CONTIGUOUS-linear assumption that would HIDE a
        receive-side placement bug), this dumps every local slot's bytes AS-IS plus the
        engine's authoritative ``_expert_map`` (global->local, -1 if absent). The driver
        then does a NON-circular cross-expert nearest-match vs the DISK base checkpoint,
        so a slot carrying the wrong global expert (D2 placement) surfaces as m!=j even
        though a per-name readback could not see it.

        Returns dict:
          __ranks__        : {tp_rank, tp_size, ep_rank, ep_size}
          expert_map       : list[int] length global_num_experts (global->local slot, -1 absent)
          slot_to_global   : list[int] length local_num_experts (inverse; -1 if ambiguous)
          local_num_experts, global_num_experts, w13_inter_half (I), placement_strategy
          slots            : {local_slot -> {"w13": cpu_fp32 [2I,H], "w2": cpu_fp32 [H,I]}}
        Heavy (full local expert stack as fp32 CPU); call for ONE layer.
        """
        import torch as _torch

        model = self.model_runner.model
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        all_params = {**params, **buffers}

        try:
            from vllm.distributed import parallel_state as _ps

            tp_rank = _ps.get_tensor_model_parallel_rank()
            tp_size = _ps.get_tensor_model_parallel_world_size()
        except Exception:
            tp_rank, tp_size = 0, 1
        try:
            ep_rank = _ps.get_ep_group().rank_in_group
            ep_size = _ps.get_ep_group().world_size
        except Exception:
            ep_rank, ep_size = 0, 1

        out = {"__ranks__": {"tp_rank": tp_rank, "tp_size": tp_size, "ep_rank": ep_rank, "ep_size": ep_size}}
        prefix = f"model.layers.{layer_idx}.mlp"
        w13 = all_params.get(f"{prefix}.experts.routed_experts.w13_weight")
        w2 = all_params.get(f"{prefix}.experts.routed_experts.w2_weight")
        if w13 is None or w2 is None:
            cand = [k for k in all_params if k.startswith(f"{prefix}.experts.") and k.endswith("weight")]
            out["error"] = f"no w13/w2 under {prefix}; candidates={cand}"
            return out

        # Find the FusedMoE module to read its authoritative expert_map + counts.
        emap = None
        local_num = int(w13.shape[0])
        global_num = None
        placement = None
        for mod_name, mod in model.named_modules():
            routed_experts_name = f"{prefix}.experts.routed_experts"
            if mod_name == routed_experts_name or mod_name.endswith(routed_experts_name):
                emap = getattr(mod, "_expert_map", None)
                if emap is None:
                    emap = getattr(mod, "expert_map", None)
                local_num = int(getattr(mod, "local_num_experts", local_num))
                global_num = getattr(mod, "global_num_experts", None)
                placement = getattr(mod, "expert_placement_strategy", None)
                break

        # global->local slot map (the engine's OWN authority on placement).
        if emap is not None:
            emap_list = [int(x) for x in emap.detach().cpu().tolist()]
        else:
            emap_list = None
        out["expert_map"] = emap_list
        out["local_num_experts"] = local_num
        out["global_num_experts"] = int(global_num) if global_num is not None else None
        out["placement_strategy"] = str(placement) if placement is not None else None
        out["w13_inter_half"] = int(w13.shape[1] // 2)

        # Inverse: local slot -> global expert (per the engine's expert_map).
        slot_to_global = [-1] * local_num
        if emap_list is not None:
            for g, loc in enumerate(emap_list):
                if 0 <= loc < local_num:
                    slot_to_global[loc] = g
        out["slot_to_global"] = slot_to_global

        slots = {}
        for s in range(local_num):
            slots[s] = {
                "w13": w13[s].detach().to("cpu", dtype=_torch.float32).contiguous(),
                "w2": w2[s].detach().to("cpu", dtype=_torch.float32).contiguous(),
            }
        out["slots"] = slots
        return out

    def report_host(self):
        """TEST-ONLY (disaggregation proof): this engine worker's hostname (one per
        TP/EP worker via collective_rpc) so the driver can prove the engine node is
        DISJOINT from the policy nodes (=> the broadcast is genuinely cross-node)."""
        import socket

        return socket.gethostname()

    def report_runtime_installation(self, expected_vllm_engine_sha256: str):
        """Return and verify the MarinSkyRL module loaded by this engine worker."""
        import hashlib
        from pathlib import Path

        import skyrl_train

        if len(expected_vllm_engine_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_vllm_engine_sha256
        ):
            raise ValueError("expected_vllm_engine_sha256 must be a lowercase SHA256 digest")

        module_path = Path(__file__).resolve()
        actual_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
        payload = {
            "skyrl_train_file": str(Path(skyrl_train.__file__).resolve()),
            "vllm_engine_file": str(module_path),
            "vllm_engine_sha256": actual_sha256,
            "expected_vllm_engine_sha256": expected_vllm_engine_sha256,
            "matches_checkout": actual_sha256 == expected_vllm_engine_sha256,
        }
        print(f"SKYRL_ENGINECORE_RUNTIME {json.dumps(payload, sort_keys=True)}", flush=True)
        if not payload["matches_checkout"]:
            raise RuntimeError(f"EngineCore MarinSkyRL source mismatch: {payload}")
        return payload


class BaseVLLMInferenceEngine(InferenceEngineInterface):
    """Base class containing shared logic between sync and async VLLM engines."""

    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        setup_envvars_for_vllm(kwargs, bundle_indices)
        lm_head_compute_dtype = kwargs.pop("lm_head_compute_dtype", None)
        # vLLM may construct the model in a separate EngineCore process, so the
        # explicit constructor setting crosses that process boundary via env.
        if lm_head_compute_dtype is None:
            os.environ.pop(VLLM_LM_HEAD_COMPUTE_DTYPE_ENV, None)
        else:
            os.environ[VLLM_LM_HEAD_COMPUTE_DTYPE_ENV] = lm_head_compute_dtype
        configured_model_classes = configure_vllm_qwen3_5_lm_head_compute_dtype(lm_head_compute_dtype)
        if configured_model_classes:
            action = "enabled" if lm_head_compute_dtype is not None else "restored"
            logger.info(f"{action.capitalize()} lm_head compute for {', '.join(configured_model_classes)}")
        vllm_v1_disable_multiproc = kwargs.pop("vllm_v1_disable_multiproc", False)
        logger.info(
            f"BaseVLLMInferenceEngine: vllm_v1_disable_multiproc={vllm_v1_disable_multiproc}, "
            f"vllm.__version__={vllm.__version__}, "
            f"VLLM_ENABLE_V1_MULTIPROCESSING={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', '<unset>')}"
        )
        if vllm_v1_disable_multiproc or vllm.__version__ == "0.8.2":
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("BaseVLLMInferenceEngine: set VLLM_ENABLE_V1_MULTIPROCESSING=0")

        # Store common attributes
        self._tp_size = kwargs.get("tensor_parallel_size", 1)
        self._pp_size = kwargs.get("pipeline_parallel_size", 1)
        self._dp_size = kwargs.get("data_parallel_size", 1)
        self._is_lora = kwargs.get("enable_lora", False)

        if "rope_scaling" in kwargs:
            kwargs.pop("rope_scaling")
        # Let subclass create the appropriate engine
        try:
            self.llm = self._create_engine(*args, **kwargs)
        finally:
            if lm_head_compute_dtype is not None:
                restored_model_classes = configure_vllm_qwen3_5_lm_head_compute_dtype(None)
                if restored_model_classes:
                    logger.info(f"Restored lm_head compute for {', '.join(restored_model_classes)}")

        # Set NUMA affinity for TP>1 workers via collective_rpc
        if self._tp_size > 1 or self._pp_size > 1:
            try:
                self.llm.collective_rpc("set_numa_affinity")
            except Exception:
                pass

        # Weight loader is created by subclass after engine initialization
        self._weight_loader = None

    def tp_size(self):
        return self._tp_size

    def pp_size(self):
        return self._pp_size

    def dp_size(self):
        return self._dp_size

    def _create_engine(self, *args, **kwargs):
        """Abstract method for subclasses to implement engine creation."""
        raise NotImplementedError("Subclasses must implement _create_engine")

    def _preprocess_prompts(self, input_batch: InferenceEngineInput):
        """Common prompt preprocessing logic."""
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        request_sampling_params = input_batch.get("sampling_params")

        assert prompts is None and prompt_token_ids is not None, (
            "VLLMInferenceEngine only accepts `prompt_token_ids`, not `prompts`."
        )

        sampling_params = (
            SamplingParams(**request_sampling_params) if request_sampling_params is not None else SamplingParams()
        )

        return prompt_token_ids, sampling_params

    def _postprocess_outputs(self, outputs):
        """Common output processing logic."""
        responses: List[str] = []
        stop_reasons: List[str] = []
        response_ids: List[List[int]] = []
        response_logprobs: Optional[List[List[float]]] = []
        all_prompt_logprobs: Optional[List] = None

        for output in outputs:
            # TODO(tgriggs): Support n>1 sampling.
            assert len(output.outputs) == 1, (
                "Each prompt should have only one responses. n>1 sampling is supported by copying prompts."
            )
            resp = output.outputs[0]
            responses.append(resp.text)
            stop_reasons.append(resp.finish_reason)
            response_ids.append(resp.token_ids)
            _logprobs = None
            if resp.logprobs:
                _logprobs = []
                for i, token_logprobs in enumerate(resp.logprobs):
                    token_logprobs: Dict[str, Logprob]
                    token_id = resp.token_ids[i]
                    logprob = token_logprobs[token_id].logprob
                    _logprobs.append(logprob)
                    del token_logprobs
            response_logprobs.append(_logprobs)

            # Extract prompt_logprobs if available (used for teacher scoring)
            if hasattr(output, "prompt_logprobs") and output.prompt_logprobs is not None:
                if all_prompt_logprobs is None:
                    all_prompt_logprobs = []
                # Convert vLLM's List[Optional[Dict[int, Logprob]]] to
                # List[Optional[Dict[int, float]]] (extract .logprob from Logprob objects)
                prompt_lps = []
                for pos_logprobs in output.prompt_logprobs:
                    if pos_logprobs is None:
                        prompt_lps.append(None)
                    else:
                        prompt_lps.append(
                            {
                                token_id: lp.logprob if hasattr(lp, "logprob") else lp
                                for token_id, lp in pos_logprobs.items()
                            }
                        )
                all_prompt_logprobs.append(prompt_lps)

        if len(response_logprobs) and response_logprobs[0] is None:
            response_logprobs = None  # hack: assume uniform sampling params

        return InferenceEngineOutput(
            responses=responses,
            stop_reasons=stop_reasons,
            response_ids=response_ids,
            response_logprobs=response_logprobs,
            prompt_logprobs=all_prompt_logprobs,
        )

    def _get_engine(self):
        """Get the underlying engine for RPC calls."""
        return self.llm.engine if hasattr(self.llm, "engine") else self.llm

    def _is_lora_disk_loading_request(self, request: NamedWeightsUpdateRequest) -> bool:
        """Check if this is a LoRA disk loading request."""
        is_lora = request["names"][0] == "lora_disk_load"
        if is_lora:
            assert request.get("extras") and len(request["extras"]) > 0 and "lora_disk_path" in request["extras"][0], (
                "vLLM LoRA weight update requests must contain the disk load path under key `lora_disk_path`"
            )
        return is_lora

    def reset_prefix_cache(self):
        """Reset the prefix cache. Subclasses override for async version."""
        return self.llm.llm_engine.reset_prefix_cache()

    async def pause_generation(self) -> None:
        raise NotImplementedError("Pausing generation is only supported for AsyncVLLMInferenceEngine.")

    async def resume_generation(self) -> None:
        raise NotImplementedError("Resuming generation is only supported for AsyncVLLMInferenceEngine.")


class VLLMInferenceEngine(BaseVLLMInferenceEngine):
    """Synchronous VLLM engine."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_loader = VLLMWeightLoader(self.llm, is_async=False)

    def _create_engine(self, *args, **kwargs):
        # Pipeline parallelism requires AsyncLLMEngine
        if kwargs.get("pipeline_parallel_size", 1) > 1:
            raise ValueError(
                "Pipeline parallelism is only supported with AsyncVLLMInferenceEngine. "
                "Please set `generator.async_engine=true` in your config."
            )
        # Strip OpenAI-serving-only kwargs (e.g. openai_sampling_params, tool
        # parser) that the config layer injects for all engines. The sync
        # vllm.LLM/EngineArgs path does not accept these — only the async
        # OpenAI server consumes them. Mirror the async engine's pop so the
        # sync engine (async_engine=false, used by the batched OPD path) does
        # not pass them through to EngineArgs and raise TypeError.
        openai_kwargs = pop_openai_kwargs(kwargs)
        self._openai_sampling_params = openai_kwargs.pop("openai_sampling_params", {})
        # Pop enable_ray_prometheus_stats - only supported for async engine
        enable_ray_prometheus_stats = kwargs.pop("enable_ray_prometheus_stats", False)
        if enable_ray_prometheus_stats:
            logger.warning(
                "enable_ray_prometheus_stats is only supported with AsyncVLLMInferenceEngine. "
                "Set `generator.async_engine=true` to enable Ray Prometheus stats logging."
            )
        return vllm.LLM(*args, **kwargs)

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        prompt_token_ids, sampling_params = self._preprocess_prompts(input_batch)

        # Check if LoRA is enabled and create LoRA requests
        lora_requests = None
        if self._is_lora:
            lora_int_ids = list(self.llm.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                batch_size = len(prompt_token_ids)
                # dummy_lora_path for placeholder (actual loading done in add_lora())
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/dummy_lora_path")
                ] * batch_size

        outputs = await asyncio.to_thread(
            self.llm.generate,
            prompts=[TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids],
            sampling_params=sampling_params,
            lora_request=lora_requests,
        )

        return self._postprocess_outputs(outputs)

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Only supported in AsyncVLLMInferenceEngine."""
        raise NotImplementedError()

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Only supported in AsyncVLLMInferenceEngine."""
        raise NotImplementedError()

    async def wake_up(self, *args: Any, **kwargs: Any):
        await asyncio.to_thread(self.llm.wake_up, tags=kwargs.get("tags", None))

    async def sleep(self, *args: Any, **kwargs: Any):
        engine = self._get_engine().llm_engine
        output_processor = engine.output_processor
        if output_processor.has_unfinished_requests():
            logger.warning(
                "Calling sleep() with unfinished requests in vLLM engine. This is unexpected since all "
                "generation should be done before sleep() is called. Check for potential failures or "
                "dangling requests in your Generator/Env. Aborting all unfinished requests."
            )
            unfinished_request_ids = list(output_processor.request_states.keys())
            await asyncio.to_thread(engine.abort_request, unfinished_request_ids)

        level = 1 if self._is_lora else kwargs.get("level", 2)
        await asyncio.to_thread(self.llm.sleep, level=level)

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        engine = self._get_engine()
        return await asyncio.to_thread(
            engine.collective_rpc,
            "init_weight_update_communicator",
            args=(master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing),
        )

    async def _load_lora_from_disk(self, lora_path: str):
        """Load LoRA adapters from disk using vLLM's native add_lora method."""
        lora_id = int(time.time_ns() % 0x7FFFFFFF)
        lora_request = LoRARequest(lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path=lora_path)
        result = self.llm.llm_engine.add_lora(lora_request)
        return result

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        if "names" not in request:
            raise ValueError(f"Expected update weight request with 'names' entry, got keys: {request.keys()}")

        if not len(request["names"]):
            raise ValueError("Update weight request should have at least one entry in 'names'")

        # Handle LoRA disk loading request
        if self._is_lora_disk_loading_request(request):
            lora_path = request["extras"][0]["lora_disk_path"]
            return await self._load_lora_from_disk(lora_path)

        # Use the weight loader to coordinate weight transfer
        return await self._weight_loader.load_weights(request)

    async def teardown(self):
        await self._destroy_weights_update_group()

    async def reset_prefix_cache(self):
        return await asyncio.to_thread(self.llm.llm_engine.reset_prefix_cache)

    async def _destroy_weights_update_group(self):
        engine = self._get_engine()
        return await asyncio.to_thread(engine.collective_rpc, "destroy_weights_update_group")


class V1LoggingStatLoggerFixed(LoggingStatLogger):
    """
    A fixed version of LoggingStatLogger that actually logs during the record method.
    The log method is otherwise not called in the VLLM codebase.

    Also stores aggregated stats in a class-level registry for programmatic access
    (used by VLLMStatsCallback to bypass Ray log-to-driver unreliability).

    Stats are accumulated throughout a step:
    - Request counts (running, waiting): track peak and median values
    - Throughput metrics: track peak and median values observed during active periods
    - Cache metrics: track peak and median usage
    """

    # Class-level registry mapping engine IDs to their accumulated stats
    _stats_registry: Dict[int, Dict[str, Any]] = {}
    _registry_lock = threading.Lock()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.log_interval = 5
        self._engine_id: Optional[int] = None

    def set_engine_id(self, engine_id: int) -> None:
        """Set the engine ID for this stat logger instance."""
        self._engine_id = engine_id

    def record(self, *args: Any, **kwargs: Any) -> None:
        # Call parent with original arguments - important to preserve vLLM's calling convention
        super().record(*args, **kwargs)

        # Accumulate stats in registry if engine ID is set
        if self._engine_id is not None:
            # Extract scheduler_stats from vLLM v1 API:
            # vLLM calls record(scheduler_stats, iteration_stats, ...) with positional args
            # or record(scheduler_stats=..., iteration_stats=...) with keyword args
            scheduler_stats = None
            if args:
                scheduler_stats = args[0]
            elif "scheduler_stats" in kwargs:
                scheduler_stats = kwargs["scheduler_stats"]

            current_running = 0
            current_waiting = 0
            current_cache_usage = 0.0
            prefix_cache_stats = None

            if scheduler_stats is not None:
                current_running = getattr(scheduler_stats, "num_running_reqs", 0)
                current_waiting = getattr(scheduler_stats, "num_waiting_reqs", 0)
                current_cache_usage = getattr(scheduler_stats, "kv_cache_usage", 0.0) * 100.0  # Convert to percentage
                prefix_cache_stats = scheduler_stats.prefix_cache_stats

            # Extract iteration_stats (second positional arg) for per-request latency data
            iteration_stats = None
            if len(args) > 1:
                iteration_stats = args[1]
            elif "iteration_stats" in kwargs:
                iteration_stats = kwargs["iteration_stats"]

            # Collect per-request latency samples from finished requests
            finished_prefill_times: List[float] = []
            finished_decode_times: List[float] = []
            finished_e2e_latencies: List[float] = []
            finished_queued_times: List[float] = []
            finished_ttfts: List[float] = []
            finished_num_preempted = 0
            if iteration_stats is not None:
                # Time-to-first-token samples from this iteration
                ttft_iter = getattr(iteration_stats, "time_to_first_tokens_iter", None)
                if ttft_iter:
                    finished_ttfts.extend(ttft_iter)
                # Preemption count
                finished_num_preempted = getattr(iteration_stats, "num_preempted_reqs", 0)
                # Per-request stats from completed requests
                for req in getattr(iteration_stats, "finished_requests", []):
                    prefill_t = getattr(req, "prefill_time", 0.0)
                    decode_t = getattr(req, "decode_time", 0.0)
                    e2e_t = getattr(req, "e2e_latency", 0.0)
                    queued_t = getattr(req, "queued_time", 0.0)
                    if prefill_t > 0:
                        finished_prefill_times.append(prefill_t)
                    if decode_t > 0:
                        finished_decode_times.append(decode_t)
                    if e2e_t > 0:
                        finished_e2e_latencies.append(e2e_t)
                    if queued_t > 0:
                        finished_queued_times.append(queued_t)

            # Throughput is computed by parent class LoggingStatLogger after super().record()
            # These are stored as instance attributes
            current_prompt_tp = getattr(self, "last_prompt_throughput", 0.0) or 0.0
            current_gen_tp = getattr(self, "last_generation_throughput", 0.0) or 0.0

            is_active = current_running > 0 or current_waiting > 0

            with V1LoggingStatLoggerFixed._registry_lock:
                existing = V1LoggingStatLoggerFixed._stats_registry.get(self._engine_id)

                if existing is None:
                    # Initialize with sample lists for median calculation
                    existing = {
                        # Sample lists for computing median (only active samples)
                        "_samples_prompt_tp": [current_prompt_tp] if is_active else [],
                        "_samples_gen_tp": [current_gen_tp] if is_active else [],
                        "_samples_running": [current_running] if is_active else [],
                        "_samples_waiting": [current_waiting] if is_active else [],
                        "_samples_cache": [current_cache_usage] if is_active else [],
                        "_prefix_hit": PrefixCacheHitRateAccumulator(),
                        # Per-request latency samples (accumulated from finished requests)
                        "_samples_prefill_time": list(finished_prefill_times),
                        "_samples_decode_time": list(finished_decode_times),
                        "_samples_e2e_latency": list(finished_e2e_latencies),
                        "_samples_queued_time": list(finished_queued_times),
                        "_samples_ttft": list(finished_ttfts),
                        "_total_preempted": finished_num_preempted,
                        # Peak values
                        "_peak_prompt_tp": current_prompt_tp,
                        "_peak_gen_tp": current_gen_tp,
                        "_peak_running": current_running,
                        "_peak_waiting": current_waiting,
                        "_peak_cache": current_cache_usage,
                        # Counters
                        "_num_samples": 1,
                        "_num_active_samples": 1 if is_active else 0,
                        "timestamp": time.time(),
                    }
                    V1LoggingStatLoggerFixed._stats_registry[self._engine_id] = existing
                else:
                    # Update peak values
                    existing["_peak_prompt_tp"] = max(existing["_peak_prompt_tp"], current_prompt_tp)
                    existing["_peak_gen_tp"] = max(existing["_peak_gen_tp"], current_gen_tp)
                    existing["_peak_running"] = max(existing["_peak_running"], current_running)
                    existing["_peak_waiting"] = max(existing["_peak_waiting"], current_waiting)
                    existing["_peak_cache"] = max(existing["_peak_cache"], current_cache_usage)

                    # Accumulate per-request latency samples
                    existing["_samples_prefill_time"].extend(finished_prefill_times)
                    existing["_samples_decode_time"].extend(finished_decode_times)
                    existing["_samples_e2e_latency"].extend(finished_e2e_latencies)
                    existing["_samples_queued_time"].extend(finished_queued_times)
                    existing["_samples_ttft"].extend(finished_ttfts)
                    existing["_total_preempted"] += finished_num_preempted

                    # Append to sample lists (only for active samples to get meaningful medians)
                    if is_active:
                        existing["_samples_prompt_tp"].append(current_prompt_tp)
                        existing["_samples_gen_tp"].append(current_gen_tp)
                        existing["_samples_running"].append(current_running)
                        existing["_samples_waiting"].append(current_waiting)
                        existing["_samples_cache"].append(current_cache_usage)
                        existing["_num_active_samples"] += 1

                    existing["_num_samples"] += 1
                    existing["timestamp"] = time.time()

                existing["_prefix_hit"].observe(prefix_cache_stats, is_active=is_active)

        now = time.monotonic()
        if now - self.last_log_time > self.log_interval:
            self.log()
            self.last_log_time = now

    @staticmethod
    def _compute_median(samples: List[float]) -> float:
        """Compute median of a list of samples."""
        if not samples:
            return 0.0
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_samples[mid - 1] + sorted_samples[mid]) / 2.0
        return sorted_samples[mid]

    @classmethod
    def get_stats_by_engine_id(cls, engine_id: int, reset: bool = True) -> Optional[Dict[str, Any]]:
        """Get the accumulated stats for a given engine ID.

        Args:
            engine_id: The engine ID to get stats for.
            reset: If True, reset the accumulated stats after reading (default True).
                   This ensures each training step gets fresh stats.

        Returns:
            Dict with accumulated stats, or None if no stats recorded yet.
            Includes peak values, median values, and computed averages.
        """
        with cls._registry_lock:
            stats = cls._stats_registry.get(engine_id)
            if stats is None:
                return None

            # Compute medians from sample lists
            median_prompt_tp = cls._compute_median(stats["_samples_prompt_tp"])
            median_gen_tp = cls._compute_median(stats["_samples_gen_tp"])
            median_running = cls._compute_median(stats["_samples_running"])
            median_waiting = cls._compute_median(stats["_samples_waiting"])
            median_cache = cls._compute_median(stats["_samples_cache"])
            median_prefix_hit = cls._compute_median(stats["_prefix_hit"].samples)

            # Compute means from sample lists
            num_active = stats["_num_active_samples"]
            if num_active > 0:
                mean_prompt_tp = sum(stats["_samples_prompt_tp"]) / num_active
                mean_gen_tp = sum(stats["_samples_gen_tp"]) / num_active
            else:
                mean_prompt_tp = 0.0
                mean_gen_tp = 0.0

            # Compute per-request latency statistics
            prefill_samples = stats["_samples_prefill_time"]
            decode_samples = stats["_samples_decode_time"]
            e2e_samples = stats["_samples_e2e_latency"]
            queued_samples = stats["_samples_queued_time"]
            ttft_samples = stats["_samples_ttft"]

            def _mean(s: List[float]) -> float:
                return sum(s) / len(s) if s else 0.0

            def _p90(s: List[float]) -> float:
                if not s:
                    return 0.0
                sorted_s = sorted(s)
                idx = int(len(sorted_s) * 0.9)
                return sorted_s[min(idx, len(sorted_s) - 1)]

            result = {
                # Peak values
                "peak_prompt_throughput": stats["_peak_prompt_tp"],
                "peak_generation_throughput": stats["_peak_gen_tp"],
                "peak_running_reqs": stats["_peak_running"],
                "peak_waiting_reqs": stats["_peak_waiting"],
                "peak_gpu_cache_usage_perc": stats["_peak_cache"],
                "peak_prefix_cache_hit_rate": stats["_prefix_hit"].peak,
                # Median values
                "median_prompt_throughput": median_prompt_tp,
                "median_generation_throughput": median_gen_tp,
                "median_running_reqs": median_running,
                "median_waiting_reqs": median_waiting,
                "median_gpu_cache_usage_perc": median_cache,
                "median_prefix_cache_hit_rate": median_prefix_hit,
                # Mean values
                "mean_prompt_throughput": mean_prompt_tp,
                "mean_generation_throughput": mean_gen_tp,
                # Per-request latency stats (seconds)
                "latency_prefill_mean": _mean(prefill_samples),
                "latency_prefill_median": cls._compute_median(prefill_samples),
                "latency_prefill_p90": _p90(prefill_samples),
                "latency_decode_mean": _mean(decode_samples),
                "latency_decode_median": cls._compute_median(decode_samples),
                "latency_decode_p90": _p90(decode_samples),
                "latency_e2e_mean": _mean(e2e_samples),
                "latency_e2e_median": cls._compute_median(e2e_samples),
                "latency_e2e_p90": _p90(e2e_samples),
                "latency_queued_mean": _mean(queued_samples),
                "latency_queued_median": cls._compute_median(queued_samples),
                "latency_queued_p90": _p90(queued_samples),
                "latency_ttft_mean": _mean(ttft_samples),
                "latency_ttft_median": cls._compute_median(ttft_samples),
                "latency_ttft_p90": _p90(ttft_samples),
                "latency_num_finished_requests": len(e2e_samples),
                "total_preempted_reqs": stats["_total_preempted"],
                # Legacy field names for backwards compatibility (use peak values)
                "avg_prompt_throughput": stats["_peak_prompt_tp"],
                "avg_generation_throughput": stats["_peak_gen_tp"],
                "num_running_reqs": stats["_peak_running"],
                "num_waiting_reqs": stats["_peak_waiting"],
                "gpu_cache_usage_perc": stats["_peak_cache"],
                "prefix_cache_hit_rate": stats["_prefix_hit"].peak,
                # Metadata
                "timestamp": stats["timestamp"],
                "num_samples": stats["_num_samples"],
                "num_active_samples": stats["_num_active_samples"],
            }

            if reset:
                # Reset for next step
                del cls._stats_registry[engine_id]

            return result


class AsyncVLLMInferenceEngine(BaseVLLMInferenceEngine):
    """Asynchronous VLLM engine."""

    def __init__(self, *args, **kwargs):
        # Generate unique engine ID before calling super().__init__() which calls _create_engine
        self._stats_engine_id = id(self)
        self._batch_admission_lock = asyncio.Lock()
        super().__init__(*args, **kwargs)
        self._weight_loader = VLLMWeightLoader(self.llm, is_async=True)

    def _create_stat_logger_factory(self):
        """Create a factory that produces stat loggers with the engine ID set."""
        engine_id = self._stats_engine_id

        def factory(*args, **kwargs):
            logger_instance = V1LoggingStatLoggerFixed(*args, **kwargs)
            logger_instance.set_engine_id(engine_id)
            return logger_instance

        return factory

    def _create_engine(self, *args, **kwargs):
        openai_kwargs = pop_openai_kwargs(kwargs)
        # Store sampling params for OpenAI-style requests (Harbor rollouts)
        self._openai_sampling_params = openai_kwargs.pop("openai_sampling_params", {})
        if self._openai_sampling_params:
            logger.warning(
                f"OpenAI API sampling params overridden: "
                f"temperature={self._openai_sampling_params.get('temperature', 1.0)}, "
                f"top_p={self._openai_sampling_params.get('top_p', 1.0)}, "
                f"top_k={self._openai_sampling_params.get('top_k', -1)}"
            )
        enable_ray_prometheus_stats = kwargs.pop("enable_ray_prometheus_stats", False)

        # TODO (erictang000): potentially enable log requests for a debugging mode
        custom_chat_template_path = kwargs.pop("custom_chat_template_chat_completion_path", None)
        # Use factory to inject engine ID into stat logger
        stat_loggers = [self._create_stat_logger_factory()]

        # vLLM >= 0.10 renamed AsyncEngineArgs' `disable_log_requests=True` to
        # `enable_log_requests=False` (and removed the old kwarg). Gate on the
        # ACTUAL field set rather than a parsed version number: source-built
        # vLLM forks report PEP 440-valid dev versions like "0.1.dev16611+g..."
        # which parse as 0.1 (< 0.10.0) even though they ship the NEW signature,
        # so `_parse_vllm_version() >= 0.10.0` wrongly took the old branch and
        # crashed with `unexpected keyword argument 'disable_log_requests'`.
        try:
            _engine_arg_fields = {f.name for f in _dataclass_fields(vllm.AsyncEngineArgs)}
        except TypeError:
            _engine_arg_fields = set()
        if "enable_log_requests" in _engine_arg_fields:
            engine_args = vllm.AsyncEngineArgs(enable_log_requests=False, **kwargs)
        elif "disable_log_requests" in _engine_arg_fields:
            engine_args = vllm.AsyncEngineArgs(disable_log_requests=True, **kwargs)
        else:
            engine_args = vllm.AsyncEngineArgs(**kwargs)

        # Add Ray Prometheus stat loggers if enabled
        if enable_ray_prometheus_stats:
            ray_loggers = self._create_ray_prometheus_stat_loggers()
            if ray_loggers:
                stat_loggers.extend(ray_loggers)

        # Stagger engine startup to avoid TOCTOU port collisions (EADDRINUSE).
        # vLLM's get_open_port() queries a free port then releases the socket;
        # if multiple engines on the same node call it simultaneously, they can
        # get the same port. A random pre-startup delay desynchronises the
        # within-job case.
        #
        # The retry loop below additionally addresses the *cross-job* race
        # we hit on Jupiter A3 RL chain restarts (job 485102, 2026-05-23):
        # Slurm reaps the prior chain leader on TIMEOUT, allocates the same
        # nodes to the next-in-chain immediately, but kernel socket TIME_WAIT
        # can hold the prior holder's bound port for up to ~60 s. A 1.5-3 s
        # stagger doesn't bridge that gap, so without a retry the new chain
        # head exits 1 in ~20 min with EADDRINUSE and the chain visibly
        # "loses" a restart slot until the next dependency-satisfied slot
        # finally gets a fresh port.
        #
        # 5 attempts with exponential backoff (15→30→60→120→240 s) bridges
        # the TIME_WAIT window cleanly while staying well under the outer
        # wait_for_engine_startup deadline.
        import random
        import time
        from torch.distributed import DistNetworkError

        def _is_port_collision(exc: BaseException) -> bool:
            """True if exc (or any cause in its chain) is an EADDRINUSE port race.

            The within-node port collision can surface two ways:
              - directly as torch DistNetworkError(EADDRINUSE) on the legacy path;
              - WRAPPED by vLLM V1 as ``RuntimeError("Engine core initialization
                failed. ...")`` because the bind happens in the EngineCore child
                process, where the DistNetworkError is logged but the parent only
                sees the generic wrapper. Match both: the message OR the wrapper.
            """
            seen = set()
            cur: BaseException | None = exc
            while cur is not None and id(cur) not in seen:
                seen.add(id(cur))
                msg = str(cur).lower()
                if "eaddrinuse" in msg or "address already in use" in msg or "already in use" in msg:
                    return True
                if isinstance(cur, DistNetworkError):
                    return True
                if isinstance(cur, RuntimeError) and "engine core initialization failed" in msg:
                    # V1 EngineCore child died during distributed init — the
                    # overwhelmingly common transient cause on colocated/host-
                    # network multi-engine starts is a TCPStore port collision.
                    return True
                cur = cur.__cause__ or cur.__context__
            return False

        _MAX_INIT_ATTEMPTS = 5
        _BACKOFF_BASE_SEC = 15.0
        engine = None
        for _attempt in range(_MAX_INIT_ATTEMPTS):
            _stagger = random.uniform(1.5, 3.0)
            logger.info(
                f"Engine startup stagger: sleeping {_stagger:.2f}s "
                f"(attempt {_attempt + 1}/{_MAX_INIT_ATTEMPTS}) to avoid port collisions"
            )
            time.sleep(_stagger)
            try:
                engine = vllm.AsyncLLMEngine.from_engine_args(engine_args, stat_loggers=stat_loggers)
                break
            except (DistNetworkError, RuntimeError) as e:
                if not _is_port_collision(e):
                    raise
                if _attempt == _MAX_INIT_ATTEMPTS - 1:
                    logger.error(f"Engine init still hit EADDRINUSE after {_MAX_INIT_ATTEMPTS} attempts; giving up")
                    raise
                _backoff = _BACKOFF_BASE_SEC * (2**_attempt)
                logger.warning(
                    f"Engine init hit a port collision (EADDRINUSE / engine-core init) on attempt "
                    f"{_attempt + 1}/{_MAX_INIT_ATTEMPTS}; retrying in {_backoff:.0f}s: {str(e).splitlines()[0]}"
                )
                time.sleep(_backoff)
        assert engine is not None  # loop either breaks with engine set or raises

        # Adapted from https://github.com/volcengine/verl/blob/e90f18c40aa639cd25092b78a5ff7e2d2508c088/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L327
        model_config = engine.model_config
        model_path = kwargs.get("model")
        # Allow overriding the served model name (similar to vLLM's --served-model-name flag).
        # Useful for Harbor/LiteLLM compatibility where model names must have exactly one '/'.
        # See https://github.com/NovaSky-AI/SkyRL/pull/238#discussion_r2326561295
        served_model_name = kwargs.get("served_model_name")
        model_name = served_model_name if served_model_name else model_path

        base_model_paths = [BaseModelPath(name=model_name, model_path=model_path)]

        # vLLM API compatibility via try/except:
        # - vLLM >= 0.13: model_config removed (obtained internally from engine_client)
        # - vLLM < 0.13: model_config is required as a parameter
        # Try newer API first, fall back to older API if TypeError
        try:
            models = OpenAIServingModels(
                engine_client=engine,
                base_model_paths=base_model_paths,
            )
        except TypeError:
            logger.info(f"vLLM {vllm.__version__}: using legacy API with model_config")
            models = OpenAIServingModels(
                engine_client=engine,
                model_config=model_config,
                base_model_paths=base_model_paths,
            )

        # TODO(Charlie): adding custom chat template for chat completion. Hacky!
        if custom_chat_template_path:
            with open(custom_chat_template_path, "r") as f:
                custom_chat_template_content = f.read()
            logger.info(
                f"Initializing OpenAIServingChat with custom_chat_template read from: {custom_chat_template_path}"
            )
        else:
            custom_chat_template_content = None

        # vLLM >= 0.20.2rc0 moved chat-template / tool-parsing into a separate
        # ``OpenAIServingRender`` object that both OpenAIServingChat and
        # OpenAIServingCompletion now take as a REQUIRED keyword-only
        # ``openai_serving_render`` arg (and dropped ``model_config``). Build it
        # lazily here; ``None`` on older vLLM where the class doesn't exist, in
        # which case the legacy try/except branches below are taken (byte-
        # identical to the prior behavior on vLLM 0.16 / <0.20.2).
        #
        # In vLLM >= 0.20.2rc0 the tool-calling config (``enable_auto_tools``,
        # ``tool_parser``) lives on the RENDER object, not on ``OpenAIServingChat``.
        # Pop them from ``openai_kwargs`` here and pass to the render constructor.
        # On the legacy path (no render API), restore them so ``OpenAIServingChat``
        # receives them as before.
        enable_auto_tools = openai_kwargs.pop("enable_auto_tools", False)
        tool_parser = openai_kwargs.pop("tool_parser", None)

        openai_serving_render = None
        try:
            from vllm.entrypoints.serve.render.serving import OpenAIServingRender

            openai_serving_render = OpenAIServingRender(
                model_config=model_config,
                renderer=engine.renderer,
                model_registry=models.registry,
                request_logger=None,
                chat_template=custom_chat_template_content,
                chat_template_content_format="auto",
                enable_auto_tools=enable_auto_tools,
                tool_parser=tool_parser,
            )
        except ImportError:
            openai_serving_render = None
            # Legacy path: OpenAIServingChat owns the tool-calling kwargs
            openai_kwargs["enable_auto_tools"] = enable_auto_tools
            openai_kwargs["tool_parser"] = tool_parser

        # Try the vLLM >= 0.20.2rc0 render API first, then newer (>=0.13, no
        # model_config), then legacy (<0.13, with model_config).
        if openai_serving_render is not None:
            # ``enable_auto_tools``/``tool_parser`` were popped from ``openai_kwargs``
            # above and passed to the RENDER object, but the render API's
            # OpenAIServingChat STILL gates tool-call parsing on its OWN
            # ``self.enable_auto_tools``/``self.tool_parser`` (see
            # ``_should_stream_with_auto_tool_parsing``). Without them here they default
            # to False/None, so an opencode/agentic request with tools has its
            # well-formed ``<tool_call>`` output returned as plain CONTENT (never parsed
            # into ``tool_calls``) -> the agent executes nothing (tool_use=0). Pass them
            # to OpenAIServingChat too so the auto-tool path actually engages.
            self.openai_serving_chat = OpenAIServingChat(
                engine_client=engine,
                models=models,
                response_role="assistant",
                openai_serving_render=openai_serving_render,
                request_logger=None,
                chat_template=custom_chat_template_content,
                chat_template_content_format="auto",
                enable_auto_tools=enable_auto_tools,
                tool_parser=tool_parser,
                **openai_kwargs,
            )
        else:
            try:
                self.openai_serving_chat = OpenAIServingChat(
                    engine_client=engine,
                    models=models,
                    response_role="assistant",
                    request_logger=None,
                    chat_template=custom_chat_template_content,
                    chat_template_content_format="auto",
                    **openai_kwargs,
                )
            except TypeError:
                self.openai_serving_chat = OpenAIServingChat(
                    engine_client=engine,
                    model_config=model_config,
                    models=models,
                    response_role="assistant",
                    request_logger=None,
                    chat_template=custom_chat_template_content,
                    chat_template_content_format="auto",
                    **openai_kwargs,
                )

        # TODO(Charlie): revisit kwargs `return_tokens_as_token_ids`,
        # `enable_prompt_tokens_details`, `enable_force_include_usage`.
        # Same three-way API selection as OpenAIServingChat above.
        if openai_serving_render is not None:
            self.openai_serving_completion = OpenAIServingCompletion(
                engine_client=engine,
                models=models,
                openai_serving_render=openai_serving_render,
                request_logger=None,
            )
        else:
            try:
                self.openai_serving_completion = OpenAIServingCompletion(
                    engine_client=engine,
                    models=models,
                    request_logger=None,
                )
            except TypeError:
                self.openai_serving_completion = OpenAIServingCompletion(
                    engine_client=engine,
                    model_config=model_config,
                    models=models,
                    request_logger=None,
                )
        return engine

    def _create_ray_prometheus_stat_loggers(self):
        """Create Ray Prometheus stat loggers for vLLM metrics.

        Returns stat_loggers in the format expected by vLLM's from_engine_args().
        For vLLM v1 (0.9.0+), this returns a list of StatLoggerFactory callables.
        For older versions where the v1 API is not available, this returns `None`.

        See: https://docs.vllm.ai/en/latest/api/vllm/v1/metrics/ray_wrappers/
        """
        try:
            # Try vLLM v1 API first (0.9.0+)
            from vllm.v1.metrics.ray_wrappers import RayPrometheusStatLogger

            logger.info("Enabling RayPrometheusStatLogger for vLLM inference engine metrics")
            # For v1, stat_loggers is a list of factory callables
            return [RayPrometheusStatLogger]
        except ImportError:
            logger.warning(
                "RayPrometheusStatLogger not available in this vLLM version. "
                "For Ray-integrated metrics, upgrade to vLLM >= 0.9.0. "
                "Stat logging will be disabled."
            )
            return None

    async def _load_lora_from_disk(self, lora_path: str):
        """Load LoRA adapters from disk using vLLM's native add_lora method."""
        lora_id = int(time.time_ns() % 0x7FFFFFFF)
        lora_request = LoRARequest(lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path=lora_path)
        result = await self.llm.add_lora(lora_request)
        return result

    async def _collect_outputs(self, prompt_token_ids, request_id: str, sampling_params: SamplingParams):
        """Collect outputs for a single prompt."""
        # Check if LoRA is enabled and create LoRA request
        final_output = None
        lora_request = None

        if self._is_lora:
            lora_int_ids = list(await self.llm.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                # dummy_lora_path for placeholder (actual loading done in add_lora())
                lora_request = LoRARequest(
                    lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/dummy_lora_path"
                )

        async for request_output in self.llm.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        ):
            final_output = request_output

        return final_output

    async def _collect_admitted_output(self, queue):
        """Collect one request that was admitted before scheduler resume."""
        final_output = None
        finished = False
        while not finished:
            request_output = queue.get_nowait() or await queue.get()
            if request_output is STREAM_FINISHED:
                break
            final_output = request_output
            finished = request_output.finished
        return final_output

    async def _admit_batch(self, prompt_token_ids, sampling_params: SamplingParams, request_ids: list[str]):
        """Admit one logical batch before vLLM can execute its first step."""
        lora_request = None
        if self._is_lora:
            lora_int_ids = list(await self.llm.list_loras())
            if lora_int_ids:
                lora_int_id = lora_int_ids[0]
                lora_request = LoRARequest(
                    lora_name=f"{lora_int_id}",
                    lora_int_id=lora_int_id,
                    lora_path="/dummy_lora_path",
                )
        queues = []
        pause_scheduler = len(prompt_token_ids) > 1
        async with self._batch_admission_lock:
            if pause_scheduler:
                await self.llm.pause_generation(mode="keep", clear_cache=False)
            try:
                for prompt, request_id in zip(prompt_token_ids, request_ids, strict=True):
                    queues.append(
                        await self.llm.add_request(
                            request_id=request_id,
                            prompt=TokensPrompt(prompt_token_ids=prompt),
                            params=sampling_params,
                            lora_request=lora_request,
                        )
                    )
            finally:
                if pause_scheduler:
                    await self.llm.resume_generation()
        return queues

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        """Generate responses using vLLM's async engine.

        v2 (2026-05-26): wrapped in try/except to explicitly abort all
        sibling in-flight vLLM requests when any task in this batch raises
        (typical case: vLLM serving_chat ValueError on 32k-token validation).
        Without this, the failed task unwinds Python but its sibling tasks'
        Ray ObjectRefs leak into the entrypoint actor's distributed-refcount
        state — that's the `reference_count.cc:1619` SIGABRT pattern that's
        been killing the v3 maxgn09_hint chain links one after another even
        after the harbor rollback_on_exception hook landed (the hook fires
        AFTER the trial, but the ObjectRefs leak DURING the generate batch).
        See agent_logs/2026-05-25_v6a-agrs_507771_moe_combine_stack_pinned.md
        and project_ray_workercrashed_harbor_rollback.md for the chain of
        evidence.
        """
        prompt_token_ids, sampling_params = self._preprocess_prompts(input_batch)

        request_ids = [str(uuid4().hex) for _ in prompt_token_ids]
        tasks = []
        try:
            queues = await self._admit_batch(prompt_token_ids, sampling_params, request_ids)
            tasks = [asyncio.create_task(self._collect_admitted_output(queue)) for queue in queues]
            outputs = await asyncio.gather(*tasks)
        except BaseException as e:
            # Cancel any sibling asyncio tasks still in flight.
            for t in tasks:
                if not t.done():
                    t.cancel()
            # Abort their vLLM-side request state so Ray releases the
            # ObjectRefs and the entrypoint's distributed-refcount table
            # doesn't accumulate orphan entries. We tolerate failures of
            # the abort itself — the goal is best-effort cleanup, not a
            # second hard exception.
            try:
                engine = self._get_engine()
                await engine.abort(request_ids)
            except Exception as abort_exc:
                logger.warning(
                    "generate() failed with %r and vllm engine.abort cleanup "
                    "also failed with %r — Ray ObjectRefs may leak",
                    e,
                    abort_exc,
                )
            raise

        return self._postprocess_outputs(outputs)

    async def wake_up(self, *args: Any, **kwargs: Any):
        await self.llm.wake_up(tags=kwargs.get("tags", None))

    async def sleep(self, *args: Any, **kwargs: Any):
        engine = self._get_engine()
        output_processor = engine.output_processor
        # make sure that the engine is alive
        engine.engine_core.ensure_alive()
        if output_processor.has_unfinished_requests():
            logger.warning(
                "Calling sleep() with unfinished requests in vLLM engine. This is unexpected since all "
                "generation should be done before sleep() is called. Check for potential failures or "
                "dangling requests in your Generator/Env. Aborting all unfinished requests."
            )
            unfinished_request_ids = list(output_processor.request_states.keys())
            await engine.abort(unfinished_request_ids)

        # TODO(team): remove once vllm fixes this
        # otherwise waking it up will output gibberish: https://github.com/vllm-project/vllm/issues/17103
        await self.reset_prefix_cache()
        level = 1 if self._is_lora else kwargs.get("level", 2)
        await self.llm.sleep(level=level)

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        engine = self._get_engine()
        return await engine.collective_rpc(
            "init_weight_update_communicator",
            args=(master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing),
        )

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        if "names" not in request:
            raise ValueError(f"Expected update weight request with 'names' entry, got keys: {request.keys()}")

        if not len(request["names"]):
            raise ValueError("Update weight request should have atleast one entry in 'names'")

        # Check for LoRA disk loading request
        if self._is_lora_disk_loading_request(request):
            lora_path = request["extras"][0]["lora_disk_path"]
            return await self._load_lora_from_disk(lora_path)

        # Use the weight loader to coordinate weight transfer
        return await self._weight_loader.load_weights(request)

    async def begin_weight_update(self):
        """Signal engines to start accumulating weights for batched loading."""
        engine = self._get_engine()
        return await engine.collective_rpc("begin_weight_update")

    async def end_weight_update(self):
        """Flush accumulated weights via model.load_weights()."""
        engine = self._get_engine()
        return await engine.collective_rpc("end_weight_update")

    async def read_engine_weights(
        self,
        hf_names,
        dump_inventory: bool = False,
    ):
        """Read engine-side weights back
        under the trainer's HF parameter names, gathered across all TP/EP workers.

        Returns ``List[Dict]`` (one dict per worker rank), each as produced by
        ``WorkerWrap.read_named_weights``.
        """
        engine = self._get_engine()
        return await engine.collective_rpc("read_named_weights", args=(list(hf_names), dump_inventory))

    async def fingerprint_engine_weights(self, hf_names, expected_shapes):
        """Return compact in-actor fingerprints for engine weights."""
        engine = self._get_engine()
        return await engine.collective_rpc(
            "fingerprint_named_weights",
            args=(list(hf_names), dict(expected_shapes)),
        )

    async def begin_head_input_capture(self, selected_token: int):
        """Start a bounded, test-only capture in each live engine worker."""
        engine = self._get_engine()
        return await engine.collective_rpc("begin_head_input_capture", args=(int(selected_token),))

    async def end_head_input_capture(self):
        """Stop a bounded, test-only capture and return its compact payload."""
        engine = self._get_engine()
        return await engine.collective_rpc("end_head_input_capture")

    async def read_engine_expert_slots_raw(self, layer_idx: int):
        """TEST-ONLY (D1/D2 diag): per-engine-worker RAW FusedMoE local-slot weights +
        the engine's own expert_map for ``layer_idx``. Returns List[Dict] (one per
        engine worker rank). See ``WorkerWrap.read_expert_slots_raw``."""
        engine = self._get_engine()
        return await engine.collective_rpc("read_expert_slots_raw", args=(int(layer_idx),))

    async def report_engine_hosts(self):
        """TEST-ONLY (disaggregation proof): hostname of every engine TP/EP worker."""
        engine = self._get_engine()
        return await engine.collective_rpc("report_host")

    async def report_runtime_installation(self, expected_vllm_engine_sha256: str):
        """Return and verify MarinSkyRL provenance inside each engine worker."""
        engine = self._get_engine()
        return await engine.collective_rpc(
            "report_runtime_installation",
            args=(expected_vllm_engine_sha256,),
        )

    async def begin_weight_reload(self):
        """#1685 fix: open the layerwise-reload bracket on every engine worker so the
        multi-chunk RL sync defers processing; finalize re-runs process_weights_after_loading
        (re-applies the FlashInfer-CUTLASS w13 swap). See WorkerWrap.skyrl_begin_weight_reload."""
        engine = self._get_engine()
        return await engine.collective_rpc("skyrl_begin_weight_reload")

    async def finish_weight_reload(self):
        """#1685 fix: close the layerwise-reload bracket -> finalize_layerwise_reload ->
        process_weights_after_loading (swap_w13_to_w31) re-applied EXACTLY once."""
        engine = self._get_engine()
        return await engine.collective_rpc("skyrl_finish_weight_reload")

    async def teardown(self):
        await self._destroy_weights_update_group()

    async def reset_prefix_cache(self):
        engine = self._get_engine()
        await engine.reset_prefix_cache()

    async def _destroy_weights_update_group(self):
        engine = self._get_engine()
        return await engine.collective_rpc("destroy_weights_update_group")

    # ----------------------------------------
    # Methods for handling OpenAI API requests
    # ----------------------------------------

    async def _handle_openai_request(self, request_payload: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """Handle OpenAI API request."""
        assert endpoint in ["/chat/completions", "/completions"]

        body = request_payload.get("json", {})
        headers = request_payload.get("headers", {})

        # Apply configured sampling params from generator config.
        # Harbor requests may include their own sampling params; we override
        # with the SkyRL generator config so rollout exploration is consistent.
        sp = getattr(self, "_openai_sampling_params", {})
        body.update(
            {
                "temperature": sp.get("temperature", 1.0),
                "top_p": sp.get("top_p", 1.0),
                "top_k": sp.get("top_k", -1),
                "min_p": sp.get("min_p", 0.0),
            }
        )

        # 1. Build request
        try:
            if endpoint == "/chat/completions":
                request = ChatCompletionRequest(**body)
            else:
                request = CompletionRequest(**body)
            assert request.stream is False, "Streaming is not supported in SkyRL yet, please set stream to False."
        except Exception as e:
            return _build_error_response(str(e), HTTPStatus.BAD_REQUEST.phrase, HTTPStatus.BAD_REQUEST.value)

        # 2. Call vllm engine
        try:
            # Create a minimal request-like object with attributes used by vLLM
            minimal_request = _MinimalRequest(headers)
            if endpoint == "/chat/completions":
                generator = await self.openai_serving_chat.create_chat_completion(request, minimal_request)
                assert isinstance(generator, (ChatCompletionResponse, ErrorResponse))
            else:
                generator = await self.openai_serving_completion.create_completion(request, minimal_request)
                assert isinstance(generator, (CompletionResponse, ErrorResponse))
            return generator.model_dump()

        except Exception as e:
            # Handle it here so we can surface the error from a ray worker.
            #
            # Input-overflow (VLLMValidationError raised at serving.py during
            # input validation, e.g. "You passed 32769 input tokens ... context
            # length is only 32768") is a *deterministic* client error: retrying
            # the identical over-budget prompt can never succeed. Classify it as
            # HTTP 400 (BAD_REQUEST) so that downstream LiteLLM/Harbor map it to
            # a ContextWindowExceededError (non-retryable) instead of treating a
            # generic 500 as a transient server error and retrying. Retrying the
            # doomed request across many concurrent trials is what exhausts the
            # entrypoint actor's file descriptors and aborts its uvloop event
            # loop (uv__epoll_ctl_prep SIGABRT) -> ray.WorkerCrashedError.
            # See project notes: nemotron-junit a3 #11 (chain 521442-448).
            is_input_overflow = False
            try:
                from vllm.exceptions import VLLMValidationError

                if isinstance(e, VLLMValidationError):
                    param = getattr(e, "parameter", None)
                    is_input_overflow = param == "input_tokens" or "input tokens" in str(e)
            except ImportError:
                # Older vLLM without VLLMValidationError: fall back to message match.
                is_input_overflow = "input tokens" in str(e) and "context length" in str(e)

            status = HTTPStatus.BAD_REQUEST if is_input_overflow else HTTPStatus.INTERNAL_SERVER_ERROR
            if is_input_overflow:
                logger.warning("Input-overflow rejected by vLLM serving (returning 400, non-retryable): %s", e)

            return _build_error_response(str(e), status.phrase, status.value)

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible HTTP endpoint for handling `/chat/completions` in Python vLLM engine.

        Accepts a JSON-serializable payload: {"json": <request-body>, "headers": <headers-dict>}.
        Constructs a minimal request-like object for vLLM's openai_serving_chat.
        Returns a plain dict, either a ChatCompletionResponse or an ErrorResponse, both defined
        in vllm.entrypoints.openai.protocol.
        """
        return await self._handle_openai_request(request_payload, endpoint="/chat/completions")

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible HTTP endpoint for handling `/completions` in Python vLLM engine.

        Accepts a JSON-serializable payload: {"json": <request-body>, "headers": <headers-dict>}.
        Constructs a minimal request-like object for vLLM's openai_serving_completion.
        Returns a plain dict, either a CompletionResponse or an ErrorResponse, both defined
        in vllm.entrypoints.openai.protocol.
        """
        return await self._handle_openai_request(request_payload, endpoint="/completions")

    @ray.method(num_returns="streaming")
    async def chat_completion_stream(self, request_payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Streaming chat completion yielding SSE-formatted ``data: …`` strings.

        Delegates to vLLM's ``openai_serving_chat.create_chat_completion`` with
        ``stream=True`` and surfaces each SSE chunk through Ray's streaming-
        generator transport (``num_returns="streaming"``).  Each ``yield``
        produces one ``ObjectRef`` that the caller resolves to the SSE string.

        ``_ensure_token_ids_in_sse_chunk`` is applied per-chunk so that
        ``provider_specific_fields.token_ids`` is present for harbor's literal
        accumulator.
        """
        body = request_payload.get("json", {})
        headers = request_payload.get("headers", {})

        sp = getattr(self, "_openai_sampling_params", {})
        body.update(
            {
                "temperature": sp.get("temperature", 1.0),
                "top_p": sp.get("top_p", 1.0),
                "top_k": sp.get("top_k", -1),
                "min_p": sp.get("min_p", 0.0),
            }
        )
        body["stream"] = True
        body["return_token_ids"] = True  # force vLLM to emit per-chunk token_ids

        try:
            request = ChatCompletionRequest(**body)
            minimal_request = _MinimalRequest(headers)
            result = await self.openai_serving_chat.create_chat_completion(request, minimal_request)

            if isinstance(result, ErrorResponse):
                err = result.model_dump()
                yield f"data: {json.dumps(err)}\n\n"
                yield "data: [DONE]\n\n"
                return

            async for chunk in result:
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8")
                yield ensure_token_ids_in_sse_chunk(chunk)

        except Exception as e:
            is_input_overflow = False
            try:
                from vllm.exceptions import VLLMValidationError

                if isinstance(e, VLLMValidationError):
                    param = getattr(e, "parameter", None)
                    is_input_overflow = param == "input_tokens" or "input tokens" in str(e)
            except ImportError:
                is_input_overflow = "input tokens" in str(e) and "context length" in str(e)

            status = HTTPStatus.BAD_REQUEST if is_input_overflow else HTTPStatus.INTERNAL_SERVER_ERROR
            err = _build_error_response(str(e), status.phrase, status.value)
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"

    async def get_stats(self) -> Dict[str, Any]:
        """Get accumulated vLLM engine statistics for the current step.

        Returns a dict with the following keys:
        - peak_*: Peak values observed during the step
        - median_*: Median values across active samples
        - mean_*: Mean values across active samples
        - num_samples: Total number of stat samples collected
        - num_active_samples: Number of samples with active requests
        - timestamp: Unix timestamp of last sample
        - engine_id: Unique identifier for this engine instance

        Note: Stats are reset after reading to provide fresh stats per training step.

        Used by VLLMStatsCallback to collect and aggregate stats across engines.
        """
        # Reset=True ensures each training step gets fresh stats
        stats = V1LoggingStatLoggerFixed.get_stats_by_engine_id(self._stats_engine_id, reset=True)
        if stats is None:
            # Return empty stats if no data recorded yet
            stats = {
                # Peak values
                "peak_prompt_throughput": 0.0,
                "peak_generation_throughput": 0.0,
                "peak_running_reqs": 0,
                "peak_waiting_reqs": 0,
                "peak_gpu_cache_usage_perc": 0.0,
                "peak_prefix_cache_hit_rate": 0.0,
                # Median values
                "median_prompt_throughput": 0.0,
                "median_generation_throughput": 0.0,
                "median_running_reqs": 0.0,
                "median_waiting_reqs": 0.0,
                "median_gpu_cache_usage_perc": 0.0,
                "median_prefix_cache_hit_rate": 0.0,
                # Mean values
                "mean_prompt_throughput": 0.0,
                "mean_generation_throughput": 0.0,
                # Per-request latency stats
                "latency_prefill_mean": 0.0,
                "latency_prefill_median": 0.0,
                "latency_prefill_p90": 0.0,
                "latency_decode_mean": 0.0,
                "latency_decode_median": 0.0,
                "latency_decode_p90": 0.0,
                "latency_e2e_mean": 0.0,
                "latency_e2e_median": 0.0,
                "latency_e2e_p90": 0.0,
                "latency_queued_mean": 0.0,
                "latency_queued_median": 0.0,
                "latency_queued_p90": 0.0,
                "latency_ttft_mean": 0.0,
                "latency_ttft_median": 0.0,
                "latency_ttft_p90": 0.0,
                "latency_num_finished_requests": 0,
                "total_preempted_reqs": 0,
                # Legacy field names
                "avg_prompt_throughput": 0.0,
                "avg_generation_throughput": 0.0,
                "num_running_reqs": 0,
                "num_waiting_reqs": 0,
                "gpu_cache_usage_perc": 0.0,
                "prefix_cache_hit_rate": 0.0,
                # Metadata
                "num_samples": 0,
                "num_active_samples": 0,
                "timestamp": time.time(),
            }
        stats["engine_id"] = self._stats_engine_id
        return stats

    async def pause_generation(self) -> None:
        """Abort outstanding requests and hold the EngineCore scheduler idle for weight reload."""
        engine = self._get_engine()
        outstanding_requests = len(engine.output_processor.request_states)
        # vLLM's scheduler-level pause is a utility RPC into EngineCore. In abort
        # mode it aborts running/waiting requests, waits for the scheduler to reach
        # its paused state, and clears the KV/prefix cache before returning. Unlike
        # AsyncLLM.abort(), it cannot report success merely because the frontend
        # output_processor already removed the request IDs.
        await engine.pause_generation(mode="abort", clear_cache=True)
        logger.info(f"pause_generation() finished, aborted {outstanding_requests} requests and paused EngineCore")

    async def resume_generation(self) -> None:
        """Release the EngineCore scheduler after the weight reload completes."""
        await self._get_engine().resume_generation()
        logger.info("resume_generation() finished, EngineCore scheduler released")


class _MinimalRequest:
    """
    Minimal request-like object for vLLM's openai_serving_chat and openai_serving_completion.

    We cannot use the original user Request object because it cannot be serialized and hence
    cannot be a ray method argument. Instead we take the original request's headers and
    reconstruct an instance of _MinimalRequest to mimic the FastAPI Request object.

    The fields depend on what vLLM accesses internally.
    """

    def __init__(self, headers):
        self.headers = headers  # Expect a mapping with .get support
        self.state = SimpleNamespace()  # vLLM sets raw_request.state.request_metadata


class VLLMWeightTransferReceiver:
    """Receives weights via broadcast or CUDA IPC for vLLM.

    Handles both transfer strategies based on the request contents.
    Created locally in WorkerWrap with worker-specific state.
    """

    def __init__(self, model_update_group: Any, model_config: Any, device: torch.device) -> None:
        """Initialize the receiver with worker-local state.

        Args:
            model_update_group: Torch process group for weight updates.
            model_config: vLLM model configuration.
            device: CUDA device for this worker.
        """
        self.model_update_group = model_update_group
        self.model_config = model_config
        self.device = device

    def _is_fp32_grug_router_bias(self, name: str, dtype: torch.dtype) -> bool:
        hf_config = getattr(self.model_config, "hf_text_config", None)
        if hf_config is None:
            hf_config = getattr(self.model_config, "hf_config", None)
        return is_grug_router_bias(getattr(hf_config, "model_type", None), name) and dtype == torch.float32

    def receive_weights(self, request: NamedWeightsUpdateRequest) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights and yield (name, tensor) tuples.

        Args:
            request: Weight update request with names, dtypes, shapes, and optionally IPC handles.
        """
        extras = request.get("extras")
        is_ipc = extras and len(extras) > 0 and "ipc_handles" in extras[0]

        if is_ipc:
            yield from self._receive_ipc(request)
        else:
            yield from self._receive_broadcast(request)

    def _receive_broadcast(self, request: NamedWeightsUpdateRequest) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights via torch.distributed.broadcast."""
        _fuse = bool(request.get("packed", False))
        for name, dtype_str, shape in zip(request["names"], request["dtypes"], request["shapes"]):
            dtype = str_to_torch_dtype(dtype_str)
            if not _fuse and not self._is_fp32_grug_router_bias(name, dtype):
                assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
            # Always receive in sender's dtype, load_weights handles conversion
            weight = torch.empty(shape, dtype=dtype, device="cuda")
            torch.distributed.broadcast(weight, 0, group=self.model_update_group)
            yield name, weight

    def _receive_ipc(self, request: NamedWeightsUpdateRequest) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights via CUDA IPC handles."""
        names = request["names"]
        dtypes = request["dtypes"]
        shapes = request["shapes"]
        sizes = request.get("sizes", [])
        ipc_handles = [extra["ipc_handles"] for extra in request["extras"]]
        packed = request.get("packed", False)

        if packed:
            assert len(ipc_handles) == 1, "packed weight update should receive one ipc handle for all tensors"
            assert len(set(dtypes)) == 1, "packed weight update should have all tensors with the same dtype"
            assert str_to_torch_dtype(dtypes[0]) == self.model_config.dtype, (
                f"mismatch dtype: src {dtypes[0]}, dst {self.model_config.dtype}"
            )
            assert len(sizes) == len(names), "sizes must be provided for packed weight update"
            assert all(isinstance(size, int) for size in sizes), "sizes should be a list of integers"

            cuda_device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(cuda_device)
            physical_gpu_id = str(props.uuid)

            handle = ipc_handles[0][physical_gpu_id]
            device_id = self.device.index
            func, args = handle
            list_args = list(args)
            list_args[6] = device_id
            packed_tensor = func(*list_args)

            offset = 0
            for name, shape, size in zip(names, shapes, sizes):
                yield name, packed_tensor[offset : offset + size].view(*shape)
                offset += size
        else:
            cuda_device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(cuda_device)
            physical_gpu_id = str(props.uuid)
            for name, dtype_str, shape, ipc_handle in zip(names, dtypes, shapes, ipc_handles):
                dtype = str_to_torch_dtype(dtype_str)
                if not self._is_fp32_grug_router_bias(name, dtype):
                    assert dtype == self.model_config.dtype, (
                        f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
                    )

                handle = ipc_handle[physical_gpu_id]
                device_id = self.device.index
                func, args = handle
                list_args = list(args)
                list_args[6] = device_id
                weight = func(*list_args)
                yield name, weight


class VLLMWeightLoader(WeightLoader):
    """Loads weights into vLLM engine, managing RPC coordination.

    This loader encapsulates the collective_rpc calls to workers.
    Workers create VLLMWeightTransferReceiver locally for the actual weight transfer.
    """

    def __init__(self, engine: Any, is_async: bool = False) -> None:
        """Initialize the loader.

        Args:
            engine: The vLLM engine (LLM or AsyncLLMEngine).
            is_async: Whether this is for AsyncVLLMInferenceEngine.
        """
        self._engine = engine.engine if hasattr(engine, "engine") else engine
        self._is_async = is_async

    async def load_weights(self, request: NamedWeightsUpdateRequest) -> None:
        """Load weights by coordinating RPC to workers.

        Sends the request to workers via collective_rpc. Workers create
        the receiver locally and use it to receive and load weights.

        Args:
            request: Weight update request containing names, dtypes, shapes,
                    and optionally IPC handles.
        """
        if self._is_async:
            await self._engine.collective_rpc(
                "load_weights",
                args=(request,),
            )
        else:
            await asyncio.to_thread(
                self._engine.collective_rpc,
                "load_weights",
                args=(request,),
            )


VLLMRayActor = ray.remote(VLLMInferenceEngine)
AsyncVLLMRayActor = ray.remote(AsyncVLLMInferenceEngine)
