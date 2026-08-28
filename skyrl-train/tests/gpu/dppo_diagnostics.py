"""Ray worker diagnostics used by the DPPO GPU integration test."""

import hashlib
import importlib
import importlib.metadata
import sys
from pathlib import Path

import ray
import torch
from torch.distributed.tensor import DTensor
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    chunk_gated_delta_rule as transformers_chunk_gated_delta_rule,
    is_fast_path_available,
    torch_chunk_gated_delta_rule,
)

from skyrl_train.utils import str_to_torch_dtype
from skyrl_train.utils.tensor_fingerprint import canonical_tensor_fingerprint
from skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase, FSDPWeightExtractor


def _tensor_layout(tensor: torch.Tensor) -> dict:
    return {
        "contiguous": tensor.is_contiguous(),
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "storage_nbytes": tensor.untyped_storage().nbytes(),
        "storage_offset": tensor.storage_offset(),
        "stride": list(tensor.stride()),
    }


def _exact_error_summary(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = (actual.float() - expected.float()).abs().reshape(-1)
    mismatch_indices = torch.nonzero(difference, as_tuple=False).reshape(-1)
    first_mismatch = int(mismatch_indices[0].item()) if mismatch_indices.numel() else None
    return {
        "exact": bool(torch.equal(actual, expected)),
        "first_mismatch": first_mismatch,
        "l2": float(torch.linalg.vector_norm(difference).item()),
        "max": float(difference.max().item()),
        "mismatch_count": int(torch.count_nonzero(difference).item()),
        "nonfinite_count": int(
            torch.count_nonzero(~torch.isfinite(actual)).item() + torch.count_nonzero(~torch.isfinite(expected)).item()
        ),
        "p95": float(torch.quantile(difference, 0.95).item()),
        "shape": list(actual.shape),
    }


class DPPOPolicyWorker(FSDPPolicyWorkerBase):
    def fingerprint_broadcast_weights(self, names=None):
        wanted = None if names is None else set(names)
        fingerprints = {}
        generator_dtype = str_to_torch_dtype(self.cfg.generator.model_dtype)
        is_rank0 = torch.distributed.get_rank() == 0
        for chunk in self.weight_extractor.extract_weights(generator_dtype):
            for name, tensor in zip(chunk.names, chunk.tensors):
                if is_rank0 and (wanted is None or name in wanted):
                    fingerprints[name] = canonical_tensor_fingerprint(tensor)
        return fingerprints

    def perturb_weight(self, name: str, delta: float):
        matches = []
        for parameter_name, parameter in self.model.model.named_parameters():
            normalized_name = parameter_name.replace(FSDPWeightExtractor._FSDP_SEG, ".")
            if normalized_name == name:
                matches.append((parameter_name, parameter))
        if len(matches) != 1:
            raise KeyError(f"Expected one live parameter named {name!r}, found {[entry[0] for entry in matches]}")

        parameter_name, parameter = matches[0]
        local_parameter = parameter.to_local() if isinstance(parameter, DTensor) else parameter
        flat_parameter = local_parameter.reshape(-1)
        if flat_parameter.numel() == 0:
            return {
                "name": parameter_name,
                "changed": False,
                "before": None,
                "after": None,
                "rank": torch.distributed.get_rank(),
            }

        with torch.no_grad():
            before = flat_parameter[0].float().item()
            flat_parameter[0].copy_((flat_parameter[0].float() + delta).to(dtype=flat_parameter.dtype))
            after = flat_parameter[0].float().item()
        return {
            "name": parameter_name,
            "changed": before != after,
            "before": before,
            "after": after,
            "rank": torch.distributed.get_rank(),
        }

    def score_next_token(self, prompt_token_ids, selected_token: int, capture_head_input: bool = False):
        if not is_fast_path_available:
            raise RuntimeError("Qwen3.5 learner parity requires the flash-linear-attention and causal-conv1d fast path")
        device = torch.cuda.current_device()
        input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        model = self.model.model
        was_training = model.training
        captured_head_inputs = []
        layer_captures = []
        hooks = []
        learner_fla_restore = None
        finalize_fla_capture = None
        if capture_head_input:

            def tensor_payload(tensor):
                if tensor.ndim == 3:
                    if tensor.shape[0] != 1:
                        raise RuntimeError(f"Expected learner batch size 1, got {tensor.shape}")
                    tensor = tensor[0, -1]
                elif tensor.ndim == 2:
                    tensor = tensor[-1]
                else:
                    raise RuntimeError(f"Expected learner hidden states with rank 2 or 3, got {tensor.shape}")
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

            output_embeddings = model.get_output_embeddings()

            def capture_output_embedding_input(_module, args):
                hidden_states = args[0]
                if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
                    raise RuntimeError(f"Expected learner head input [1, sequence, hidden], got {hidden_states.shape}")
                captured_head_inputs.append(
                    {
                        "head_input": hidden_states[0, -1].detach().float().cpu(),
                        "head_input_dtype": str(hidden_states.dtype),
                        "head_input_shape": list(hidden_states[0, -1].shape),
                        "output_embedding_input_shape": list(hidden_states.shape),
                    }
                )

            hooks.append(output_embeddings.register_forward_pre_hook(capture_output_embedding_input))

            text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
            for layer_index, layer in enumerate(text_model.layers):
                mixer_name = "linear_attn" if layer.layer_type == "linear_attention" else "self_attn"
                mixer = getattr(layer, mixer_name)
                entry = {"layer": layer_index, "mixer": mixer_name}
                layer_captures.append(entry)

                if layer_index == 0 and mixer_name == "linear_attn":
                    projections = {}
                    entry["projections"] = projections
                    entry["mixer_stages"] = {}
                    fla_captures = []
                    entry["fla_core"] = fla_captures

                    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as released_chunk_gated_delta_rule

                    live_chunk_gated_delta_rule = mixer.chunk_gated_delta_rule
                    learner_fla_restore = (mixer, live_chunk_gated_delta_rule)
                    backend_module = importlib.import_module(live_chunk_gated_delta_rule.__module__)
                    backend_source = Path(backend_module.__file__)
                    backend_source_sha256 = hashlib.sha256(backend_source.read_bytes()).hexdigest()

                    def capture_fla_core(
                        q,
                        k,
                        v,
                        *,
                        g,
                        beta,
                        scale=None,
                        initial_state=None,
                        output_final_state=False,
                        use_qk_l2norm_in_kernel=False,
                        cu_seqlens=None,
                        **kwargs,
                    ):
                        if initial_state is not None or output_final_state or cu_seqlens is not None:
                            raise RuntimeError(
                                "Expected an uncached fixed-length learner prefill for the bounded FLA diagnostic"
                            )
                        if kwargs:
                            raise RuntimeError(f"Unexpected learner FLA options: {sorted(kwargs)}")
                        captured_inputs = {
                            name: tensor.detach().clone() if tensor is not None else None
                            for name, tensor in {
                                "q": q,
                                "k": k,
                                "v": v,
                                "g": g,
                                "beta": beta,
                                "initial_state": initial_state,
                            }.items()
                        }
                        live_output, live_final_state = live_chunk_gated_delta_rule(
                            q,
                            k,
                            v,
                            g=g,
                            beta=beta,
                            scale=scale,
                            initial_state=initial_state,
                            output_final_state=output_final_state,
                            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                            cu_seqlens=cu_seqlens,
                            **kwargs,
                        )
                        post_live_inputs = {
                            name: tensor
                            for name, tensor in {
                                "q": q,
                                "k": k,
                                "v": v,
                                "g": g,
                                "beta": beta,
                                "initial_state": initial_state,
                            }.items()
                            if tensor is not None
                        }
                        fla_captures.append(
                            {
                                "captured_inputs": captured_inputs,
                                "grad_enabled": torch.is_grad_enabled(),
                                "live_final_state": (
                                    live_final_state.detach().clone() if live_final_state is not None else None
                                ),
                                "live_output": live_output.detach().clone(),
                                "post_live_input_errors": {
                                    name: _exact_error_summary(tensor, captured_inputs[name])
                                    for name, tensor in post_live_inputs.items()
                                },
                                "options": {
                                    "cu_seqlens": None,
                                    "output_final_state": output_final_state,
                                    "scale": scale,
                                    "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
                                },
                            }
                        )
                        return live_output, live_final_state

                    mixer.chunk_gated_delta_rule = capture_fla_core

                    def finalize_fla_capture(capture):
                        captured_inputs = capture["captured_inputs"]
                        options = capture["options"]
                        replay_output, replay_final_state = released_chunk_gated_delta_rule(
                            captured_inputs["q"].clone(),
                            captured_inputs["k"].clone(),
                            captured_inputs["v"].clone(),
                            g=captured_inputs["g"].clone(),
                            beta=captured_inputs["beta"].clone(),
                            scale=options["scale"],
                            initial_state=None,
                            output_final_state=options["output_final_state"],
                            use_qk_l2norm_in_kernel=options["use_qk_l2norm_in_kernel"],
                            cu_seqlens=None,
                        )
                        zero_state = torch.zeros(
                            captured_inputs["q"].shape[0],
                            captured_inputs["q"].shape[2],
                            captured_inputs["k"].shape[-1],
                            captured_inputs["v"].shape[-1],
                            dtype=torch.float32,
                            device=captured_inputs["q"].device,
                        )
                        zero_state_output, _ = released_chunk_gated_delta_rule(
                            captured_inputs["q"].clone(),
                            captured_inputs["k"].clone(),
                            captured_inputs["v"].clone(),
                            g=captured_inputs["g"].clone(),
                            beta=captured_inputs["beta"].clone(),
                            scale=options["scale"],
                            initial_state=zero_state,
                            output_final_state=False,
                            use_qk_l2norm_in_kernel=options["use_qk_l2norm_in_kernel"],
                            cu_seqlens=None,
                        )
                        live_output = capture["live_output"]
                        live_final_state = capture["live_final_state"]
                        return {
                            "backend": {
                                "live_callable_is_released": (
                                    live_chunk_gated_delta_rule is released_chunk_gated_delta_rule
                                ),
                                "live_callable_is_torch_fallback": (
                                    live_chunk_gated_delta_rule is torch_chunk_gated_delta_rule
                                ),
                                "live_callable_is_transformers_global": (
                                    live_chunk_gated_delta_rule is transformers_chunk_gated_delta_rule
                                ),
                                "method": getattr(
                                    live_chunk_gated_delta_rule,
                                    "__qualname__",
                                    type(live_chunk_gated_delta_rule).__qualname__,
                                ),
                                "method_module": getattr(
                                    live_chunk_gated_delta_rule,
                                    "__module__",
                                    type(live_chunk_gated_delta_rule).__module__,
                                ),
                                "package_version": importlib.metadata.version("flash-linear-attention"),
                                "source_sha256": backend_source_sha256,
                            },
                            "context": {
                                "configured_use_sample_packing": bool(self.cfg.trainer.use_sample_packing),
                                "grad_enabled": capture["grad_enabled"],
                                "scored_batch_size": input_ids.shape[0],
                                "scored_sequence_count": input_ids.shape[0],
                            },
                            "inputs": {
                                name: (
                                    {
                                        "fingerprint": canonical_tensor_fingerprint(tensor),
                                        "layout": _tensor_layout(tensor),
                                    }
                                    if tensor is not None
                                    else None
                                )
                                for name, tensor in captured_inputs.items()
                            },
                            "zero_initial_state": {
                                "fingerprint": canonical_tensor_fingerprint(zero_state),
                                "layout": _tensor_layout(zero_state),
                            },
                            "live": {
                                "final_state": (
                                    {
                                        "fingerprint": canonical_tensor_fingerprint(live_final_state),
                                        "layout": _tensor_layout(live_final_state),
                                    }
                                    if live_final_state is not None
                                    else None
                                ),
                                "output": {
                                    "fingerprint": canonical_tensor_fingerprint(live_output),
                                    "layout": _tensor_layout(live_output),
                                },
                            },
                            "live_vs_released_replay": {
                                "final_state": (
                                    _exact_error_summary(live_final_state, replay_final_state)
                                    if live_final_state is not None and replay_final_state is not None
                                    else None
                                ),
                                "output": _exact_error_summary(live_output, replay_output),
                            },
                            "inputs_unchanged_by_live_call": capture["post_live_input_errors"],
                            "none_vs_zero_initial_state": _exact_error_summary(live_output, zero_state_output),
                            "options": options,
                        }

                    def capture_projection(_module, _args, output, *, destination, key):
                        destination.setdefault(key, []).append(tensor_payload(output))

                    for projection_name, key in (
                        ("in_proj_qkv", "qkv"),
                        ("in_proj_z", "z"),
                        ("in_proj_b", "b"),
                        ("in_proj_a", "a"),
                    ):
                        projection = getattr(mixer, projection_name)
                        hooks.append(
                            projection.register_forward_hook(
                                lambda module, args, output, destination=projections, key=key: capture_projection(
                                    module,
                                    args,
                                    output,
                                    destination=destination,
                                    key=key,
                                )
                            )
                        )

                def capture_input(_module, args, kwargs, *, destination=entry, key="mixer_input"):
                    hidden_states = kwargs.get("hidden_states")
                    if hidden_states is None:
                        hidden_states = args[0]
                    destination.setdefault(key, []).append(tensor_payload(hidden_states))

                def capture_output(_module, args, kwargs, output, *, destination=entry, key="mixer_output"):
                    del args
                    hidden_states = kwargs.get("output")
                    if hidden_states is None:
                        hidden_states = output[0] if isinstance(output, tuple) else output
                    destination.setdefault(key, []).append(tensor_payload(hidden_states))

                if layer_index == 0 and mixer_name == "linear_attn":
                    stages = entry["mixer_stages"]
                    num_heads = mixer.num_v_heads

                    def capture_norm_input(_module, args, kwargs, *, destination=stages, heads=num_heads):
                        core_attn_out = kwargs.get("x")
                        if core_attn_out is None:
                            core_attn_out = args[0]
                        gate = kwargs.get("gate")
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
                        heads=num_heads,
                    ):
                        destination.setdefault("norm_output", []).append(token_heads_payload(output, heads))

                    hooks.append(mixer.norm.register_forward_pre_hook(capture_norm_input, with_kwargs=True))
                    hooks.append(mixer.norm.register_forward_hook(capture_norm_output, with_kwargs=True))
                    hooks.append(
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
                    hooks.append(
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

                hooks.append(mixer.register_forward_pre_hook(capture_input, with_kwargs=True))
                hooks.append(mixer.register_forward_hook(capture_output, with_kwargs=True))
                hooks.append(
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
                hooks.append(
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
        model.eval()
        try:
            with torch.no_grad():
                model_output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                if learner_fla_restore is not None:
                    fla_mixer, live_chunk_gated_delta_rule = learner_fla_restore
                    fla_mixer.chunk_gated_delta_rule = live_chunk_gated_delta_rule
                    learner_fla_restore = None
                if capture_head_input:
                    if finalize_fla_capture is None:
                        raise RuntimeError("Missing learner FLA capture finalizer")
                    for layer_entry in layer_captures:
                        fla_core = layer_entry.get("fla_core")
                        if fla_core is None:
                            continue
                        if len(fla_core) != 1:
                            raise RuntimeError(
                                f"Expected one learner FLA core capture in layer "
                                f"{layer_entry['layer']}, got {len(fla_core)}"
                            )
                        fla_core[0] = finalize_fla_capture(fla_core[0])
                logits = model_output.logits[0, -1]
                if logits.dtype != torch.float32:
                    raise TypeError(f"Expected learner FP32 final-token logits, got {logits.dtype}")
                logsumexp = logits.logsumexp(dim=-1)
                selected_logit = logits[selected_token]
                top_logits, top_tokens = logits.topk(k=2)
                top_logprobs = top_logits - logsumexp
                result = {
                    "rank": torch.distributed.get_rank(),
                    "top1": int(logits.argmax(dim=-1).item()),
                    "top_candidates": [
                        {
                            "token": int(token.item()),
                            "logit": float(logit.item()),
                            "logprob": float(logprob.item()),
                        }
                        for token, logit, logprob in zip(top_tokens, top_logits, top_logprobs, strict=True)
                    ],
                    "top1_margin": float((top_logits[0] - top_logits[1]).item()),
                    "selected_token": selected_token,
                    "selected_logit": float(selected_logit.item()),
                    "logsumexp": float(logsumexp.item()),
                    "selected_logprob": float((selected_logit - logsumexp).item()),
                    "logits_dtype": str(logits.dtype),
                }
                if capture_head_input:
                    if len(captured_head_inputs) != 1:
                        raise RuntimeError(
                            f"Expected one learner output-head invocation, captured {len(captured_head_inputs)}"
                        )
                    head_input = captured_head_inputs[0]
                    result.update(head_input)
                    result["head_input"] = head_input["head_input"].tolist()
                    for layer_entry in layer_captures:
                        for key in ("mixer_input", "mixer_output", "mlp_input", "mlp_output"):
                            values = layer_entry[key]
                            if len(values) != 1:
                                raise RuntimeError(
                                    f"Expected one learner {key} capture in layer "
                                    f"{layer_entry['layer']}, got {len(values)}"
                                )
                            layer_entry[key] = values[0]
                        for key, values in layer_entry.get("projections", {}).items():
                            if len(values) != 1:
                                raise RuntimeError(
                                    f"Expected one learner {key} projection capture in layer "
                                    f"{layer_entry['layer']}, got {len(values)}"
                                )
                            layer_entry["projections"][key] = values[0]
                        for key, values in layer_entry.get("mixer_stages", {}).items():
                            if len(values) != 1:
                                raise RuntimeError(
                                    f"Expected one learner {key} stage capture in layer "
                                    f"{layer_entry['layer']}, got {len(values)}"
                                )
                            layer_entry["mixer_stages"][key] = values[0]
                        fla_core = layer_entry.get("fla_core")
                        if fla_core is not None:
                            if len(fla_core) != 1:
                                raise RuntimeError(
                                    f"Expected one learner FLA core capture in layer "
                                    f"{layer_entry['layer']}, got {len(fla_core)}"
                                )
                            layer_entry["fla_core"] = fla_core[0]
                    result["layer_trace"] = layer_captures
        finally:
            if learner_fla_restore is not None:
                mixer, live_chunk_gated_delta_rule = learner_fla_restore
                mixer.chunk_gated_delta_rule = live_chunk_gated_delta_rule
            for hook in hooks:
                hook.remove()
            if was_training:
                model.train()
        return result


# Pytest adds ``skyrl-train`` to the driver's import path, but Ray workers do
# not inherit that test-only path. Serialize this helper module by value so the
# diagnostic actor does not require ``tests`` to be installed on every worker.
ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
PolicyWorker = ray.remote(num_gpus=1)(DPPOPolicyWorker)
