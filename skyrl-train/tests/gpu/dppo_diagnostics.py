"""Ray worker diagnostics used by the DPPO GPU integration test."""

import hashlib
import importlib
import importlib.metadata
import sys
from functools import partial
from pathlib import Path

import ray
import torch
from fla.modules.l2norm import l2norm_fwd
from torch.distributed.tensor import DTensor
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    apply_rotary_pos_emb,
    chunk_gated_delta_rule as transformers_chunk_gated_delta_rule,
    is_fast_path_available,
    repeat_kv,
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


def _token_fingerprints(tensor: torch.Tensor) -> list[dict]:
    if tensor.ndim < 2:
        raise ValueError(f"Expected a token dimension, got {tensor.shape}")
    if tensor.ndim >= 3:
        if tensor.shape[0] != 1:
            raise ValueError(f"Expected batch size one, got {tensor.shape}")
        return [canonical_tensor_fingerprint(tensor[:, index]) for index in range(tensor.shape[1])]
    return [canonical_tensor_fingerprint(tensor[index : index + 1]) for index in range(tensor.shape[0])]


def compact_layer_capture(layer_entry: dict, capture_key: str, capture_label: str) -> None:
    captures = layer_entry.get(capture_key)
    if captures is None:
        return
    if len(captures) != 1:
        raise RuntimeError(
            f"Expected one learner {capture_label} capture in layer {layer_entry['layer']}, got {len(captures)}"
        )
    layer_entry[capture_key] = captures[0]


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

    def score_next_token(
        self,
        prompt_token_ids,
        selected_token: int,
        capture_head_input: bool = False,
        prefill_token_count: int | None = None,
    ):
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
        hook_registrations = []
        learner_fla_capture = None
        learner_fla_restore = None
        learner_conv_capture = None
        learner_conv_restore = None
        finalize_fla_capture = None
        finalize_conv_capture = None

        def restore_diagnostic_state():
            nonlocal learner_conv_restore, learner_fla_restore
            if learner_conv_restore is not None:
                mixer, live_causal_conv1d_fn = learner_conv_restore
                mixer.causal_conv1d_fn = live_causal_conv1d_fn
                learner_conv_restore = None
            if learner_fla_restore is not None:
                mixer, live_chunk_gated_delta_rule = learner_fla_restore
                mixer.chunk_gated_delta_rule = live_chunk_gated_delta_rule
                learner_fla_restore = None
            for hook in hooks:
                hook.remove()
            hooks.clear()
            if was_training:
                model.train()

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

            def attention_payload(sequence_heads):
                if sequence_heads.ndim != 3:
                    raise RuntimeError(
                        f"Expected attention tensor [tokens, heads, head_dim], got {sequence_heads.shape}"
                    )
                last_token = sequence_heads[-1].reshape(-1)
                return {
                    "values": last_token.detach().float().cpu().tolist(),
                    "dtype": str(sequence_heads.dtype),
                    "shape": list(last_token.shape),
                    "sequence_shape": list(sequence_heads.shape),
                    "sequence_fingerprint": canonical_tensor_fingerprint(sequence_heads),
                    "token_fingerprints": [
                        canonical_tensor_fingerprint(sequence_heads[index : index + 1])
                        for index in range(sequence_heads.shape[0])
                    ],
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

            hook_registrations.append(
                partial(output_embeddings.register_forward_pre_hook, capture_output_embedding_input)
            )

            text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
            diagnostic_gdn_layer = 2
            for layer_index, layer in enumerate(text_model.layers):
                mixer_name = "linear_attn" if layer.layer_type == "linear_attention" else "self_attn"
                mixer = getattr(layer, mixer_name)
                entry = {"layer": layer_index, "mixer": mixer_name}
                layer_captures.append(entry)

                if layer_index == diagnostic_gdn_layer and mixer_name == "linear_attn":
                    projections = {}
                    entry["projections"] = projections
                    entry["mixer_stages"] = {}
                    fla_captures = []
                    entry["fla_core"] = fla_captures
                    conv_captures = []
                    entry["causal_conv"] = conv_captures

                    from causal_conv1d import causal_conv1d_fn as released_causal_conv1d_fn
                    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as released_chunk_gated_delta_rule

                    live_causal_conv1d_fn = mixer.causal_conv1d_fn

                    def capture_causal_conv1d(*args, **kwargs):
                        if conv_captures:
                            raise RuntimeError("Expected one learner causal-convolution call")
                        x = kwargs.get("x", args[0] if args else None)
                        weight = kwargs.get("weight", args[1] if len(args) > 1 else None)
                        bias = kwargs.get("bias", args[2] if len(args) > 2 else None)
                        seq_idx = kwargs.get("seq_idx", args[3] if len(args) > 3 else None)
                        activation = kwargs.get("activation")
                        if x is None or weight is None:
                            raise RuntimeError("Missing learner causal-convolution inputs")
                        captured_x = x.detach().clone()
                        captured_weight = weight.detach().clone()
                        captured_bias = bias.detach().clone() if bias is not None else None
                        live_output = live_causal_conv1d_fn(*args, **kwargs)
                        if isinstance(live_output, tuple):
                            raise RuntimeError("Expected learner causal convolution without a returned final state")
                        conv_captures.append(
                            {
                                "activation": activation,
                                "bias": captured_bias,
                                "live_output": live_output.detach().clone(),
                                "seq_idx": seq_idx.detach().clone() if isinstance(seq_idx, torch.Tensor) else seq_idx,
                                "weight": captured_weight,
                                "x": captured_x,
                            }
                        )
                        return live_output

                    learner_conv_capture = (mixer, live_causal_conv1d_fn, capture_causal_conv1d)

                    live_chunk_gated_delta_rule = mixer.chunk_gated_delta_rule
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

                    learner_fla_capture = (mixer, live_chunk_gated_delta_rule, capture_fla_core)

                    def finalize_causal_conv_capture(capture):
                        captured_x = capture["x"]
                        captured_weight = capture["weight"]
                        captured_bias = capture["bias"]
                        activation = capture["activation"]
                        live_output = capture["live_output"]
                        released_output = released_causal_conv1d_fn(
                            captured_x.clone(),
                            captured_weight.clone(),
                            bias=captured_bias.clone() if captured_bias is not None else None,
                            seq_idx=capture["seq_idx"],
                            activation=activation,
                        )
                        token0_x = captured_x[:, :, :1]
                        token0_raw = released_causal_conv1d_fn(
                            token0_x.clone(),
                            captured_weight.clone(),
                            bias=captured_bias.clone() if captured_bias is not None else None,
                            seq_idx=None,
                            activation=None,
                        ).transpose(1, 2)
                        token0_silu = released_causal_conv1d_fn(
                            token0_x.clone(),
                            captured_weight.clone(),
                            bias=captured_bias.clone() if captured_bias is not None else None,
                            seq_idx=None,
                            activation="silu",
                        ).transpose(1, 2)
                        manual_token0_raw = captured_x[:, :, 0].float() * captured_weight[:, -1].float()
                        if captured_bias is not None:
                            manual_token0_raw = manual_token0_raw + captured_bias.float()
                        manual_token0_raw = manual_token0_raw.to(token0_raw.dtype).unsqueeze(1)
                        manual_token0_silu = torch.nn.functional.silu(manual_token0_raw.float()).to(token0_silu.dtype)

                        def tensor_payload(tensor):
                            token_major = tensor.transpose(1, 2)
                            return {
                                "fingerprint": canonical_tensor_fingerprint(token_major),
                                "layout": _tensor_layout(tensor),
                                "token_fingerprints": _token_fingerprints(token_major),
                            }

                        return {
                            "backend": {
                                "live_callable_is_released": live_causal_conv1d_fn is released_causal_conv1d_fn,
                                "method": getattr(
                                    live_causal_conv1d_fn,
                                    "__qualname__",
                                    type(live_causal_conv1d_fn).__qualname__,
                                ),
                                "method_module": getattr(
                                    live_causal_conv1d_fn,
                                    "__module__",
                                    type(live_causal_conv1d_fn).__module__,
                                ),
                            },
                            "inputs": {
                                "x": tensor_payload(captured_x),
                                "weight": {
                                    "fingerprint": canonical_tensor_fingerprint(captured_weight),
                                    "layout": _tensor_layout(captured_weight),
                                },
                                "bias": (
                                    {
                                        "fingerprint": canonical_tensor_fingerprint(captured_bias),
                                        "layout": _tensor_layout(captured_bias),
                                    }
                                    if captured_bias is not None
                                    else None
                                ),
                            },
                            "live": tensor_payload(live_output),
                            "released_replay": tensor_payload(released_output),
                            "comparisons": {
                                "live_vs_released": _exact_error_summary(live_output, released_output),
                                "token0_raw": {
                                    "released_vs_manual": _exact_error_summary(token0_raw, manual_token0_raw),
                                },
                                "token0_silu": {
                                    "released_vs_manual": _exact_error_summary(token0_silu, manual_token0_silu),
                                },
                            },
                            "options": {
                                "activation": activation,
                                "seq_idx": (
                                    capture["seq_idx"].detach().cpu().tolist()
                                    if isinstance(capture["seq_idx"], torch.Tensor)
                                    else capture["seq_idx"]
                                ),
                            },
                        }

                    finalize_conv_capture = finalize_causal_conv_capture

                    def finalize_fla_capture(capture):
                        captured_inputs = capture["captured_inputs"]
                        options = capture["options"]
                        sequence_length = captured_inputs["q"].shape[1]
                        if prefill_token_count is None or not 0 < prefill_token_count <= sequence_length:
                            raise ValueError(
                                f"Expected a prefill boundary in [1, {sequence_length}], got {prefill_token_count}"
                            )
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
                        whole_state_output, whole_final_state = released_chunk_gated_delta_rule(
                            captured_inputs["q"].clone(),
                            captured_inputs["k"].clone(),
                            captured_inputs["v"].clone(),
                            g=captured_inputs["g"].clone(),
                            beta=captured_inputs["beta"].clone(),
                            scale=options["scale"],
                            initial_state=None,
                            output_final_state=True,
                            use_qk_l2norm_in_kernel=options["use_qk_l2norm_in_kernel"],
                            cu_seqlens=None,
                        )
                        segment_boundaries = [(0, prefill_token_count)] + [
                            (token_index, token_index + 1)
                            for token_index in range(prefill_token_count, sequence_length)
                        ]
                        segmented_outputs = []
                        segmented_state = None
                        for start, end in segment_boundaries:
                            segmented_output, segmented_state = released_chunk_gated_delta_rule(
                                captured_inputs["q"][:, start:end].clone(),
                                captured_inputs["k"][:, start:end].clone(),
                                captured_inputs["v"][:, start:end].clone(),
                                g=captured_inputs["g"][:, start:end].clone(),
                                beta=captured_inputs["beta"][:, start:end].clone(),
                                scale=options["scale"],
                                initial_state=segmented_state,
                                output_final_state=True,
                                use_qk_l2norm_in_kernel=options["use_qk_l2norm_in_kernel"],
                                cu_seqlens=None,
                            )
                            segmented_outputs.append(segmented_output)
                        segmented_output = torch.cat(segmented_outputs, dim=1)
                        response_score_positions = [
                            {
                                "response_token_index": position - prefill_token_count + 1,
                                "sequence_position": position,
                                "error": _exact_error_summary(
                                    whole_state_output[:, position], segmented_output[:, position]
                                ),
                            }
                            for position in range(prefill_token_count - 1, sequence_length)
                        ]
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
                        input_payloads = {}
                        for name, tensor in captured_inputs.items():
                            if tensor is None:
                                input_payloads[name] = None
                                continue
                            payload = {
                                "fingerprint": canonical_tensor_fingerprint(tensor),
                                "layout": _tensor_layout(tensor),
                            }
                            if name != "initial_state":
                                payload["token_fingerprints"] = _token_fingerprints(tensor)
                            if name in {"q", "k"}:
                                normalized, _ = l2norm_fwd(tensor.clone())
                                payload["normalized_fingerprint"] = canonical_tensor_fingerprint(normalized)
                                payload["normalized_token_fingerprints"] = _token_fingerprints(normalized)
                            input_payloads[name] = payload
                        post_conv_qkv = torch.cat(
                            [
                                captured_inputs["q"].flatten(2),
                                captured_inputs["k"].flatten(2),
                                captured_inputs["v"].flatten(2),
                            ],
                            dim=-1,
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
                                "prefill_token_count": prefill_token_count,
                                "production_sample_packing_exercised": False,
                                "scored_response_token_index": sequence_length - prefill_token_count,
                                "scored_batch_size": input_ids.shape[0],
                                "scored_sequence_count": input_ids.shape[0],
                                "segment_boundaries": [list(boundary) for boundary in segment_boundaries],
                            },
                            "inputs": input_payloads,
                            "post_conv_qkv": {
                                "fingerprint": canonical_tensor_fingerprint(post_conv_qkv),
                                "layout": _tensor_layout(post_conv_qkv),
                                "token_fingerprints": _token_fingerprints(post_conv_qkv),
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
                            "released_whole_vs_segmented": {
                                "final_state": _exact_error_summary(whole_final_state, segmented_state),
                                "live_options_vs_whole_state_output": _exact_error_summary(
                                    replay_output, whole_state_output
                                ),
                                "output": _exact_error_summary(whole_state_output, segmented_output),
                                "response_score_positions": response_score_positions,
                                "scored_position": response_score_positions[-1],
                            },
                            "inputs_unchanged_by_live_call": capture["post_live_input_errors"],
                            "none_vs_zero_initial_state": _exact_error_summary(live_output, zero_state_output),
                            "options": options,
                        }

                    def capture_projection(_module, _args, output, *, destination, key):
                        payload = tensor_payload(output)
                        payload["token_fingerprints"] = _token_fingerprints(output)
                        destination.setdefault(key, []).append(payload)

                    for projection_name, key in (
                        ("in_proj_qkv", "qkv"),
                        ("in_proj_z", "z"),
                        ("in_proj_b", "b"),
                        ("in_proj_a", "a"),
                    ):
                        projection = getattr(mixer, projection_name)
                        hook_registrations.append(
                            partial(
                                projection.register_forward_hook,
                                lambda module, args, output, destination=projections, key=key: capture_projection(
                                    module,
                                    args,
                                    output,
                                    destination=destination,
                                    key=key,
                                ),
                            )
                        )

                if layer_index == 3 and mixer_name == "self_attn":
                    attention_stages = {}
                    attention_replays = {}
                    attention_inputs = []
                    entry["attention_stages"] = attention_stages
                    entry["attention_replays"] = attention_replays

                    def capture_attention_input(
                        _module,
                        args,
                        kwargs,
                        *,
                        inputs=attention_inputs,
                    ):
                        hidden_states = kwargs.get("hidden_states")
                        if hidden_states is None:
                            hidden_states = args[0]
                        position_embeddings = kwargs.get("position_embeddings")
                        if position_embeddings is None:
                            position_embeddings = args[1]
                        attention_mask_value = kwargs.get("attention_mask")
                        if attention_mask_value is None and len(args) > 2:
                            attention_mask_value = args[2]
                        inputs.append((hidden_states, position_embeddings, attention_mask_value))

                    def capture_q_gate(_module, _args, output, *, destination=attention_stages, layer_mixer=mixer):
                        input_shape = output.shape[:-1]
                        q_gate = output.view(*input_shape, layer_mixer.config.num_attention_heads, -1)
                        query, gate = torch.chunk(q_gate, 2, dim=-1)
                        destination.setdefault("q_raw", []).append(attention_payload(query[0]))
                        destination.setdefault("gate", []).append(attention_payload(gate[0]))

                    def capture_kv(
                        _module,
                        _args,
                        output,
                        *,
                        destination=attention_stages,
                        key,
                        layer_mixer=mixer,
                    ):
                        heads = layer_mixer.config.num_key_value_heads
                        sequence_heads = output.view(output.shape[0], output.shape[1], heads, layer_mixer.head_dim)[0]
                        destination.setdefault(key, []).append(attention_payload(sequence_heads))

                    def capture_attention_norm(
                        _module,
                        _args,
                        output,
                        *,
                        destination=attention_stages,
                        key,
                    ):
                        destination.setdefault(key, []).append(attention_payload(output[0]))

                    def replay_attention(
                        _module,
                        _args,
                        _kwargs,
                        _output,
                        *,
                        inputs=attention_inputs,
                        destination=attention_replays,
                        layer_mixer=mixer,
                    ):
                        if len(inputs) != 1:
                            raise RuntimeError(f"Expected one learner attention input, got {len(inputs)}")
                        hidden_states, position_embeddings, attention_mask_value = inputs[0]
                        input_shape = hidden_states.shape[:-1]
                        hidden_shape = (*input_shape, -1, layer_mixer.head_dim)
                        q_gate = torch.nn.functional.linear(
                            hidden_states,
                            layer_mixer.q_proj.weight,
                            layer_mixer.q_proj.bias,
                        ).view(*input_shape, -1, layer_mixer.head_dim * 2)
                        query, gate = torch.chunk(q_gate, 2, dim=-1)
                        gate = gate.reshape(*input_shape, -1)
                        query = layer_mixer.q_norm.forward(query.view(hidden_shape)).transpose(1, 2)
                        key = layer_mixer.k_norm.forward(
                            torch.nn.functional.linear(
                                hidden_states,
                                layer_mixer.k_proj.weight,
                                layer_mixer.k_proj.bias,
                            ).view(hidden_shape)
                        ).transpose(1, 2)
                        value = (
                            torch.nn.functional.linear(
                                hidden_states,
                                layer_mixer.v_proj.weight,
                                layer_mixer.v_proj.bias,
                            )
                            .view(hidden_shape)
                            .transpose(1, 2)
                        )
                        cos, sin = position_embeddings
                        query, key = apply_rotary_pos_emb(query, key, cos, sin)
                        destination.setdefault("q_rope", []).append(attention_payload(query[0].transpose(0, 1)))
                        destination.setdefault("k_rope", []).append(attention_payload(key[0].transpose(0, 1)))
                        destination.setdefault("v", []).append(attention_payload(value[0].transpose(0, 1)))

                        repeated_key = repeat_kv(key, layer_mixer.num_key_value_groups)
                        repeated_value = repeat_kv(value, layer_mixer.num_key_value_groups)
                        weights = torch.matmul(query, repeated_key.transpose(2, 3)) * layer_mixer.scaling
                        if attention_mask_value is not None:
                            weights = weights + attention_mask_value
                        weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
                        core = torch.matmul(weights, repeated_value).transpose(1, 2).contiguous()
                        destination.setdefault("attention_core", []).append(attention_payload(core[0]))
                        post_gate = core.reshape(*input_shape, -1).contiguous() * torch.sigmoid(gate)
                        destination.setdefault("post_gate", []).append(tensor_payload(post_gate))
                        projected = torch.nn.functional.linear(
                            post_gate,
                            layer_mixer.o_proj.weight,
                            layer_mixer.o_proj.bias,
                        )
                        destination.setdefault("out_proj", []).append(tensor_payload(projected))

                    hook_registrations.extend(
                        (
                            partial(mixer.register_forward_pre_hook, capture_attention_input, with_kwargs=True),
                            partial(mixer.q_proj.register_forward_hook, capture_q_gate),
                            partial(
                                mixer.k_proj.register_forward_hook,
                                lambda module, args, output, destination=attention_stages: capture_kv(
                                    module, args, output, destination=destination, key="k_raw"
                                ),
                            ),
                            partial(
                                mixer.v_proj.register_forward_hook,
                                lambda module, args, output, destination=attention_stages: capture_kv(
                                    module, args, output, destination=destination, key="v_raw"
                                ),
                            ),
                            partial(
                                mixer.q_norm.register_forward_hook,
                                lambda module, args, output, destination=attention_stages: capture_attention_norm(
                                    module, args, output, destination=destination, key="q_norm"
                                ),
                            ),
                            partial(
                                mixer.k_norm.register_forward_hook,
                                lambda module, args, output, destination=attention_stages: capture_attention_norm(
                                    module, args, output, destination=destination, key="k_norm"
                                ),
                            ),
                            partial(
                                mixer.o_proj.register_forward_pre_hook,
                                lambda module, args, destination=attention_stages: destination.setdefault(
                                    "post_gate", []
                                ).append(tensor_payload(args[0])),
                            ),
                            partial(
                                mixer.o_proj.register_forward_hook,
                                lambda module, args, output, destination=attention_stages: destination.setdefault(
                                    "out_proj", []
                                ).append(tensor_payload(output)),
                            ),
                            partial(mixer.register_forward_hook, replay_attention, with_kwargs=True),
                        )
                    )

                def capture_input(
                    _module,
                    args,
                    kwargs,
                    *,
                    destination=entry,
                    key="mixer_input",
                    trace_tokens=False,
                ):
                    hidden_states = kwargs.get("hidden_states")
                    if hidden_states is None:
                        hidden_states = args[0]
                    payload = tensor_payload(hidden_states)
                    if trace_tokens:
                        payload["token_fingerprints"] = _token_fingerprints(hidden_states)
                    destination.setdefault(key, []).append(payload)

                def capture_output(
                    _module,
                    args,
                    kwargs,
                    output,
                    *,
                    destination=entry,
                    key="mixer_output",
                    trace_tokens=False,
                ):
                    del args
                    hidden_states = kwargs.get("output")
                    if hidden_states is None:
                        hidden_states = output[0] if isinstance(output, tuple) else output
                    payload = tensor_payload(hidden_states)
                    if trace_tokens:
                        payload["token_fingerprints"] = _token_fingerprints(hidden_states)
                    destination.setdefault(key, []).append(payload)

                if layer_index == diagnostic_gdn_layer and mixer_name == "linear_attn":
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

                    hook_registrations.append(
                        partial(mixer.norm.register_forward_pre_hook, capture_norm_input, with_kwargs=True)
                    )
                    hook_registrations.append(
                        partial(mixer.norm.register_forward_hook, capture_norm_output, with_kwargs=True)
                    )
                    hook_registrations.append(
                        partial(
                            mixer.out_proj.register_forward_pre_hook,
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
                    hook_registrations.append(
                        partial(
                            mixer.out_proj.register_forward_hook,
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
                    entry["mlp_stages"] = mlp_stages
                    for module, key in (
                        (layer.mlp.gate_proj, "gate"),
                        (layer.mlp.up_proj, "up"),
                        (layer.mlp.act_fn, "activation"),
                    ):
                        hook_registrations.append(
                            partial(
                                module.register_forward_hook,
                                lambda module, args, kwargs, output, destination=mlp_stages, key=key: capture_output(
                                    module,
                                    args,
                                    kwargs,
                                    output,
                                    destination=destination,
                                    key=key,
                                ),
                                with_kwargs=True,
                            )
                        )
                    hook_registrations.append(
                        partial(
                            layer.mlp.down_proj.register_forward_pre_hook,
                            lambda module, args, kwargs, destination=mlp_stages: capture_input(
                                module,
                                args,
                                kwargs,
                                destination=destination,
                                key="product",
                            ),
                            with_kwargs=True,
                        )
                    )
                    hook_registrations.append(
                        partial(
                            layer.mlp.down_proj.register_forward_hook,
                            lambda module, args, kwargs, output, destination=mlp_stages: capture_output(
                                module,
                                args,
                                kwargs,
                                output,
                                destination=destination,
                                key="down",
                            ),
                            with_kwargs=True,
                        )
                    )
                hook_registrations.append(
                    partial(
                        mixer.register_forward_pre_hook,
                        partial(capture_input, trace_tokens=layer_index < 4),
                        with_kwargs=True,
                    )
                )
                hook_registrations.append(
                    partial(
                        mixer.register_forward_hook,
                        partial(capture_output, trace_tokens=layer_index < 4),
                        with_kwargs=True,
                    )
                )
                hook_registrations.append(
                    partial(
                        layer.mlp.register_forward_pre_hook,
                        lambda module, args, kwargs, destination=entry, trace_tokens=layer_index < 4: capture_input(
                            module,
                            args,
                            kwargs,
                            destination=destination,
                            key="mlp_input",
                            trace_tokens=trace_tokens,
                        ),
                        with_kwargs=True,
                    )
                )
                hook_registrations.append(
                    partial(
                        layer.mlp.register_forward_hook,
                        lambda module, args, kwargs, output, destination=entry, trace_tokens=layer_index < 4: (
                            capture_output(
                                module,
                                args,
                                kwargs,
                                output,
                                destination=destination,
                                key="mlp_output",
                                trace_tokens=trace_tokens,
                            )
                        ),
                        with_kwargs=True,
                    )
                )
        try:
            hooks.extend(register_hook() for register_hook in hook_registrations)
            model.eval()
            if learner_fla_capture is not None:
                mixer, live_chunk_gated_delta_rule, capture_fla_core = learner_fla_capture
                mixer.chunk_gated_delta_rule = capture_fla_core
                learner_fla_restore = (mixer, live_chunk_gated_delta_rule)
            if learner_conv_capture is not None:
                mixer, live_causal_conv1d_fn, capture_causal_conv1d = learner_conv_capture
                mixer.causal_conv1d_fn = capture_causal_conv1d
                learner_conv_restore = (mixer, live_causal_conv1d_fn)
        except BaseException:
            restore_diagnostic_state()
            raise
        try:
            with torch.no_grad():
                model_output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                if learner_conv_restore is not None:
                    conv_mixer, live_causal_conv1d_fn = learner_conv_restore
                    conv_mixer.causal_conv1d_fn = live_causal_conv1d_fn
                    learner_conv_restore = None
                if learner_fla_restore is not None:
                    fla_mixer, live_chunk_gated_delta_rule = learner_fla_restore
                    fla_mixer.chunk_gated_delta_rule = live_chunk_gated_delta_rule
                    learner_fla_restore = None
                if capture_head_input:
                    if finalize_fla_capture is None:
                        raise RuntimeError("Missing learner FLA capture finalizer")
                    for layer_entry in layer_captures:
                        causal_conv = layer_entry.get("causal_conv")
                        if causal_conv is not None:
                            if finalize_conv_capture is None or len(causal_conv) != 1:
                                raise RuntimeError(
                                    f"Expected one learner causal-convolution capture in layer "
                                    f"{layer_entry['layer']}, got {len(causal_conv)}"
                                )
                            causal_conv[0] = finalize_conv_capture(causal_conv[0])
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
                        for key, values in layer_entry.get("mlp_stages", {}).items():
                            if len(values) != 1:
                                raise RuntimeError(
                                    f"Expected one learner {key} MLP stage capture in layer "
                                    f"{layer_entry['layer']}, got {len(values)}"
                                )
                            layer_entry["mlp_stages"][key] = values[0]
                        for capture_name in ("attention_stages", "attention_replays"):
                            for key, values in layer_entry.get(capture_name, {}).items():
                                if len(values) != 1:
                                    raise RuntimeError(
                                        f"Expected one learner {key} {capture_name} capture in layer "
                                        f"{layer_entry['layer']}, got {len(values)}"
                                    )
                                layer_entry[capture_name][key] = values[0]
                        compact_layer_capture(layer_entry, "fla_core", "FLA core")
                        compact_layer_capture(layer_entry, "causal_conv", "causal-convolution")
                    result["layer_trace"] = layer_captures
        finally:
            restore_diagnostic_state()
        return result


# Pytest adds ``skyrl-train`` to the driver's import path, but Ray workers do
# not inherit that test-only path. Serialize this helper module by value so the
# diagnostic actor does not require ``tests`` to be installed on every worker.
ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
PolicyWorker = ray.remote(num_gpus=1)(DPPOPolicyWorker)
