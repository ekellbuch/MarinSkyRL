"""Ray worker diagnostics used by the DPPO GPU integration test."""

import ray
import torch
from torch.distributed.tensor import DTensor

from skyrl_train.utils import str_to_torch_dtype
from skyrl_train.utils.tensor_fingerprint import canonical_tensor_fingerprint
from skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase, FSDPWeightExtractor


class DPPOPolicyWorker(FSDPPolicyWorkerBase):
    def fingerprint_broadcast_weights(self, names):
        wanted = set(names)
        fingerprints = {}
        generator_dtype = str_to_torch_dtype(self.cfg.generator.model_dtype)
        is_rank0 = torch.distributed.get_rank() == 0
        for chunk in self.weight_extractor.extract_weights(generator_dtype):
            for name, tensor in zip(chunk.names, chunk.tensors):
                if is_rank0 and name in wanted:
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
            return {"name": parameter_name, "changed": False, "rank": torch.distributed.get_rank()}

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

    def score_next_token(self, prompt_token_ids, selected_token: int):
        from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available

        if not is_fast_path_available:
            raise RuntimeError("Qwen3.5 learner parity requires the flash-linear-attention and causal-conv1d fast path")
        device = torch.cuda.current_device()
        input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        model = self.model.model
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1]
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
                    "gdn_fast_path": True,
                }
        finally:
            if was_training:
                model.train()
        return result


PolicyWorker = ray.remote(num_gpus=1)(DPPOPolicyWorker)
