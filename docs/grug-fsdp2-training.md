# Grug FSDP2 training

This is a policy-only, EP1 FSDP2 implementation. The trainer uses the
canonical PyTorch model in `skyrl_train.models.grug_moe`; vLLM serves the same HF
checkpoint with TP1/DP2/EP2. Packing, FlashAttention, trainer EP/CP, R3/router
replay, grouped MoE, LoRA/4-bit loading, and PKO are intentionally rejected.

## Runtime image

Grug serving requires the Marin vLLM fork at commit `4b55591306c9`. The shared
`DEFAULT_RL_DOCKER_IMAGE` does not yet include that fork and must not be used for
Grug. Until a Grug-capable image is published and pinned by immutable digest,
launches must pass an explicit verified image. Check the launcher constant when
this release constraint changes. The S3 wheel artifact used by validation is a
build fallback, not a production image.

## Query bias

For every optimizer window, each rank counts non-padding tokens and uses
`q = max(1, floor(tokens * top_k / num_experts))`. Each router retains only its
per-expert top-q values of `unbiased_logit - biased_(K+1)th_logit`; concatenating
and top-k reducing these candidates is exactly equivalent to retaining the full
token-by-expert matrix. The q-th values are averaged across ranks. After a real
optimizer step, the next persistent FP32 bias is `center(-beta)`. A skipped
non-finite step discards the observation and preserves the previous bias.

This padding exclusion is the RL adaptation of Levanter's fixed, padding-free
batch geometry. The strict cross-framework fixture is padding-free.

## Step-42150 memory gate

The step-42150 checkpoint has roughly 67B parameters. A conservative two-rank BF16
budget, before framework overhead, is:

| State | Aggregate host memory |
| --- | ---: |
| BF16 parameters | 134 GB |
| BF16 gradients | 134 GB |
| FP32 Adam first and second moments | 536 GB |
| Total steady training state | 804 GB |
| Rank-0 full-state load transient | up to 134 GB |

The largest routed layer is about 2.5B parameters, or 5 GB in BF16, so FSDP2
CPU offload plus gradient checkpointing bounds the per-layer GPU materialization;
host RAM is the harder gate. Load/forward and one update fit a 1 TiB cgroup, but
the measured same-process optimizer restore reached 1024/1024 GiB while both
old and loaded state were live. Reserve 1.5 TiB for save/resume and the
colocated full-cycle gate; record RSS/cgroup peaks rather than relying on the
budget alone. A save/resume gate also needs 1 TiB of ephemeral disk while the
125 GiB staged export and sharded model/Adam checkpoint coexist; 512 GiB is
sufficient only for load/forward. vLLM must sleep at level 2 before training,
and the trainer must offload before the next rollout. Do not exceed two H100s.

## Validation tiers

1. Tiny FP32 unit tests: state names, eager forward/loss/backward, gradient
   checkpointing, save/load, query bias, and skipped steps.
2. Tiny one-H100 JAX/PyTorch parity: separate environments exchange only an HF
   checkpoint and compressed observations; MarinSkyRL has no JAX dependency.
3. Tiny two-H100 cycle: FSDP2 EP1 train, vLLM TP1/DP2/EP2 rollout, weight/bias
   sync, and a changed second-rollout logprob.
4. Step-42150 gates: load/forward, then backward/update/resume, then the short RL
   cycle, recording host/GPU peaks and elapsed time at each boundary.

The real-checkpoint test is opt-in and uses one harness for every progressive
gate. Stage the content-addressed HF export locally on Iris, then set
`GRUG_REAL_CHECKPOINT_DIR` and choose `GRUG_REAL_GATE=forward`, `train`,
`resume`, or `full`. Set `GRUG_REAL_CHECKPOINT_SOURCE` to the canonical S3 URI
and `GRUG_REAL_RESULT_JSON` to retain the compact timing, finite-step, and
per-rank host/GPU-memory report. Before allocating the model, the gate hashes
the full staged tree with Marin's canonical digest algorithm. The fixed export
identity is `781bc3291c81ce282be6762520280ebd5ef5b85e88ba65129c2d0162d48ee632`.
