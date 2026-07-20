# gpu-rl image — wheel cache for fast rebuilds

The `gpu-rl` image's `wheel-builder` stage compiles two things FROM SOURCE with
`nvcc` against torch 2.11 / CUDA 12.8 / cp312 / x86_64 — and those are the only
slow parts of the build:

- the **vLLM fork** (`marin-community/vllm` @ `4b555913`) — Grug serving plus SkyRL runtime patches.
- **flash-attn 2.8.3** — its `flash_attn_2_cuda` extension.

To avoid recompiling them on every rebuild, `Dockerfile.gpu-rl` is split into a
**`wheel-builder`** stage (the only stage that runs nvcc) that emits relocatable
wheels, and an **`rl`** stage that just `uv pip install`s those wheels.

## Build host / mechanism

The production mechanism is `docker/build_gpu_rl_kaniko.sh` on Iris; a local
x86 builder can use **`docker buildx build … --push`** directly. The production
`gpu-rl` image is **single-platform `linux/amd64`** (verified on the live digest)
— CoreWeave H100 + the x86 CUDA build are amd64-only. So everything below
targets `linux/amd64`.

> This must run on an **x86_64 / linux/amd64** host (real x86 build host, GPU
> build pod, or x86 CI runner). The nvcc compiles need `MAX_JOBS=8 * ~5GB/job ≈
> 40GB RAM`. On the arm64 dev Mac the amd64 pass runs under QEMU emulation
> (impractically slow + RAM-bound on Docker Desktop) — do NOT build there.

## The wheel cache

| | |
|---|---|
| **Cache location** | `docker/wheelhouse/` on the build host (gitignored; only `.keep` is committed). Holds `vllm-*.whl`, `flash_attn-*.whl`, `MANIFEST`. |
| **Cache key (what invalidates a wheel)** | the tuple `{ VLLM_FORK_COMMIT, FLASH_ATTN_VERSION, torch 2.11.0, CUDA 12.8, cp312, x86_64, TORCH_CUDA_ARCH_LIST "8.0;9.0" }` — all declared as `ARG`s at the top of `Dockerfile.gpu-rl`. Change any → rebuild the wheels. `MANIFEST` records the key the wheels were built at. |
| **Not cached** | torchtitan (pure-python, trivial — installed fresh each build) and all pip-resolved deps. |

## Commands

### One-time / on a pin change — build the wheels

```bash
source ~/secrets.env          # ghcr auth, if pushing later
./docker/build_wheels.sh      # -> docker/wheelhouse/{vllm,flash_attn}-*.whl + MANIFEST
```

Re-run this **only** when a pin/ABI in the cache key changes.

### Every rebuild — build + push from a populated wheelhouse (NO nvcc)

```bash
source ~/secrets.env
echo "$GITHUB_TOKEN" | docker login ghcr.io -u <user> --password-stdin
docker buildx build -f docker/Dockerfile.gpu-rl \
  --platform linux/amd64 --target rl \
  --build-arg WHEEL_SOURCE=prebuilt-wheelhouse \
  -t ghcr.io/<owner>/<package>:gpu-rl-<gitsha> --push .
# capture the pushed @sha256 digest from the immutable :gpu-rl-<gitsha> tag:
docker buildx imagetools inspect ghcr.io/<owner>/<package>:gpu-rl-<gitsha>
```

The Kaniko launcher can fetch a wheelhouse tarball directly from Iris storage:

```bash
export GITSHA=$(git rev-parse --short HEAD)
export DOCKER_USER_ID="YOUR_GHCR_USER"
export GHCR_TOKEN="YOUR_WRITE_PACKAGES_TOKEN"
export GHCR_IMAGE_REPOSITORY="ghcr.io/YOUR_GHCR_USER/YOUR_PACKAGE"
export WHEEL_SOURCE=prebuilt-wheelhouse
export PREBUILT_WHEEL_ARTIFACT_URI=s3://marin-us-east-02a/iris/grug-vllm-wheels/4b55591306c9-torch211-cu128-cp312-fb6ff59.tar.gz
export KANIKO_CACHE=0
./docker/build_gpu_rl_kaniko.sh
```

The archive must contain `wheels/MANIFEST`, one `vllm-*.whl`, and one
`flash_attn-*.whl`. The launcher verifies the exact ABI/pin manifest and exits
on any fetch or validation failure; it never falls back to a source compile.
The URI above is the validated, content-keyed wheelhouse for the checked-in
vLLM, Torch, CUDA, Python, and architecture manifest. Keep it until one of those
inputs changes; a new manifest requires a new artifact rather than overwriting it.

To publish a new wheel artifact, an Iris build may set
`WHEEL_ARTIFACT_URI=s3://...tar.gz`. The build script then runs only the exact
`wheel-builder` stage with kaniko `--no-push`, extracts `/wheels`, and uploads
that tarball without requiring GHCR credentials.

### No prebuilt wheelhouse? — compile inline (fallback, slow)

```bash
docker buildx build -f docker/Dockerfile.gpu-rl \
  --platform linux/amd64 --target rl \
  --build-arg WHEEL_SOURCE=wheel-builder \
  -t ghcr.io/<owner>/<package>:gpu-rl --push .
```

`WHEEL_SOURCE=wheel-builder` makes the `rl` stage take its `/wheels` from the
in-build `wheel-builder` stage (compiled fresh; buildx layer-caches it for the
next rebuild on the same builder). Equivalent result, but pays the nvcc cost.

## After a rebuild — bump the launcher digest

The image is pinned by **immutable `@sha256:` digest** in
`cloud/iris/launch_rl_iris.py` (`DEFAULT_RL_DOCKER_IMAGE`). After pushing, set it
to the new digest (the `:gpu-rl-<gitsha>` tag's digest, never the floating
`:gpu-rl`) and update the provenance comment.

## What proves the build is good

The `rl` stage asserts at build time:

- `import flash_attn, flash_attn_2_cuda` — the CUDA extension EXISTS (from the wheel).
- `import torch, vllm, skyrl_train, flash_attn, flash_attn_2_cuda`.
- `from torchtitan.distributed.expert_parallel import ExpertParallel` — the
  **EP>1 MoE unblock**; if this line prints `... import OK`, `apply_ep` will
  resolve `ExpertParallel` and the CoreWeave EP=8 RL jobs can launch.
