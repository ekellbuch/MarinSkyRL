#!/usr/bin/env bash
# build_gpu_rl_kaniko.sh — in-cluster kaniko build of the MarinSkyRL gpu-rl image.
#
# Runs INSIDE an iris job whose task-image is docker.io/library/ubuntu:22.04
# (kaniko's executor image is distroless / has no bash, so it cannot be the task
# image directly). We crane-export the kaniko executor rootfs over / and run
# /kaniko/executor. Context = the iris-synced /app bundle (this repo).
# See .agents/skills/build-gpu-rl-image-iris/SKILL.md in this repository.
#
# Required env (passed by the iris launch as -e):
#   GITSHA          MarinSkyRL commit sha for the immutable pinned tag
#   GHCR_IMAGE_REPOSITORY  full repository, e.g. ghcr.io/<owner>/<package>
#   DOCKER_USER_ID         GHCR user
#   GHCR_TOKEN             GitHub token with write:packages
# Optional:
#   WHEEL_SOURCE       prebuilt-wheelhouse (default) | wheel-builder
#   INSTALL_MEGATRON   0 (default) | 1  -> builds the megatron variant
#   TAG_PREFIX         gpu-rl (default) | gpu-rl-megatron  -> the pinned tag prefix
#   SINGLE_SNAPSHOT    0 (default here) | 1
#   PUSH_FLOATING      0 (default here — experimental; leave :gpu-rl untouched) | 1
#   KANIKO_CACHE       1 (default) | 0
#   KANIKO_CACHE_REPOSITORY  required when KANIKO_CACHE=1
#   PREBUILT_WHEEL_ARTIFACT_URI  s3://...tar.gz containing wheels/{MANIFEST,*.whl};
#                      required when WHEEL_SOURCE=prebuilt-wheelhouse
# SECURITY: NO `set -x` before the ghcr token is consumed (would echo GHCR_TOKEN
# / the base64 AUTH into the R2-persisted finelog). Tracing is enabled AFTER the
# config.json write, so build steps are traced but the secret never is.
set -euo pipefail

: "${GITSHA:?}"

WHEEL_SOURCE="${WHEEL_SOURCE:-prebuilt-wheelhouse}"
INSTALL_MEGATRON="${INSTALL_MEGATRON:-0}"
TAG_PREFIX="${TAG_PREFIX:-gpu-rl}"
# harbor is baked non-editably into the image (no runtime --harbor-ref); bump this to
# bake a new harbor commit. Default matches the Dockerfile ARG default.
HARBOR_COMMIT="${HARBOR_COMMIT:-1319eb29}"
PREBUILT_WHEEL_ARTIFACT_URI="${PREBUILT_WHEEL_ARTIFACT_URI:-}"
GHCR_IMAGE_REPOSITORY="${GHCR_IMAGE_REPOSITORY:-}"
GHCR_TOKEN="${GHCR_TOKEN:-}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile.gpu-rl}"
: "${GHCR_IMAGE_REPOSITORY:?}"
: "${DOCKER_USER_ID:?}"
: "${GHCR_TOKEN:?}"
case "$WHEEL_SOURCE" in
  prebuilt-wheelhouse)
    : "${PREBUILT_WHEEL_ARTIFACT_URI:?Required when WHEEL_SOURCE=prebuilt-wheelhouse}"
    ;;
  wheel-builder) ;;
  *)
    echo "unsupported WHEEL_SOURCE: $WHEEL_SOURCE" >&2
    exit 2
    ;;
esac

# SINGLE_SNAPSHOT=0 (default) => per-instruction layers (each small enough to pull
# + retry over the CoreWeave->ghcr egress). =1 collapses to ONE ~16 GB layer that
# CANNOT be pulled (containerd EOFs the single-blob GET) — the build looks green
# but every pod ImagePullBackOffs. --compressed-caching=false keeps multi-layer
# snapshotting within the memory budget.
SINGLE_SNAPSHOT="${SINGLE_SNAPSHOT:-0}"
SNAPSHOT_FLAGS=()
if [ "$SINGLE_SNAPSHOT" = "1" ]; then SNAPSHOT_FLAGS=(--single-snapshot); fi

if [ "${KANIKO_CACHE:-1}" = "0" ]; then
  CACHE_FLAGS=(--cache=false)
else
  : "${KANIKO_CACHE_REPOSITORY:?Required when KANIKO_CACHE=1}"
  CACHE_FLAGS=(--cache=true "--cache-repo=${KANIKO_CACHE_REPOSITORY}")
fi

# Consumers pin the DIGEST; the floating tag is only moved when PUSH_FLOATING=1.
DEST_FLOATING="${GHCR_IMAGE_REPOSITORY}:${TAG_PREFIX}"
DEST_PINNED="${GHCR_IMAGE_REPOSITORY}:${TAG_PREFIX}-${GITSHA}"
FLOATING_DEST_FLAGS=(--destination "$DEST_FLOATING")
if [ "${PUSH_FLOATING:-0}" != "1" ]; then FLOATING_DEST_FLAGS=(); fi

# --- 1. fetch crane (static binary) ---
apt-get update -y && apt-get install -y --no-install-recommends ca-certificates curl tar
cd /tmp
CRANE_VER=v0.20.2
curl -fsSL "https://github.com/google/go-containerregistry/releases/download/${CRANE_VER}/go-containerregistry_Linux_x86_64.tar.gz" -o crane.tgz
tar -xzf crane.tgz crane
install -m 0755 crane /usr/local/bin/crane

# --- 2. crane-export the kaniko executor rootfs over / ---
crane export gcr.io/kaniko-project/executor:latest - | tar -xf - -C / || true

# --- 3. write the ghcr auth config AFTER the overlay (kaniko clobbers /kaniko otherwise) ---
export DOCKER_CONFIG=/kaniko/.docker
mkdir -p "$DOCKER_CONFIG"
AUTH=$(printf '%s:%s' "$DOCKER_USER_ID" "$GHCR_TOKEN" | base64 | tr -d '\n')
cat > "$DOCKER_CONFIG/config.json" <<EOF
{"auths":{"ghcr.io":{"auth":"${AUTH}"}}}
EOF
unset AUTH
unset GHCR_TOKEN
set -x  # ghcr PAT consumed — safe to trace the build steps (token never traced)

prepare_wheel_artifact_io() {
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip
  python3 -m pip install --no-cache-dir fsspec s3fs
}

download_wheel_artifact() {
  python3 - "$1" "$2" <<'PY'
import shutil
import sys

import fsspec

uri, local_path = sys.argv[1:]
with fsspec.open(uri, "rb") as source, open(local_path, "wb") as destination:
    shutil.copyfileobj(source, destination)
print(f"downloaded wheel artifact: {uri}")
PY
}

# --- 3.5. fetch and validate the prebuilt wheelhouse, when selected ---
# This path is deliberately strict: a fetch or cache-key mismatch is fatal and
# can never fall through to the multi-hour wheel-builder stage.
if [ "$WHEEL_SOURCE" = "prebuilt-wheelhouse" ]; then
  prepare_wheel_artifact_io
  download_wheel_artifact "$PREBUILT_WHEEL_ARTIFACT_URI" /tmp/grug-vllm-wheels.tar.gz

  mkdir -p /tmp/grug-wheel-artifact /app/docker/wheelhouse
  tar -xzf /tmp/grug-vllm-wheels.tar.gz -C /tmp/grug-wheel-artifact
  WHEEL_DIR=/tmp/grug-wheel-artifact/wheels
  test -s "$WHEEL_DIR/MANIFEST"
  DOCKERFILE_PATH="$DOCKERFILE"
  if [[ "$DOCKERFILE_PATH" != /* ]]; then DOCKERFILE_PATH="/app/$DOCKERFILE_PATH"; fi
  dockerfile_arg() {
    sed -n "s/^ARG $1=//p" "$DOCKERFILE_PATH" | head -n 1 | tr -d '"'
  }
  bash /app/docker/write_gpu_rl_wheel_manifest.sh /tmp/expected-grug-wheel-manifest \
    "$(dockerfile_arg VLLM_FORK_COMMIT)" \
    "$(dockerfile_arg FLASH_ATTN_VERSION)" \
    "$(dockerfile_arg TORCH_VERSION)" \
    "$(dockerfile_arg BASE_IMAGE)" \
    "$(dockerfile_arg WHEEL_PYTHON_VERSION)" \
    "$(dockerfile_arg TARGET_PLATFORM)" \
    "$(dockerfile_arg TORCH_CUDA_ARCH_LIST)"
  cmp /tmp/expected-grug-wheel-manifest "$WHEEL_DIR/MANIFEST"
  test "$(find "$WHEEL_DIR" -maxdepth 1 -type f -name 'vllm-*.whl' | wc -l)" -eq 1
  test "$(find "$WHEEL_DIR" -maxdepth 1 -type f -name 'flash_attn-*.whl' | wc -l)" -eq 1
  install -m 0644 "$WHEEL_DIR/MANIFEST" /app/docker/wheelhouse/MANIFEST
  find "$WHEEL_DIR" -maxdepth 1 -type f -name '*.whl' -exec install -m 0644 '{}' /app/docker/wheelhouse/ \;
  echo "validated and staged prebuilt Grug wheelhouse"
fi

# --- 4. run kaniko ---
# --skip-unused-stages prunes the wheel-builder (nvcc) stage on the prebuilt path.
exec /kaniko/executor \
  --context dir:///app \
  --dockerfile "$DOCKERFILE" \
  --build-arg WHEEL_SOURCE="$WHEEL_SOURCE" \
  --build-arg INSTALL_MEGATRON="$INSTALL_MEGATRON" \
  --build-arg GITSHA="$GITSHA" \
  --build-arg HARBOR_COMMIT="$HARBOR_COMMIT" \
  --skip-unused-stages \
  "${SNAPSHOT_FLAGS[@]}" \
  --compressed-caching=false \
  "${CACHE_FLAGS[@]}" \
  "${FLOATING_DEST_FLAGS[@]}" \
  --destination "${DEST_PINNED}"
