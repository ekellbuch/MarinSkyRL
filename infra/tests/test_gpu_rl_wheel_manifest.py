import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
MANIFEST_SCRIPT = REPO_ROOT / "docker" / "write_gpu_rl_wheel_manifest.sh"


def test_wheel_manifest_records_compiled_abi(tmp_path: Path) -> None:
    output = tmp_path / "MANIFEST"

    subprocess.run(
        [
            MANIFEST_SCRIPT,
            output,
            "4b55591306c934cdc21461f091c9ea22ad008007",
            "2.8.3",
            "2.11.0",
            "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04",
            "3.12",
            "linux_x86_64",
            "8.0;9.0",
        ],
        check=True,
    )

    assert output.read_text() == (
        "VLLM_FORK_COMMIT=4b55591306c934cdc21461f091c9ea22ad008007\n"
        "FLASH_ATTN_VERSION=2.8.3\n"
        "TORCH_VERSION=2.11.0\n"
        "TORCH_CUDA_ARCH_LIST=8.0;9.0\n"
        "CUDA=12.8 PY=cp312 PLATFORM=linux_x86_64\n"
    )


def test_wheel_manifest_rejects_base_without_cuda_version(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            MANIFEST_SCRIPT,
            tmp_path / "MANIFEST",
            "commit",
            "2.8.3",
            "2.11.0",
            "ubuntu:22.04",
            "3.12",
            "linux_x86_64",
            "9.0",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
