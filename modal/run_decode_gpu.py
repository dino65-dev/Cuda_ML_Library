"""Build, test, and benchmark decode_kernels on a Modal A10G GPU.

Run from the repository root:
    .venv/bin/python -m modal run modal/run_decode_gpu.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REMOTE_PACKAGE = "/opt/cuda_ml/decode_kernels"
REMOTE_DSPARK = "/opt/cuda_ml/DSpark"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel")
    .pip_install("pytest>=8.3", "ninja>=1.11")
    .add_local_dir(REPOSITORY_ROOT / "decode_kernels", remote_path=REMOTE_PACKAGE, copy=True)
    .add_local_dir(REPOSITORY_ROOT / "DSpark", remote_path=REMOTE_DSPARK, copy=True)
    .run_commands(
        f"cd {REMOTE_PACKAGE} && python -m pip install --no-build-isolation --no-deps .",
        f"cd {REMOTE_DSPARK} && python -m pip install --no-build-isolation --no-deps .",
        env={"TORCH_CUDA_ARCH_LIST": "8.6", "MAX_JOBS": "4"},
    )
)

app = modal.App("cuda-ml-decode-validation", image=image)


def run_checked(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout


@app.function(gpu="A10G", cpu=4.0, memory=16384, timeout=1_200)
def validate_on_gpu(warmup: int, iterations: int) -> dict:
    test_output = run_checked(
        ["python", "-m", "pytest", "-q", f"{REMOTE_PACKAGE}/tests", f"{REMOTE_DSPARK}/tests"]
    )
    benchmark_output = run_checked(
        [
            "python",
            f"{REMOTE_PACKAGE}/benchmarks/run_benchmarks.py",
            "--traces",
            f"{REMOTE_PACKAGE}/benchmarks/decode_traces.json",
            "--warmup",
            str(warmup),
            "--iterations",
            str(iterations),
        ]
    )
    rank34_benchmark_output = run_checked(
        [
            "python",
            f"{REMOTE_PACKAGE}/benchmarks/run_rank34_benchmarks.py",
            "--warmup",
            str(warmup),
            "--iterations",
            str(iterations),
        ]
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "modal_gpu_request": "A10G",
        "tests": test_output.strip(),
        "benchmark": json.loads(benchmark_output),
        "rank34_benchmark": json.loads(rank34_benchmark_output),
    }


@app.local_entrypoint()
def main(warmup: int = 10, iterations: int = 30):
    result = validate_on_gpu.remote(warmup, iterations)
    artifact = REPOSITORY_ROOT / "artifacts" / "modal_a10g_rank1_to_rank4_validation.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(result, indent=2) + "\n")
    print(f"GPU validation complete: {result['tests']}")
    print(f"Evidence written to {artifact}")
