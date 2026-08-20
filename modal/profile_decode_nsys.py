"""Capture a real Nsight Systems trace for Decode Kernels on Modal A10G.

Run from the repository root, then download the resulting directory:

    .venv/bin/python -m modal run modal/profile_decode_nsys.py
    .venv/bin/python -m modal volume get cuda-ml-nsys / artifacts/nsys --force
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REMOTE_PACKAGE = "/opt/cuda_ml/decode_kernels"
ARTIFACT_ROOT = "/artifacts"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel")
    .pip_install("ninja>=1.11")
    .apt_install("gnupg", "ca-certificates", "wget")
    .run_commands(
        "echo 'deb https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /' > /etc/apt/sources.list.d/nvidia-devtools.list",
        "wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add -",
        "apt-get update",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nsight-systems-cli",
        "nsys --version",
    )
    .add_local_dir(REPOSITORY_ROOT / "decode_kernels", remote_path=REMOTE_PACKAGE, copy=True)
    .run_commands(
        f"cd {REMOTE_PACKAGE} && python -m pip install --no-build-isolation --no-deps .",
        env={"TORCH_CUDA_ARCH_LIST": "8.6", "MAX_JOBS": "4"},
    )
)

volume = modal.Volume.from_name("cuda-ml-nsys", create_if_missing=True)
app = modal.App("cuda-ml-decode-nsys", image=image)


def run_checked(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout + completed.stderr


@app.function(
    gpu="A10G",
    cpu=4.0,
    memory=16384,
    timeout=1_200,
    volumes={ARTIFACT_ROOT: volume},
)
def capture_nsys(iterations: int) -> dict:
    report_base = f"{ARTIFACT_ROOT}/decode_a10_nsys"
    report = f"{report_base}.nsys-rep"
    sqlite = f"{report_base}.sqlite"
    stats = f"{ARTIFACT_ROOT}/decode_a10_nsys_stats.csv"

    profile_command = [
            "nsys",
            "profile",
            "--trace=cuda-sw,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-event-trace=false",
            "--force-overwrite=true",
            "--output",
            report_base,
            "python",
            f"{REMOTE_PACKAGE}/benchmarks/nsys_workload.py",
            "--traces",
            f"{REMOTE_PACKAGE}/benchmarks/decode_traces.json",
            "--iterations",
            str(iterations),
        ]
    profile_completed = subprocess.run(profile_command, check=False, capture_output=True, text=True)
    profile_output = profile_completed.stdout + profile_completed.stderr
    if profile_completed.returncode != 0 and not Path(report).exists():
        raise RuntimeError(
            f"profile failed ({profile_completed.returncode}) without generating {report}:\n{profile_output}"
        )
    volume.commit()
    export_output = run_checked(
        ["nsys", "export", "--type=sqlite", "--force-overwrite=true", "--output", sqlite, report]
    )
    volume.commit()
    for old_stats in Path(ARTIFACT_ROOT).glob("decode_a10_nsys_stats.csv_*.csv"):
        old_stats.unlink()
    stats_output = run_checked(
        [
            "nsys",
            "stats",
            "--force-export=true",
            "--report",
            "cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,nvtx_gpu_proj_sum",
            "--format",
            "csv",
            "--output",
            stats,
            report,
        ]
    )
    version = run_checked(["nsys", "--version"]).strip()
    volume.commit()
    return {
        "nsys_version": version,
        "iterations_per_range": iterations,
        "files": [report, sqlite, stats],
        "profile_log": profile_output[-4000:],
        "export_log": export_output[-2000:],
        "stats_log": stats_output[-2000:],
    }


@app.local_entrypoint()
def main(iterations: int = 20):
    result = capture_nsys.remote(iterations)
    print(result)
    print("Download with:")
    print(".venv/bin/python -m modal volume get cuda-ml-nsys / artifacts/nsys --force")
