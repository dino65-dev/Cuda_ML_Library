"""GPU correctness, distortion, and direct-packed STQ1_0 GEMV benchmark."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from mixstq import amax_stq, quantize_weighted_stq, unpack_stq, weighted_squared_error
from mixstq.cuda import stq_gemv


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def time_cuda(function: Callable[[], object], *, warmup: int, iterations: int) -> dict[str, float]:
    torch.cuda.synchronize()
    start_wall = time.perf_counter_ns()
    function()
    torch.cuda.synchronize()
    first_call_us = (time.perf_counter_ns() - start_wall) / 1_000
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    # Queue all events behind an untimed lead-in.  This prevents device-side
    # event timing from absorbing Python launch gaps on very short kernels.
    torch.cuda._sleep(50_000_000)
    pairs = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        pairs.append((start, end))
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) * 1_000 for start, end in pairs]
    return {
        "first_call_us": first_call_us,
        "mean_us": statistics.fmean(samples),
        "p20_us": percentile(samples, 0.2),
        "p50_us": percentile(samples, 0.5),
        "p80_us": percentile(samples, 0.8),
        "min_us": min(samples),
    }


def metadata() -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "multiprocessors": properties.multi_processor_count,
        "memory_bytes": properties.total_memory,
        "driver": driver,
    }


def make_weight(rows: int, columns: int) -> torch.Tensor:
    """A deterministic MoE-like synthetic weight with rare scale outliers."""

    weight = torch.randn(rows, columns, device="cuda", dtype=torch.float32) * 0.16
    # PTQ failure mode: rare values should not set every 256-wide scale.
    row_indices = torch.arange(rows, device="cuda")
    weight[row_indices, (row_indices * 131) % columns] *= 18.0
    return weight


def run_case(rows: int, columns: int, *, warmup: int, iterations: int) -> dict[str, object]:
    weight = make_weight(rows, columns)
    calibration = torch.randn(192, columns, device="cuda", dtype=torch.float32)
    # Give the synthetic calibration distribution a nonuniform activation
    # profile, exactly the information an imatrix diagonal supplies.
    calibration[:, ::29] *= 5.0
    importance = calibration.square().mean(dim=0).clamp_min(1e-6)

    baseline = amax_stq(weight)
    weighted = quantize_weighted_stq(weight, importance, rounds=3)
    dense_stq = unpack_stq(weighted).contiguous()
    activation = torch.randn(1, columns, device="cuda", dtype=torch.float32)
    direct = stq_gemv(weighted, activation)
    reference = F.linear(activation, dense_stq)
    max_abs_error = float((direct - reference).abs().max().item())

    direct_timing = time_cuda(lambda: stq_gemv(weighted, activation), warmup=warmup, iterations=iterations)
    dense_timing = time_cuda(lambda: F.linear(activation, weight), warmup=warmup, iterations=iterations)
    dense_stq_timing = time_cuda(lambda: F.linear(activation, dense_stq), warmup=warmup, iterations=iterations)
    parameters = rows * columns
    direct_p50 = direct_timing["p50_us"]
    return {
        "shape": {"batch": 1, "out_features": rows, "in_features": columns},
        "compression": {
            "dense_fp32_bytes": parameters * 4,
            "packed_stq_bytes": weighted.storage_bytes,
            "bits_per_weight": weighted.bits_per_weight,
            "storage_reduction_vs_fp32": parameters * 4 / weighted.storage_bytes,
        },
        "distortion": {
            "amax_weighted_error": weighted_squared_error(weight, baseline, importance),
            "weighted_stq_weighted_error": weighted_squared_error(weight, weighted, importance),
            "weighted_error_reduction_vs_amax": 1
            - weighted_squared_error(weight, weighted, importance) / weighted_squared_error(weight, baseline, importance),
        },
        "correctness": {"direct_vs_materialized_max_abs_error": max_abs_error},
        "timing_us": {
            "packed_stq_direct": direct_timing,
            "dense_fp32_cublas": dense_timing,
            "dense_materialized_stq_cublas": dense_stq_timing,
        },
        "packed_effective_weight_read_gbps_at_p50": weighted.storage_bytes / (direct_p50 * 1_000),
        "direct_speedup_vs_dense_fp32_p50": dense_timing["p50_us"] / direct_p50,
        "direct_speedup_vs_dense_materialized_stq_p50": dense_stq_timing["p50_us"] / direct_p50,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(20260904)
    result = {
        "schema_version": 1,
        "metadata": metadata(),
        "method": "Device-only CUDA Event timings queued behind an untimed GPU lead-in. Warm cache, batch=1 GEMV. The packed kernel reads 42-byte STQ blocks directly; no dense dequantized STQ matrix is materialized.",
        "results": [
            run_case(4096, 4096, warmup=args.warmup, iterations=args.iterations),
            run_case(14336, 4096, warmup=args.warmup, iterations=args.iterations),
        ],
        "limitations": [
            "Synthetic weights and calibration activations test the format/algorithm, not model quality on HY4 or any language benchmark.",
            "The direct kernel is a readable experimental GEMV kernel. It is not a tuned llama.cpp MUL_MAT_ID or an end-to-end token/s benchmark.",
            "The GTX 1050 Ti is Pascal (SM 6.1), so Tensor-Core-oriented results should not be generalized to modern GPUs.",
        ],
    }
    rendered = json.dumps(result, indent=2)
    args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
