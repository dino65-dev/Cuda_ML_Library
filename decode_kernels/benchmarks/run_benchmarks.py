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

import cuda_ml_decode as ops
from cuda_ml_decode import reference


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def time_cuda(
    function: Callable[[], object],
    warmup: int,
    iterations: int,
    cold_cache: bool,
) -> dict[str, float]:
    torch.cuda.synchronize()
    start_wall = time.perf_counter_ns()
    function()
    torch.cuda.synchronize()
    first_call_us = (time.perf_counter_ns() - start_wall) / 1_000.0

    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    flush = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8) if cold_cache else None
    event_pairs = []
    # Keep the GPU occupied while Python queues every measured event/function
    # pair. Otherwise a very short kernel can execute its start event before
    # the host has enqueued the kernel, and CUDA Event time incorrectly absorbs
    # Python dispatch latency. The sleep is before every start event and is not
    # part of any measurement.
    torch.cuda._sleep(50_000_000)
    for _ in range(iterations):
        if flush is not None:
            flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        event_pairs.append((start, end))
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) * 1_000.0 for start, end in event_pairs]
    return {
        "first_call_us": first_call_us,
        "mean_us": statistics.fmean(samples),
        "p20_us": percentile(samples, 0.20),
        "p50_us": percentile(samples, 0.50),
        "p80_us": percentile(samples, 0.80),
        "min_us": min(samples),
    }


def make_case(trace: dict) -> tuple[Callable[[], object], Callable[[], object], int | None]:
    dtype = DTYPES[trace["dtype"]]
    op = trace["op"]
    if op == "residual_rms_norm":
        shape = (trace["tokens"], trace["hidden"])
        input = torch.randn(shape, device="cuda", dtype=dtype)
        residual = torch.randn_like(input)
        weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
        custom = lambda: ops.residual_rms_norm(input, residual, weight)
        baseline = lambda: reference.residual_rms_norm(input, residual, weight)
        bytes_moved = (4 * input.numel() + weight.numel()) * input.element_size()
    elif op == "rms_norm_quantize":
        shape = (trace["tokens"], trace["hidden"])
        input = torch.randn(shape, device="cuda", dtype=dtype)
        weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
        custom = lambda: ops.rms_norm_quantize(input, weight)
        baseline = lambda: reference.rms_norm_quantize(input, weight)
        bytes_moved = (input.numel() + weight.numel()) * input.element_size() + input.numel()
    elif op == "rope_qk_norm":
        q_shape = (trace["batch"], trace["sequence"], trace["q_heads"], trace["head_dim"])
        k_shape = (trace["batch"], trace["sequence"], trace["kv_heads"], trace["head_dim"])
        q = torch.randn(q_shape, device="cuda", dtype=dtype)
        k = torch.randn(k_shape, device="cuda", dtype=dtype)
        q_weight = torch.randn(trace["head_dim"], device="cuda", dtype=dtype)
        k_weight = torch.randn_like(q_weight)
        angle = torch.randn(trace["sequence"], trace["head_dim"] // 2, device="cuda")
        cos, sin = angle.cos().to(dtype), angle.sin().to(dtype)
        custom = lambda: ops.rope_qk_norm(q, k, q_weight, k_weight, cos, sin)
        baseline = lambda: reference.rope_qk_norm(q, k, q_weight, k_weight, cos, sin)
        bytes_moved = (2 * (q.numel() + k.numel()) + q_weight.numel() + k_weight.numel() + cos.numel() + sin.numel()) * q.element_size()
    elif op == "kv_cache_append":
        cache_shape = (trace["capacity"], trace["kv_heads"], trace["head_dim"])
        update_shape = (trace["tokens"], trace["kv_heads"], trace["head_dim"])
        key_cache = torch.empty(cache_shape, device="cuda", dtype=dtype)
        value_cache = torch.empty_like(key_cache)
        key = torch.randn(update_shape, device="cuda", dtype=dtype)
        value = torch.randn_like(key)
        slots = torch.arange(trace["tokens"], device="cuda", dtype=torch.long)
        custom = lambda: ops.kv_cache_append(key_cache, value_cache, slots, key, value)
        baseline = lambda: reference.kv_cache_append(key_cache, value_cache, slots, key, value)
        bytes_moved = 4 * key.numel() * key.element_size() + slots.numel() * slots.element_size()
    elif op == "bias_swiglu":
        shape = (trace["tokens"], trace["hidden"])
        gate = torch.randn(shape, device="cuda", dtype=dtype)
        up = torch.randn_like(gate)
        gate_bias = torch.randn(shape[-1], device="cuda", dtype=dtype)
        up_bias = torch.randn_like(gate_bias)
        custom = lambda: ops.bias_swiglu(gate, up, gate_bias, up_bias)
        baseline = lambda: reference.bias_swiglu(gate, up, gate_bias, up_bias)
        bytes_moved = (3 * gate.numel() + gate_bias.numel() + up_bias.numel()) * gate.element_size()
    elif op == "sample_logits":
        logits = torch.randn(trace["tokens"], trace["vocabulary"], device="cuda", dtype=dtype)
        uniforms = torch.rand(trace["tokens"], device="cuda")
        arguments = (logits, uniforms, 1.0, trace["top_k"], trace["top_p"], trace["min_p"])
        custom = lambda: ops.sample_logits(*arguments)
        baseline = custom
        bytes_moved = None
    elif op == "small_n_linear":
        input = torch.randn(trace["tokens"], trace["input_features"], device="cuda", dtype=dtype)
        weight = torch.randn(trace["output_features"], trace["input_features"], device="cuda", dtype=dtype)
        bias = torch.randn(trace["output_features"], device="cuda", dtype=dtype)
        custom = lambda: ops.small_n_linear(input, weight, bias)
        baseline = custom
        bytes_moved = (input.numel() + weight.numel() + bias.numel() + trace["tokens"] * trace["output_features"]) * input.element_size()
    else:
        raise ValueError(f"unknown operator {op}")
    return custom, baseline, bytes_moved


def metadata() -> dict:
    properties = torch.cuda.get_device_properties(0)
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        driver = "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "multiprocessors": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "l2_cache_bytes": getattr(properties, "L2_cache_size", None),
        "driver_and_memory": driver,
    }


def run(trace_path: Path, warmup: int, iterations: int) -> dict:
    payload = json.loads(trace_path.read_text())
    results = []
    for trace in payload["traces"]:
        custom, baseline, bytes_moved = make_case(trace)
        warm = time_cuda(custom, warmup, iterations, cold_cache=False)
        cold = time_cuda(custom, warmup, iterations, cold_cache=True)
        baseline_warm = time_cuda(baseline, warmup, iterations, cold_cache=False)
        record = {
            "trace": trace,
            "custom_warm": warm,
            "custom_cold_cache": cold,
            "reference_warm": baseline_warm,
            "speedup_p50": baseline_warm["p50_us"] / warm["p50_us"],
        }
        if bytes_moved is not None:
            record["estimated_bytes"] = bytes_moved
            record["effective_bandwidth_gbps"] = bytes_moved / (warm["p50_us"] * 1_000.0)
        results.append(record)
    return {
        "schema_version": 1,
        "metadata": metadata(),
        "trace_provenance": payload["provenance"],
        "timing": {
            "warmup": warmup,
            "iterations": iterations,
            "method": "Device-only queued CUDA Event pairs behind an untimed GPU lead-in; p20/p50/p80; 64 MiB L2 flush before each cold-cache sample",
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, default=Path(__file__).with_name("decode_traces.json"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.traces, args.warmup, args.iterations)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
