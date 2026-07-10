"""Microbenchmarks for the DSpark CUDA primitives."""

from __future__ import annotations

import argparse
import statistics

import torch

from .dspark import cuda_extension_available, markov_logits, schedule
from .reference import markov_logits_reference, schedule_reference


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _time_cuda(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1_000.0)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--proposal-length", type=int, default=7)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    if not cuda_extension_available():
        raise SystemExit("Build DSpark/install.sh before running the benchmark")

    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    logits = torch.randn(
        args.requests,
        args.vocab_size,
        device=device,
        dtype=dtype,
    )
    ids = torch.randint(args.vocab_size, (args.requests,), device=device)
    embedding = torch.randn(
        args.vocab_size,
        args.rank,
        device=device,
        dtype=dtype,
    )
    projection_t = torch.randn(
        args.rank,
        args.vocab_size,
        device=device,
        dtype=dtype,
    )

    confidence = torch.randn(
        args.requests,
        args.proposal_length,
        device=device,
        dtype=dtype,
    )
    temperatures = torch.ones(args.proposal_length, device=device)
    curve_size = args.requests * (args.proposal_length + 1) + 1
    curve_index = torch.arange(curve_size, device=device, dtype=torch.float32)
    step_curve = 1_000.0 / (1.0 + curve_index / 256.0)

    with torch.inference_mode():
        custom_markov_us = _time_cuda(
            lambda: markov_logits(logits, ids, embedding, projection_t),
            args.warmup,
            args.iterations,
        )
        torch_markov_us = _time_cuda(
            lambda: markov_logits_reference(logits, ids, embedding, projection_t),
            args.warmup,
            args.iterations,
        )
        custom_schedule_us = _time_cuda(
            lambda: schedule(confidence, step_curve, temperatures),
            args.warmup,
            args.iterations,
        )
        torch_schedule_us = _time_cuda(
            lambda: schedule_reference(confidence, step_curve, temperatures),
            args.warmup,
            args.iterations,
        )

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"dtype={args.dtype}, requests={args.requests}")
    print(f"Markov CUDA:     {custom_markov_us:9.2f} us")
    print(f"Markov PyTorch:  {torch_markov_us:9.2f} us")
    print(f"Scheduler CUDA:  {custom_schedule_us:9.2f} us")
    print(f"Scheduler Torch: {torch_schedule_us:9.2f} us")


if __name__ == "__main__":
    main()
