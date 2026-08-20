"""NVTX-instrumented decode workload for Nsight Systems.

The workload intentionally profiles both the CUDA implementation and its
independent PyTorch reference under separate NVTX ranges. Random tensors are
constructed before the capture window, and each range uses a fixed iteration
count so launch count and per-iteration GPU time can be compared directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from run_benchmarks import make_case


def nvtx_range(name: str):
    class Range:
        def __enter__(self):
            torch.cuda.nvtx.range_push(name)

        def __exit__(self, exc_type, exc, traceback):
            torch.cuda.nvtx.range_pop()

    return Range()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, default=Path(__file__).with_name("decode_traces.json"))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    payload = json.loads(args.traces.read_text())
    cases = []
    for trace in payload["traces"]:
        custom, reference, _ = make_case(trace)
        cases.append((trace["name"], custom, reference))

    for _, custom, reference in cases:
        for _ in range(args.warmup):
            custom()
            reference()
    torch.cuda.synchronize()

    with nvtx_range("decode_profile_window"):
        for name, custom, reference in cases:
            with nvtx_range(f"custom::{name}"):
                for iteration in range(args.iterations):
                    with nvtx_range(f"custom_iteration::{name}::{iteration}"):
                        custom()
            with nvtx_range(f"reference::{name}"):
                for iteration in range(args.iterations):
                    with nvtx_range(f"reference_iteration::{name}::{iteration}"):
                        reference()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
