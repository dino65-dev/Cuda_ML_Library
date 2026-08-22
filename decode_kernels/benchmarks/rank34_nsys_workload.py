"""Deterministic NVTX workload for Rank 3/4 Nsight Systems evidence."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cuda_ml_decode import PagedDecodeWorkspace, paged_decode_attention
from DSpark import DSparkGreedyGraph, DSparkMarkovHead
from run_rank34_benchmarks import build_cache


def main(iterations: int = 12) -> None:
    torch.manual_seed(3404)
    sequence = 2048
    query = torch.randn(1, 8, 64, device="cuda", dtype=torch.float16)
    float_cache, dense_key, dense_value = build_cache(sequence, quantized=False)
    int8_cache, _, _ = build_cache(sequence, quantized=True)
    float_workspace = PagedDecodeWorkspace.allocate(1, 8, 64, 8, device="cuda", dtype=torch.float16)
    int8_workspace = PagedDecodeWorkspace.allocate(1, 8, 64, 8, device="cuda", dtype=torch.float16)
    repeated_key = dense_key.repeat_interleave(4, dim=1)
    repeated_value = dense_value.repeat_interleave(4, dim=1)

    head = DSparkMarkovHead(4096, 64).cuda().half().eval()
    base = torch.randn(1, 7, 4096, device="cuda", dtype=torch.float16)
    previous = torch.randint(4096, (1,), device="cuda")
    graph = DSparkGreedyGraph(head, batch=1, proposal_length=7, dtype=torch.float16)
    for _ in range(4):
        paged_decode_attention(query, float_cache, float_workspace, num_splits=8)
        graph.replay(base, previous)
    torch.cuda.synchronize()

    for _ in range(iterations):
        with torch.cuda.nvtx.range("rank3/dense_sdpa/S2048"):
            F.scaled_dot_product_attention(query[:, :, None, :], repeated_key, repeated_value, dropout_p=0.0)
        with torch.cuda.nvtx.range("rank3/paged_fp16/S2048/split8"):
            paged_decode_attention(query, float_cache, float_workspace, num_splits=8)
        with torch.cuda.nvtx.range("rank3/paged_int8/S2048/split8"):
            paged_decode_attention(query, int8_cache, int8_workspace, num_splits=8)
        with torch.cuda.nvtx.range("rank4/dspark_eager/B1_K7_V4096_R64"):
            head.sample_block(base, previous)
        with torch.cuda.nvtx.range("rank4/dspark_graph/B1_K7_V4096_R64"):
            graph.replay(base, previous)
    torch.cuda.synchronize()


if __name__ == "__main__": main()
