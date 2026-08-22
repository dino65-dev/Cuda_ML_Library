"""CUDA Event benchmarks for paged decode and DSpark graph replay."""

from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from cuda_ml_decode import PagedDecodeWorkspace, PagedKVCache, paged_decode_attention
from DSpark import DSparkGreedyGraph, DSparkMarkovHead


def latency_us(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup): function()
    torch.cuda.synchronize()
    starts, ends = [], []
    lead = torch.cuda.Event(); lead.record()
    for _ in range(iterations):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record(); function(); end.record()
        starts.append(start); ends.append(end)
    ends[-1].synchronize()
    return sum(start.elapsed_time(end) for start, end in zip(starts, ends)) * 1000 / iterations


def build_cache(sequence: int, *, quantized: bool):
    block_size, heads, dim = 16, 2, 64
    blocks = math.ceil(sequence / block_size)
    table = torch.arange(blocks, device="cuda", dtype=torch.long)[None]
    cache = PagedKVCache.allocate(
        num_blocks=blocks, block_size=block_size, kv_heads=heads, head_dim=dim,
        block_tables=table, sequence_lengths=torch.tensor([sequence], device="cuda"),
        dtype=torch.float16, quantized=quantized,
    )
    dense_key = torch.randn(1, heads, sequence, dim, device="cuda", dtype=torch.float16)
    dense_value = torch.randn_like(dense_key)
    page_key = dense_key.transpose(1, 2).reshape(blocks, block_size, heads, dim)
    page_value = dense_value.transpose(1, 2).reshape_as(page_key)
    if quantized:
        ks = page_key.float().abs().amax(-1).div(127).clamp_min(1e-12)
        vs = page_value.float().abs().amax(-1).div(127).clamp_min(1e-12)
        cache.key.copy_((page_key.float() / ks[..., None]).round().clamp(-127, 127).to(torch.int8))
        cache.value.copy_((page_value.float() / vs[..., None]).round().clamp(-127, 127).to(torch.int8))
        cache.key_scales.copy_(ks); cache.value_scales.copy_(vs)
    else:
        cache.key.copy_(page_key); cache.value.copy_(page_value)
    return cache, dense_key, dense_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    torch.manual_seed(34)
    records = []
    for sequence in (128, 512, 2048):
        query = torch.randn(1, 8, 64, device="cuda", dtype=torch.float16)
        float_cache, dense_key, dense_value = build_cache(sequence, quantized=False)
        int8_cache, _, _ = build_cache(sequence, quantized=True)
        splits = min(8, max(1, math.ceil(sequence / 256)))
        float_workspace = PagedDecodeWorkspace.allocate(1, 8, 64, splits, device="cuda", dtype=torch.float16)
        int8_workspace = PagedDecodeWorkspace.allocate(1, 8, 64, splits, device="cuda", dtype=torch.float16)
        repeated_key = dense_key.repeat_interleave(4, dim=1)
        repeated_value = dense_value.repeat_interleave(4, dim=1)
        baseline = lambda: F.scaled_dot_product_attention(
            query[:, :, None, :], repeated_key, repeated_value, dropout_p=0.0
        )
        paged_float = lambda: paged_decode_attention(query, float_cache, float_workspace, num_splits=splits)
        paged_int8 = lambda: paged_decode_attention(query, int8_cache, int8_workspace, num_splits=splits)
        baseline_us = latency_us(baseline, args.warmup, args.iterations)
        for name, function in (("paged_fp16", paged_float), ("paged_int8", paged_int8)):
            value = latency_us(function, args.warmup, args.iterations)
            records.append({
                "family": "rank3_paged_decode", "variant": name, "sequence": sequence,
                "batch": 1, "query_heads": 8, "kv_heads": 2, "head_dim": 64,
                "splits": splits, "latency_us": value,
                "dense_sdpa_latency_us": baseline_us, "speedup_vs_dense_sdpa": baseline_us / value,
            })

    vocab, rank, batch, proposal = 4096, 64, 1, 7
    head = DSparkMarkovHead(vocab, rank).cuda().half().eval()
    base = torch.randn(batch, proposal, vocab, device="cuda", dtype=torch.float16)
    previous = torch.randint(vocab, (batch,), device="cuda")
    graph = DSparkGreedyGraph(head, batch=batch, proposal_length=proposal, dtype=torch.float16)
    eager = lambda: head.sample_block(base, previous)
    replay = lambda: graph.replay(base, previous)
    eager_us = latency_us(eager, args.warmup, args.iterations)
    graph_us = latency_us(replay, args.warmup, args.iterations)
    records.append({
        "family": "rank4_dspark_block", "variant": "cuda_graph_replay",
        "batch": batch, "proposal_length": proposal, "vocab": vocab, "rank": rank,
        "latency_us": graph_us, "eager_latency_us": eager_us,
        "speedup_vs_eager": eager_us / graph_us,
    })
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(), "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__, "warmup": args.warmup, "iterations": args.iterations,
        "records": records,
    }, indent=2))


if __name__ == "__main__":
    main()
