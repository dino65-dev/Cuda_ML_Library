"""Dependency-free Rank 3/4 CUDA smoke/correctness validation."""

from __future__ import annotations

import json
import math

import torch

from cuda_ml_decode import PagedDecodeWorkspace, PagedKVCache, paged_decode_attention
from cuda_ml_decode import reference
from DSpark import DSparkMarkovHead, DSparkGreedyGraph, markov_greedy, verify_speculative
from DSpark.reference import markov_logits_reference


def main() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    torch.manual_seed(2026)
    evidence = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}

    rank3 = []
    for quantized in (False, True):
        tables = torch.tensor([[3, 0, 4], [2, 1, 5]], device=device, dtype=torch.long)
        lengths = torch.tensor([7, 11], device=device, dtype=torch.long)
        cache = PagedKVCache.allocate(
            num_blocks=6, block_size=4, kv_heads=2, head_dim=32,
            block_tables=tables, sequence_lengths=lengths,
            dtype=torch.float32, quantized=quantized,
        )
        keys = torch.randn(2, 11, 2, 32, device=device)
        values = torch.randn_like(keys)
        for position in range(11):
            positions = torch.tensor([position if position < 7 else -1, position], device=device)
            cache.append(keys[:, position], values[:, position], positions)
        query = torch.randn(2, 8, 32, device=device)
        workspace = PagedDecodeWorkspace.allocate(2, 8, 32, 4, device=device, dtype=query.dtype)
        actual = paged_decode_attention(query, cache, workspace, num_splits=4).clone()
        expected = torch.empty_like(query)
        reference.paged_decode_attention_out(
            query, cache.key, cache.value, cache.key_scales, cache.value_scales,
            cache.block_tables, cache.sequence_lengths, expected, 1 / math.sqrt(32),
        )
        error = (actual - expected).abs().max().item()
        assert error < (0.05 if quantized else 3e-5), error
        rank3.append({"quantized": quantized, "max_abs_error": error})

    tables = torch.tensor([[0, 1]], device=device)
    cache = PagedKVCache.allocate(
        num_blocks=2, block_size=4, kv_heads=1, head_dim=32,
        block_tables=tables, sequence_lengths=torch.tensor([3], device=device),
        dtype=torch.float32,
    )
    cache.key.zero_(); cache.value.zero_()
    query = torch.randn(1, 4, 32, device=device)
    key = torch.randn(1, 1, 32, device=device); value = torch.randn_like(key)
    workspace = PagedDecodeWorkspace.allocate(1, 4, 32, 2, device=device, dtype=query.dtype)
    for _ in range(3): cache.append(key, value); paged_decode_attention(query, cache, workspace, num_splits=2)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cache.append(key, value)
        graph_output = paged_decode_attention(query, cache, workspace, num_splits=2)
    graph.replay(); torch.cuda.synchronize()
    assert torch.isfinite(graph_output).all()
    evidence["rank3"] = rank3
    evidence["rank3_cuda_graph"] = "pass"

    vocab, rank, batch, proposal = 257, 16, 2, 4
    head = DSparkMarkovHead(vocab, rank).to(device).eval()
    base = torch.randn(batch, vocab, device=device)
    previous = torch.randint(vocab, (batch,), device=device)
    greedy = markov_greedy(base, previous, head.token_embedding, head.projection_t)
    expected = markov_logits_reference(base, previous, head.token_embedding, head.projection_t)
    markov_error = (greedy.corrected_logits - expected).abs().max().item()
    assert markov_error < 3e-5 and torch.equal(greedy.token_ids, expected.argmax(-1))
    graph_draft = DSparkGreedyGraph(head, batch=batch, proposal_length=proposal, dtype=torch.float32)
    block = torch.randn(batch, proposal, vocab, device=device)
    replayed = graph_draft.replay(block, previous)
    torch.cuda.synchronize()
    eager = head.sample_block(block, previous)
    assert torch.equal(replayed.token_ids, eager.token_ids)
    verification = verify_speculative(
        torch.tensor([[1, 2, 3], [4, 5, 6]], device=device),
        torch.tensor([[1, 0, 3, 7], [4, 5, 6, 8]], device=device),
        torch.tensor([3, 2], device=device, dtype=torch.int32),
    )
    assert verification.accepted_lengths.tolist() == [1, 2]
    evidence["rank4"] = {
        "markov_greedy_max_abs_error": markov_error,
        "greedy_graph": "pass",
        "speculative_verification": "pass",
    }
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
