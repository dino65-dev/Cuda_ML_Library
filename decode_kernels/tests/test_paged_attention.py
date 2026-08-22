from __future__ import annotations

import math

import pytest
import torch

from cuda_ml_decode import (
    PagedDecodeWorkspace,
    PagedKVCache,
    paged_decode_attention,
    prefill_attention,
)
from cuda_ml_decode import reference


def _case(device: str, *, quantized: bool = False, dtype=torch.float32):
    torch.manual_seed(91)
    tables = torch.tensor([[3, 0, 4], [2, 1, 5]], device=device, dtype=torch.long)
    lengths = torch.tensor([7, 11], device=device, dtype=torch.long)
    cache = PagedKVCache.allocate(
        num_blocks=6, block_size=4, kv_heads=2, head_dim=32,
        block_tables=tables, sequence_lengths=lengths,
        dtype=dtype, quantized=quantized,
    )
    keys = torch.randn(2, 12, 2, 32, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    for position in range(12):
        active = torch.tensor(
            [position if position < 7 else -1, position if position < 11 else -1],
            device=device, dtype=torch.long,
        )
        cache.append(keys[:, position], values[:, position], active)
    query = torch.randn(2, 8, 32, device=device, dtype=dtype)
    return cache, query


@pytest.mark.parametrize("quantized", [False, True])
def test_cpu_ragged_gqa_paged_decode_matches_reference(quantized):
    cache, query = _case("cpu", quantized=quantized)
    workspace = PagedDecodeWorkspace.allocate(2, 8, 32, 3, device="cpu", dtype=query.dtype)
    actual = paged_decode_attention(query, cache, workspace, num_splits=3)
    expected = torch.empty_like(query)
    reference.paged_decode_attention_out(
        query, cache.key, cache.value, cache.key_scales, cache.value_scales,
        cache.block_tables, cache.sequence_lengths, expected, 1 / math.sqrt(32),
    )
    tolerance = 0.035 if quantized else 1e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_prefill_is_explicit_and_supports_gqa():
    query = torch.randn(2, 8, 5, 32)
    key = torch.randn(2, 2, 5, 32)
    value = torch.randn_like(key)
    output = prefill_attention(query, key, value)
    assert output.shape == query.shape


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("quantized", [False, True])
def test_cuda_paged_decode_split_kv_matches_reference(dtype, quantized):
    cache, query = _case("cuda", quantized=quantized, dtype=dtype)
    workspace = PagedDecodeWorkspace.allocate(2, 8, 32, 4, device="cuda", dtype=dtype)
    actual = paged_decode_attention(query, cache, workspace, num_splits=4).clone()
    expected = torch.empty_like(query)
    reference.paged_decode_attention_out(
        query, cache.key, cache.value, cache.key_scales, cache.value_scales,
        cache.block_tables, cache.sequence_lengths, expected, 1 / math.sqrt(32),
    )
    tolerance = 0.04 if quantized else (0.004 if dtype == torch.float16 else 2e-5)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.cuda
def test_paged_append_attention_cuda_graph_replay():
    tables = torch.tensor([[0, 1]], device="cuda", dtype=torch.long)
    lengths = torch.tensor([3], device="cuda", dtype=torch.long)
    cache = PagedKVCache.allocate(
        num_blocks=2, block_size=4, kv_heads=1, head_dim=32,
        block_tables=tables, sequence_lengths=lengths, dtype=torch.float16,
    )
    cache.key.zero_(); cache.value.zero_()
    query = torch.randn(1, 4, 32, device="cuda", dtype=torch.float16)
    key = torch.randn(1, 1, 32, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    workspace = PagedDecodeWorkspace.allocate(1, 4, 32, 2, device="cuda", dtype=torch.float16)
    for _ in range(3):
        cache.append(key, value)
        paged_decode_attention(query, cache, workspace, num_splits=2)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cache.append(key, value)
        captured = paged_decode_attention(query, cache, workspace, num_splits=2)
    graph.replay()
    expected = torch.empty_like(query)
    reference.paged_decode_attention_out(
        query, cache.key, cache.value, cache.key_scales, cache.value_scales,
        cache.block_tables, cache.sequence_lengths, expected, 1 / math.sqrt(32),
    )
    torch.testing.assert_close(captured, expected, atol=4e-3, rtol=4e-3)
