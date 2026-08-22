"""Serving-oriented paged KV cache and single-token ragged decode attention."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .ops import paged_decode_attention_raw, paged_kv_append_raw


@dataclass
class PagedDecodeWorkspace:
    """Caller-owned split-KV reduction buffers, safe to reuse in CUDA Graphs."""

    partial_output: Tensor
    partial_max: Tensor
    partial_sum: Tensor
    output: Tensor

    @classmethod
    def allocate(
        cls, batch: int, query_heads: int, head_dim: int, num_splits: int,
        *, device: torch.device | str, dtype: torch.dtype,
    ) -> "PagedDecodeWorkspace":
        if min(batch, query_heads, head_dim, num_splits) < 1:
            raise ValueError("workspace dimensions must be positive")
        return cls(
            torch.empty(batch, query_heads, num_splits, head_dim, device=device, dtype=torch.float32),
            torch.empty(batch, query_heads, num_splits, device=device, dtype=torch.float32),
            torch.empty(batch, query_heads, num_splits, device=device, dtype=torch.float32),
            torch.empty(batch, query_heads, head_dim, device=device, dtype=dtype),
        )

    @property
    def num_splits(self) -> int:
        return self.partial_output.shape[2]


@dataclass
class PagedKVCache:
    """Physical KV pages plus a per-request logical-to-physical block table."""

    key: Tensor
    value: Tensor
    key_scales: Tensor
    value_scales: Tensor
    block_tables: Tensor
    sequence_lengths: Tensor

    @classmethod
    def allocate(
        cls, *, num_blocks: int, block_size: int, kv_heads: int, head_dim: int,
        block_tables: Tensor, sequence_lengths: Tensor,
        dtype: torch.dtype = torch.float16, quantized: bool = False,
    ) -> "PagedKVCache":
        if min(num_blocks, block_size, kv_heads, head_dim) < 1:
            raise ValueError("cache dimensions must be positive")
        device = block_tables.device
        cache_dtype = torch.int8 if quantized else dtype
        shape = (num_blocks, block_size, kv_heads, head_dim)
        scale_shape = shape[:-1] if quantized else (0,)
        return cls(
            torch.empty(shape, device=device, dtype=cache_dtype),
            torch.empty(shape, device=device, dtype=cache_dtype),
            torch.empty(scale_shape, device=device, dtype=torch.float32),
            torch.empty(scale_shape, device=device, dtype=torch.float32),
            block_tables.to(device=device, dtype=torch.long).contiguous(),
            sequence_lengths.to(device=device, dtype=torch.long).contiguous(),
        )

    @property
    def quantized(self) -> bool:
        return self.key.dtype == torch.int8

    @torch.no_grad()
    def append(self, key: Tensor, value: Tensor, positions: Tensor | None = None) -> None:
        """Append one token per request; allocation and length advance remain explicit."""
        if positions is None:
            positions = self.sequence_lengths
        paged_kv_append_raw(
            self.key, self.value, self.key_scales, self.value_scales,
            self.block_tables, positions.contiguous(), key.contiguous(), value.contiguous(),
        )

    @torch.no_grad()
    def advance_(self, active: Tensor | None = None) -> None:
        """Advance logical lengths on-device after a successful append."""
        if active is None:
            self.sequence_lengths.add_(1)
        else:
            self.sequence_lengths.add_(active.to(self.sequence_lengths.dtype))


@torch.no_grad()
def paged_decode_attention(
    query: Tensor, cache: PagedKVCache, workspace: PagedDecodeWorkspace | None = None,
    *, num_splits: int = 1, scale: float | None = None,
) -> Tensor:
    """Attend one query token per ragged request over a paged MHA/GQA/MQA cache."""
    if query.ndim != 3:
        raise ValueError("query must have shape [batch, query_heads, head_dim]")
    batch, query_heads, head_dim = query.shape
    kv_heads = cache.key.shape[2]
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    if workspace is None:
        workspace = PagedDecodeWorkspace.allocate(
            batch, query_heads, head_dim, num_splits, device=query.device, dtype=query.dtype
        )
    if workspace.partial_output.shape != (batch, query_heads, num_splits, head_dim):
        raise ValueError("workspace shape does not match query and num_splits")
    attention_scale = float(scale if scale is not None else 1.0 / math.sqrt(head_dim))
    paged_decode_attention_raw(
        query.contiguous(), cache.key, cache.value, cache.key_scales, cache.value_scales,
        cache.block_tables, cache.sequence_lengths, workspace.partial_output,
        workspace.partial_max, workspace.partial_sum, workspace.output,
        num_splits, attention_scale,
    )
    return workspace.output


def prefill_attention(
    query: Tensor, key: Tensor, value: Tensor, *, causal: bool = True,
    scale: float | None = None,
) -> Tensor:
    """Explicit dense prefill API; decode callers should use paged_decode_attention."""
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("query/key/value must be [batch, heads, sequence, head_dim]")
    if query.shape[1] % key.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    if key.shape[1] != query.shape[1]:
        repeats = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=causal, scale=scale,
    )
