"""High-precision PyTorch references for every public decode operator."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def _rms_norm_float(input: Tensor, weight: Tensor, eps: float) -> Tensor:
    value = input.float()
    inv_rms = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    return (value * inv_rms * weight.float()).to(input.dtype)


def residual_rms_norm(
    input: Tensor, residual: Tensor, weight: Tensor, eps: float = 1.0e-6
) -> tuple[Tensor, Tensor]:
    summed_float = input.float() + residual.float()
    inv_rms = torch.rsqrt(summed_float.square().mean(dim=-1, keepdim=True) + eps)
    output = (summed_float * inv_rms * weight.float()).to(input.dtype)
    return output, summed_float.to(input.dtype)


def rms_norm_quantize(
    input: Tensor, weight: Tensor, eps: float = 1.0e-6
) -> tuple[Tensor, Tensor]:
    normalized = _rms_norm_float(input, weight, eps).float()
    scale = normalized.abs().amax(dim=-1).div(127.0).clamp_min(1.0e-12)
    quantized = torch.round(normalized / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def rope_qk_norm(
    q: Tensor,
    k: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    eps: float = 1.0e-6,
) -> tuple[Tensor, Tensor]:
    def normalize_and_rotate(value: Tensor, weight: Tensor) -> Tensor:
        normalized = _rms_norm_float(value, weight, eps).float()
        first, second = normalized.chunk(2, dim=-1)
        cosine = cos.float().view(1, cos.shape[0], 1, cos.shape[1])
        sine = sin.float().view(1, sin.shape[0], 1, sin.shape[1])
        return torch.cat((first * cosine - second * sine, first * sine + second * cosine), dim=-1).to(value.dtype)

    return normalize_and_rotate(q, q_weight), normalize_and_rotate(k, k_weight)


def kv_cache_append(
    key_cache: Tensor,
    value_cache: Tensor,
    slots: Tensor,
    key: Tensor,
    value: Tensor,
) -> None:
    valid = (slots >= 0) & (slots < key_cache.shape[0])
    valid_slots = slots[valid]
    key_cache.index_copy_(0, valid_slots, key[valid])
    value_cache.index_copy_(0, valid_slots, value[valid])


def bias_swiglu(
    gate: Tensor,
    up: Tensor,
    gate_bias: Optional[Tensor] = None,
    up_bias: Optional[Tensor] = None,
) -> Tensor:
    gate_float = gate.float()
    up_float = up.float()
    if gate_bias is not None:
        gate_float = gate_float + gate_bias.float()
    if up_bias is not None:
        up_float = up_float + up_bias.float()
    return (F.silu(gate_float) * up_float).to(gate.dtype)


def small_n_linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    activation: str = "none",
) -> Tensor:
    """Small-batch linear path with a composable epilogue.

    PyTorch/cuBLAS remains the measured baseline until a shape/architecture
    crossover demonstrates that a specialized GEMV is worthwhile.
    """
    output = F.linear(input, weight, bias)
    if residual is not None:
        output = output + residual
    if activation == "silu":
        output = F.silu(output)
    elif activation != "none":
        raise ValueError(f"unsupported activation: {activation}")
    return output


def sample_logits(
    logits: Tensor,
    uniforms: Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
) -> Tensor:
    """Deterministic, graph-capturable logits filtering and sampling.

    Random uniforms are explicit inputs so replaying a CUDA Graph never hides
    or accidentally freezes RNG state. The result has one token id per row.
    """
    if logits.ndim < 2:
        raise ValueError("logits must have shape [..., vocabulary]")
    if uniforms.shape != logits.shape[:-1]:
        raise ValueError("uniforms must have shape logits.shape[:-1]")
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if not 0.0 <= min_p <= 1.0:
        raise ValueError("min_p must be in [0, 1]")

    vocabulary = logits.shape[-1]
    keep = vocabulary if top_k <= 0 else min(top_k, vocabulary)
    sorted_logits, sorted_indices = torch.topk(logits.float() / temperature, keep, dim=-1)
    probabilities = torch.softmax(sorted_logits, dim=-1)

    if min_p > 0.0:
        probabilities = torch.where(
            probabilities >= probabilities[..., :1] * min_p,
            probabilities,
            torch.zeros_like(probabilities),
        )
    if top_p < 1.0:
        cumulative = probabilities.cumsum(dim=-1)
        remove = cumulative - probabilities >= top_p
        probabilities = probabilities.masked_fill(remove, 0.0)

    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
    cumulative = probabilities.cumsum(dim=-1)
    selected = (cumulative < uniforms.float().unsqueeze(-1)).sum(dim=-1).clamp_max(keep - 1)
    return sorted_indices.gather(-1, selected.unsqueeze(-1)).squeeze(-1)


def paged_kv_append(
    key_cache: Tensor,
    value_cache: Tensor,
    key_scales: Tensor,
    value_scales: Tensor,
    block_tables: Tensor,
    positions: Tensor,
    key: Tensor,
    value: Tensor,
) -> None:
    """Reference page-table append; negative positions are inactive rows."""
    block_size = key_cache.shape[1]
    quantized = key_cache.dtype == torch.int8
    for request in range(key.shape[0]):
        position = int(positions[request])
        if position < 0:
            continue
        logical_block, offset = divmod(position, block_size)
        if logical_block >= block_tables.shape[1]:
            continue
        physical_block = int(block_tables[request, logical_block])
        if not 0 <= physical_block < key_cache.shape[0]:
            continue
        if quantized:
            k_scale = key[request].float().abs().amax(-1).div(127).clamp_min(1e-12)
            v_scale = value[request].float().abs().amax(-1).div(127).clamp_min(1e-12)
            key_scales[physical_block, offset].copy_(k_scale)
            value_scales[physical_block, offset].copy_(v_scale)
            key_cache[physical_block, offset].copy_(
                (key[request].float() / k_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
            )
            value_cache[physical_block, offset].copy_(
                (value[request].float() / v_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
            )
        else:
            key_cache[physical_block, offset].copy_(key[request])
            value_cache[physical_block, offset].copy_(value[request])


def paged_decode_attention_out(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    key_scales: Tensor,
    value_scales: Tensor,
    block_tables: Tensor,
    sequence_lengths: Tensor,
    output: Tensor,
    scale: float,
) -> None:
    """High-precision logical-cache reference for ragged MHA/GQA/MQA decode."""
    batch, query_heads, head_dim = query.shape
    kv_heads = key_cache.shape[2]
    group_size = query_heads // kv_heads
    block_size = key_cache.shape[1]
    output.zero_()
    for request in range(batch):
        length = int(sequence_lengths[request])
        if length <= 0:
            continue
        for query_head in range(query_heads):
            kv_head = query_head // group_size
            keys, values = [], []
            for position in range(length):
                logical_block, offset = divmod(position, block_size)
                physical_block = int(block_tables[request, logical_block])
                key_row = key_cache[physical_block, offset, kv_head].float()
                value_row = value_cache[physical_block, offset, kv_head].float()
                if key_cache.dtype == torch.int8:
                    key_row = key_row * key_scales[physical_block, offset, kv_head]
                    value_row = value_row * value_scales[physical_block, offset, kv_head]
                keys.append(key_row)
                values.append(value_row)
            logical_key = torch.stack(keys)
            logical_value = torch.stack(values)
            probabilities = torch.softmax(
                logical_key @ query[request, query_head].float() * scale, dim=0
            )
            output[request, query_head].copy_(
                (probabilities[:, None] * logical_value).sum(0).to(output.dtype)
            )
