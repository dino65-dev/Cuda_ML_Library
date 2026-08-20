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
