"""Stable torch.library schemas backed by CUDA kernels and CPU references."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from . import reference

try:
    from . import _C
except ImportError:  # CPU-only reference and schema validation environments
    _C = None


def _require_cuda_extension():
    if _C is None:
        raise RuntimeError(
            "cuda_ml_decode._C is not installed; build decode_kernels on a host with the CUDA toolkit"
        )
    return _C


@torch.library.custom_op("cuda_ml::residual_rms_norm", mutates_args=(), device_types="cuda")
def residual_rms_norm(
    input: Tensor, residual: Tensor, weight: Tensor, eps: float = 1.0e-6
) -> tuple[Tensor, Tensor]:
    return _require_cuda_extension().residual_rms_norm(input, residual, weight, eps)


@residual_rms_norm.register_kernel("cpu")
def _residual_rms_norm_cpu(
    input: Tensor, residual: Tensor, weight: Tensor, eps: float = 1.0e-6
) -> tuple[Tensor, Tensor]:
    return reference.residual_rms_norm(input, residual, weight, eps)


@residual_rms_norm.register_fake
def _residual_rms_norm_fake(
    input: Tensor, residual: Tensor, weight: Tensor, eps: float = 1.0e-6
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(input), torch.empty_like(input)


def _residual_setup_context(ctx, inputs, output) -> None:
    input, residual, weight, eps = inputs
    ctx.save_for_backward(input, residual, weight)
    ctx.eps = eps


def _residual_backward(ctx, grad_output: Tensor, grad_residual_out: Tensor):
    input, residual, weight = ctx.saved_tensors
    needs = ctx.needs_input_grad[:3]
    with torch.enable_grad():
        differentiable = [tensor.detach().requires_grad_(need) for tensor, need in zip((input, residual, weight), needs)]
        output = reference.residual_rms_norm(*differentiable, ctx.eps)
        grads = torch.autograd.grad(
            output,
            differentiable,
            (grad_output, grad_residual_out),
            allow_unused=True,
        )
    return (*grads, None)


residual_rms_norm.register_autograd(_residual_backward, setup_context=_residual_setup_context)


@torch.library.custom_op("cuda_ml::rms_norm_quantize", mutates_args=(), device_types="cuda")
def rms_norm_quantize(
    input: Tensor, weight: Tensor, eps: float = 1.0e-6
) -> tuple[Tensor, Tensor]:
    return _require_cuda_extension().rms_norm_quantize(input, weight, eps)


@rms_norm_quantize.register_kernel("cpu")
def _rms_norm_quantize_cpu(
    input: Tensor, weight: Tensor, eps: float = 1.0e-6
) -> tuple[Tensor, Tensor]:
    return reference.rms_norm_quantize(input, weight, eps)


@rms_norm_quantize.register_fake
def _rms_norm_quantize_fake(
    input: Tensor, weight: Tensor, eps: float = 1.0e-6
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(input, dtype=torch.int8), torch.empty(input.shape[:-1], device=input.device, dtype=torch.float32)


@torch.library.custom_op("cuda_ml::rope_qk_norm", mutates_args=(), device_types="cuda")
def rope_qk_norm(
    q: Tensor,
    k: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    eps: float = 1.0e-6,
) -> tuple[Tensor, Tensor]:
    return _require_cuda_extension().rope_qk_norm(q, k, q_weight, k_weight, cos, sin, eps)


@rope_qk_norm.register_kernel("cpu")
def _rope_qk_norm_cpu(
    q: Tensor,
    k: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    eps: float = 1.0e-6,
) -> tuple[Tensor, Tensor]:
    return reference.rope_qk_norm(q, k, q_weight, k_weight, cos, sin, eps)


@rope_qk_norm.register_fake
def _rope_qk_norm_fake(
    q: Tensor,
    k: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    eps: float = 1.0e-6,
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(q), torch.empty_like(k)


def _rope_setup_context(ctx, inputs, output) -> None:
    *tensors, eps = inputs
    ctx.save_for_backward(*tensors)
    ctx.eps = eps


def _rope_backward(ctx, grad_q: Tensor, grad_k: Tensor):
    saved = ctx.saved_tensors
    needs = ctx.needs_input_grad[:6]
    with torch.enable_grad():
        differentiable = [tensor.detach().requires_grad_(need) for tensor, need in zip(saved, needs)]
        output = reference.rope_qk_norm(*differentiable, ctx.eps)
        grads = torch.autograd.grad(output, differentiable, (grad_q, grad_k), allow_unused=True)
    return (*grads, None)


rope_qk_norm.register_autograd(_rope_backward, setup_context=_rope_setup_context)


@torch.library.custom_op(
    "cuda_ml::kv_cache_append",
    mutates_args=("key_cache", "value_cache"),
    device_types="cuda",
)
def kv_cache_append(
    key_cache: Tensor,
    value_cache: Tensor,
    slots: Tensor,
    key: Tensor,
    value: Tensor,
) -> None:
    _require_cuda_extension().kv_cache_append(key_cache, value_cache, slots, key, value)


@kv_cache_append.register_kernel("cpu")
def _kv_cache_append_cpu(
    key_cache: Tensor,
    value_cache: Tensor,
    slots: Tensor,
    key: Tensor,
    value: Tensor,
) -> None:
    reference.kv_cache_append(key_cache, value_cache, slots, key, value)


@kv_cache_append.register_fake
def _kv_cache_append_fake(
    key_cache: Tensor,
    value_cache: Tensor,
    slots: Tensor,
    key: Tensor,
    value: Tensor,
) -> None:
    return None


@torch.library.custom_op("cuda_ml::bias_swiglu", mutates_args=(), device_types="cuda")
def bias_swiglu(
    gate: Tensor,
    up: Tensor,
    gate_bias: Optional[Tensor] = None,
    up_bias: Optional[Tensor] = None,
) -> Tensor:
    return _require_cuda_extension().bias_swiglu(gate, up, gate_bias, up_bias)


@bias_swiglu.register_kernel("cpu")
def _bias_swiglu_cpu(
    gate: Tensor,
    up: Tensor,
    gate_bias: Optional[Tensor] = None,
    up_bias: Optional[Tensor] = None,
) -> Tensor:
    return reference.bias_swiglu(gate, up, gate_bias, up_bias)


@bias_swiglu.register_fake
def _bias_swiglu_fake(
    gate: Tensor,
    up: Tensor,
    gate_bias: Optional[Tensor] = None,
    up_bias: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(gate)


def _swiglu_setup_context(ctx, inputs, output) -> None:
    gate, up, gate_bias, up_bias = inputs
    sentinel = torch.empty(0, device=gate.device, dtype=gate.dtype)
    ctx.save_for_backward(gate, up, gate_bias if gate_bias is not None else sentinel, up_bias if up_bias is not None else sentinel)
    ctx.has_gate_bias = gate_bias is not None
    ctx.has_up_bias = up_bias is not None


def _swiglu_backward(ctx, grad_output: Tensor):
    gate, up, saved_gate_bias, saved_up_bias = ctx.saved_tensors
    gate_bias = saved_gate_bias if ctx.has_gate_bias else None
    up_bias = saved_up_bias if ctx.has_up_bias else None
    values = (gate, up, gate_bias, up_bias)
    needs = ctx.needs_input_grad[:4]
    with torch.enable_grad():
        differentiable = [
            value.detach().requires_grad_(need) if value is not None else None
            for value, need in zip(values, needs)
        ]
        output = reference.bias_swiglu(*differentiable)
        grad_inputs = [value for value in differentiable if value is not None]
        computed = torch.autograd.grad(output, grad_inputs, grad_output, allow_unused=True)
    iterator = iter(computed)
    return tuple(next(iterator) if value is not None else None for value in differentiable)


bias_swiglu.register_autograd(_swiglu_backward, setup_context=_swiglu_setup_context)


small_n_linear = reference.small_n_linear
sample_logits = reference.sample_logits
