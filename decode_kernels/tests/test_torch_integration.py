import pytest
import torch

import cuda_ml_decode as ops


@pytest.mark.cuda
@pytest.mark.parametrize("operator", ["residual", "rope", "swiglu", "quant", "kv_append"])
def test_opcheck(operator):
    if operator == "residual":
        args = (
            torch.randn(2, 64, device="cuda", requires_grad=True),
            torch.randn(2, 64, device="cuda", requires_grad=True),
            torch.randn(64, device="cuda", requires_grad=True),
        )
        op = ops.residual_rms_norm
    elif operator == "rope":
        args = (
            torch.randn(1, 2, 4, 32, device="cuda", requires_grad=True),
            torch.randn(1, 2, 2, 32, device="cuda", requires_grad=True),
            torch.randn(32, device="cuda", requires_grad=True),
            torch.randn(32, device="cuda", requires_grad=True),
            torch.randn(2, 16, device="cuda").cos().requires_grad_(),
            torch.randn(2, 16, device="cuda").sin().requires_grad_(),
        )
        op = ops.rope_qk_norm
    elif operator == "swiglu":
        args = (
            torch.randn(2, 64, device="cuda", requires_grad=True),
            torch.randn(2, 64, device="cuda", requires_grad=True),
            torch.randn(64, device="cuda", requires_grad=True),
            torch.randn(64, device="cuda", requires_grad=True),
        )
        op = ops.bias_swiglu
    elif operator == "quant":
        args = (
            torch.randn(2, 64, device="cuda"),
            torch.randn(64, device="cuda"),
        )
        op = ops.rms_norm_quantize
    else:
        args = (
            torch.zeros(8, 2, 32, device="cuda"),
            torch.zeros(8, 2, 32, device="cuda"),
            torch.tensor([5, 1], device="cuda", dtype=torch.long),
            torch.randn(2, 2, 32, device="cuda"),
            torch.randn(2, 2, 32, device="cuda"),
        )
        op = ops.kv_cache_append
    result = torch.library.opcheck(op, args, raise_exception=False)
    failures = {name: value for name, value in result.items() if value != "SUCCESS"}
    assert not failures, failures


@pytest.mark.cuda
def test_torch_compile_fullgraph():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")
    if torch.cuda.get_device_capability()[0] < 7:
        pytest.skip("PyTorch Inductor/Triton does not support pre-Volta GPUs")

    def function(input, residual, weight):
        normalized, residual_out = ops.residual_rms_norm(input, residual, weight)
        return normalized + residual_out

    compiled = torch.compile(function, fullgraph=True)
    input = torch.randn(3, 128, device="cuda")
    residual = torch.randn_like(input)
    weight = torch.randn(128, device="cuda")
    torch.testing.assert_close(compiled(input, residual, weight), function(input, residual, weight))


@pytest.mark.cuda
def test_cuda_graph_capture_and_replay():
    input = torch.randn(4, 256, device="cuda")
    residual = torch.randn_like(input)
    weight = torch.randn(256, device="cuda")
    gate = torch.randn_like(input)
    up = torch.randn_like(input)

    for _ in range(3):
        ops.residual_rms_norm(input, residual, weight)
        ops.bias_swiglu(gate, up)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        normalized, residual_out = ops.residual_rms_norm(input, residual, weight)
        activated = ops.bias_swiglu(gate, up)
        captured = normalized + residual_out + activated
    graph.replay()
    expected = sum(ops.residual_rms_norm(input, residual, weight)) + ops.bias_swiglu(gate, up)
    torch.testing.assert_close(captured, expected)


@pytest.mark.cuda
def test_cuda_graph_capture_rope_quantize_and_cache_append():
    dtype = torch.float16
    q = torch.randn(2, 1, 4, 64, device="cuda", dtype=dtype)
    k = torch.randn(2, 1, 2, 64, device="cuda", dtype=dtype)
    q_weight = torch.randn(64, device="cuda", dtype=dtype)
    k_weight = torch.randn(64, device="cuda", dtype=dtype)
    angle = torch.randn(1, 32, device="cuda")
    cos, sin = angle.cos().to(dtype), angle.sin().to(dtype)
    cache_key = torch.zeros(8, 2, 64, device="cuda", dtype=dtype)
    cache_value = torch.zeros_like(cache_key)
    slots = torch.tensor([6, 2], device="cuda", dtype=torch.long)
    update_key = torch.randn(2, 2, 64, device="cuda", dtype=dtype)
    update_value = torch.randn_like(update_key)

    for _ in range(3):
        ops.rope_qk_norm(q, k, q_weight, k_weight, cos, sin)
        ops.rms_norm_quantize(q.reshape(-1, 64), q_weight)
        ops.kv_cache_append(cache_key, cache_value, slots, update_key, update_value)
    torch.cuda.synchronize()
    cache_key.zero_()
    cache_value.zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q_out, k_out = ops.rope_qk_norm(q, k, q_weight, k_weight, cos, sin)
        quantized, scales = ops.rms_norm_quantize(q.reshape(-1, 64), q_weight)
        ops.kv_cache_append(cache_key, cache_value, slots, update_key, update_value)
    graph.replay()

    assert q_out.shape == q.shape and k_out.shape == k.shape
    assert quantized.dtype == torch.int8 and scales.dtype == torch.float32
    torch.testing.assert_close(cache_key[slots], update_key)
    torch.testing.assert_close(cache_value[slots], update_value)
