import pytest
import torch

import cuda_ml_decode as ops
from cuda_ml_decode import reference


DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def tolerance(dtype):
    if dtype == torch.float32:
        return {"rtol": 2.0e-5, "atol": 2.0e-5}
    if dtype == torch.float16:
        return {"rtol": 3.0e-3, "atol": 3.0e-3}
    return {"rtol": 1.0e-2, "atol": 1.0e-2}


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, 32), (7, 127), (3, 4096)])
def test_residual_rms_norm_matches_reference_and_does_not_alias(dtype, shape):
    torch.manual_seed(11)
    input = torch.randn(shape, device="cuda", dtype=dtype)
    residual = torch.randn_like(input)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    input_before = input.clone()
    residual_before = residual.clone()
    actual = ops.residual_rms_norm(input, residual, weight, 1.0e-6)
    expected = reference.residual_rms_norm(input, residual, weight, 1.0e-6)
    torch.testing.assert_close(actual[0], expected[0], **tolerance(dtype))
    torch.testing.assert_close(actual[1], expected[1], **tolerance(dtype))
    torch.testing.assert_close(input, input_before)
    torch.testing.assert_close(residual, residual_before)
    assert actual[0].data_ptr() not in (input.data_ptr(), residual.data_ptr())


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_norm_quantize_matches_reference(dtype):
    torch.manual_seed(12)
    input = torch.randn(5, 513, device="cuda", dtype=dtype)
    weight = torch.randn(513, device="cuda", dtype=dtype)
    actual_q, actual_scale = ops.rms_norm_quantize(input, weight)
    expected_q, expected_scale = reference.rms_norm_quantize(input, weight)
    assert actual_q.dtype == torch.int8
    assert actual_scale.dtype == torch.float32
    torch.testing.assert_close(actual_scale, expected_scale, **tolerance(dtype))
    assert (actual_q.int() - expected_q.int()).abs().max().item() <= 1


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads", [(8, 8), (8, 2)])
def test_rope_qk_norm_supports_mha_and_gqa(dtype, heads):
    torch.manual_seed(13)
    q_heads, k_heads = heads
    q = torch.randn(2, 3, q_heads, 64, device="cuda", dtype=dtype)
    k = torch.randn(2, 3, k_heads, 64, device="cuda", dtype=dtype)
    q_weight = torch.randn(64, device="cuda", dtype=dtype)
    k_weight = torch.randn(64, device="cuda", dtype=dtype)
    angle = torch.randn(3, 32, device="cuda", dtype=torch.float32)
    cos, sin = angle.cos().to(dtype), angle.sin().to(dtype)
    actual = ops.rope_qk_norm(q, k, q_weight, k_weight, cos, sin)
    expected = reference.rope_qk_norm(q, k, q_weight, k_weight, cos, sin)
    torch.testing.assert_close(actual[0], expected[0], **tolerance(dtype))
    torch.testing.assert_close(actual[1], expected[1], **tolerance(dtype))


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("with_bias", [False, True])
def test_bias_swiglu_matches_reference(dtype, with_bias):
    torch.manual_seed(14)
    gate = torch.randn(9, 1101, device="cuda", dtype=dtype)
    up = torch.randn_like(gate)
    gate_bias = torch.randn(1101, device="cuda", dtype=dtype) if with_bias else None
    up_bias = torch.randn(1101, device="cuda", dtype=dtype) if with_bias else None
    actual = ops.bias_swiglu(gate, up, gate_bias, up_bias)
    expected = reference.bias_swiglu(gate, up, gate_bias, up_bias)
    torch.testing.assert_close(actual, expected, **tolerance(dtype))


@pytest.mark.cuda
def test_kv_cache_append_valid_invalid_and_nonsequential_slots():
    torch.manual_seed(15)
    key_cache = torch.full((8, 2, 16), -1.0, device="cuda")
    value_cache = torch.full_like(key_cache, -2.0)
    key = torch.randn(4, 2, 16, device="cuda")
    value = torch.randn_like(key)
    slots = torch.tensor([6, -1, 2, 9], device="cuda", dtype=torch.long)
    ops.kv_cache_append(key_cache, value_cache, slots, key, value)
    torch.testing.assert_close(key_cache[6], key[0])
    torch.testing.assert_close(key_cache[2], key[2])
    torch.testing.assert_close(value_cache[6], value[0])
    assert (key_cache[[0, 1, 3, 4, 5, 7]] == -1.0).all()


@pytest.mark.cuda
def test_current_stream_semantics():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        input = torch.full((4, 1024), 2.0, device="cuda")
        residual = torch.full_like(input, 3.0)
        weight = torch.ones(1024, device="cuda")
        output, residual_out = ops.residual_rms_norm(input, residual, weight)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(residual_out, torch.full_like(residual_out, 5.0))
    torch.testing.assert_close(output, torch.ones_like(output), rtol=1.0e-5, atol=1.0e-5)


@pytest.mark.cuda
def test_rejects_noncontiguous_and_mismatched_shapes():
    input = torch.randn(4, 8, device="cuda").t()
    residual = torch.randn_like(input)
    weight = torch.ones(4, device="cuda")
    with pytest.raises(RuntimeError, match="contiguous"):
        ops.residual_rms_norm(input, residual, weight)
    with pytest.raises(RuntimeError, match="identical shapes"):
        ops.residual_rms_norm(torch.randn(2, 8, device="cuda"), torch.randn(3, 8, device="cuda"), torch.ones(8, device="cuda"))


@pytest.mark.cuda
@pytest.mark.multigpu
def test_device_guard_on_nondefault_device():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    with torch.cuda.device(1):
        input = torch.randn(2, 64, device="cuda:1")
        output, _ = ops.residual_rms_norm(input, torch.zeros_like(input), torch.ones(64, device="cuda:1"))
    assert output.device.index == 1
