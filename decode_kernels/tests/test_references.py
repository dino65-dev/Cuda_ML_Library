import pytest
import torch

import cuda_ml_decode as ops
from cuda_ml_decode import reference
from cuda_ml_decode import sample_logits, small_n_linear


def test_cpu_residual_reference_preserves_contract():
    torch.manual_seed(1)
    input = torch.randn(3, 17)
    residual = torch.randn_like(input)
    weight = torch.randn(17)
    output, residual_out = reference.residual_rms_norm(input, residual, weight)
    assert output.shape == input.shape
    assert residual_out.shape == input.shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(residual_out, input + residual)


def test_public_cpu_dispatch_uses_reference_kernel():
    input = torch.randn(2, 19)
    residual = torch.randn_like(input)
    weight = torch.randn(19)
    actual = ops.residual_rms_norm(input, residual, weight)
    expected = reference.residual_rms_norm(input, residual, weight)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_sampling_filters_are_deterministic():
    logits = torch.tensor([[9.0, 8.0, 1.0, -2.0], [1.0, 3.0, 2.0, 0.0]])
    uniforms = torch.tensor([0.0, 0.999])
    selected = sample_logits(logits, uniforms, top_k=2, top_p=0.95, min_p=0.01)
    assert selected.tolist() == [0, 2]
    greedy = sample_logits(logits, uniforms, temperature=0.0)
    assert greedy.tolist() == [0, 1]


@pytest.mark.parametrize("activation", ["none", "silu"])
def test_small_n_linear_matches_explicit_epilogue(activation):
    input = torch.randn(2, 11)
    weight = torch.randn(5, 11)
    bias = torch.randn(5)
    residual = torch.randn(2, 5)
    actual = small_n_linear(input, weight, bias, residual, activation)
    expected = torch.nn.functional.linear(input, weight, bias) + residual
    if activation == "silu":
        expected = torch.nn.functional.silu(expected)
    torch.testing.assert_close(actual, expected)


def test_sampling_rejects_invalid_controls():
    logits = torch.randn(2, 8)
    uniforms = torch.rand(2)
    with pytest.raises(ValueError, match="top_p"):
        sample_logits(logits, uniforms, top_p=0.0)
    with pytest.raises(ValueError, match="uniforms"):
        sample_logits(logits, torch.rand(3))
