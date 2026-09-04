from __future__ import annotations

import unittest

import torch

from mixstq import quantize_weighted_stq, unpack_stq


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class STQCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from mixstq.cuda import stq_gemv

        cls.stq_gemv = staticmethod(stq_gemv)

    def test_packed_cuda_gemv_matches_materialized_stq(self) -> None:
        torch.manual_seed(9)
        weight = torch.randn(19, 512, device="cuda")
        importance = torch.rand(512, device="cuda") + 0.05
        packed = quantize_weighted_stq(weight, importance)
        activation = torch.randn(3, 512, device="cuda")
        actual = self.stq_gemv(packed, activation)
        expected = activation @ unpack_stq(packed).T
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
