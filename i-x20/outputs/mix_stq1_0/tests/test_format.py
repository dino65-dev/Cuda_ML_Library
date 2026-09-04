from __future__ import annotations

import unittest

import torch

from mixstq import (
    BLOCK_BYTES,
    BITS_PER_WEIGHT,
    amax_stq,
    build_codebook,
    quantize_weighted_stq,
    unpack_stq,
    weighted_squared_error,
)


class STQFormatTests(unittest.TestCase):
    def test_codebook_has_exactly_32_unique_3_of_4_patterns(self) -> None:
        codebook = build_codebook()
        self.assertEqual(codebook.shape, (32, 4))
        self.assertEqual(torch.unique(codebook, dim=0).shape[0], 32)
        self.assertTrue(torch.all((codebook == 0).sum(dim=1) == 1))
        self.assertTrue(torch.all((codebook != 0).sum(dim=1) == 3))
        self.assertTrue(torch.all(torch.isin(codebook, torch.tensor([-1, 0, 1], dtype=torch.int8))))

    def test_exact_42_byte_block_layout(self) -> None:
        weight = torch.randn(3, 512)
        packed = amax_stq(weight)
        self.assertEqual(BLOCK_BYTES, 42)
        self.assertEqual(packed.codes.shape, (3, 2, 40))
        self.assertEqual(packed.scales.shape, (3, 2))
        self.assertEqual(packed.storage_bytes, 3 * 2 * 42)
        self.assertEqual(packed.bits_per_weight, BITS_PER_WEIGHT)
        self.assertEqual(BITS_PER_WEIGHT, 1.3125)

    def test_packed_codes_round_trip_to_legal_scaled_patterns(self) -> None:
        weight = torch.randn(2, 512)
        packed = quantize_weighted_stq(weight, torch.ones(512))
        reconstructed = unpack_stq(packed).reshape(2, 2, 64, 4)
        scales = packed.scales.float()[..., None, None]
        normalized = torch.where(scales == 0, torch.zeros_like(reconstructed), reconstructed / scales)
        self.assertTrue(torch.all((normalized == 0).sum(dim=-1) == 1))
        self.assertTrue(torch.all((normalized != 0).sum(dim=-1) == 3))
        self.assertTrue(torch.all(torch.isin(normalized, torch.tensor([-1.0, 0.0, 1.0]))))

    def test_weighted_coordinate_descent_beats_amax_on_weighted_error(self) -> None:
        torch.manual_seed(7)
        weight = torch.randn(12, 512) * 0.2
        # A sparse outlier stresses the amax scale.  The alternating weighted
        # LS fit should protect the high-energy channel instead.
        weight[:, 31] = 3.0
        importance = torch.linspace(0.1, 4.0, 512).square()
        baseline = amax_stq(weight)
        weighted = quantize_weighted_stq(weight, importance, rounds=3)
        self.assertLess(
            weighted_squared_error(weight, weighted, importance),
            weighted_squared_error(weight, baseline, importance),
        )


if __name__ == "__main__":
    unittest.main()
