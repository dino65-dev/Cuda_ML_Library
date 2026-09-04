"""STQ1_0 packing and imatrix-weighted post-training quantization.

The storage contract is exact: a 256-element block contains 64 3:4 ternary
groups.  Each group has 32 legal states and is stored in five bits; the whole
block has a 16-bit scale, for 40 + 2 = 42 bytes (1.3125 bpw).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


GROUP_SIZE = 4
GROUPS_PER_BLOCK = 64
BLOCK_SIZE = GROUP_SIZE * GROUPS_PER_BLOCK
CODE_BITS = 5
CODE_BYTES = GROUPS_PER_BLOCK * CODE_BITS // 8
SCALE_BYTES = 2
BLOCK_BYTES = CODE_BYTES + SCALE_BYTES
BITS_PER_WEIGHT = BLOCK_BYTES * 8 / BLOCK_SIZE


def build_codebook(*, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return all 32 legal patterns indexed by the STQ1_0 five-bit code.

    Bits 0--1 encode the zero position.  Bits 2--4 encode the signs of the
    three nonzero positions in left-to-right order (one means +1).
    """

    patterns = torch.empty((32, 4), dtype=torch.int8, device=device)
    for code in range(32):
        zero = code & 0b11
        signs = code >> 2
        pattern = []
        sign_bit = 0
        for position in range(4):
            if position == zero:
                pattern.append(0)
            else:
                pattern.append(1 if (signs >> sign_bit) & 1 else -1)
                sign_bit += 1
        patterns[code] = torch.tensor(pattern, dtype=torch.int8, device=device)
    return patterns


def _check_weight(weight: torch.Tensor) -> None:
    if weight.ndim != 2:
        raise ValueError("weight must have shape [out_features, in_features]")
    if weight.shape[1] % BLOCK_SIZE:
        raise ValueError(f"in_features must be a multiple of {BLOCK_SIZE}")
    if not weight.is_floating_point():
        raise TypeError("weight must be floating point")


@dataclass(frozen=True)
class PackedSTQ:
    """A directly executable STQ1_0 matrix.

    `codes` is `[out_features, blocks_per_row, 40]` uint8.  `scales` is
    `[out_features, blocks_per_row]` FP16.  `logical_codes` is retained only
    for validation/inspection and is not part of the 42-byte block payload.
    """

    codes: torch.Tensor
    scales: torch.Tensor
    logical_codes: torch.Tensor

    @property
    def shape(self) -> tuple[int, int]:
        return (self.codes.shape[0], self.codes.shape[1] * BLOCK_SIZE)

    @property
    def storage_bytes(self) -> int:
        return self.codes.numel() + self.scales.numel() * self.scales.element_size()

    @property
    def bits_per_weight(self) -> float:
        rows, columns = self.shape
        return self.storage_bytes * 8 / (rows * columns)


def _make_codes(zero_position: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Build five-bit codes from zero positions and four sign values."""

    codes = torch.zeros_like(zero_position, dtype=torch.int16)
    for zero in range(GROUP_SIZE):
        selected = zero_position == zero
        candidate = torch.full_like(codes, zero)
        sign_bit = 0
        for position in range(GROUP_SIZE):
            if position == zero:
                continue
            candidate |= ((signs[..., position] > 0).to(torch.int16) << (2 + sign_bit))
            sign_bit += 1
        codes = torch.where(selected, candidate, codes)
    return codes.to(torch.uint8)


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack 64 five-bit codes into 40 bytes per STQ block."""

    if codes.shape[-1] != GROUPS_PER_BLOCK:
        raise ValueError("expected 64 group codes per block")
    packed = torch.zeros((*codes.shape[:-1], CODE_BYTES), dtype=torch.int16, device=codes.device)
    values = codes.to(torch.int16)
    for group in range(GROUPS_PER_BLOCK):
        bit = group * CODE_BITS
        byte = bit // 8
        shift = bit % 8
        packed[..., byte] |= values[..., group] << shift
        if shift > 3:
            packed[..., byte + 1] |= values[..., group] >> (8 - shift)
    return packed.to(torch.uint8).contiguous()


def _unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    if packed.ndim != 3 or packed.shape[-1] != CODE_BYTES or packed.dtype != torch.uint8:
        raise ValueError("codes must have shape [out_features, blocks, 40] and dtype uint8")
    values = packed.to(torch.int16)
    codes = []
    for group in range(GROUPS_PER_BLOCK):
        bit = group * CODE_BITS
        byte = bit // 8
        shift = bit % 8
        code = values[..., byte] >> shift
        if shift > 3:
            code |= values[..., byte + 1] << (8 - shift)
        codes.append((code & 0x1F).to(torch.uint8))
    return torch.stack(codes, dim=-1)


def _packed_from_zeroes_and_scale(weight: torch.Tensor, zero_position: torch.Tensor, scale: torch.Tensor) -> PackedSTQ:
    rows, columns = weight.shape
    blocks = columns // BLOCK_SIZE
    grouped = weight.reshape(rows, blocks, GROUPS_PER_BLOCK, GROUP_SIZE)
    signs = torch.where(grouped >= 0, torch.ones_like(grouped), -torch.ones_like(grouped))
    logical_codes = _make_codes(zero_position, signs)
    return PackedSTQ(_pack_codes(logical_codes), scale.to(torch.float16).contiguous(), logical_codes)


def amax_stq(weight: torch.Tensor) -> PackedSTQ:
    """Structured ternary baseline: smallest magnitude zero, amax scale."""

    _check_weight(weight)
    rows, columns = weight.shape
    blocks = columns // BLOCK_SIZE
    grouped = weight.reshape(rows, blocks, GROUPS_PER_BLOCK, GROUP_SIZE)
    zero_position = grouped.abs().argmin(dim=-1)
    scale = grouped.abs().amax(dim=(-1, -2))
    # A zero block has a legal arbitrary code and a zero scale.
    return _packed_from_zeroes_and_scale(weight, zero_position, scale)


def quantize_weighted_stq(
    weight: torch.Tensor,
    importance: torch.Tensor,
    *,
    rounds: int = 3,
) -> PackedSTQ:
    """Run imatrix-weighted STQ coordinate descent for a row-major matrix.

    `importance[j]` is the diagonal activation-energy estimate E[x_j^2].
    The encoder starts from minimum-magnitude zero placement, then alternates
    weighted LS scale fitting and delta-cost zero placement for `rounds` rounds.
    """

    _check_weight(weight)
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if importance.ndim != 1 or importance.numel() != weight.shape[1]:
        raise ValueError("importance must have shape [in_features]")
    if not importance.is_floating_point() or importance.device != weight.device:
        raise ValueError("importance must be floating point and on the weight device")
    if torch.any(importance < 0):
        raise ValueError("importance must be non-negative")

    rows, columns = weight.shape
    blocks = columns // BLOCK_SIZE
    values = weight.reshape(rows, blocks, GROUPS_PER_BLOCK, GROUP_SIZE)
    abs_values = values.abs()
    omega = importance.reshape(1, blocks, GROUPS_PER_BLOCK, GROUP_SIZE).to(weight.dtype)
    zero_position = abs_values.argmin(dim=-1)

    for _ in range(rounds):
        retained = torch.ones_like(values, dtype=torch.bool)
        retained.scatter_(-1, zero_position.unsqueeze(-1), False)
        retained_weight = omega * retained
        denominator = retained_weight.sum(dim=(-1, -2)).clamp_min(torch.finfo(weight.dtype).eps)
        scale = (retained_weight * abs_values).sum(dim=(-1, -2)) / denominator
        extra_cost = omega * (values.square() - (abs_values - scale[..., None, None]).square())
        zero_position = extra_cost.argmin(dim=-1)

    retained = torch.ones_like(values, dtype=torch.bool)
    retained.scatter_(-1, zero_position.unsqueeze(-1), False)
    retained_weight = omega * retained
    denominator = retained_weight.sum(dim=(-1, -2)).clamp_min(torch.finfo(weight.dtype).eps)
    scale = (retained_weight * abs_values).sum(dim=(-1, -2)) / denominator
    return _packed_from_zeroes_and_scale(weight, zero_position, scale)


def unpack_stq(packed: PackedSTQ) -> torch.Tensor:
    """Materialize a dense FP32 weight matrix for correctness checking only."""

    codes = _unpack_codes(packed.codes)
    codebook = build_codebook(device=packed.codes.device).to(torch.float32)
    ternary = codebook[codes.long()]
    return (ternary * packed.scales.to(torch.float32)[..., None, None]).reshape(*packed.shape)


def weighted_squared_error(weight: torch.Tensor, packed: PackedSTQ, importance: torch.Tensor) -> float:
    """Return sum_j E[x_j^2] * (W_ij - W_hat_ij)^2 as a Python float."""

    _check_weight(weight)
    if importance.ndim != 1 or importance.numel() != weight.shape[1]:
        raise ValueError("importance must have shape [in_features]")
    reconstructed = unpack_stq(packed).to(weight.dtype)
    return float(((weight - reconstructed).square() * importance).sum().item())
