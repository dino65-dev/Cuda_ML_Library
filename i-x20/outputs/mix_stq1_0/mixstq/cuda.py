"""Lazy compilation and invocation of the packed STQ1_0 CUDA GEMV."""

from __future__ import annotations

import os
import importlib
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import torch

from .format import CODE_BYTES, PackedSTQ


@lru_cache(maxsize=1)
def _extension():
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required for stq_gemv")
    root = Path(__file__).resolve().parent.parent
    try:
        return importlib.import_module("mix_stq1_0_cuda")
    except ModuleNotFoundError:
        # The CUDA 11.8 container used for the GTX 1050 Ti has GNU make but no
        # Ninja.  setup.py explicitly selects PyTorch's compatible distutils
        # builder, so this remains a one-command reproducible experiment.
        environment = os.environ.copy()
        environment.setdefault("TORCH_CUDA_ARCH_LIST", "6.1")
        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=root,
            check=True,
            env=environment,
        )
        importlib.invalidate_caches()
        return importlib.import_module("mix_stq1_0_cuda")


def stq_gemv(packed: PackedSTQ, activation: torch.Tensor) -> torch.Tensor:
    """Compute `activation @ dequantized_weight.T` from packed STQ bytes.

    Activations must be contiguous CUDA FP32 `[batch, in_features]`.  The
    result is FP32 `[batch, out_features]`.  This is intentionally GEMV-focused
    (batch 1 is the decode path); no dense dequantized weight is materialized.
    """

    if activation.ndim != 2 or activation.dtype != torch.float32:
        raise ValueError("activation must be a FP32 tensor of shape [batch, in_features]")
    if not activation.is_cuda or not activation.is_contiguous():
        raise ValueError("activation must be a contiguous CUDA tensor")
    if packed.codes.shape[-1] != CODE_BYTES or packed.codes.dtype != torch.uint8:
        raise ValueError("invalid STQ1_0 code storage")
    if packed.scales.dtype != torch.float16:
        raise ValueError("STQ1_0 scales must be FP16")
    if not packed.codes.is_cuda or packed.codes.device != activation.device or packed.scales.device != activation.device:
        raise ValueError("packed weights and activations must be on the same CUDA device")
    if activation.shape[1] != packed.shape[1]:
        raise ValueError("activation width and packed weight width differ")
    return _extension().stq_gemv(packed.codes, packed.scales, activation)
