"""Cross-generation CUDA decode microkernels with auditable references."""

from .ops import (
    bias_swiglu,
    kv_cache_append,
    residual_rms_norm,
    rms_norm_quantize,
    rope_qk_norm,
    sample_logits,
    small_n_linear,
)

__all__ = [
    "bias_swiglu",
    "kv_cache_append",
    "residual_rms_norm",
    "rms_norm_quantize",
    "rope_qk_norm",
    "sample_logits",
    "small_n_linear",
]

__version__ = "0.1.0"
