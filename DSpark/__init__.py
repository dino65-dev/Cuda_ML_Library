"""CUDA-accelerated primitives for DeepSeek DSpark."""

from .dspark import (
    DraftBlockResult,
    DSparkMarkovHead,
    DSparkScheduler,
    ScheduleResult,
    cuda_extension_available,
    markov_logits,
    markov_logits_raw_cuda,
    schedule,
)

__all__ = [
    "DraftBlockResult",
    "DSparkMarkovHead",
    "DSparkScheduler",
    "ScheduleResult",
    "cuda_extension_available",
    "markov_logits",
    "markov_logits_raw_cuda",
    "schedule",
]
