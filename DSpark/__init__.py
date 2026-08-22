"""CUDA-accelerated primitives for DeepSeek DSpark."""

from .dspark import (
    DraftBlockResult,
    DSparkMarkovHead,
    DSparkScheduler,
    DSparkGreedyGraph,
    DSparkSchedulerGraph,
    ScheduleResult,
    cuda_extension_available,
    markov_logits,
    markov_logits_raw_cuda,
    markov_greedy,
    schedule,
)
from .speculative import (
    DSparkSpeculativeEngine,
    SpeculativeMetrics,
    VerificationResult,
    load_deepspec_checkpoint,
    prompt_lookup_candidates,
    verify_speculative,
)

__all__ = [
    "DraftBlockResult",
    "DSparkMarkovHead",
    "DSparkScheduler",
    "DSparkGreedyGraph",
    "DSparkSchedulerGraph",
    "ScheduleResult",
    "cuda_extension_available",
    "markov_logits",
    "markov_logits_raw_cuda",
    "markov_greedy",
    "schedule",
    "DSparkSpeculativeEngine",
    "SpeculativeMetrics",
    "VerificationResult",
    "load_deepspec_checkpoint",
    "prompt_lookup_candidates",
    "verify_speculative",
]
