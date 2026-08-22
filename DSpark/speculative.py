"""End-to-end DSpark verification, checkpoint integration, and serving metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import torch
from torch import Tensor


class VerificationResult(NamedTuple):
    accepted_lengths: Tensor
    emitted_tokens: Tensor
    emitted_lengths: Tensor
    verification_waste: Tensor


@torch.no_grad()
def verify_speculative(
    draft_tokens: Tensor, target_tokens: Tensor, verification_lengths: Tensor,
) -> VerificationResult:
    """Exact greedy speculative verification with one target bonus/mismatch token.

    ``target_tokens[:, :K]`` are the target's choices at draft positions and
    column ``K`` is the bonus token. Output rows are padded with ``-1``.
    """
    if draft_tokens.ndim != 2 or target_tokens.ndim != 2:
        raise ValueError("draft_tokens and target_tokens must be two-dimensional")
    batch, proposal_length = draft_tokens.shape
    if target_tokens.shape != (batch, proposal_length + 1):
        raise ValueError("target_tokens must have shape [batch, proposal_length + 1]")
    if verification_lengths.shape != (batch,):
        raise ValueError("verification_lengths must have shape [batch]")
    positions = torch.arange(proposal_length, device=draft_tokens.device)[None, :]
    selected = positions < verification_lengths[:, None]
    mismatches = (draft_tokens != target_tokens[:, :-1]) & selected
    mismatch_position = torch.where(
        mismatches, positions, torch.full_like(positions, proposal_length)
    ).amin(dim=1)
    accepted = torch.minimum(mismatch_position, verification_lengths.to(torch.long))
    emitted_lengths = accepted + 1
    output_positions = torch.arange(proposal_length + 1, device=draft_tokens.device)[None, :]
    accepted_mask = output_positions < accepted[:, None]
    terminal_mask = output_positions == accepted[:, None]
    draft_padded = torch.nn.functional.pad(draft_tokens, (0, 1), value=0)
    terminal_source = target_tokens.gather(1, accepted[:, None]).squeeze(1)
    emitted = torch.full_like(target_tokens, -1)
    emitted = torch.where(accepted_mask, draft_padded, emitted)
    emitted = torch.where(terminal_mask, terminal_source[:, None], emitted)
    waste = verification_lengths.to(torch.long) - accepted
    return VerificationResult(accepted, emitted, emitted_lengths, waste)


@dataclass(frozen=True)
class SpeculativeMetrics:
    requests: int
    proposed_tokens: int
    verified_tokens: int
    accepted_tokens: int
    emitted_tokens: int
    elapsed_ms: float

    @property
    def mean_acceptance_length(self) -> float:
        return self.accepted_tokens / max(self.requests, 1)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / max(self.verified_tokens, 1)

    @property
    def verification_waste(self) -> int:
        return self.verified_tokens - self.accepted_tokens

    @property
    def tokens_per_second(self) -> float:
        return self.emitted_tokens * 1000.0 / max(self.elapsed_ms, 1e-9)

    @property
    def latency_per_request_ms(self) -> float:
        return self.elapsed_ms / max(self.requests, 1)


class SpeculativeStepResult(NamedTuple):
    draft_tokens: Tensor
    verification: VerificationResult
    schedule: Any
    metrics: SpeculativeMetrics


class DSparkSpeculativeEngine:
    """Model-agnostic DSpark step around a trained head and target verifier."""

    def __init__(self, head, scheduler) -> None:
        self.head = head
        self.scheduler = scheduler

    @torch.no_grad()
    def step(
        self, base_logits: Tensor, previous_ids: Tensor, confidence_logits: Tensor,
        target_tokens: Tensor, step_curve: Tensor,
    ) -> SpeculativeStepResult:
        if base_logits.is_cuda:
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
        else:
            start_time = time.perf_counter()
        draft = self.head.sample_block(base_logits, previous_ids, temperature=0.0)
        scheduled = self.scheduler(confidence_logits, step_curve)
        verification = verify_speculative(draft.token_ids, target_tokens, scheduled.lengths)
        if base_logits.is_cuda:
            end.record(); end.synchronize(); elapsed_ms = start.elapsed_time(end)
        else:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        metrics = SpeculativeMetrics(
            requests=base_logits.shape[0], proposed_tokens=draft.token_ids.numel(),
            verified_tokens=int(scheduled.lengths.sum().item()),
            accepted_tokens=int(verification.accepted_lengths.sum().item()),
            emitted_tokens=int(verification.emitted_lengths.sum().item()), elapsed_ms=elapsed_ms,
        )
        return SpeculativeStepResult(draft.token_ids, verification, scheduled, metrics)


def prompt_lookup_candidates(input_ids: Tensor, proposal_length: int, max_ngram: int = 4) -> Tensor:
    """Deterministic prompt-lookup baseline used by the same evaluation harness."""
    if input_ids.ndim != 1 or proposal_length < 1 or max_ngram < 1:
        raise ValueError("input_ids must be 1-D and lengths must be positive")
    count = input_ids.numel()
    for ngram in range(min(max_ngram, count - 1), 0, -1):
        suffix = input_ids[-ngram:]
        for start in range(count - ngram - 1, -1, -1):
            if torch.equal(input_ids[start:start + ngram], suffix):
                continuation = input_ids[start + ngram:start + ngram + proposal_length]
                if continuation.numel():
                    return continuation
    return input_ids.new_empty((0,))


def load_deepspec_checkpoint(head, checkpoint: str | Path | Mapping[str, Tensor]) -> Any:
    """Load a real DeepSpec state dict without depending on its model wrapper."""
    state: Mapping[str, Any]
    if isinstance(checkpoint, (str, Path)):
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = loaded.get("state_dict", loaded) if isinstance(loaded, Mapping) else loaded
    else:
        state = checkpoint
    def find(suffix: str) -> Tensor:
        matches = [value for key, value in state.items() if key.endswith(suffix)]
        if len(matches) != 1:
            raise KeyError(f"expected exactly one checkpoint key ending in {suffix!r}")
        return matches[0]
    return head.load_deepspec_(find("markov_w1.weight"), find("markov_w2.weight"))
