"""PyTorch interface for DeepSeek's DSpark inference primitives."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import nn

from .reference import (
    ScheduleResult,
    markov_logits_reference,
    schedule_reference,
)


class DraftBlockResult(NamedTuple):
    """Tokens and corrected logits from semi-autoregressive block sampling."""

    token_ids: torch.Tensor
    corrected_logits: torch.Tensor


try:
    import _dspark_cuda
except ImportError:
    _dspark_cuda = None


def cuda_extension_available() -> bool:
    """Return whether the compiled CUDA extension was imported successfully."""

    return _dspark_cuda is not None


def _check_markov_inputs(
    base_logits: torch.Tensor,
    previous_token_ids: torch.Tensor,
    token_embedding: torch.Tensor,
    projection_t: torch.Tensor,
) -> None:
    if base_logits.ndim != 2:
        raise ValueError("base_logits must have shape [batch, vocab]")
    if previous_token_ids.ndim != 1:
        raise ValueError("previous_token_ids must have shape [batch]")
    if token_embedding.ndim != 2:
        raise ValueError("token_embedding must have shape [vocab, rank]")
    if projection_t.ndim != 2:
        raise ValueError("projection_t must have shape [rank, vocab]")

    batch, vocab = base_logits.shape
    embedding_vocab, rank = token_embedding.shape
    if previous_token_ids.shape[0] != batch:
        raise ValueError("previous_token_ids batch dimension does not match logits")
    if embedding_vocab != vocab:
        raise ValueError("token_embedding and base_logits vocabulary sizes differ")
    if projection_t.shape != (rank, vocab):
        raise ValueError("projection_t must have shape [rank, vocab]")
    if previous_token_ids.dtype != torch.long:
        raise TypeError("previous_token_ids must use torch.int64")
    if not (base_logits.dtype == token_embedding.dtype == projection_t.dtype):
        raise TypeError("base_logits, token_embedding, and projection_t need one dtype")
    if not (
        base_logits.device
        == previous_token_ids.device
        == token_embedding.device
        == projection_t.device
    ):
        raise ValueError("all Markov-head inputs must be on one device")


def markov_logits(
    base_logits: torch.Tensor,
    previous_token_ids: torch.Tensor,
    token_embedding: torch.Tensor,
    projection_t: torch.Tensor,
) -> torch.Tensor:
    """Apply DSpark's default low-rank Markov head.

    On CUDA inference workloads this fuses embedding lookup, low-rank
    projection, and base-logit addition. During autograd it intentionally uses
    native PyTorch operations so parameter gradients remain exact.
    """

    _check_markov_inputs(
        base_logits,
        previous_token_ids,
        token_embedding,
        projection_t,
    )
    needs_autograd = torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (base_logits, token_embedding, projection_t)
    )
    if base_logits.is_cuda and _dspark_cuda is not None and not needs_autograd:
        return _dspark_cuda.markov_logits(
            base_logits.contiguous(),
            previous_token_ids.contiguous(),
            token_embedding.contiguous(),
            projection_t.contiguous(),
        )
    return markov_logits_reference(
        base_logits,
        previous_token_ids,
        token_embedding,
        projection_t,
    )


def markov_logits_raw_cuda(
    base_logits: torch.Tensor,
    previous_token_ids: torch.Tensor,
    token_embedding: torch.Tensor,
    projection_t: torch.Tensor,
) -> torch.Tensor:
    """Run the original scalar CUDA research kernel for comparison.

    This path intentionally remains available for microbatch experiments, but
    it rereads the projection per request and is not the production default.
    """

    _check_markov_inputs(
        base_logits,
        previous_token_ids,
        token_embedding,
        projection_t,
    )
    if not base_logits.is_cuda or _dspark_cuda is None:
        raise RuntimeError("the compiled DSpark CUDA extension is required")
    return _dspark_cuda.markov_logits_raw(
        base_logits.contiguous(),
        previous_token_ids.contiguous(),
        token_embedding.contiguous(),
        projection_t.contiguous(),
    )


def _temperatures_tensor(
    temperatures: float | Sequence[float] | torch.Tensor | None,
    *,
    proposal_length: int,
    device: torch.device,
) -> torch.Tensor:
    if temperatures is None:
        result = torch.ones(proposal_length, device=device, dtype=torch.float32)
    elif isinstance(temperatures, torch.Tensor):
        result = temperatures.to(device=device, dtype=torch.float32)
    elif isinstance(temperatures, (float, int)):
        result = torch.full(
            (proposal_length,),
            float(temperatures),
            device=device,
            dtype=torch.float32,
        )
    else:
        result = torch.as_tensor(
            list(temperatures),
            device=device,
            dtype=torch.float32,
        )
    if result.shape != (proposal_length,):
        raise ValueError("temperatures must contain one value per proposal position")
    # Avoid launching a validation reduction on every decode step. Reusable
    # DSparkScheduler temperatures are validated on CPU at construction time;
    # direct CUDA callers are required to supply positive values.
    if not result.is_cuda and not bool(torch.all(result > 0).item()):
        raise ValueError("all sequential calibration temperatures must be positive")
    return result.contiguous()


@torch.no_grad()
def schedule(
    confidence_logits: torch.Tensor,
    step_curve: torch.Tensor | Sequence[float],
    temperatures: float | Sequence[float] | torch.Tensor | None = None,
) -> ScheduleResult:
    """Select per-request DSpark verification lengths.

    Args:
        confidence_logits: Conditional acceptance logits with shape
            ``[active_requests, proposal_length]``.
        step_curve: Profiled target-model steps/second. Entry ``b`` is the
            throughput for a physical verification batch of ``b`` tokens. It
            must cover indices through ``requests * (proposal_length + 1)``.
        temperatures: Optional STS calibration temperatures, one per proposal
            position. A scalar applies the same temperature at every position.

    Returns:
        A :class:`ScheduleResult`; ``lengths`` is the number of draft tokens to
        verify for every request. Every returned tensor stays on the input
        device, and the CUDA path does not synchronize with the host.
    """

    if confidence_logits.ndim != 2:
        raise ValueError(
            "confidence_logits must have shape [active_requests, proposal_length]"
        )
    request_count, proposal_length = confidence_logits.shape
    if request_count < 1:
        raise ValueError("at least one active request is required")
    if not 1 <= proposal_length <= 32:
        raise ValueError("proposal_length must be in [1, 32]")
    if confidence_logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("confidence_logits must be float16, bfloat16, or float32")

    curve = torch.as_tensor(
        step_curve,
        dtype=torch.float32,
        device=confidence_logits.device,
    ).contiguous()
    if curve.ndim != 1:
        raise ValueError("step_curve must be one-dimensional")
    required_curve_size = request_count * (proposal_length + 1) + 1
    if curve.numel() < required_curve_size:
        raise ValueError(
            f"step_curve needs at least {required_curve_size} entries; "
            f"received {curve.numel()}"
        )
    calibrated_temperatures = _temperatures_tensor(
        temperatures,
        proposal_length=proposal_length,
        device=confidence_logits.device,
    )

    if confidence_logits.is_cuda and _dspark_cuda is not None:
        tensors = _dspark_cuda.schedule(
            confidence_logits.contiguous(),
            curve,
            calibrated_temperatures,
        )
        return ScheduleResult(*tensors)
    return schedule_reference(
        confidence_logits,
        curve,
        calibrated_temperatures,
    )


class DSparkScheduler(nn.Module):
    """Reusable module holding DSpark's sequential calibration temperatures."""

    def __init__(
        self,
        proposal_length: int = 7,
        temperatures: float | Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not 1 <= proposal_length <= 32:
            raise ValueError("proposal_length must be in [1, 32]")
        values = _temperatures_tensor(
            temperatures,
            proposal_length=proposal_length,
            device=torch.device("cpu"),
        )
        self.proposal_length = proposal_length
        self.register_buffer("temperatures", values)

    def forward(
        self,
        confidence_logits: torch.Tensor,
        step_curve: torch.Tensor | Sequence[float],
    ) -> ScheduleResult:
        if confidence_logits.shape[-1] != self.proposal_length:
            raise ValueError(
                f"expected proposal length {self.proposal_length}, "
                f"received {confidence_logits.shape[-1]}"
            )
        return schedule(confidence_logits, step_curve, self.temperatures)


class DSparkMarkovHead(nn.Module):
    """Default semi-autoregressive DSpark Markov head.

    The projection is stored as ``[rank, vocab]``—the transpose of
    ``nn.Linear.weight``—to make vocabulary lanes contiguous in the CUDA
    kernel. Use :meth:`load_deepspec_` to import an official DeepSpec head.
    """

    def __init__(self, vocab_size: int, rank: int = 256) -> None:
        super().__init__()
        if vocab_size < 1 or rank < 1:
            raise ValueError("vocab_size and rank must be positive")
        self.vocab_size = int(vocab_size)
        self.rank = int(rank)
        self.token_embedding = nn.Parameter(torch.empty(vocab_size, rank))
        self.projection_t = nn.Parameter(torch.empty(rank, vocab_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding, std=1.0 / math.sqrt(self.rank))
        nn.init.xavier_uniform_(self.projection_t)

    @torch.no_grad()
    def load_deepspec_(
        self,
        markov_w1_weight: torch.Tensor,
        markov_w2_weight: torch.Tensor,
    ) -> "DSparkMarkovHead":
        """Load ``markov_w1.weight`` and ``markov_w2.weight`` from DeepSpec."""

        if markov_w1_weight.shape != self.token_embedding.shape:
            raise ValueError("markov_w1 weight shape does not match this head")
        if markov_w2_weight.shape != (self.vocab_size, self.rank):
            raise ValueError("markov_w2 weight must have shape [vocab, rank]")
        self.token_embedding.copy_(markov_w1_weight)
        self.projection_t.copy_(markov_w2_weight.t())
        return self

    def forward(
        self,
        base_logits: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        return markov_logits(
            base_logits,
            previous_token_ids,
            self.token_embedding,
            self.projection_t,
        )

    @torch.no_grad()
    def sample_block(
        self,
        base_logits: torch.Tensor,
        first_previous_token_ids: torch.Tensor,
        *,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> DraftBlockResult:
        """Sample a DSpark block left-to-right through the Markov head.

        ``base_logits`` is the parallel drafter output with shape
        ``[batch, proposal_length, vocab]``. A non-positive temperature uses
        greedy decoding; a positive value samples from the corrected
        distribution just like the reference DeepSpec loop.
        """

        if base_logits.ndim != 3 or base_logits.shape[-1] != self.vocab_size:
            raise ValueError(
                "base_logits must have shape [batch, proposal_length, vocab_size]"
            )
        if first_previous_token_ids.shape != (base_logits.shape[0],):
            raise ValueError("first_previous_token_ids must have shape [batch]")
        if first_previous_token_ids.dtype != torch.long:
            raise TypeError("first_previous_token_ids must use torch.int64")

        proposal_length = base_logits.shape[1]
        if proposal_length == 0:
            return DraftBlockResult(
                torch.empty(
                    base_logits.shape[0],
                    0,
                    dtype=torch.long,
                    device=base_logits.device,
                ),
                base_logits,
            )

        previous = first_previous_token_ids
        sampled = []
        corrected = []
        for position in range(proposal_length):
            position_logits = self(base_logits[:, position, :], previous)
            corrected.append(position_logits)
            if temperature <= 0.0:
                next_tokens = position_logits.argmax(dim=-1)
            else:
                probabilities = torch.softmax(
                    position_logits.float() / float(temperature),
                    dim=-1,
                )
                next_tokens = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                ).squeeze(-1)
            sampled.append(next_tokens)
            previous = next_tokens
        return DraftBlockResult(
            torch.stack(sampled, dim=1),
            torch.stack(corrected, dim=1),
        )
