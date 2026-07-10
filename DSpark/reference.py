"""Exact PyTorch reference operations for the DSpark CUDA extension."""

from __future__ import annotations

from typing import NamedTuple

import torch


class ScheduleResult(NamedTuple):
    """Result of one hardware-aware DSpark scheduling decision."""

    lengths: torch.Tensor
    survival: torch.Tensor
    selected_count: torch.Tensor
    expected_tokens: torch.Tensor
    expected_throughput: torch.Tensor


def markov_logits_reference(
    base_logits: torch.Tensor,
    previous_token_ids: torch.Tensor,
    token_embedding: torch.Tensor,
    projection_t: torch.Tensor,
) -> torch.Tensor:
    """Apply the default DSpark low-rank Markov correction with PyTorch ops."""

    latent = token_embedding[previous_token_ids.long()]
    return base_logits + latent @ projection_t


def schedule_reference(
    confidence_logits: torch.Tensor,
    step_curve: torch.Tensor,
    temperatures: torch.Tensor,
) -> ScheduleResult:
    """Vectorized reference for Algorithm 1 in the DSpark paper.

    ``step_curve[b]`` is the profiled target-model steps/second for a physical
    verification batch containing ``b`` tokens. One anchor/bonus token per
    request is included before draft candidates are admitted.
    """

    request_count, proposal_length = confidence_logits.shape
    conditional = torch.sigmoid(confidence_logits.float() / temperatures)
    survival = conditional.cumprod(dim=-1)

    flat_survival = survival.flatten()
    # Stable order plus row-major flattening keeps an earlier position ahead of
    # a later equal-probability position from the same request.
    order = torch.argsort(flat_survival, descending=True, stable=True)
    sorted_survival = flat_survival[order]
    prefix_sums = sorted_survival.cumsum(dim=0)

    candidate_count = flat_survival.numel()
    candidate_index = torch.arange(
        1,
        candidate_count + 1,
        device=confidence_logits.device,
        dtype=torch.long,
    )
    expected = float(request_count) + prefix_sums
    throughputs = expected * step_curve[request_count + candidate_index]
    baseline = step_curve.new_tensor(float(request_count)) * step_curve[request_count]
    previous = torch.cat([baseline.reshape(1), throughputs[:-1]])
    drops = ~(throughputs > previous)

    first_drop = torch.argmax(drops.to(torch.int64))
    selected_count_long = torch.where(
        drops.any(),
        first_drop,
        first_drop.new_tensor(candidate_count),
    )
    selected_count = selected_count_long.to(torch.int32).reshape(1)

    selected_mask = torch.arange(
        candidate_count,
        device=confidence_logits.device,
    ) < selected_count_long
    selected_ids = order[selected_mask]
    selected_requests = torch.div(
        selected_ids,
        proposal_length,
        rounding_mode="floor",
    )
    selected_positions = selected_ids.remainder(proposal_length).to(torch.int32) + 1
    lengths = torch.zeros(
        request_count,
        dtype=torch.int32,
        device=confidence_logits.device,
    )
    lengths.scatter_reduce_(
        0,
        selected_requests,
        selected_positions,
        reduce="amax",
        include_self=True,
    )

    selected_prefix = prefix_sums[(selected_count_long - 1).clamp_min(0)]
    final_expected = torch.where(
        selected_count_long > 0,
        selected_prefix + float(request_count),
        prefix_sums.new_tensor(float(request_count)),
    ).reshape(1)
    final_throughput = (
        final_expected * step_curve[request_count + selected_count_long]
    ).reshape(1)
    return ScheduleResult(
        lengths,
        survival,
        selected_count,
        final_expected,
        final_throughput,
    )
