from __future__ import annotations

import pytest
import torch

from DSpark import (
    DSparkMarkovHead,
    DSparkScheduler,
    cuda_extension_available,
    markov_logits,
    markov_logits_raw_cuda,
    schedule,
    markov_greedy,
    verify_speculative,
    load_deepspec_checkpoint,
    prompt_lookup_candidates,
    DSparkGreedyGraph,
    DSparkSchedulerGraph,
)
from DSpark.reference import markov_logits_reference, schedule_reference


def _curve(values: dict[int, float], size: int = 9) -> torch.Tensor:
    curve = torch.zeros(size, dtype=torch.float32)
    for index, value in values.items():
        curve[index] = value
    return curve


def test_markov_reference_matches_formula_and_backpropagates() -> None:
    torch.manual_seed(4)
    base = torch.randn(3, 11, requires_grad=True)
    ids = torch.tensor([2, 7, 4])
    embedding = torch.randn(11, 5, requires_grad=True)
    projection_t = torch.randn(5, 11, requires_grad=True)

    actual = markov_logits(base, ids, embedding, projection_t)
    expected = base + embedding[ids] @ projection_t
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert base.grad is not None
    assert embedding.grad is not None
    assert projection_t.grad is not None


def test_scheduler_stops_at_first_throughput_drop() -> None:
    probability = torch.tensor(0.9)
    logits = torch.logit(probability).expand(2, 3).clone()
    curve = _curve({2: 100.0, 3: 95.0, 4: 90.0, 5: 70.0, 6: 60.0, 7: 50.0, 8: 40.0})

    result = schedule(logits, curve)

    assert result.selected_count.item() == 2
    assert result.lengths.tolist() == [1, 1]
    torch.testing.assert_close(
        result.survival,
        torch.tensor([[0.9, 0.81, 0.729], [0.9, 0.81, 0.729]]),
    )
    torch.testing.assert_close(result.expected_tokens, torch.tensor([3.8]))
    torch.testing.assert_close(result.expected_throughput, torch.tensor([342.0]))


def test_scheduler_can_select_none_or_everything() -> None:
    logits = torch.zeros(2, 3)
    none_curve = _curve({2: 100.0, 3: 10.0, 4: 9.0, 5: 8.0, 6: 7.0, 7: 6.0, 8: 5.0})
    none = schedule(logits, none_curve)
    assert none.selected_count.item() == 0
    assert none.lengths.tolist() == [0, 0]

    all_curve = torch.ones(9)
    all_candidates = schedule(logits, all_curve)
    assert all_candidates.selected_count.item() == 6
    assert all_candidates.lengths.tolist() == [3, 3]


def test_sequential_temperature_scaling_is_applied_before_cumprod() -> None:
    logits = torch.tensor([[2.0, 2.0]])
    curve = torch.ones(4)
    result = schedule(logits, curve, temperatures=[2.0, 1.0])
    expected = torch.tensor(
        [[torch.sigmoid(torch.tensor(1.0)),
          torch.sigmoid(torch.tensor(1.0)) * torch.sigmoid(torch.tensor(2.0))]]
    )
    torch.testing.assert_close(result.survival, expected)


def test_deepspec_weight_loader_transposes_projection() -> None:
    head = DSparkMarkovHead(vocab_size=13, rank=4)
    w1 = torch.randn(13, 4)
    w2 = torch.randn(13, 4)
    head.load_deepspec_(w1, w2)
    torch.testing.assert_close(head.token_embedding, w1)
    torch.testing.assert_close(head.projection_t, w2.t())


def test_markov_head_samples_block_sequentially() -> None:
    head = DSparkMarkovHead(vocab_size=5, rank=2)
    with torch.no_grad():
        head.token_embedding.zero_()
        head.projection_t.zero_()
    base = torch.tensor(
        [[[0.0, 3.0, 1.0, 0.0, -1.0], [0.0, 1.0, 4.0, 0.0, -1.0]]]
    )
    result = head.sample_block(base, torch.tensor([0]))
    assert result.token_ids.tolist() == [[1, 2]]
    torch.testing.assert_close(result.corrected_logits, base)


def test_markov_greedy_matches_separate_reference() -> None:
    base = torch.randn(3, 17)
    ids = torch.tensor([1, 5, 9])
    embedding = torch.randn(17, 4)
    projection = torch.randn(4, 17)
    result = markov_greedy(base, ids, embedding, projection)
    expected = markov_logits_reference(base, ids, embedding, projection)
    torch.testing.assert_close(result.corrected_logits, expected)
    torch.testing.assert_close(result.token_ids, expected.argmax(-1))


def test_speculative_verification_accepts_prefix_and_emits_target_terminal() -> None:
    draft = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 9, 9, 9]])
    target = torch.tensor([[1, 2, 0, 7, 8], [0, 6, 7, 8, 9], [9, 9, 9, 9, 3]])
    lengths = torch.tensor([4, 3, 2], dtype=torch.int32)
    result = verify_speculative(draft, target, lengths)
    assert result.accepted_lengths.tolist() == [2, 0, 2]
    assert result.emitted_lengths.tolist() == [3, 1, 3]
    assert result.emitted_tokens.tolist() == [[1, 2, 0, -1, -1], [0, -1, -1, -1, -1], [9, 9, 9, -1, -1]]
    assert result.verification_waste.tolist() == [2, 3, 0]


def test_checkpoint_adapter_and_prompt_lookup() -> None:
    head = DSparkMarkovHead(13, 4)
    w1, w2 = torch.randn(13, 4), torch.randn(13, 4)
    load_deepspec_checkpoint(head, {"drafter.markov_w1.weight": w1, "drafter.markov_w2.weight": w2})
    torch.testing.assert_close(head.token_embedding, w1)
    torch.testing.assert_close(head.projection_t, w2.t())
    prompt = torch.tensor([4, 5, 6, 4, 5])
    assert prompt_lookup_candidates(prompt, 3).tolist() == [6, 4, 5]


@pytest.mark.skipif(
    not torch.cuda.is_available() or not cuda_extension_available(),
    reason="compiled DSpark CUDA extension is unavailable",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cuda_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(5)
    device = torch.device("cuda")
    base = torch.randn(4, 257, device=device, dtype=dtype)
    ids = torch.randint(257, (4,), device=device)
    embedding = torch.randn(257, 16, device=device, dtype=dtype)
    projection_t = torch.randn(16, 257, device=device, dtype=dtype)
    with torch.inference_mode():
        actual = markov_logits(base, ids, embedding, projection_t)
        raw = markov_logits_raw_cuda(base, ids, embedding, projection_t)
        expected = markov_logits_reference(base, ids, embedding, projection_t)
    tolerance = 2e-2 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(raw, expected, atol=tolerance, rtol=tolerance)
    greedy = markov_greedy(base, ids, embedding, projection_t)
    torch.testing.assert_close(greedy.corrected_logits, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(greedy.token_ids, expected.argmax(-1))

    # Exercise a small fused CTA, the exact 128x7 benchmark shape, and the
    # >1024-candidate CUB fallback.
    for requests in (17, 128, 160):
        confidence = torch.randn(requests, 7, device=device, dtype=dtype)
        temperatures = torch.ones(7, device=device)
        curve = torch.ones(requests * 8 + 1, device=device)
        actual_schedule = schedule(confidence, curve, temperatures)
        expected_schedule = schedule_reference(confidence, curve, temperatures)
        torch.testing.assert_close(
            actual_schedule.survival,
            expected_schedule.survival,
            atol=2e-3,
            rtol=2e-3,
        )
        torch.testing.assert_close(
            actual_schedule.lengths,
            expected_schedule.lengths,
        )
        torch.testing.assert_close(
            actual_schedule.selected_count,
            expected_schedule.selected_count,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not cuda_extension_available(),
    reason="compiled DSpark CUDA extension is unavailable",
)
def test_cuda_graph_wrappers_match_eager() -> None:
    device = torch.device("cuda")
    head = DSparkMarkovHead(257, 16).to(device).eval()
    base = torch.randn(2, 3, 257, device=device)
    previous = torch.randint(257, (2,), device=device)
    draft_graph = DSparkGreedyGraph(head, batch=2, proposal_length=3, dtype=torch.float32)
    actual = draft_graph.replay(base, previous)
    expected = head.sample_block(base, previous)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.token_ids, expected.token_ids)

    scheduler = DSparkScheduler(3).to(device)
    confidence = torch.randn(2, 3, device=device)
    curve = torch.ones(9, device=device)
    scheduler_graph = DSparkSchedulerGraph(scheduler, requests=2, device=device, dtype=torch.float32)
    actual_schedule = scheduler_graph.replay(confidence, curve)
    expected_schedule = scheduler(confidence, curve)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual_schedule.lengths, expected_schedule.lengths)
    torch.testing.assert_close(actual_schedule.survival, expected_schedule.survival)
