"""Minimal DSpark CUDA integration example."""

import torch

from DSpark import DSparkMarkovHead, DSparkScheduler


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This example requires an NVIDIA GPU")

    device = torch.device("cuda")
    requests, proposal_length = 32, 7
    vocab_size, rank = 32_000, 256

    head = DSparkMarkovHead(vocab_size, rank).to(device).eval()
    scheduler = DSparkScheduler(proposal_length).to(device)

    base_logits = torch.randn(
        requests,
        vocab_size,
        device=device,
        dtype=torch.float16,
    )
    previous_tokens = torch.randint(vocab_size, (requests,), device=device)
    confidence_logits = torch.randn(
        requests,
        proposal_length,
        device=device,
        dtype=torch.float16,
    )

    max_batch = requests * (proposal_length + 1)
    batch_tokens = torch.arange(max_batch + 1, device=device)
    profiled_steps_per_second = 1_000.0 / (1.0 + batch_tokens / 128.0)

    with torch.inference_mode():
        corrected_logits = head(base_logits, previous_tokens)
        decision = scheduler(confidence_logits, profiled_steps_per_second)

    print("corrected logits:", corrected_logits.shape)
    print("verification lengths:", decision.lengths)
    print("expected tokens/step:", decision.expected_tokens)
    print("expected tokens/second:", decision.expected_throughput)


if __name__ == "__main__":
    main()
