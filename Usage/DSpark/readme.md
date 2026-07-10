# DSpark usage

Build the extension in `DSpark/`, then run from the repository root:

```bash
python Usage/DSpark/dspark_usage.py
```

The example applies the fused low-rank Markov correction and feeds confidence
logits plus a target-engine step curve into the hardware-aware prefix scheduler.
See [`DSpark/README.md`](../../DSpark/README.md) for the full tensor contract,
weight import instructions, correctness tests, and benchmarks.
