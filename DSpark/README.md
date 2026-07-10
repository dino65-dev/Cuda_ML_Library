# DSpark CUDA primitives

Low-level PyTorch/CUDA inference primitives for **DSpark: Confidence-Scheduled
Speculative Decoding with Semi-Autoregressive Generation** (DeepSeek-AI,
2026).

This module accelerates the two DSpark operations that sit directly on the
decode hot path:

1. the default low-rank Markov head used to inject one-token local
   autoregression into a parallel draft block; and
2. the hardware-aware prefix scheduler that decides how many draft tokens from
   every active request should be sent to the target model.

It is intentionally a kernel library, not a duplicate of DeepSeek's complete
DeepSpec training/evaluation stack. Use a trained DSpark drafter (or the
official checkpoints) to produce base logits and conditional confidence logits,
then pass those tensors to these primitives.

## Kernel design

### Markov logit correction

For previous token `x`, the default head computes

```text
corrected_logits = base_logits + W2 @ W1[x]
```

The optimized CUDA path stores `W2.T` as `[rank, vocab]`, gathers all `W1[x]`
rows, and submits one dense `addmm` update to cuBLAS:

```text
output = 1 * (latent @ W2.T) + 1 * base_logits
```

This reuses projection tiles across requests, enables Tensor Cores for aligned
FP16/BF16 shapes, and folds the base-logit addition into the GEMM update. The
original scalar CUDA kernel remains available as `markov_logits_raw_cuda` for
research comparisons. Production dispatch uses it only for batch sizes 1–2 on
pre-Volta GPUs, where launch/setup cost dominates and Tensor Cores are absent;
larger batches and newer architectures use cuBLAS. During autograd the API uses
native PyTorch operations so parameter gradients remain exact.

### Confidence scheduler

Given conditional acceptance estimates `c[r, k]`, calibrated prefix survival is

```text
p[r, k] = product(sigmoid(logit[r, j] / temperature[j]), j=0..k)
```

For the production envelope of up to 896 candidates, the extension uses:

- one fused CUDA launch for survival construction, sorting, admission, and
  scatter;
- a CTA-local CUB radix sort with four candidates per thread and a 64-bit key
  whose low word preserves stable row-major order on exact ties; and
- an exact causal first-throughput-drop scan matching DSpark Algorithm 1.

Larger candidate sets automatically use the CUB radix-sort/scan implementation.
Both paths run on the current PyTorch stream without explicit host
synchronization.

For `R` requests and `K` draft positions, `K` may be 1–32; the paper's default
is 7. The fused path is selected when `R*K <= 896`, exactly covering the common
`128*7` serving case while larger sets retain the numerically matched CUB scan.

## Install

Requirements:

- Linux with an NVIDIA GPU
- CUDA toolkit compatible with the installed PyTorch
- PyTorch 2.1 or newer
- a C++17 compiler

```bash
cd DSpark
chmod +x install.sh
./install.sh
```

The installer performs a regular wheel-style installation without build
isolation, so it also works with older setuptools releases that do not support
PEP 660 editable builds.

PyTorch chooses the local GPU architecture automatically. To build a portable
binary, set `TORCH_CUDA_ARCH_LIST`, for example:

```bash
TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;8.9;9.0" ./install.sh
```

Compute capability 6.1 includes the GTX 1050 Ti. Newer architectures use the
same instruction-level path and benefit from larger caches and memory bandwidth.

## Python API

```python
import torch
from DSpark import DSparkMarkovHead, DSparkScheduler

device = "cuda"
requests, proposal_length = 128, 7
vocab_size, rank = 32_000, 256

# Semi-autoregressive Markov correction for one draft position.
head = DSparkMarkovHead(vocab_size, rank).to(device).eval()
base_logits = torch.randn(requests, vocab_size, device=device, dtype=torch.float16)
previous_ids = torch.randint(vocab_size, (requests,), device=device)
with torch.inference_mode():
    corrected_logits = head(base_logits, previous_ids)

# Or correct and sample every position from parallel drafter logits [R, K, V].
parallel_logits = torch.randn(
    requests, proposal_length, vocab_size, device=device, dtype=torch.float16
)
with torch.inference_mode():
    draft = head.sample_block(parallel_logits, previous_ids, temperature=0.0)

# Confidence-scheduled verification.
confidence_logits = torch.randn(
    requests, proposal_length, device=device, dtype=torch.float16
)
max_physical_batch = requests * (proposal_length + 1)

# step_curve[b] = profiled target-model steps/second at b verification tokens.
batch_sizes = torch.arange(max_physical_batch + 1, device=device)
step_curve = 900.0 / (1.0 + batch_sizes / 256.0)

scheduler = DSparkScheduler(
    proposal_length,
    temperatures=[1.08, 1.04, 1.0, 0.98, 0.96, 0.95, 0.94],
).to(device)
result = scheduler(confidence_logits, step_curve)

# Keep these tensors on-device when wiring them into a serving engine.
verification_lengths = result.lengths
expected_tokens = result.expected_tokens
expected_token_throughput = result.expected_throughput
```

`step_curve[batch_tokens]` must contain the profiled target-model steps per
second at that physical verification batch size. The scheduler includes one
anchor/bonus token per active request, so the table must cover indices through
`requests * (proposal_length + 1)`.

If the extension is not built, both operations use equivalent PyTorch code. The
fallback is useful for CPU correctness checks and integration development.

## Import official DeepSpec weights

DeepSpec stores the second Markov matrix as `nn.Linear.weight` with shape
`[vocab, rank]`. This library stores the transpose for coalesced CUDA access:

```python
head.load_deepspec_(
    official_model.markov_head.markov_w1.weight,
    official_model.markov_head.markov_w2.weight,
)
```

## Validate and benchmark

From the repository root:

```bash
python -m pytest DSpark/tests -q
python -m DSpark.benchmark --dtype float16 --requests 128 --vocab-size 32000
```

The benchmark reports actual timings for the current GPU; this repository does
not claim a universal speedup because the Markov kernel's crossover against
cuBLAS depends on batch size, vocabulary, rank, dtype, and architecture.

## Scope and guarantees

- Inference forward path only for the custom kernels.
- Markov head: FP32, FP16, or BF16; rank up to 4096.
- Scheduler: FP32, FP16, or BF16 confidence logits; proposal length up to 32.
- The optimized Markov path uses a dense cuBLAS/Tensor Core GEMM except for
  batch sizes 1–2 on pre-Volta GPUs, which use the lower-overhead scalar path.
- All CUDA work is submitted to the active PyTorch stream and respects the
  current CUDA device guard.
- CPU and autograd paths use native PyTorch operations.

## References

- [DSpark paper (arXiv:2607.05147)](https://arxiv.org/abs/2607.05147)
- [DeepSeek-AI/DeepSpec reference implementation](https://github.com/deepseek-ai/DeepSpec)

The implementation in this directory is original MIT-licensed code written for
this repository and uses the public algorithm/tensor contracts from the sources
above.
