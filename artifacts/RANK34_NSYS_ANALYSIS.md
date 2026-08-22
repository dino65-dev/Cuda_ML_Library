# Rank 3/4 CUDA and Nsight analysis

## Evidence boundary

This report covers the physical NVIDIA GeForce GTX 1050 Ti (SM 6.1) path. The
extensions were built with CUDA 11.8 for `TORCH_CUDA_ARCH_LIST=6.1` and tested
with PyTorch 2.7.1. CUDA Event timings and Nsight Systems traces are separate
runs: Events measure steady-state latency with 8 warmups and 30 queued samples;
Nsight captures 12 NVTX-labelled iterations with CUDA/NVTX/OSRT correlation.

The checked-in raw evidence is:

- `gtx1050ti_rank34_benchmarks.json` (CUDA Event measurements);
- `gtx1050ti_rank34_validation.json` (correctness/capture summary);
- `nsys/rank34_gtx1050ti_nsys.qdstrm`, `.nsys-rep`, and `.sqlite`;
- six `nsys/rank34_gtx1050ti_stats_*.csv` structured exports; and
- editable SVG plus PNG graphs `10` through `14` under `artifacts/graphs`.

## Rank 3: paged/ragged decode

The benchmark is B1, eight query heads, two KV heads (GQA), head dimension 64,
16-token physical pages, and one decode query. The dense baseline is PyTorch
SDPA over pre-materialized contiguous KV with KV heads expanded before timing.
The paged implementation follows the physical block table directly and uses
caller-owned FP32 partial values/maxima/sums.

| Sequence | Splits | Dense SDPA | Paged FP16 | Speedup | Paged INT8 | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 703.32 us | 205.03 us | 3.43x | 163.46 us | 4.30x |
| 512 | 2 | 2,650.32 us | 340.89 us | 7.77x | 336.55 us | 7.87x |
| 2,048 | 8 | 8,824.76 us | 338.09 us | 26.10x | 299.25 us | 29.49x |

The large S2048 gain is specific to this Pascal shape and baseline. It is not a
universal attention claim. Split-KV exposes 64 CTAs (B x QH x splits), avoiding
the single dense SM50 memory-efficient attention kernel's poor B1 utilization.
The FP16 and INT8 kernels are both compute/latency limited here; INT8 reduces KV
bytes but Pascal has no INT8 Tensor Core acceleration, so it is capacity-focused
rather than consistently faster.

Nsight independently reports median projected GPU spans of 11,057.92 us for
dense SDPA, 438.30 us for paged FP16, and 424.25 us for paged INT8: 25.23x and
26.06x respectively. The CUDA Event and profiled runs agree on the direction
and approximate magnitude despite expected profiler overhead. Stage 1 accounts
for essentially all paged kernel time; the FP16 split-reduction kernel median is
9.94 us and the INT8 reduction median is 5.62 us.

## Rank 4: DSpark complete inference path

The Pascal B1 common mode fuses embedding lookup, low-rank Markov correction,
base-logit addition, corrected-logit materialization, and stable argmax into one
kernel. A seven-position block is intrinsically sequential because each chosen
token indexes the next Markov embedding; `DSparkGreedyGraph` captures that
seven-launch dependency once and replays it allocation-free.

For B1/K7/V4096/rank64, CUDA Events measure 510.44 us eager and 509.60 us graph
replay (1.002x). The negligible gain is the correct result on this hardware: each
Markov kernel is long enough that Python/launch overhead is a small fraction.
Nsight observes seven fused Markov kernels per eager range and a 729.20 us median
projected span under profiling.

## Modal A10 modern-GPU crossover

The same final sources pass 61 tests with one expected multi-GPU skip on an A10
(SM 8.6, PyTorch 2.8.0+cu128). DSpark graph replay is 87.59 us versus 501.17 us
eager, a 5.72x improvement. Paged FP16 is 107.65, 207.22, and 207.40 us at
S128/S512/S2048; it is faster in absolute time than Pascal but only 0.40x,
0.29x, and 0.23x the speed of contiguous A10 SDPA. This is an important backend
crossover, not a hidden failure: the portable page-table kernel provides direct
ragged access and graph-safe workspace, while PyTorch's contiguous baseline uses
a highly tuned modern fused-attention backend with no page gather/allocation in
the timed region. A production A10 deployment should dispatch to a FlashInfer or
CUTLASS-class paged backend behind this contract when that dependency is allowed.

Nsight Systems 2022.4 reports only the graph-launch/input-copy boundary for the
captured replay range (about 5.7 us) and does not project the captured child
kernels. That value is deliberately excluded from performance conclusions; the
CUDA Event timing is authoritative for graph replay.

The serving layer additionally provides exact prefix verification, target
mismatch/bonus emission, accepted-length and verification-waste accounting,
tokens/s and request-latency metrics, a prompt-lookup baseline, a graph-cached
scheduler wrapper, and real DeepSpec state-dict loading by canonical key suffix.
EAGLE 3.1, DFlash, and PFlash require their own model checkpoints and serving
runtimes; no cross-method speed claim is made without those external inputs.

## Correctness and limitations

- Full local suite: 60 passed, 2 expected skips (multi-GPU; pre-Volta Triton).
- Floating and INT8-cache paged attention match the logical-cache reference;
  CUDA Graph append+attention replay passes.
- Fused Markov greedy max absolute error is 4.77e-7 in the FP32 validation;
  graph replay tokens match eager; speculative prefix verification passes.
- Head dimension is intentionally limited to 256 in the portable kernel.
- Page allocation/eviction policy remains the serving engine's responsibility;
  this library owns page layout, append, attention, and sequence-length advance.
- A10G validation is pending explicit approval to upload the two source folders
  to Modal; no modern-GPU result is inferred from the Pascal trace.
