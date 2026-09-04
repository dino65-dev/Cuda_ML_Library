# MIX-STQ1_0 GPU benchmark report

Run date: 2026-09-04  
GPU: NVIDIA GeForce GTX 1050 Ti, 4 GiB, Pascal SM 6.1 (six SMs)  
Runtime: CUDA 11.8, PyTorch 2.7.1+cu118, driver 580.173.02

## Result

| GEMV (batch 1) | Dense FP32 | Packed STQ1_0 | Storage reduction vs FP32 | Direct STQ p50 | cuBLAS FP32 p50 | STQ speedup |
|---|---:|---:|---:|---:|---:|---:|
| 4,096 x 4,096 | 64.00 MiB | 2.625 MiB | 24.38x | 538.624 us | 667.648 us | 1.24x |
| 14,336 x 4,096 | 224.00 MiB | 9.188 MiB | 24.38x | 1,728.512 us | 2,375.680 us | 1.37x |

The packed format is exactly **1.3125 bits/weight**, therefore it is also
12.19x smaller than a BF16 matrix.  The table uses FP32 as the execution
baseline because this Pascal-targeted prototype accumulates in FP32.

The direct kernel agreed with a materialized STQ weight matrix to a maximum
absolute error of **7.63e-6** for both benchmark shapes. It does not create
that dense matrix in the timed path.

## PTQ result

The synthetic test weights include rare large outliers and calibration-derived,
non-uniform activation energies. This intentionally exercises the failure mode
of a per-block `amax` scale.

| Shape | Amax weighted error | 3-round weighted-STQ error | Reduction |
|---|---:|---:|---:|
| 4,096 x 4,096 | 13,625,884.0 | 243,572.72 | 98.21% |
| 14,336 x 4,096 | 47,350,740.0 | 866,862.0 | 98.17% |

This shows the expected direction for this controlled outlier stress test; it
is not an accuracy result for HY4, another LLM, or a language benchmark.

## Method

Each p50 is from 80 CUDA Event samples after 25 warm-up calls. Event pairs were
queued behind an untimed GPU lead-in so host dispatch gaps are excluded. The
benchmark runs warm-cache, batch-1 GEMV only. The packed kernel loads each
five-bit 3:4 ternary code directly from the 40-byte payload, applies the
per-256-weight FP16 scale, and accumulates in FP32.

All five tests passed, including 32-state codebook validation, 42-byte layout,
pack/decode legality, weighted-PTQ improvement over the amax baseline, and
CUDA GEMV agreement with the materialized STQ reference.

## Reading the result correctly

The 1.24--1.37x latency gain is real for this readable prototype on this GPU,
but it is much smaller than the 24.38x FP32 memory reduction. Five-bit unpack,
sign expansion, reduction overhead, and the GTX 1050 Ti's six Pascal SMs limit
the kernel. This is not an end-to-end LLM token/s claim and not a replacement
for a tuned llama.cpp `MUL_MAT_ID` kernel. A production next step is warp-level
bit unpacking plus an MoE-routing-aware multiple-row kernel, then re-measure on
Ampere/Hopper separately.

The complete machine-readable record is `benchmark_results.json`.
