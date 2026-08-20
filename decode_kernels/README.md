# Decode Kernels

This package implements the first two items in the repository roadmap: a
correctness/benchmark foundation and a bounded GPU-native decode microkernel
pack. It is an inference research package, not a blanket production claim.

## Operator contracts

| API | Contract | CUDA implementation |
|---|---|---|
| `residual_rms_norm` | Returns `(RMSNorm(input + residual), input + residual)` without mutating inputs | One CTA per row |
| `rms_norm_quantize` | Per-row RMSNorm followed by symmetric INT8 quantization; returns `(int8, FP32 scale)` | One CTA, two reductions per row |
| `rope_qk_norm` | Q/K RMSNorm plus half-split RoPE for MHA, GQA, or MQA tensors `[B,S,H,D]` | One CTA per token/head |
| `kv_cache_append` | Appends `[tokens, kv_heads, head_dim]` into explicit cache slots | Coalesced scatter; invalid slots are safely skipped |
| `bias_swiglu` | Optional broadcast biases followed by `SiLU(gate) * up` | Single elementwise kernel |
| `sample_logits` | Temperature, top-k, top-p, and min-p with explicit random uniforms | Exact graph-visible ATen pipeline |
| `small_n_linear` | Linear plus optional residual/SiLU epilogue | cuBLAS/PyTorch baseline pending measured crossover |

CUDA kernels accept contiguous FP32, FP16, and BF16 tensors, use the current
PyTorch CUDA stream, and install a device guard for non-default GPUs. Q/K may
have different head counts. Cache slots may be non-sequential; callers should
not supply duplicate slots in the same append because their write order is not
defined.

The last two APIs intentionally remain visible PyTorch compositions. This keeps
`torch.compile` able to optimize them and avoids claiming a specialized kernel
before trace measurements establish a crossover against cuBLAS/ATen.

## Correctness policy

- FP32: `rtol=atol=2e-5`.
- FP16: `rtol=atol=3e-3`.
- BF16: `rtol=atol=1e-2`.
- INT8 quantization: FP32 scales must meet the source-dtype tolerance and codes
  may differ from the reference by at most one integer due to reduction order.
- Every CUDA operator has an independent, higher-precision PyTorch reference.
- Tests cover random and boundary shapes, GQA, non-sequential/invalid cache
  slots, aliasing, current streams, non-default devices, FakeTensor/opcheck,
  autograd registration, `torch.compile`, and CUDA Graph replay.

## Build and test

```bash
cd decode_kernels
python -m pip install --no-build-isolation .
python -m pytest -q tests
```

To build, test, and benchmark on a Modal A10G:

```bash
.venv/bin/python -m modal run modal/run_decode_gpu.py
```

The run writes `artifacts/modal_a10g_decode_validation.json` locally. The
artifact records test output, exact GPU/software metadata, first-call latency,
warm and cold-cache CUDA Event distributions, reference comparisons, estimated
bytes, and effective bandwidth.

## Performance graphs and Nsight Systems

The full visual and profiler analysis is checked in at
[`artifacts/NSYS_PERFORMANCE_ANALYSIS.md`](../artifacts/NSYS_PERFORMANCE_ANALYSIS.md).
It includes nine reproducible PNG/SVG graphs and the native `.nsys-rep`, SQLite,
CSV, and derived JSON evidence behind them.

Modal supplies the A10 correctness/latency run and a real CUDA API/NVTX Nsight
trace. Modal's gVisor environment did not expose GPU workload activities to
Nsight, even with explicit `cuda-sw` tracing, so kernel-level Nsight analysis is
reported separately from the physical GTX 1050 Ti trace. The report never
substitutes API duration for GPU duration.

Regenerate the figures with:

```bash
python -m pip install -r decode_kernels/requirements-analysis.txt
python decode_kernels/benchmarks/generate_analysis_graphs.py
```

The Pascal capture is reproducible through
`decode_kernels/benchmarks/profile_pascal_nsys.sh`; the Modal capture is defined
in `modal/profile_decode_nsys.py`.

## Trace provenance

`benchmarks/decode_traces.json` contains representative synthetic shapes based
on common 7B-class GQA decoders. They are labelled assumptions, not a claim that
production traffic was captured. Replace or extend them with real serving
traces before drawing deployment conclusions.
