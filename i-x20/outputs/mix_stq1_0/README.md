# MIX-STQ1_0 experimental implementation

This is a focused, reproducible implementation of the part of MIX-STQ1_0 that
can be validated without a 770B model:

* the exact 42-byte STQ1_0 block layout (`256` weights, `64` 5-bit 3:4 ternary
  codes, one FP16 scale),
* the 32-state one-zero/three-signs codebook,
* an `amax` structured-ternary baseline,
* three-round imatrix-weighted least-squares coordinate descent,
* a packed CUDA GEMV kernel that consumes the 1.3125-bpw representation
  directly, and
* tests plus a CUDA-event benchmark that compares dense FP32 GEMV,
  dequantized dense GEMV, and packed STQ1_0 GEMV.

It is an experimental microkernel, not a drop-in GGUF or llama.cpp replacement.
The benchmark is deliberately an MoE-like GEMV exercise (batch 1); it does not
claim end-to-end LLM tokens/s.

## Run on the GTX 1050 Ti host

```bash
apptainer exec --nv \
  --bind /home/spedrox/Documents/Codex/2026-09-04/i-x20/outputs/mix_stq1_0:/workspace \
  /home/spedrox/gpu-prof/work/cuda118 \
  bash -lc 'cd /workspace && TORCH_EXTENSIONS_DIR=/workspace/.torch_extensions TORCH_CUDA_ARCH_LIST=6.1 python3 benchmark.py --output benchmark_results.json'
```

The CUDA compiler builds the small extension on first run (the bundled builder
also works without Ninja). Run the tests first:

```bash
python3 -m unittest discover -s tests -v
```

## Interpretation

`weighted_stq` is optimized for diagonal activation-energy weights rather than
plain parameter MSE.  It therefore need not win unweighted MSE on every tensor;
the relevant result is `weighted_error`.

The kernel decodes 5-bit codes in registers and multiplies directly with FP32
activations.  It stores no materialized dense STQ weight matrix at inference.
