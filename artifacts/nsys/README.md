# Nsight Systems evidence

This directory contains the native profiler evidence behind
[`NSYS_PERFORMANCE_ANALYSIS.md`](../NSYS_PERFORMANCE_ANALYSIS.md).

## Modal A10

- `decode_a10_nsys.nsys-rep`: native Nsight Systems 2026.4.1 report.
- `decode_a10_nsys.sqlite`: full export used for diagnostics and API analysis.
- `decode_a10_nsys_stats.csv_*`: built-in CUDA API and NVTX summaries.

The capture used explicit `--trace=cuda-sw,nvtx,osrt`. Modal/gVisor exposed
CUDA API and NVTX activities but no `CUPTI_ACTIVITY_KIND_KERNEL` table. Empty
kernel and GPU-projection CSV files are retained as evidence of that boundary.

## Physical GTX 1050 Ti

- `decode_gtx1050ti_nsys.qdstrm`: raw in-container collection stream.
- `decode_gtx1050ti_nsys.nsys-rep`: native imported report.
- `decode_gtx1050ti_nsys.sqlite`: export containing 4,389 kernel activities.
- `decode_gtx1050ti_stats_*.csv`: built-in kernel, API, correlation, NVTX, and
  GPU-projection reports.

The trace used Nsight Systems 2022.4.2 inside the CUDA 11.8/PyTorch 2.7.1
container and was imported on the host. `profile_pascal_nsys.sh` reproduces the
collection and export sequence.

`manifest.json` records byte sizes, SHA-256 hashes, profiler versions, and core
SQLite table counts for auditability.
