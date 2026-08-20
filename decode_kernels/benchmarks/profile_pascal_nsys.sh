#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CUDA118_CONTAINER_PATH="${CUDA118_CONTAINER_PATH:-/home/spedrox/gpu-prof/work/cuda118}"
NSYS_TARGET_DIR="${NSYS_TARGET_DIR:-/usr/lib/x86_64-linux-gnu/nsight-systems}"
NSYS_IMPORTER="${NSYS_IMPORTER:-/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter}"
PROFILE_ITERATIONS="${PROFILE_ITERATIONS:-12}"
ARTIFACT_DIR="$PROJECT_ROOT/artifacts/nsys"
REPORT_BASE="$ARTIFACT_DIR/decode_gtx1050ti_nsys"

mkdir -p "$ARTIFACT_DIR"

apptainer exec --nv \
  --bind "$PROJECT_ROOT:/workspace" \
  "$CUDA118_CONTAINER_PATH" \
  bash -lc "cd /workspace/decode_kernels && TORCH_CUDA_ARCH_LIST=6.1 MAX_JOBS=4 python3 setup.py build_ext --inplace"

apptainer exec --nv \
  --bind "$PROJECT_ROOT:/workspace" \
  --bind "$NSYS_TARGET_DIR:/opt/nsys" \
  "$CUDA118_CONTAINER_PATH" \
  bash -lc "cd /workspace/decode_kernels/benchmarks && PYTHONPATH=/workspace/decode_kernels /opt/nsys/target-linux-x64/nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true --output /workspace/artifacts/nsys/decode_gtx1050ti_nsys python3 nsys_workload.py --traces decode_traces.json --iterations '$PROFILE_ITERATIONS'"

"$NSYS_IMPORTER" \
  --input-file "$REPORT_BASE.qdstrm" \
  --output-file "$REPORT_BASE.nsys-rep" \
  --force-overwrite

nsys export \
  --type sqlite \
  --force-overwrite=true \
  --output "$REPORT_BASE.sqlite" \
  "$REPORT_BASE.nsys-rep"

nsys stats \
  --force-export=true \
  --force-overwrite=true \
  --report gpukernsum,cudaapisum,kernexecsum,nvtxkernsum,nvtxppsum,nvtxgpuproj \
  --format csv \
  --output "$ARTIFACT_DIR/decode_gtx1050ti_stats" \
  "$REPORT_BASE.nsys-rep"

echo "Nsight Systems artifacts written under $ARTIFACT_DIR"
