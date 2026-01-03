#!/bin/bash
# Quick installation script for FlashAttention CUDA extension

set -e

echo "======================================"
echo "FlashAttention CUDA Extension Setup"
echo "======================================"

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA (nvcc) not found. Please install CUDA toolkit."
    exit 1
fi

# Check for PyTorch
python -c "import torch" 2>/dev/null || {
    echo "PyTorch not found. Installing..."
    pip install torch
}

# Detect GPU
echo ""
echo "Detecting GPU..."
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)

if [ -n "$GPU_CC" ]; then
    echo "GPU Compute Capability: $GPU_CC"
    
    # Convert to sm_XX format
    SM_ARCH="sm_${GPU_CC/./}"
    echo "Using architecture: $SM_ARCH"
    export TORCH_CUDA_ARCH_LIST=$SM_ARCH
else
    echo "Could not detect GPU. Using default sm_75 (T4/RTX 2080)"
    export TORCH_CUDA_ARCH_LIST="sm_75"
fi

# Set environment variables
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CXX=${CXX:-g++}

echo ""
echo "Build Configuration:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  CXX: $CXX"
echo "  Architecture: $TORCH_CUDA_ARCH_LIST"

# Build and install
echo ""
echo "Building extension..."
pip install -e .

# Test
echo ""
echo "Testing installation..."
python -c "import flash_attention; print('✓ Extension imported successfully')"

echo ""
echo "======================================"
echo "Installation complete!"
echo "======================================"
echo ""
echo "Run: python flash_attention.py"
