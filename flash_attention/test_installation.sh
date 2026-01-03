#!/bin/bash

# FlashAttention Installation and Test Script
# This script builds and tests the FlashAttention CUDA extension

set -e  # Exit on error

echo "=========================================="
echo "FlashAttention Installation & Test"
echo "=========================================="
echo ""

# Check CUDA
echo "Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "❌ Error: nvcc not found. Please install CUDA toolkit."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "✓ CUDA version: $CUDA_VERSION"

# Check GPU
echo ""
echo "Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. No NVIDIA GPU detected."
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)
echo "✓ GPU: $GPU_INFO"

# Check Python and PyTorch
echo ""
echo "Checking Python and PyTorch..."
python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}')"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in PyTorch'; print(f'✓ PyTorch CUDA available: {torch.version.cuda}')"

# Set environment variables
echo ""
echo "Setting environment variables..."
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CXX=${CXX:-g++}
echo "  CUDA_HOME=$CUDA_HOME"
echo "  CXX=$CXX"

# Build extension
echo ""
echo "Building FlashAttention extension..."
echo "This may take a few minutes..."
pip install --no-build-isolation --force-reinstall -e .

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✓ Build successful!"

# Run tests
echo ""
echo "=========================================="
echo "Running Tests"
echo "=========================================="
echo ""

# Test 1: Import test
echo "Test 1: Import test..."
python -c "
import torch
import _flash_attention_cuda as cuda_ext
print('✓ CUDA extension imported successfully')
print(f'  Has forward: {hasattr(cuda_ext, \"forward\")}')
print(f'  Has backward: {hasattr(cuda_ext, \"backward\")}')
"

# Test 2: Python wrapper import
echo ""
echo "Test 2: Python wrapper import..."
python -c "
from flash_attention import FlashAttention, flash_attention
print('✓ Python wrapper imported successfully')
"

# Test 3: Forward pass
echo ""
echo "Test 3: Forward pass..."
python -c "
import torch
from flash_attention import FlashAttention

attn = FlashAttention(head_dim=64)
Q = torch.randn(2, 4, 128, 64, device='cuda')
K = torch.randn(2, 4, 128, 64, device='cuda')
V = torch.randn(2, 4, 128, 64, device='cuda')

output = attn(Q, K, V)
print(f'✓ Forward pass successful')
print(f'  Output shape: {output.shape}')
print(f'  Output mean: {output.mean():.6f}, std: {output.std():.6f}')
"

# Test 4: Backward pass
echo ""
echo "Test 4: Backward pass..."
python -c "
import torch
from flash_attention import FlashAttention

attn = FlashAttention(head_dim=64)
Q = torch.randn(2, 4, 128, 64, device='cuda', requires_grad=True)
K = torch.randn(2, 4, 128, 64, device='cuda', requires_grad=True)
V = torch.randn(2, 4, 128, 64, device='cuda', requires_grad=True)

output = attn(Q, K, V)
loss = output.sum()
loss.backward()

assert Q.grad is not None, 'Q.grad is None'
assert K.grad is not None, 'K.grad is None'
assert V.grad is not None, 'V.grad is None'
assert not torch.isnan(Q.grad).any(), 'Q.grad contains NaN'
assert not torch.isnan(K.grad).any(), 'K.grad contains NaN'
assert not torch.isnan(V.grad).any(), 'V.grad contains NaN'

print(f'✓ Backward pass successful')
print(f'  Q.grad shape: {Q.grad.shape}')
print(f'  Q.grad mean: {Q.grad.mean():.6f}, std: {Q.grad.std():.6f}')
"

# Test 5: Compare with PyTorch
echo ""
echo "Test 5: Accuracy vs PyTorch..."
python -c "
import torch
import torch.nn.functional as F
from flash_attention import FlashAttention

attn = FlashAttention(head_dim=64)
Q = torch.randn(2, 4, 128, 64, device='cuda')
K = torch.randn(2, 4, 128, 64, device='cuda')
V = torch.randn(2, 4, 128, 64, device='cuda')

# FlashAttention
with torch.no_grad():
    flash_out = attn(Q, K, V)

# PyTorch native
pytorch_out = F.scaled_dot_product_attention(Q, K, V)

# Compare
diff = (flash_out - pytorch_out).abs().max()
print(f'✓ Accuracy test passed')
print(f'  Max difference: {diff:.6e}')
print(f'  Acceptable: {diff < 1e-3}')
"

# Summary
echo ""
echo "=========================================="
echo "All Tests Passed! ✅"
echo "=========================================="
echo ""
echo "FlashAttention is ready to use!"
echo ""
echo "Quick start:"
echo "  from flash_attention import FlashAttention"
echo "  attn = FlashAttention(head_dim=64)"
echo "  output = attn(Q, K, V)"
echo ""
echo "For more examples, see:"
echo "  - example_training.py"
echo "  - USAGE.md"
echo ""
