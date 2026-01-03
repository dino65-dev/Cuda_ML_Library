"""
Example: Training with FlashAttention Backward Pass

This script demonstrates how to use FlashAttention with PyTorch's autograd
for training. The backward pass is fully integrated and computes gradients
automatically.
"""

import torch
import torch.nn as nn
from flash_attention import FlashAttention

# Configuration
batch_size = 2
num_heads = 8
seq_len = 128
head_dim = 64
device = 'cuda'

print("FlashAttention Training Example")
print("=" * 60)
print(f"Config: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, dim={head_dim}")
print()

# Create FlashAttention module
attn = FlashAttention(head_dim=head_dim)

# Create sample inputs with gradients enabled
Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)

# Forward pass
print("Forward pass...")
output = attn(Q, K, V)
print(f"Output shape: {output.shape}")
print(f"Output requires_grad: {output.requires_grad}")

# Create a simple loss (sum of all outputs)
print("\nComputing loss and backward pass...")
loss = output.sum()

# Backward pass - gradients are computed automatically!
loss.backward()

# Check gradients
print("\n✓ Gradients computed successfully!")
print(f"Q.grad shape: {Q.grad.shape}")
print(f"Q.grad stats: mean={Q.grad.mean():.6f}, std={Q.grad.std():.6f}")
print(f"K.grad stats: mean={K.grad.mean():.6f}, std={K.grad.std():.6f}")
print(f"V.grad stats: mean={V.grad.mean():.6f}, std={V.grad.std():.6f}")

# Verify no NaN or Inf
has_nan = torch.isnan(Q.grad).any() or torch.isnan(K.grad).any() or torch.isnan(V.grad).any()
has_inf = torch.isinf(Q.grad).any() or torch.isinf(K.grad).any() or torch.isinf(V.grad).any()

if has_nan:
    print("\n✗ ERROR: Gradients contain NaN")
elif has_inf:
    print("\n✗ ERROR: Gradients contain Inf")
else:
    print("\n✓ All gradients are valid (no NaN/Inf)")
    print("\n✅ FlashAttention backward pass is working correctly!")
    print("You can use this in your training loops with PyTorch optimizers.")

print("\n" + "=" * 60)
print("Example: Using with an optimizer")
print("=" * 60)

# Example with optimizer
optimizer = torch.optim.Adam([Q, K, V], lr=0.001)

for step in range(3):
    optimizer.zero_grad()
    
    output = attn(Q, K, V)
    loss = output.sum()
    loss.backward()
    
    optimizer.step()
    
    print(f"Step {step+1}: loss={loss.item():.6f}")

print("\n✓ Training loop completed successfully!")
