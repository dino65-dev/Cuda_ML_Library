# FlashAttention - Usage Guide

Complete guide for using FlashAttention with PyTorch, including training examples.

## Table of Contents
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Training with Optimizers](#training-with-optimizers)
- [Integration with Models](#integration-with-models)
- [Advanced Examples](#advanced-examples)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
```bash
# Check CUDA is available
nvcc --version

# Check PyTorch with CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Build and Install
```bash
cd flash_attention
export CUDA_HOME=/usr/local/cuda
export CXX=g++
pip install -e .
```

## Basic Usage

### Functional Interface (Inference Only)

```python
import torch
from flash_attention import flash_attention

# Create test tensors
Q = torch.randn(2, 4, 128, 64, device='cuda')
K = torch.randn(2, 4, 128, 64, device='cuda')
V = torch.randn(2, 4, 128, 64, device='cuda')

# Compute attention (no gradients)
output = flash_attention(Q, K, V)
print(f"Output shape: {output.shape}")  # [2, 4, 128, 64]
```

### Module Interface (With Gradients)

```python
import torch
from flash_attention import FlashAttention

# Initialize module
attn = FlashAttention(head_dim=64)

# Inputs with gradients enabled
Q = torch.randn(2, 8, 512, 64, device='cuda', requires_grad=True)
K = torch.randn(2, 8, 512, 64, device='cuda', requires_grad=True)
V = torch.randn(2, 8, 512, 64, device='cuda', requires_grad=True)

# Forward pass
output = attn(Q, K, V)

# Backward pass - gradients computed automatically!
loss = output.sum()
loss.backward()

# Access gradients
print(f"Q.grad shape: {Q.grad.shape}")  # [2, 8, 512, 64]
print(f"Q.grad mean: {Q.grad.mean():.6f}")
print(f"All gradients valid: {not torch.isnan(Q.grad).any()}")
```

## Training with Optimizers

### Example 1: Simple Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from flash_attention import FlashAttention

# Model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = FlashAttention(head_dim=64)
        
    def forward(self, q, k, v):
        return self.attention(q, k, v)

# Setup
model = SimpleModel().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    # Generate random data
    Q = torch.randn(2, 8, 128, 64, device='cuda', requires_grad=True)
    K = torch.randn(2, 8, 128, 64, device='cuda', requires_grad=True)
    V = torch.randn(2, 8, 128, 64, device='cuda', requires_grad=True)
    target = torch.randn(2, 8, 128, 64, device='cuda')
    
    # Forward
    optimizer.zero_grad()
    output = model(Q, K, V)
    
    # Loss
    loss = nn.functional.mse_loss(output, target)
    
    # Backward
    loss.backward()
    
    # Update
    optimizer.step()
    
    print(f"Epoch {epoch}: loss={loss.item():.4f}")
```

### Example 2: With Learning Rate Scheduler

```python
import torch
import torch.nn as nn
import torch.optim as optim
from flash_attention import FlashAttention

class AttentionModel(nn.Module):
    def __init__(self, num_heads=8, head_dim=64):
        super().__init__()
        self.attention = FlashAttention(head_dim=head_dim)
        self.num_heads = num_heads
        self.head_dim = head_dim
        
    def forward(self, q, k, v):
        return self.attention(q, k, v)

# Initialize
model = AttentionModel().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training
num_epochs = 20
for epoch in range(num_epochs):
    # Your data here
    Q = torch.randn(4, 8, 256, 64, device='cuda', requires_grad=True)
    K = torch.randn(4, 8, 256, 64, device='cuda', requires_grad=True)
    V = torch.randn(4, 8, 256, 64, device='cuda', requires_grad=True)
    
    optimizer.zero_grad()
    output = model(Q, K, V)
    loss = output.mean()  # Dummy loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
```

## Integration with Models

### Example 3: Multi-Head Attention Layer

```python
import torch
import torch.nn as nn
from flash_attention import FlashAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # FlashAttention
        self.attention = FlashAttention(head_dim=self.head_dim)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Project and reshape
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply attention
        attn_output = self.attention(Q, K, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.out_proj(attn_output)
        
        return output

# Usage
model = MultiHeadAttention(d_model=512, num_heads=8).cuda()
x = torch.randn(4, 128, 512, device='cuda')
output = model(x)
print(f"Output shape: {output.shape}")  # [4, 128, 512]
```

### Example 4: Transformer Block

```python
import torch
import torch.nn as nn
from flash_attention import FlashAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = FlashAttention(head_dim=self.head_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        
        # Project
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = self.attention(Q, K, V)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        x = self.out_proj(attn)
        x = self.dropout(x)
        x = x + residual
        
        # Feed-forward with residual
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + residual
        
        return x

# Usage in a full model
class SimpleTransformer(nn.Module):
    def __init__(self, num_layers=6, d_model=512, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Train it
model = SimpleTransformer(num_layers=6).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

x = torch.randn(8, 128, 512, device='cuda')
output = model(x)
loss = output.mean()
loss.backward()
optimizer.step()
print("✅ Training step successful!")
```

## Advanced Examples

### Example 5: Gradient Clipping

```python
import torch
import torch.nn as nn
from flash_attention import FlashAttention

model = FlashAttention(head_dim=64).cuda()
optimizer = torch.optim.Adam(model.parameters())

Q = torch.randn(2, 4, 128, 64, device='cuda', requires_grad=True)
K = torch.randn(2, 4, 128, 64, device='cuda', requires_grad=True)
V = torch.randn(2, 4, 128, 64, device='cuda', requires_grad=True)

output = model(Q, K, V)
loss = output.sum()
loss.backward()

# Clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

### Example 6: Mixed Precision Training (Note: Currently FP32 only)

```python
# FlashAttention currently only supports FP32
# This example shows the intended usage pattern for future FP16 support

import torch
from flash_attention import FlashAttention

model = FlashAttention(head_dim=64).cuda()

# For now, use FP32
Q = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float32, requires_grad=True)
K = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float32, requires_grad=True)
V = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float32, requires_grad=True)

output = model(Q, K, V)
loss = output.sum()
loss.backward()
```

## Troubleshooting

### Common Issues

**Q: Gradients are None after backward()**
```python
# Make sure requires_grad=True
Q = torch.randn(2, 4, 128, 64, device='cuda', requires_grad=True)
```

**Q: CUDA out of memory**
```python
# Reduce batch size or sequence length
Q = torch.randn(1, 4, 64, 64, device='cuda', requires_grad=True)  # Smaller
```

**Q: Loss is NaN**
```python
# Check for valid inputs
assert not torch.isnan(Q).any()
assert not torch.isinf(Q).any()

# Check gradients after backward
loss.backward()
assert not torch.isnan(Q.grad).any()
```

**Q: Slow training**
```python
# Make sure you're using CUDA
assert Q.device.type == 'cuda'

# Use contiguous tensors
Q = Q.contiguous()
```

## Validation

### Check Forward Pass
```python
import torch
from flash_attention import flash_attention

Q = torch.randn(2, 4, 128, 64).cuda()
K = torch.randn(2, 4, 128, 64).cuda()
V = torch.randn(2, 4, 128, 64).cuda()

# FlashAttention
flash_out = flash_attention(Q, K, V)

# PyTorch native
torch_out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

# Compare
diff = (flash_out - torch_out).abs().max()
print(f"Max difference: {diff:.6f}")
assert diff < 1e-3, "Forward pass accuracy issue!"
```

### Check Backward Pass
```python
import torch
from flash_attention import FlashAttention

attn = FlashAttention(head_dim=64)

Q = torch.randn(2, 4, 128, 64, device='cuda', requires_grad=True)
K = torch.randn(2, 4, 128, 64, device='cuda', requires_grad=True)
V = torch.randn(2, 4, 128, 64, device='cuda', requires_grad=True)

output = attn(Q, K, V)
loss = output.sum()
loss.backward()

# Verify gradients exist and are valid
assert Q.grad is not None, "Q.grad should not be None"
assert K.grad is not None, "K.grad should not be None"
assert V.grad is not None, "V.grad should not be None"

assert not torch.isnan(Q.grad).any(), "Q.grad contains NaN"
assert not torch.isinf(Q.grad).any(), "Q.grad contains Inf"

print("✅ Backward pass validation successful!")
```

## Performance Tips

1. **Use contiguous tensors**: Always call `.contiguous()` before passing to FlashAttention
2. **Batch operations**: Process multiple sequences together for better GPU utilization
3. **Profile first**: Use `torch.cuda.synchronize()` for accurate timing
4. **Monitor memory**: Use `torch.cuda.memory_allocated()` to track usage

## Next Steps

- See `example_training.py` for a complete working example
- Check `README.md` for installation troubleshooting
- Read the FlashAttention paper for algorithm details

---

**Ready to train your models with FlashAttention! 🚀**
