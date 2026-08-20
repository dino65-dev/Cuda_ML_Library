# CUDA ML Library [![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=plastic&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/dino65-dev/Cuda_ML_Library.git)

A reproducible CUDA-kernel research lab for studying inference operators from
Pascal-class consumer GPUs through modern cloud GPUs. DSpark and Decode Kernels
are the validated research modules; SVM, Random Forest, and the original
FlashAttention directory are retained as educational/incomplete prototypes.

## 🚀 Features

- **Decode Kernels**: Fused residual/RMSNorm, RMSNorm-to-INT8, QK norm + RoPE,
  KV-cache append, and bias + SwiGLU kernels with FP32/FP16/BF16 support.
- **Modern PyTorch Integration**: Stable `torch.library` schemas, FakeTensor,
  autograd registrations, `torch.compile`, current-stream, and CUDA Graph tests.
- **DeepSeek DSpark**: Architecture-aware Markov dispatch and a fused
  confidence-scheduled verification path.
- **Evidence First**: Independent PyTorch references, adversarial correctness
  tests, labelled trace assumptions, and machine-readable warm/cold benchmarks.
- **Cross-generation Research**: Raw CUDA paths retain a Pascal lane while Modal
  supplies reproducible SM 8.6 cloud-GPU validation.

## 📋 System Requirements

### Hardware Requirements
- **GPU**: Requirements are component-specific. DSpark has a validated Pascal
  path; the checked-in Decode Kernels cloud build currently targets SM 8.6.
- **CPU (Required)**: Any modern x86_64 processor
- **RAM**: 4GB+ system memory (8GB+ recommended for large datasets)

### Software Requirements
- **CUDA Toolkit**: Match it to the installed PyTorch build. The recorded Modal
  run used CUDA 12.8 and PyTorch 2.8.0.
- **Python**: 3.10+ for Decode Kernels; legacy modules may have different bounds.
- **Dependencies**: PyTorch ≥2.5 for Decode Kernels; PyTorch ≥2.1 for DSpark.

Only Linux CUDA builds are currently validated. CPU PyTorch references are
available for correctness, but they do not make the legacy CUDA modules
cross-platform packages.

## 🛠️ Installation

### Option 1: Install from PyPI (Not yet configured)

```bash
pip install cuda-ml-library
```

### Option 2: Build from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/dino65-dev/Cuda_ML_Library.git
cd Cuda_ML_Library

# Install dependencies
pip install numpy scikit-learn

# Build the CUDA library
cd SVM
make clean
make

# Install the package
cd ..
pip install -e .
```

Build the optional DSpark PyTorch extension separately:

```bash
cd DSpark
./install.sh
```

Build Decode Kernels separately on a CUDA development host:

```bash
cd decode_kernels
python -m pip install --no-build-isolation .
python -m pytest -q tests
```

Or use the authenticated Modal client from the repository root:

```bash
.venv/bin/python -m modal run modal/run_decode_gpu.py
```

See [Decode Kernels](./decode_kernels/README.md) for contracts and the
machine-readable validation artifact. The detailed speedup figures, raw Nsight
Systems captures, CUDA API/kernel analysis, and reproduction commands are in
[the performance analysis](./artifacts/NSYS_PERFORMANCE_ANALYSIS.md).

## 🚀 Quick Start

### Classification Example

```python
from SVM.cuda_svm import CudaSVC
import numpy as np

# Generate sample data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Create and train the model (automatically uses CUDA if available)
svc = CudaSVC(C=1.0, kernel='rbf', gamma='scale')
svc.fit(X, y)

# Make predictions
predictions = svc.predict(X_test)
probabilities = svc.predict_proba(X_test)  # If probability=True

print(f"Accuracy: {accuracy_score(y_test, predictions)}")
```

### Regression Example

```python
from SVM.cuda_svm import CudaSVR
import numpy as np

# Generate sample data
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=20, random_state=42)

# Create and train the model
svr = CudaSVR(C=1.0, epsilon=0.1, kernel='rbf', gamma='auto')
svr.fit(X, y)

# Make predictions
predictions = svr.predict(X_test)

print(f"R² Score: {r2_score(y_test, predictions)}")
```

### FlashAttention Example

```python
import torch
from flash_attention import FlashAttention

# Initialize FlashAttention module
attn = FlashAttention(head_dim=64)

# Create input tensors (batch_size, num_heads, seq_len, head_dim)
Q = torch.randn(2, 8, 512, 64, device='cuda', requires_grad=True)
K = torch.randn(2, 8, 512, 64, device='cuda', requires_grad=True)
V = torch.randn(2, 8, 512, 64, device='cuda', requires_grad=True)

# Forward pass with automatic gradient support
output = attn(Q, K, V)

# Use in training with any optimizer
optimizer = torch.optim.Adam(attn.parameters())
optimizer.zero_grad()
loss = output.sum()
loss.backward()  # Gradients computed automatically!
optimizer.step()

print(f"Output shape: {output.shape}")  # [2, 8, 512, 64]
print(f"Memory efficient: O(N) instead of O(N²)")
```

### DeepSeek DSpark Example

```python
import torch
from DSpark import DSparkMarkovHead, DSparkScheduler

requests, proposal_length = 128, 7
vocab_size, rank = 32_000, 256

head = DSparkMarkovHead(vocab_size, rank).cuda().eval()
scheduler = DSparkScheduler(proposal_length).cuda()

base_logits = torch.randn(requests, vocab_size, device="cuda", dtype=torch.float16)
previous_ids = torch.randint(vocab_size, (requests,), device="cuda")
confidence_logits = torch.randn(
    requests, proposal_length, device="cuda", dtype=torch.float16
)

max_batch = requests * (proposal_length + 1)
batch_tokens = torch.arange(max_batch + 1, device="cuda")
step_curve = 1_000.0 / (1.0 + batch_tokens / 256.0)

with torch.inference_mode():
    corrected_logits = head(base_logits, previous_ids)
    decision = scheduler(confidence_logits, step_curve)

verification_lengths = decision.lengths
```

See the [DSpark CUDA documentation](./DSpark/README.md) for the kernel design,
DeepSpec weight import, step-curve contract, tests, and GPU benchmark.

## 📚 API Reference

### CudaSVC (Classification)

```python
CudaSVC(
    svm_type='c_svc',     # 'c_svc' or 'nu_svc'
    kernel='rbf',         # 'linear', 'rbf', 'poly', 'sigmoid'
    C=1.0,               # Regularization parameter
    gamma='scale',        # Kernel coefficient
    coef0=0.0,           # Independent term for poly/sigmoid
    degree=3,            # Degree for polynomial kernel
    nu=0.5,              # Nu parameter for nu-SVM
    tolerance=1e-3,      # Tolerance for stopping criterion
    max_iter=1000,       # Maximum iterations
    shrinking=True,      # Use shrinking heuristic
    probability=False    # Enable probability estimates
)
```

### CudaSVR (Regression)

```python
CudaSVR(
    svm_type='epsilon_svr',  # 'epsilon_svr' or 'nu_svr'
    kernel='rbf',            # 'linear', 'rbf', 'poly', 'sigmoid'
    C=1.0,                   # Regularization parameter
    epsilon=0.1,             # Epsilon for epsilon-SVR
    gamma='scale',           # Kernel coefficient
    coef0=0.0,              # Independent term
    degree=3,               # Polynomial degree
    nu=0.5,                 # Nu parameter
    tolerance=1e-3,         # Stopping tolerance
    max_iter=1000          # Maximum iterations
)
```

### FlashAttention

```python
FlashAttention(
    head_dim=64             # Dimension of each attention head (currently fixed at 64)
)

# Functional interface (inference only)
flash_attention(
    Q,                      # Query tensor: [batch, heads, seq_len, head_dim]
    K,                      # Key tensor: [batch, heads, seq_len, head_dim]
    V                       # Value tensor: [batch, heads, seq_len, head_dim]
)
```

This is an educational FP32, fixed-head-dimension implementation. It is not the
validated serving-attention path and should not be used as a production
FlashAttention replacement.

### DSpark

```python
DSparkMarkovHead(vocab_size, rank=256)
DSparkScheduler(proposal_length=7, temperatures=None)
```

- FP32, FP16, and BF16 inference kernels
- Tensor-Core dense Markov update with fused single-CTA scheduling for common batches
- Paper-compatible first-throughput-drop admission rule
- No host synchronization on the CUDA scheduling path
- Native PyTorch fallback for CPU execution and Markov-head autograd

## 🔧 Advanced Usage

### Hardware Detection

```python
from SVM.cuda_svm import CudaSVC

# The library automatically detects and uses available hardware
svc = CudaSVC()
print("CUDA SVM initialized successfully")

# Hardware detection and optimization happen automatically
svc.fit(X_train, y_train)
```

### Kernel Customization

```python
# RBF Kernel with custom gamma
svc_rbf = CudaSVC(kernel='rbf', gamma=0.001)

# Polynomial Kernel
svc_poly = CudaSVC(kernel='poly', degree=4, coef0=1.0, gamma='auto')

# Linear Kernel (fastest)
svc_linear = CudaSVC(kernel='linear')

# Sigmoid Kernel
svc_sigmoid = CudaSVC(kernel='sigmoid', gamma='scale', coef0=0.0)
```

## ⚠️ Important Notes

### Current Status

- **Decode Kernels**: Validated research microkernels and benchmark foundation;
  integrate and re-profile before deployment.
- **DSpark**: Validated CUDA inference primitives plus exact PyTorch fallback;
  a complete serving integration remains future work.
- **SVM**: Incomplete solver prototype; it does not currently update SMO alphas.
- **Random Forest**: Educational placeholder, not a trained forest.
- **FlashAttention**: Educational fixed-shape FP32 prototype, not production-ready.

**Note**: For production transformer workloads with advanced features (FP16, variable head dimensions, attention masks), consider using the official [FlashAttention](https://github.com/Dao-AILab/flash-attention) implementation. This implementation is ideal for learning, prototyping, and small-scale training.

### Performance Tips

1. **GPU Memory**: Ensure sufficient GPU memory for large datasets
2. **Batch Processing**: For very large datasets, consider batch processing
3. **Kernel Selection**: Linear kernels are fastest, RBF kernels offer good accuracy
4. **Parameter Tuning**: Use cross-validation for optimal parameter selection

## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding new features, improving documentation, or optimizing performance, your help is appreciated.

### Ways to Contribute

- **Bug Reports**: Found a bug? [Open an issue](https://github.com/dino65-dev/Cuda_ML_Library/issues) with detailed reproduction steps
- **Feature Requests**: Have an idea? Share it through [GitHub Issues](https://github.com/dino65-dev/Cuda_ML_Library/issues)
- **Code Contributions**: Submit pull requests for bug fixes, new features, or optimizations
- **Documentation**: Help improve our docs, add examples, or fix typos
- **Testing**: Add test cases or report compatibility issues with different hardware/software configurations

### Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/dino65-dev/Cuda_ML_Library.git
   cd Cuda_ML_Library
   ```

2. **Set up your development environment**
   ```bash
   # Install dependencies
   pip install numpy scikit-learn torch
   
   # Build the project
   cd SVM && make clean && make && cd ..
   cd RandomForest && make clean && make && cd ..
   cd flash_attention && ./install.sh && cd ..
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

4. **Make your changes**
   - Write clean, readable code
   - Follow existing code style and conventions
   - Add comments for complex logic
   - Update documentation as needed

5. **Test your changes**
   ```bash
   # Run tests for the component you modified
   python -m pytest tests/
   
   # For CUDA components, test on both GPU and CPU
   python usage_example.py
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```
   
   **Commit Message Format:**
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Update:` for improvements to existing features
   - `Docs:` for documentation changes
   - `Refactor:` for code refactoring

7. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

8. **Open a Pull Request**
   - Go to the [original repository](https://github.com/dino65-dev/Cuda_ML_Library)
   - Click "New Pull Request"
   - Provide a clear description of your changes
   - Reference any related issues

### Development Guidelines

- **Code Quality**: Write clean, maintainable code with proper error handling
- **Performance**: Ensure CUDA code is optimized and memory-efficient
- **Compatibility**: Test on multiple GPU architectures when possible
- **Documentation**: Update README and inline comments for new features
- **Backward Compatibility**: Avoid breaking existing APIs unless necessary

### Areas for Contribution

We especially welcome contributions in these areas:

- **Performance Optimization**: Improve CUDA kernel efficiency
- **Hardware Support**: Test and optimize for more GPU architectures
- **New Algorithms**: Implement additional ML algorithms with CUDA acceleration
- **FP16/BF16 Support**: Add mixed-precision training capabilities
- **Distributed Training**: Multi-GPU support and distributed computing
- **Documentation**: More examples, tutorials, and API documentation
- **Testing**: Expand test coverage and add benchmarks

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on collaboration and learning
- Help others who are contributing

### Questions?

If you have questions about contributing, feel free to:
- Open a [GitHub Discussion](https://github.com/dino65-dev/Cuda_ML_Library/discussions)
- Comment on an existing issue
- Reach out to the maintainers

Thank you for making CUDA ML Library better! 🎉

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Repository**: [https://github.com/dino65-dev/Cuda_ML_Library](https://github.com/dino65-dev/Cuda_ML_Library)
- **Issues**: [https://github.com/dino65-dev/Cuda_ML_Library/issues](https://github.com/dino65-dev/Cuda_ML_Library/issues)
- **Documentation**: 
  - [SVM Usage Examples](./Usage/SVM/)
  - [Random Forest Usage Examples](./Usage/Random_forest/)
  - [FlashAttention Documentation](./flash_attention/USAGE.md)
  - [DSpark CUDA Documentation](./DSpark/README.md)

## 📊 Version

Current Version: **1.0.1**

---

**Made with ❤️ by [dino65-dev](https://github.com/dino65-dev)**
