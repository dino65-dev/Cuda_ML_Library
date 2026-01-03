# CUDA ML Library

A high-performance CUDA-accelerated Machine Learning library with automatic CPU fallback support, featuring optimized Support Vector Machine implementations for both classification and regression tasks.

## 🚀 Features

- **GPU Acceleration**: Full CUDA support for NVIDIA GPUs with Compute Capability 6.0+
- **Automatic CPU Fallback**: Seamless fallback to optimized CPU implementation when CUDA is unavailable
- **Cross-Platform Compatibility**: Linux, Windows, and macOS support
- **Multiple SVM Types**: Classification (C-SVC, Nu-SVC) and Regression (Epsilon-SVR, Nu-SVR)
- **Multiple Kernel Functions**: Linear, RBF, Polynomial, and Sigmoid kernels
- **Advanced Algorithms**: SMO (Sequential Minimal Optimization) algorithm implementation
- **FlashAttention**: Memory-efficient O(N) attention mechanism for transformer models with full training support
- **Memory Optimization**: Efficient GPU memory management with pooling
- **Easy Integration**: Scikit-learn compatible API and PyTorch integration

## 📋 System Requirements

### Hardware Requirements
- **GPU (Optional)**: NVIDIA GPU with CUDA Compute Capability 6.0+ (RTX 20 series, GTX 1050Ti+, Tesla V100+)
- **CPU (Required)**: Any modern x86_64 processor
- **RAM**: 4GB+ system memory (8GB+ recommended for large datasets)

### Software Requirements
- **CUDA Toolkit** (Optional): Version 12.0+ for GPU acceleration
- **Python**: 3.8+
- **Dependencies**: numpy ≥1.19.0, scikit-learn ≥1.0.0

### Supported Environments
- **GPU-Accelerated**: Systems with CUDA-capable NVIDIA GPUs
- **CPU-Only**: Any system (automatic fallback when CUDA unavailable)
- **Cloud Platforms**: Google Colab, AWS, Azure, etc.
- **Cross-Platform**: Linux, Windows, macOS

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

The build process will:
- Auto-detect CUDA availability and GPU architecture
- Compile CUDA kernels when GPU is available
- Create CPU fallback implementation when CUDA is unavailable
- Generate optimized shared libraries with universal compatibility

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

**Key Features:**
- O(N) memory complexity instead of O(N²)
- Full gradient support for training
- PyTorch integration with `.backward()`
- Numerical accuracy < 1e-6 vs standard attention
- Works with all PyTorch optimizers (Adam, SGD, etc.)

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

- **SVM**: Fully functional and ready for production use
- **RF**: Fully functional and ready for production use
- **FlashAttention**: Fully functional for training and inference (head_dim=64, FP32 only)

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
   git clone https://github.com/YOUR_USERNAME/Cuda_ML_Library.git
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

## 📊 Version

Current Version: **0.1.0**

---

**Made with ❤️ by [dino65-dev](https://github.com/dino65-dev)**
