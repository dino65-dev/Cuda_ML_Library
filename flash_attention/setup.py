"""
FlashAttention CUDA Extension Setup

Build configuration for the FlashAttention PyTorch CUDA extension.

Usage:
    pip install -e .

Before building, ensure:
    - CUDA_HOME is set correctly
    - CXX points to g++ or compatible compiler
    - GPU architecture matches your hardware (see -arch flag below)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Determine CUDA architecture
# Edit this based on your GPU - common values:
# sm_61: GTX 1050/1060
# sm_70: V100
# sm_75: T4, RTX 2080
# sm_80: A100
# sm_86: RTX 3090
# sm_89: RTX 4090

# Handle TORCH_CUDA_ARCH_LIST which may contain multiple archs separated by semicolons
arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', 'sm_75')
# If multiple archs, use the first one (or just use sm_75 for T4)
if ';' in arch_list:
    # Extract just the first architecture or use sm_75
    CUDA_ARCH = 'sm_75'  # Force sm_75 for T4 GPU
else:
    CUDA_ARCH = arch_list if arch_list.startswith('sm_') else f'sm_{arch_list}'

setup(
    name='flash_attention',
    version='0.1.0',
    author='Your Name',
    description='Educational FlashAttention CUDA implementation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    ext_modules=[
        CUDAExtension(
            name='_flash_attention_cuda',  # Underscore prefix to avoid conflict with .py wrapper
            sources=['flash_attention.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    f'-arch={CUDA_ARCH}',  # GPU architecture
                    '--use_fast_math',
                    '-lineinfo',
                    '-allow-unsupported-compiler',
                    '-ccbin', 'g++',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.12.0',
    ],
)
