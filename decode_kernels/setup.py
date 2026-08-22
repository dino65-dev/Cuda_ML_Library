from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent


setup(
    name="cuda-ml-decode-kernels",
    version="0.1.0",
    description="Validated CUDA decode microkernels for Cuda_ML_Library",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="cuda_ml_decode._C",
            sources=[
                str(ROOT / "csrc" / "decode_ops.cpp"),
                str(ROOT / "csrc" / "decode_ops_cuda.cu"),
                str(ROOT / "csrc" / "paged_attention_cuda.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.10",
    install_requires=["torch>=2.5"],
    zip_safe=False,
)
