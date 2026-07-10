"""Build the DSpark PyTorch CUDA extension."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent

setup(
    name="cuda-ml-dspark",
    version="0.2.0",
    description="Low-level CUDA inference primitives for DeepSeek DSpark",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=["DSpark"],
    package_dir={"DSpark": "."},
    ext_modules=[
        CUDAExtension(
            name="_dspark_cuda",
            sources=[
                "csrc/bindings.cpp",
                "csrc/dspark_cuda.cu",
            ],
            include_dirs=[str(ROOT / "csrc")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    package_data={"DSpark": ["README.md"]},
    python_requires=">=3.9",
    install_requires=["torch>=2.1"],
    zip_safe=False,
)
