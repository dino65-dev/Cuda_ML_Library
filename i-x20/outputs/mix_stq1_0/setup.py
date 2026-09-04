from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent

setup(
    name="mix-stq1-0-experiment",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="mix_stq1_0_cuda",
            sources=[str(ROOT / "csrc" / "stq_cuda.cu")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
