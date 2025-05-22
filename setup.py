from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

os.path.dirname(os.path.abspath(__file__))

setup(
    name="sobel_filter_cuda",
    ext_modules=[
        CUDAExtension(
            name="sobel_filter_cuda",
            sources=["src/sobel_filter.cu", "src/bindings.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
