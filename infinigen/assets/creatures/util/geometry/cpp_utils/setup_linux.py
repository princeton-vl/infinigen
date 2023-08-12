from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext
from Cython.Build import cythonize


extra_compile_args = [] 
extra_link_args = [] 

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize([
        Extension(
            "bnurbs",
            sources=["bnurbs.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        )
    ])
)