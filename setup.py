from pathlib import Path
import subprocess

from setuptools import setup, Extension

import numpy
from Cython.Build import cythonize

cwd = Path(__file__).parent

def get_submodule_folders():
    # Inspired by https://github.com/pytorch/pytorch/blob/main/setup.py
    with (cwd/'.gitmodules').open() as f:
        return [
            cwd/line.split("=", 1)[1].strip()
            for line in f.readlines()
            if line.strip().startswith("path")
        ]

def ensure_submodules():
    # Inspired by https://github.com/pytorch/pytorch/blob/main/setup.py

    folders = get_submodule_folders()

    isempty = lambda p: not any(p.iterdir())
    if any(not p.exists() or isempty(p) for p in folders):
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "--recursive"], cwd=cwd
        )

ensure_submodules() # not actually needed for this version of setup.py, but will be in future 
    
cython_extensions = [
    Extension(
        name="bnurbs",
        sources=["infinigen/assets/creatures/util/geometry/cpp_utils/bnurbs.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        name="infinigen.terrain.marching_cubes",
        sources=["infinigen/terrain/marching_cubes/_marching_cubes_lewiner_cy.pyx"],
        include_dirs=[numpy.get_include()]
    ),
]

setup(
    ext_modules=[
        *cythonize(cython_extensions)
    ]
    # other opts come from setup.cfg
)