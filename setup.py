from pathlib import Path
import subprocess
import sys

from setuptools import setup, Extension

import numpy
from Cython.Build import cythonize

cwd = Path(__file__).parent

TERRAIN = True
CUSTOMGT = True
FLUIDS = True
RUN_BUILD_DEPS = True

filtered_args = []
for i, arg in enumerate(sys.argv):
    if arg in ["clean", "egg_info", "sdist"]:
        RUN_BUILD_DEPS = False
    elif arg == '--noterrain':
        TERRAIN = False
    elif arg == '--nogt':
        CUSTOMGT = False
    elif arg == '--nofluids':
        FLUIDS = False
    filtered_args.append(arg)
sys.argv = filtered_args

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

    if any(not p.exists() or not any(p.iterdir()) for p in folders):
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"], cwd=cwd
        )

ensure_submodules()

def build_deps(deps):
    for dep, enabled in deps:
        if not enabled:
            continue
        print(f'Building external executable {dep}')
        try:
            # Defer to Makefile
            subprocess.run(['make', f'build_{dep}'], cwd=cwd)
        except subprocess.CalledProcessError as e:
            print(f'[WARNING] build_{dep} failed! {dep} features will not function. {e}')

if RUN_BUILD_DEPS:
    deps = [
        ('terrain', TERRAIN),
        ('custom_groundtruth', CUSTOMGT),
        ('flip_fluids', FLUIDS)
    ]
    build_deps(deps)

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
    ],
    package_data={
        "infinigen": [
            "infinigen/terrain/lib",
            "infinigen/datagen/customgt/build"
        ]
    }
    # other opts come from pyproject.toml and setup.cfg
)


