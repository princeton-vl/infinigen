# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

# Acknowledgement: This file draws inspiration from https://github.com/pytorch/pytorch/blob/main/setup.py


import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

cwd = Path(__file__).parent

str_true = "True"
MINIMAL_INSTALL = os.environ.get("INFINIGEN_MINIMAL_INSTALL", str_true) == str_true
BUILD_TERRAIN = os.environ.get("INFINIGEN_INSTALL_TERRAIN", str_true) == str_true
BUILD_OPENGL = os.environ.get("INFINIGEN_INSTALL_CUSTOMGT", "False") == str_true
# Native kernels for infinigen.perf. Off by default; without them each optimized
# implementation falls back to the default one.
BUILD_PERF = os.environ.get("INFINIGEN_INSTALL_PERF", "False") == str_true

dont_build_steps = ["clean", "egg_info", "dist_info", "sdist", "--help"]
is_build_step = not any(x in sys.argv[1] for x in dont_build_steps)


def ensure_submodules():
    # Inspired by https://github.com/pytorch/pytorch/blob/main/setup.py

    gitmodules = cwd / ".gitmodules"
    if not gitmodules.exists():
        # submodules were removed; optional components (OcMesher, infinigen_gpl,
        # customgt dependencies) are now cloned manually if needed
        return

    with gitmodules.open() as f:
        submodule_folders = [
            cwd / line.split("=", 1)[1].strip()
            for line in f.readlines()
            if line.strip().startswith("path")
        ]

    if any(not p.exists() or not any(p.iterdir()) for p in submodule_folders):
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"], cwd=cwd, check=True
        )


if not MINIMAL_INSTALL:
    ensure_submodules()

# inspired by https://github.com/pytorch/pytorch/blob/161ea463e690dcb91a30faacbf7d100b98524b6b/setup.py#L290
# theirs seems to not exclude dist_info but this causes duplicate compiling in my tests
if is_build_step and not MINIMAL_INSTALL:
    if BUILD_TERRAIN:
        subprocess.run(["make", "terrain"], cwd=cwd, check=True)
    if BUILD_OPENGL:
        subprocess.run(["make", "customgt"], cwd=cwd, check=True)

PERF_FLAGS = ["-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off"]

perf_extensions = [
    Extension(
        name="infinigen.perf.assets.utils._native_fields",
        sources=[
            "src/infinigen/perf/assets/utils/native/_native_fields.pyx",
            "src/infinigen/perf/assets/objects/leaves/native/broadleaf_native_fields.cpp",
            "src/infinigen/perf/assets/objects/leaves/native/planar_native_fields.cpp",
            "src/infinigen/perf/assets/materials/plant/native/bark_native_fields.cpp",
        ],
        depends=[
            "src/infinigen/perf/assets/utils/native/field_ops.hpp",
            "src/infinigen/perf/assets/utils/native/extended_ops.hpp",
            "src/infinigen/perf/assets/utils/native/native_fields.h",
        ],
        include_dirs=[numpy.get_include(), "src/infinigen/perf/assets/utils/native"],
        extra_compile_args=[
            "-O3",
            "-std=c++17",
            "-fno-fast-math",
            "-ffp-contract=fast",
        ],
        language="c++",
    ),
    Extension(
        name="infinigen.perf.assets.objects.trees._tree_ops",
        sources=["src/infinigen/perf/assets/objects/trees/native/_tree_ops.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
    ),
    Extension(
        name="infinigen.perf.assets.utils.heightfield._adaptive_triangulation",
        sources=[
            "src/infinigen/perf/assets/utils/heightfield/native/_adaptive_triangulation.pyx",
            "src/infinigen/perf/assets/utils/heightfield/native/third_party/meshoptimizer/simplifier.cpp",
        ],
        depends=[
            "src/infinigen/perf/assets/utils/heightfield/native/_flat_grid_preconditioner.hpp",
            "src/infinigen/perf/assets/utils/heightfield/native/_quad_triangulation.hpp",
            "src/infinigen/perf/assets/utils/heightfield/native/third_party/meshoptimizer/meshoptimizer.h",
        ],
        include_dirs=[
            numpy.get_include(),
            "src/infinigen/perf/assets/utils/heightfield",
            "src/infinigen/perf/assets/utils/heightfield/native/third_party/meshoptimizer",
        ],
        language="c++",
    ),
    # Mushroom kernels.
    Extension(
        name="infinigen.perf.assets.objects.mushroom._meshing",
        sources=["src/infinigen/perf/assets/objects/mushroom/native/_meshing.pyx"],
        depends=["src/infinigen/perf/assets/objects/mushroom/native/_meshing_core.hpp"],
        include_dirs=[
            numpy.get_include(),
            "src/infinigen/perf/assets/objects/mushroom/native",
        ],
        extra_compile_args=PERF_FLAGS,
        language="c++",
    ),
    Extension(
        name="infinigen.perf.assets.objects.mushroom._adaptive",
        sources=["src/infinigen/perf/assets/objects/mushroom/native/_adaptive.pyx"],
        depends=[
            "src/infinigen/perf/assets/objects/mushroom/native/_adaptive_core.hpp"
        ],
        include_dirs=[
            numpy.get_include(),
            "src/infinigen/perf/assets/objects/mushroom/native",
        ],
        extra_compile_args=PERF_FLAGS,
        language="c++",
    ),
]

# GPL-2.0-or-later kernels live in the optional infinigen_gpl submodule, so this
# tree carries no GPL source. Without it, their users fall back to the defaults.
GPL_PERF = cwd / "src/infinigen/infinigen_gpl/perf"
if GPL_PERF.is_dir():
    perf_extensions += [
        Extension(
            name="infinigen.infinigen_gpl.perf.particle_sampler._particle_sampler",
            sources=[
                "src/infinigen/infinigen_gpl/perf/particle_sampler/_particle_sampler.pyx"
            ],
            depends=[
                "src/infinigen/infinigen_gpl/perf/particle_sampler/_particle_sampler_core.hpp"
            ],
            include_dirs=[
                numpy.get_include(),
                "src/infinigen/infinigen_gpl/perf/particle_sampler",
            ],
            language="c++",
        ),
        Extension(
            name="infinigen.infinigen_gpl.perf.spin._spin",
            sources=["src/infinigen/infinigen_gpl/perf/spin/_spin.pyx"],
            depends=["src/infinigen/infinigen_gpl/perf/spin/_spin_core.hpp"],
            include_dirs=[numpy.get_include(), "src/infinigen/infinigen_gpl/perf/spin"],
            extra_compile_args=PERF_FLAGS,
            language="c++",
        ),
        Extension(
            name="infinigen.infinigen_gpl.perf.voronoi._voronoi",
            sources=["src/infinigen/infinigen_gpl/perf/voronoi/_voronoi.pyx"],
            depends=[
                "src/infinigen/infinigen_gpl/perf/voronoi/_voronoi_core.hpp",
                "src/infinigen/infinigen_gpl/perf/voronoi/_texture_core.hpp",
            ],
            include_dirs=[
                numpy.get_include(),
                "src/infinigen/infinigen_gpl/perf/voronoi",
            ],
            extra_compile_args=PERF_FLAGS,
            language="c++",
        ),
    ]

# Native tree meshing: ring resampling and bulk mesh writes.
perf_extensions.append(
    Extension(
        name="infinigen.perf.assets.objects.trees._cylinder_mesher",
        sources=["src/infinigen/perf/assets/objects/trees/native/_cylinder_mesher.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
    )
)

cython_extensions = []

if BUILD_PERF:
    cython_extensions += perf_extensions

if not MINIMAL_INSTALL:
    if BUILD_TERRAIN:
        cython_extensions.append(
            Extension(
                name="infinigen.terrain.marching_cubes",
                sources=[
                    "src/infinigen/terrain/marching_cubes/_marching_cubes_lewiner_cy.pyx"
                ],
                include_dirs=[numpy.get_include()],
            )
        )

ext_modules = list(cythonize(cython_extensions)) if cython_extensions else []


# The pinecone kernels are ctypes libraries (loaded with load_library, not
# imported), so they are not cythonized with the rest.
def perf_ctypes_extensions():
    # load_library resolves these relative to the optimized monocot package, so
    # they are built there.
    package = "src/infinigen/perf/assets/objects/monocot/_pinecone/native"
    gpl = "src/infinigen/infinigen_gpl/perf/pinecone"
    fields = "src/infinigen/perf/assets/utils/native"
    flags = ["-O3", "-std=c++17", "-ffp-contract=off", "-fno-fast-math"]
    result = []
    names = ["surface_grid", "subdivide", "topology", "decorate_dispatch"]
    # stucci and deform are GPL-2.0-or-later and build into the optional
    # infinigen_gpl submodule; without it the pinecone uses the default.
    if Path(gpl).is_dir():
        names += ["stucci", "deform"]
    for name in names:
        gpl_kernel = name in {"stucci", "deform"}
        sources = [f"{gpl if gpl_kernel else package}/{name}.cpp"]
        macros = []
        if name == "decorate_dispatch":
            sources.append(f"{package}/decorate.cpp")
            macros.append(("PINECONE_SCALAR_ENTRY", "1"))
            if platform.machine().lower() in {"x86_64", "amd64"}:
                sources.append(f"{package}/decorate_avx2.cpp")
                macros.append(("PINECONE_HAVE_AVX2", "1"))
        result.append(
            Extension(
                f"infinigen.infinigen_gpl.perf.pinecone.{name}"
                if gpl_kernel
                else f"infinigen.perf.assets.objects.monocot._pinecone.native.{name}",
                sources,
                include_dirs=[fields],
                depends=[f"{fields}/field_ops.hpp"],
                define_macros=macros,
                extra_compile_args=flags,
                language="c++",
            )
        )
    return result


if BUILD_PERF:
    ext_modules += perf_ctypes_extensions()

setup(
    ext_modules=ext_modules
    # other opts come from pyproject.toml
)
