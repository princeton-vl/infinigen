# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Gin settings of the optimized packages, inert unless their package is selected."""

import os

import gin


@gin.configurable
def isolation_enabled(enabled=True):
    """Build assets and animate cameras in a scratch view layer holding only their
    objects. Off uses the main view layer, as the default does."""
    return enabled


@gin.configurable
def compress_output_blend(enabled=True):
    """Write stage output .blend files with Blender's lossless compression, so later
    stages read smaller files. Off writes them uncompressed, as the default does."""
    return enabled


@gin.configurable
def only_needed_land_processes(enabled=True):
    """Skip land process simulations whose outputs the scene never reads (asset-library
    generation still runs them all). Off simulates all of them, as the default does."""
    return enabled


@gin.configurable
def prefetch_input_scene(enabled=True):
    """Pull an input scene and its terrain assets into the page cache with large reads
    before Blender reads them in small requests. Off opens them directly."""
    return enabled


@gin.configurable
def direct_monocot_leaves(enabled=True):
    """Build monocot leaf sheets, their decoration and the agave's join from arrays,
    with a different face count. Off builds just these with the default operators."""
    return enabled


@gin.configurable
def native_pinecone(enabled=True):
    """Build pinecones with the native kernels when compiled, at a slightly different
    triangle count. Off keeps the default pinecone in the optimized `monocot`."""
    return enabled


@gin.configurable
def native_tree_meshing(enabled=True):
    """Mesh tree and bush trunks as a bark-displaced tube per stem instead of voxel
    remeshing them, changing topology. Off keeps the remesh in the optimized `trees`."""
    return enabled


@gin.configurable
def native_tree_edge_length(ceiling_factor=1.7):
    """Edge length ceiling of native tree meshing, as a multiple of the target face
    size; lower gives denser trunks (the kernel's own default is 1.0)."""
    if ceiling_factor <= 0:
        raise ValueError(f"Expected a positive {ceiling_factor=}")
    return ceiling_factor


@gin.configurable
def optix_kernel_cache(enabled=True, directory=None):
    """Where GPU renders cache OptiX kernels, by default under `$XDG_CACHE_HOME` or
    `~/.cache`. Off leaves it to the driver; a set `OPTIX_CACHE_PATH` is kept."""
    if not enabled:
        return None
    if directory is None:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        directory = os.path.join(base, "infinigen", "optix")
    return os.path.expanduser(directory)


# registers land_process_pool with gin
from infinigen.perf.terrain.land_process import pool as _pool  # noqa: E402,F401
