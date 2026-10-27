# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

import pytest

from infinigen._perf_selection import selected  # noqa: E402

if not selected("TERRAIN"):
    pytest.skip(
        "exercises the optimized terrain implementation; "
        "pass --optimized=terrain (or --optimized) to run",
        allow_module_level=True,
    )


import numpy as np
import pytest

from infinigen.perf.terrain.mesher._marching_cubes_lewiner import (
    marching_cubes_zero_level,
    marching_cubes_zero_level_blocks,
)
from infinigen.terrain.mesher._marching_cubes_lewiner import marching_cubes


def _marching_cubes_built():
    """True when the compiled terrain extension behind the `marching_cubes` wrapper
    is built; the wrapper module imports without it."""
    try:
        from infinigen.terrain import marching_cubes as extension
    except ImportError:
        return False
    return hasattr(extension, "LutProvider")


needs_marching_cubes = pytest.mark.skipif(
    not _marching_cubes_built(),
    reason="needs the compiled terrain extensions (INFINIGEN_MINIMAL_INSTALL=False)",
)


def _blocks(rng, count, S):
    blocks = rng.normal(size=(count, S, S, S)).astype(np.float64)
    blocks[0] = np.abs(blocks[0])  # no surface
    blocks[1] = -np.abs(blocks[1])  # no surface
    blocks[2, 0, 0, 0] = 0.0  # exact zero values
    blocks[3] = np.where(rng.random((S, S, S)) < 0.5, 0.0, blocks[3])
    return blocks


@needs_marching_cubes
def test_zero_level_blocks_concatenates_the_per_block_meshes():
    rng = np.random.default_rng(7)
    S = 5
    blocks = rng.normal(size=(12, S, S, S, 2)).astype(np.float32)
    bounds = rng.integers(2, S, size=(12, 3)).astype(np.int32)
    block_idx = np.arange(12)
    elem_idx = rng.integers(0, 2, size=12)
    vi, vf, vs, faces, ns, ms = marching_cubes_zero_level_blocks(
        blocks, bounds, block_idx, elem_idx
    )
    v0 = f0 = 0
    for i, e in zip(block_idx, elem_idx):
        b0, b1, b2 = bounds[i]
        want_vi, want_vf, want_faces = marching_cubes_zero_level(
            blocks[i, : b0 + 1, : b1 + 1, : b2 + 1, e]
        )
        n, m = len(want_vi), len(want_faces)
        assert (ns[i], ms[i]) == (n, m)
        assert np.array_equal(vi[v0 : v0 + n], want_vi)
        assert np.array_equal(vf[v0 : v0 + n], want_vf)
        assert np.array_equal(vs[v0 : v0 + n], want_vi + want_vf)
        assert np.array_equal(faces[f0 : f0 + m], want_faces)
        v0, f0 = v0 + n, f0 + m
    assert v0 == len(vi) and f0 == len(faces)


@needs_marching_cubes
@pytest.mark.parametrize("S", [3, 5, 6])
def test_marching_cubes_zero_level_matches_wrapper(S):
    rng = np.random.default_rng(S)
    blocks = _blocks(rng, 64, S)
    straddles = ~(blocks >= 0).all(axis=(1, 2, 3)) & ~(blocks <= 0).all(axis=(1, 2, 3))
    n = 0
    for i in range(len(blocks)):
        part = blocks[i]
        if (part >= 0).all() or (part <= 0).all():
            assert not straddles[i]
            continue
        assert straddles[i]
        vi, vf, faces, _, _ = marching_cubes(part, 0)
        vi2, vf2, faces2 = marching_cubes_zero_level(part)
        assert np.array_equal(vi, vi2) and np.array_equal(vf, vf2)
        assert np.array_equal(faces, faces2) and faces2.shape[1] == 3
        n += 1
    assert n > 40
    # sliced (non-contiguous) blocks as used by the coarse loops
    part = blocks[10][:, : S - 1, 1:]
    vi, vf, faces, _, _ = marching_cubes(part, 0)
    vi2, vf2, faces2 = marching_cubes_zero_level(part)
    assert (
        np.array_equal(vi, vi2)
        and np.array_equal(vf, vf2)
        and np.array_equal(faces, faces2)
    )


def test_cube_mesher_straddle_test_matches_per_block():
    rng = np.random.default_rng(7)
    blocks = _blocks(rng, 128, 5)
    vectorised = (blocks > 0).any(axis=(1, 2, 3)) & (blocks <= 0).any(axis=(1, 2, 3))
    expected = np.array([(p > 0).any() and (p <= 0).any() for p in blocks])
    assert np.array_equal(vectorised, expected)
