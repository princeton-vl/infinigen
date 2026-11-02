# Copyright: 2009-2022 the scikit-image team
# License: BSD-3-Clause
# adapted by Zeyu Ma to avoid precision loss for large grids to achieve local determinism,
# therefore to make seamless stiched mesh

# Authors:
# - Zeyu Ma: precision-safe adaptation in the original Infinigen marching cubes (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/terrain/mesher/_marching_cubes_lewiner.py)
# - David Yan, Jia Deng: optimized implementation

import numpy as np

from infinigen.terrain import marching_cubes as _marching_cubes_lewiner_cy
from infinigen.terrain.mesher._marching_cubes_lewiner import _get_mc_luts


def marching_cubes_zero_level(volume):
    """`(vertices_integral, vertices_fractal, faces)` as `marching_cubes(volume, 0)`
    computes them with default arguments, without its checks, so 0 must lie within
    the volume's value range."""
    volume = np.ascontiguousarray(volume, np.float32)  # no copy if not necessary
    vertices_integral, vertices_fractal, faces, _, _ = (
        _marching_cubes_lewiner_cy.marching_cubes(
            volume, 0.0, _get_mc_luts(), 1, False, None
        )
    )
    if not len(vertices_integral):
        raise RuntimeError("No surface found at the given iso value.")
    vertices_integral = np.fliplr(vertices_integral)
    vertices_fractal = np.fliplr(vertices_fractal)
    faces.shape = -1, 3
    faces = np.fliplr(faces)
    vertices_fractal[np.abs(vertices_fractal) < 1e-30] = 0
    vertices_fractal[np.abs(vertices_fractal - 1) < 1e-30] = 1
    return vertices_integral, vertices_fractal, faces


def marching_cubes_zero_level_blocks(blocks, bounds, block_idx, elem_idx=None):
    """`marching_cubes_zero_level` of float32 `blocks` (count, S, S, S[, elements])
    at `block_idx` and channels `elem_idx`, concatenated in that order. `bounds`
    (count, 3) is the last index kept per axis, or None for whole blocks. Returns
    integral and fractal vertices, their float64 sum, faces indexing their own
    block's vertices, and per-block vertex and face counts."""
    blocks = np.asarray(blocks)
    if blocks.dtype != np.float32:
        raise TypeError(f"blocks must be float32, got {blocks.dtype}")
    if blocks.ndim == 4:
        blocks = blocks[..., None]
    count = blocks.shape[0]
    if bounds is None:
        bounds = np.empty((count, 3), dtype=np.int32)
        bounds[:] = [blocks.shape[1] - 1, blocks.shape[2] - 1, blocks.shape[3] - 1]
    else:
        bounds = np.ascontiguousarray(bounds, dtype=np.int32).reshape(count, 3)
    block_idx = np.ascontiguousarray(block_idx, dtype=np.int64)
    if elem_idx is None:
        elem_idx = np.zeros(len(block_idx), dtype=np.int64)
    else:
        elem_idx = np.ascontiguousarray(elem_idx, dtype=np.int64)
    vi, vf, vs, fc, ns, ms = [], [], [], [], [], []
    for i, e in zip(block_idx.tolist(), elem_idx.tolist()):
        b0, b1, b2 = bounds[i]
        v_int, v_frac, faces = marching_cubes_zero_level(
            blocks[i, : b0 + 1, : b1 + 1, : b2 + 1, e]
        )
        vi.append(v_int.astype(np.int32))
        vf.append(v_frac.astype(np.float64))
        vs.append((v_int + v_frac).astype(np.float64))
        fc.append(faces.astype(np.int32))
        ns.append(len(v_int))
        ms.append(len(faces))
    return (
        np.concatenate(vi) if vi else np.zeros((0, 3), np.int32),
        np.concatenate(vf) if vf else np.zeros((0, 3), np.float64),
        np.concatenate(vs) if vs else np.zeros((0, 3), np.float64),
        np.concatenate(fc) if fc else np.zeros((0, 3), np.int32),
        np.array(ns, dtype=np.int64),
        np.array(ms, dtype=np.int64),
    )
