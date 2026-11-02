# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

from __future__ import annotations

import numpy as np


def precondition_flat_grid_regions(
    vertices: np.ndarray,
    vertex_ids: np.ndarray,
    complete_quads: np.ndarray,
    *,
    tolerance: float = 0.0,
    minimum_block: int = 4,
    metric_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, int, int]:
    """Fan-triangulate grid blocks planar within `tolerance` and keep the other
    cells, returning the faces and the flat cell and block counts. Flat blocks keep
    every boundary lattice vertex, so they meet their neighbors without cracks."""
    vertex_array = np.ascontiguousarray(vertices, dtype=np.float32)
    ids = np.ascontiguousarray(vertex_ids, dtype=np.int32)
    complete = np.ascontiguousarray(complete_quads, dtype=np.uint8)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if ids.ndim != 2 or complete.shape != (ids.shape[0] - 1, ids.shape[1] - 1):
        raise ValueError("complete_quads must cover the cells of vertex_ids")
    if tolerance < 0 or minimum_block < 2 or minimum_block & (minimum_block - 1):
        raise ValueError("minimum_block must be a power of two of at least two")
    from . import _adaptive_triangulation

    return _adaptive_triangulation.precondition_flat_regions(
        vertex_array,
        ids,
        complete,
        float(tolerance),
        int(minimum_block),
        np.ascontiguousarray(metric_scale, dtype=np.float32),
    )


def simplify_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_budget: int | None = None,
    *,
    ratio: float | None = None,
    lock_border: bool = False,
    target_error: float = 1e30,
    regularize: str = "none",
    update_positions: bool = False,
    metric_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    features: np.ndarray | None = None,
    feature_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Edge-collapse a triangle mesh with meshoptimizer to `face_budget` or `ratio`
    (give one), stopping early at relative `target_error`. Returns the kept vertices,
    remapped faces, their source indices and the relative error. `regularize` trades
    error for even triangles, `update_positions` moves kept vertices to quadric
    optima, `metric_scale` scales XYZ only in the error so small height-field relief
    counts, and `features` (N, C) weighted by `feature_weights` (C,) join the error."""
    from . import _adaptive_triangulation

    source_vertices = np.array(vertices, dtype=np.float32, order="C", copy=True)
    face_array = np.array(faces, dtype=np.uint32, order="C", copy=True)
    if source_vertices.ndim != 2 or source_vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if len(face_array) and int(face_array.max()) >= len(source_vertices):
        raise ValueError("faces reference a vertex outside vertices")
    scale = np.asarray(metric_scale, dtype=np.float32)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("metric_scale must contain three positive finite values")
    vertex_array = np.ascontiguousarray(source_vertices * scale, dtype=np.float32)
    if (features is None) != (feature_weights is None):
        raise ValueError("features and feature_weights must be provided together")
    if features is None:
        feature_array = None
        weight_array = None
    else:
        feature_array = np.asarray(features, dtype=np.float32)
        if feature_array.ndim == 1:
            feature_array = feature_array[:, None]
        if feature_array.ndim != 2 or feature_array.shape[0] != len(source_vertices):
            raise ValueError("features must have shape (N,) or (N, C)")
        if feature_array.shape[1] > 32:
            raise ValueError("features may contain at most 32 channels")
        if not np.all(np.isfinite(feature_array)):
            raise ValueError("features must contain only finite values")
        feature_array = np.ascontiguousarray(feature_array, dtype=np.float32)
        weight_array = np.asarray(feature_weights, dtype=np.float32)
        if weight_array.shape != (feature_array.shape[1],):
            raise ValueError("feature_weights must have shape (C,)")
        if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0):
            raise ValueError("feature_weights must be non-negative and finite")
        weight_array = np.ascontiguousarray(weight_array, dtype=np.float32)
    if (face_budget is None) == (ratio is None):
        raise ValueError("provide exactly one of face_budget or ratio")
    if ratio is not None:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must be between zero and one")
        face_budget = max(1, round(len(face_array) * ratio))
    assert face_budget is not None
    regularize_value = {"none": 0, "light": 1, "full": 2}.get(regularize)
    if regularize_value is None:
        raise ValueError("regularize must be 'none', 'light', or 'full'")
    simplified, error = _adaptive_triangulation.simplify_geometry(
        vertex_array,
        face_array,
        face_budget,
        lock_border,
        target_error,
        regularize_value,
        update_positions,
        feature_array,
        weight_array,
    )
    source_indices = np.unique(simplified)
    remap = np.full(len(vertex_array), -1, dtype=np.int32)
    remap[source_indices] = np.arange(len(source_indices), dtype=np.int32)
    output_vertices = (
        vertex_array[source_indices] / scale
        if update_positions
        else source_vertices[source_indices]
    )
    return (
        np.ascontiguousarray(output_vertices, dtype=np.float32),
        np.ascontiguousarray(remap[simplified], dtype=np.int32),
        source_indices,
        float(error),
    )


def simplify_geometry_with_fields(
    vertices: np.ndarray,
    faces: np.ndarray,
    fields: np.ndarray,
    face_budget: int | None = None,
    *,
    ratio: float | None = None,
    lock_border: bool = False,
    target_error: float = 1e30,
    regularize: str = "none",
    metric_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """`simplify_geometry` moving vertices to each collapse's quadric target, with
    `fields` (N, C up to 32) interpolated there without affecting the collapse.
    Returns positions, remapped faces, source indices, fields and the error."""
    from . import _adaptive_triangulation

    source_vertices = np.asarray(vertices, dtype=np.float32)
    face_array = np.array(faces, dtype=np.uint32, order="C", copy=True)
    field_array = np.array(fields, dtype=np.float32, order="C", copy=True)
    if source_vertices.ndim != 2 or source_vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if len(face_array) and int(face_array.max()) >= len(source_vertices):
        raise ValueError("faces reference a vertex outside vertices")
    if field_array.ndim == 1:
        field_array = field_array[:, None]
    if field_array.ndim != 2 or field_array.shape[0] != len(source_vertices):
        raise ValueError("fields must have shape (N,) or (N, C)")
    if not 1 <= field_array.shape[1] <= 32:
        raise ValueError("fields must contain between 1 and 32 channels")
    if not np.all(np.isfinite(field_array)):
        raise ValueError("fields must contain only finite values")
    scale = np.asarray(metric_scale, dtype=np.float32)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("metric_scale must contain three positive finite values")
    vertex_array = np.ascontiguousarray(source_vertices * scale, dtype=np.float32)
    if (face_budget is None) == (ratio is None):
        raise ValueError("provide exactly one of face_budget or ratio")
    if ratio is not None:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must be between zero and one")
        face_budget = max(1, round(len(face_array) * ratio))
    assert face_budget is not None
    regularize_value = {"none": 0, "light": 1, "full": 2}.get(regularize)
    if regularize_value is None:
        raise ValueError("regularize must be 'none', 'light', or 'full'")

    simplified, error, transported = (
        _adaptive_triangulation.simplify_geometry_with_passive_fields(
            vertex_array,
            face_array,
            field_array,
            face_budget,
            lock_border,
            target_error,
            regularize_value,
        )
    )
    source_indices = np.unique(simplified)
    remap = np.full(len(vertex_array), -1, dtype=np.int32)
    remap[source_indices] = np.arange(len(source_indices), dtype=np.int32)
    compact_fields = transported[source_indices]
    # the fields are convex interpolations, so keep them in the input range
    compact_fields = np.clip(
        compact_fields,
        field_array.min(axis=0),
        field_array.max(axis=0),
    )
    return (
        np.ascontiguousarray(vertex_array[source_indices] / scale, dtype=np.float32),
        np.ascontiguousarray(remap[simplified], dtype=np.int32),
        source_indices,
        np.ascontiguousarray(compact_fields, dtype=np.float32),
        float(error),
    )


def sample_masked_raster(
    raster: np.ndarray,
    coordinates: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Bilinearly sample `raster` at normalized (x, y) `coordinates`. Corners outside
    `mask` get no weight (the nearest one is used if all are), so values outside an
    asset do not bleed into its contour."""
    values = np.asarray(raster, dtype=np.float32)
    points = np.asarray(coordinates, dtype=np.float32)
    if values.ndim not in (2, 3):
        raise ValueError("raster must have shape (H, W) or (H, W, C)")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2)")
    height, width = values.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("raster dimensions must be non-empty")
    if not np.all(np.isfinite(points)):
        raise ValueError("coordinates must contain only finite values")
    valid = None
    if mask is not None:
        valid = np.asarray(mask, dtype=bool)
        if valid.shape != (height, width):
            raise ValueError("mask must match the raster's first two dimensions")

    x = np.clip(points[:, 0], 0.0, 1.0) * max(width - 1, 0)
    y = np.clip(points[:, 1], 0.0, 1.0) * max(height - 1, 0)
    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = x - x0
    fy = y - y0
    weights = np.column_stack(
        ((1 - fx) * (1 - fy), fx * (1 - fy), fx * fy, (1 - fx) * fy)
    ).astype(np.float32)
    corners = ((y0, x0), (y0, x1), (y1, x1), (y1, x0))
    if valid is not None:
        weights *= np.column_stack([valid[row, column] for row, column in corners])
        totals = weights.sum(axis=1)
        empty = totals == 0
        if np.any(empty):
            nearest = np.argmax(
                np.column_stack(
                    ((1 - fx) * (1 - fy), fx * (1 - fy), fx * fy, (1 - fx) * fy)
                ),
                axis=1,
            )
            weights[empty] = 0
            weights[np.flatnonzero(empty), nearest[empty]] = 1
            totals[empty] = 1
        weights /= totals[:, None]

    samples = np.stack([values[row, column] for row, column in corners], axis=1)
    if values.ndim == 2:
        return np.asarray(np.sum(samples * weights, axis=1), dtype=np.float32)
    return np.asarray(np.sum(samples * weights[..., None], axis=1), dtype=np.float32)
