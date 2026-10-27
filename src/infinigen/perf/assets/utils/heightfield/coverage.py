# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

from dataclasses import dataclass

import numpy as np


def coverage_slices(mask, *, padding=1):
    """Row and column slices around the true samples of `mask`, padded and clipped."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or min(mask.shape) < 2:
        raise ValueError("coverage must be a 2D raster with axes of length >= 2")
    if not isinstance(padding, (int, np.integer)) or padding < 0:
        raise ValueError("padding must be a nonnegative integer")
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        raise ValueError("coverage is empty")
    return tuple(
        slice(max(0, int(ids[0]) - padding), min(size, int(ids[-1]) + padding + 1))
        for ids, size in zip((rows, columns), mask.shape)
    )


def upsample_mask(mask, full_shape):
    """Nearest-neighbor upsample of `mask` to `full_shape`, with no padding: the
    coarse mask must already cover every sample the caller may occupy."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or min(mask.shape) < 2:
        raise ValueError("coarse mask must be a 2D raster with axes of length >= 2")
    if tuple(full_shape) != tuple(map(int, full_shape)) or min(full_shape) < 2:
        raise ValueError("full shape must contain two axis lengths >= 2")
    if not mask.any():
        raise ValueError("coverage is empty")
    row_ids = np.minimum(
        np.arange(full_shape[0], dtype=np.intp) * mask.shape[0] // full_shape[0],
        mask.shape[0] - 1,
    )
    column_ids = np.minimum(
        np.arange(full_shape[1], dtype=np.intp) * mask.shape[1] // full_shape[1],
        mask.shape[1] - 1,
    )
    return mask[row_ids[:, None], column_ids[None, :]]


@dataclass(frozen=True)
class RasterCanvas:
    """Rectangle of an endpoint-aligned square sampling lattice, stops inclusive."""

    full_size: int
    x_start: int
    x_stop: int
    y_start: int
    y_stop: int


def canvas_from_coverage_mask(size: int, mask: np.ndarray) -> RasterCanvas:
    """Smallest canvas of the `size` x `size` lattice holding `mask`."""
    if mask.shape != (size, size):
        raise ValueError("an uncropped coverage_mask must have shape (size, size)")
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError("coverage_mask must contain at least one pixel")
    return RasterCanvas(
        size,
        int(columns.min()),
        int(columns.max()),
        int(rows.min()),
        int(rows.max()),
    )
