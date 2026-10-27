# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

from __future__ import annotations

import numpy as np

from .mesh import sample_masked_raster


def correct_field_residuals(raster, mask, chart, triangles, transported):
    """Move `transported` vertex values toward the raster sampled at `chart`
    (normalized raster coordinates) by one strength per channel, clipped to [0, 1],
    minimizing the piecewise-linear error over covered pixels. Returns the values and
    strengths; one scalar per channel avoids the overshoot of a per-vertex fit and
    keeps each value between its transported and resampled value."""
    raster = np.asarray(raster, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    chart = np.asarray(chart, dtype=np.float64)
    base = np.asarray(transported, dtype=np.float32)
    if raster.ndim != 3 or mask.shape != raster.shape[:2]:
        raise ValueError("raster must be (H,W,C) with a matching mask")
    if (
        base.ndim != 2
        or chart.shape != (len(base), 2)
        or base.shape[1] != raster.shape[2]
    ):
        raise ValueError("chart and transported field dimensions do not match")
    sampled = sample_masked_raster(raster, chart, mask=mask)
    residual = sampled - base
    h, w = mask.shape
    points = chart * (w - 1, h - 1)
    from ._adaptive_triangulation import field_residual_moments

    numerator, denominator = field_residual_moments(
        np.ascontiguousarray(raster),
        np.ascontiguousarray(mask, dtype=np.uint8),
        np.ascontiguousarray(points),
        np.ascontiguousarray(triangles, dtype=np.int32),
        np.ascontiguousarray(base),
        np.ascontiguousarray(residual),
    )
    strength = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-20
    )
    strength = np.clip(strength, 0, 1)
    return np.asarray(base + residual * strength, dtype=np.float32), strength
