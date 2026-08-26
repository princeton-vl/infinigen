# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging

import numpy as np
import procfunc as pf

from infinigen2 import context
from infinigen2.exporters.render_error_check.util import (
    context_meshes,
    evaluated_mesh,
)

logger = logging.getLogger(__name__)


class NonFiniteGeometryError(ValueError):
    pass


def nonfinite_vertex_counts(
    objects: list[pf.MeshObject] | None = None,
) -> dict[str, int]:
    counts = {}
    for obj in context_meshes(objects):
        if obj.type != "MESH":
            continue
        mesh = evaluated_mesh(obj)
        if mesh is None or len(mesh.vertices) == 0:
            continue
        arr = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", arr)
        bad = int(np.count_nonzero(~np.isfinite(arr)))
        if bad:
            counts[obj.name] = bad
    return counts


def assert_geometry_finite(objects: list[pf.MeshObject] | None = None):
    bad = nonfinite_vertex_counts(objects)
    if bad:
        error = NonFiniteGeometryError(
            "meshes have non-finite (NaN/Inf) vertex coordinates, which Cycles "
            f"silently drops from the BVH or renders as corruption: {bad}"
        )
        context.raise_or_warn(context.globals.error_mode_finite_geometry, error, logger)
