# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import procfunc as pf

from infinigen2.exporters.render_error_check.util import context_meshes

SINGULAR_TRANSFORM_EPS = 1e-12


class SingularTransformError(ValueError):
    pass


def singular_transform_objects(
    objects: list[pf.MeshObject] | None = None,
) -> dict[str, float]:
    bad = {}
    for obj in context_meshes(objects):
        if obj.type != "MESH":
            continue
        det = obj.matrix_world.determinant()
        if abs(det) < SINGULAR_TRANSFORM_EPS:
            bad[obj.name] = det
    return bad


def assert_transforms_nonsingular(objects: list[pf.MeshObject] | None = None):
    bad = singular_transform_objects(objects)
    if bad:
        raise SingularTransformError(
            "objects have a singular (zero-scale) world transform, so they collapse "
            f"to a plane/line and render flat or invisible: {bad}"
        )
