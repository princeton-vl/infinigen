# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import procfunc as pf

__all__ = ["estimated_eval_tricount"]


def estimated_eval_tricount(obj: pf.Object) -> int:
    """Estimate one object's render-level triangle count without evaluating it.

    Subdivision turns each n-gon into n quads at the first level and quadruples
    thereafter, so the render-level count follows from the corner count alone. Exact for
    a base mesh plus deferred subsurf; deformation-only modifiers do not affect it.
    """
    item = obj.item()
    if item.type != "MESH":
        return 0
    mesh = item.data
    levels = sum(
        modifier.render_levels
        for modifier in item.modifiers
        if modifier.type == "SUBSURF"
    )
    if levels:
        return 2 * len(mesh.loops) * 4 ** (levels - 1)
    return len(mesh.loops) - 2 * len(mesh.polygons)
