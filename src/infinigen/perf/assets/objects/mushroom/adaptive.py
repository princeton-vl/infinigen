# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Mushroom placement with a lazy surface tree per mushroom."""

import numpy as np
from mathutils import Euler


def mesh_buffers(obj):
    import bpy

    temporary = None
    if obj.modifiers:
        temporary = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = temporary.to_mesh()
    else:
        mesh = obj.data
    try:
        co = np.empty((len(mesh.vertices), 3), np.float32)
        if len(co):
            mesh.attributes["position"].data.foreach_get("vector", co.ravel())
        from infinigen.perf.assets.utils.mesh_io import read_triangles

        tri = read_triangles(mesh)
        return co, tri
    finally:
        if temporary is not None:
            temporary.to_mesh_clear()


def find_locations(keys, rotations, start_locs, directions, radius, tolerance):
    from infinigen.perf.assets.objects.mushroom._adaptive import SurfaceTree

    scales = np.tile(np.random.uniform(0.3, 1.2, len(keys))[:, None], 3)
    locations = [np.zeros(3)]
    if len(keys) == 1:
        return locations, rotations, scales
    trees = []
    try:
        for (co, tri), rotation, scale in zip(keys, rotations, scales):
            matrix = np.ascontiguousarray(
                np.array(Euler(rotation).to_matrix()) @ np.diag(scale), dtype=np.float64
            )
            trees.append(SurfaceTree(co, tri, matrix, tolerance))
        for i in range(1, len(keys)):
            for distance in np.linspace(0, 4, 20) * radius:
                offset = start_locs[i] + directions[i] * distance
                if all(
                    trees[j].clear(trees[i], offset - locations[j]) for j in range(i)
                ):
                    break
            locations.append(offset)
    finally:
        for tree in trees:
            tree.close()
    return locations, rotations, scales
