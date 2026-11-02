# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

import bpy

from infinigen.core.placement.factory import make_asset_collection
from infinigen.core.util import blender as butil


def _spawn_cube(i, **kwargs):
    bpy.ops.mesh.primitive_cube_add(location=(i, 0, 0))
    return bpy.context.active_object


def test_delete_collection_removes_an_asset_collection():
    butil.clear_scene()
    for _ in range(10):
        col = make_asset_collection(_spawn_cube, 5, "instances", verbose=False)
        name = col.name
        objects = list(col.objects)

        butil.delete_collection(col)

        assert name not in bpy.data.collections
        for o in objects:
            assert o.name not in bpy.context.scene.objects
            assert o.users == 0
