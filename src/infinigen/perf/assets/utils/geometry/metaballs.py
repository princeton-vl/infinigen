# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: original Infinigen metaballs (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/utils/geometry/metaballs.py)
# - David Yan, Jia Deng: optimized implementation

import bmesh
import bpy
import numpy as np

from infinigen.core.util import blender as butil
from infinigen.perf.core.util.isolation import IsolatedScene


class MBallStructure:
    """Default `to_object`, first making the metaball evaluable: the depsgraph skips
    objects outside the scratch view layer the optimized placement builds in."""

    def to_object(self):
        bm = bmesh.new()

        mball_obj = self.root.children[0]

        if len(self.root.children) > 1:
            first = self.root.children[1]
            mball_obj.location = first.location
            mball_obj.rotation_euler = first.rotation_euler

            # do resolution via scale, not using their settings
            for c in self.root.children:
                c.data.resolution = 1
            mball_obj.scale = np.full(3, self.resolution)

        IsolatedScene.ensure_evaluable(mball_obj)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_mesh = mball_obj.evaluated_get(depsgraph).to_mesh()
        bm.from_mesh(eval_mesh)

        mesh = bpy.data.meshes.new(self.name)
        bm.to_mesh(mesh)

        obj = bpy.data.objects.new(self.name + "_mesh", object_data=mesh)
        bpy.context.scene.collection.objects.link(obj)

        if len(self.root.children) > 1:
            obj.location = first.location
            obj.rotation_euler = first.rotation_euler
            obj.scale = np.full(3, self.resolution)
            with butil.SelectObjects(obj):
                bpy.ops.object.transform_apply(
                    location=False, rotation=False, scale=True
                )

        return obj
