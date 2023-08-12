# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy

import infinigen.core.util.blender as butil
from infinigen.core.util.logging import Suppress


def free_fall(actives, passives, place_fn, t=100):
    height = 0.
    for o in sorted(actives, key=lambda o: -o.dimensions[-1]):
        height = place_fn(o, height)
    with EnablePhysics(actives, passives):
        bpy.context.scene.frame_end = t
        with Suppress():
            bpy.ops.ptcache.bake_all(True)
        bpy.context.scene.frame_current = t
        with butil.SelectObjects(actives):
            bpy.ops.object.visual_transform_apply()


class EnablePhysics:

    def __init__(self, actives, passives):
        self.actives = actives
        self.passives = passives

    def __enter__(self):
        self.frame = bpy.context.scene.frame_current
        self.frame_start = bpy.context.scene.frame_end
        self.frame_end = bpy.context.scene.frame_start
        for a in self.actives:
            with butil.SelectObjects(a):
                bpy.ops.rigidbody.objects_add(type='ACTIVE')
                bpy.ops.rigidbody.mass_calculate()
        for p in self.passives:
            with butil.SelectObjects(p):
                bpy.ops.rigidbody.objects_add(type='PASSIVE')
                bpy.context.object.rigid_body.collision_shape = 'MESH'

    def __exit__(self, *_):
        bpy.ops.rigidbody.world_remove()
        bpy.context.scene.frame_set(self.frame)
        bpy.context.scene.frame_start = self.frame_start
        bpy.context.scene.frame_end = self.frame_end
