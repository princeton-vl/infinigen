# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from collections.abc import Iterable

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.clothes import blanket, pants, shirt
from infinigen.assets.objects.creatures.util.cloth_sim import bake_cloth
from infinigen.assets.utils.decorate import read_co, subsurf
from infinigen.core.placement.factory import make_asset_collection
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj


def cloth_sim(clothes, obj=None, end_frame=50, **kwargs):
    with (
        butil.ViewportMode(clothes, mode="OBJECT"),
        butil.SelectObjects(clothes),
        butil.Suppress(),
    ):
        bpy.ops.ptcache.free_bake_all()
    if obj is None:
        obj = []
    for o in obj if isinstance(obj, Iterable) else [obj]:
        butil.modify_mesh(o, "COLLISION", apply=False)
        o.collision.damping_factor = 0.9
        o.collision.cloth_friction = 10.0
        o.collision.friction_factor = 1.0
        o.collision.stickiness = 0.9
    frame = bpy.context.scene.frame_current
    butil.select_none()
    with butil.Suppress():
        mod = bake_cloth(clothes, kwargs, frame_start=1, frame_end=end_frame)
    bpy.context.scene.frame_set(end_frame)
    butil.apply_modifiers(clothes, mod)
    for o in obj if isinstance(obj, Iterable) else [obj]:
        with butil.SelectObjects(o):
            bpy.ops.object.modifier_remove(modifier=o.modifiers[-1].name)
    bpy.context.scene.frame_set(frame)
    with butil.Suppress():
        bpy.ops.ptcache.free_bake_all()


class ClothesCover:
    def __init__(
        self, bbox=(0.3, 0.7, 0.3, 0.7), factory_fn=None, width=None, size=None
    ):
        probs = np.array([2, 1, 1])
        if factory_fn is None:
            factory_fn = np.random.choice(
                [blanket.BlanketFactory, shirt.ShirtFactory, pants.PantsFactory],
                p=probs / probs.sum(),
            )
        self.factory = factory_fn(np.random.randint(1e5))
        if width is not None:
            self.factory.width = width
        if size is not None:
            self.factory.size = size
        self.col = make_asset_collection(
            self.factory, name="clothes", centered=True, n=3, verbose=False
        )
        self.bbox = bbox
        self.z_offset = 0.2

    def apply(self, obj, selection=None, **kwargs):
        for obj in obj if isinstance(obj, list) else [obj]:
            x, y, z = read_co(obj).T
            clothes = deep_clone_obj(
                np.random.choice(self.col.objects), keep_materials=True
            )
            clothes.parent = obj
            clothes.location = (
                uniform(self.bbox[0], self.bbox[1]) * (np.max(x) - np.min(x))
                + np.min(x),
                uniform(self.bbox[2], self.bbox[3]) * (np.max(y) - np.min(y))
                + np.min(y),
                np.max(z) + self.z_offset - np.min(read_co(clothes)[:, -1]),
            )
            clothes.rotation_euler[-1] = uniform(0, np.pi * 2)
            cloth_sim(clothes, obj, mass=0.05, tension_stiffness=2, distance_min=5e-3)
            subsurf(clothes, 2)


def apply(obj, selection=None, **kwargs):
    ClothesCover().apply(obj, selection, **kwargs)
