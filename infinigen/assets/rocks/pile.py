# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import tqdm
import numpy as np
from numpy.random import uniform

from infinigen.assets.rocks.boulder import BoulderFactory
from infinigen.assets.utils.physics import free_fall
from infinigen.core.placement.detail import remesh_with_attrs
from infinigen.core.placement.factory import AssetFactory
import infinigen.core.util.blender as butil
from infinigen.assets.utils.decorate import join_objects, multi_res, toggle_hide
from infinigen.assets.utils.draw import surface_from_func
from infinigen.core.util.blender import deep_clone_obj
from infinigen.assets.utils.tag import tag_object
from infinigen.core.util.random import log_uniform


class BoulderPileFactory(AssetFactory):

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.factory = BoulderFactory(factory_seed, coarse)

    @staticmethod
    def create_floor():
        r = 4

        def floor_fn(x, y):
            alpha = 0.01
            x = np.sqrt(x * x + y * y) - r
            return np.maximum(x, alpha * x)

        mesh = surface_from_func(floor_fn, 32, 32, 12, 12)
        obj = bpy.data.objects.new('floor', mesh)
        bpy.context.scene.collection.objects.link(obj)
        return obj

    @staticmethod
    def place_boulder(obj, height):
        obj.location = *uniform(-3, 3, 2), height
        obj.rotation_euler = 0, 0, uniform(0, np.pi * 2)
        return height + obj.dimensions[-1]

    def create_placeholder(self, **kwargs):
        n = np.random.randint(3, 5)
        empty = butil.spawn_empty('placeholder', disp_type='CUBE', s=8)
        objects = []
        for i in range(n):
            empty_ = butil.spawn_empty('placeholder', disp_type='CUBE', s=8)
            scale = [1, log_uniform(.4, .6), log_uniform(.2, .4), log_uniform(.2, .4), log_uniform(.2, .4),
                log_uniform(.1, .2)]
            p = self.factory.create_placeholder()
            p.parent = empty_
            objects.append(p.children[0])
            for s in scale[1:]:
                p_ = p.copy()
                bpy.context.scene.collection.objects.link(p_)
                o = deep_clone_obj(p.children[0])
                o.scale = [s] * 3
                o.parent = p_
                p_.parent = empty_
                objects.append(o)
            empty_.parent = empty
        floor = self.create_floor()
        free_fall(objects, [floor], BoulderPileFactory.place_boulder)
        butil.delete(floor)
        return empty

    def create_asset(self, placeholder, face_size=0.01, **params) -> bpy.types.Object:
        objects = []
        for c in tqdm.tqdm(placeholder.children, desc='Creating boulder assets'):
            p = c.children[0]
            a = self.factory.create_asset(placeholder=p)
            a.location = p.children[0].location
            a.rotation_euler = p.children[0].rotation_euler
            objects.append(a)
            for p in c.children[1:]:
                a_ = deep_clone_obj(a)
                a_.scale = p.scale
                a_.location = p.children[0].location
                a_.rotation_euler = p.children[0].rotation_euler
                objects.append(a_)
                toggle_hide(p)
        obj = join_objects(objects)
        for c in placeholder.children:
            for p in c.children:
                butil.delete(p)
            butil.delete(c)
        multi_res(obj)
        remesh_with_attrs(obj, face_size)
        tag_object(obj, 'pile')
        return obj
