# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import bmesh
import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.decorate import read_co, subdivide_edge_ring, subsurf
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.object import join_objects, new_cylinder
from infinigen.assets.utils.uv import wrap_front_back
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util.math import FixedSeed
from infinigen.core.util import blender as butil


class BottleFactory(AssetFactory):
    z_neck_offset = .05
    z_waist_offset = .15

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.z_length = uniform(.15, .25)
            self.x_length = self.z_length * uniform(.15, .25)
            self.x_cap = uniform(.3, .35)
            self.bottle_type = np.random.choice(['beer', 'bordeaux', 'champagne', 'coke', 'vintage'])
            self.bottle_width = uniform(.002, .005)
            self.z_waist = 0
            match self.bottle_type:
                case 'beer':
                    self.z_neck = uniform(.5, .6)
                    self.z_cap = uniform(.05, .08)
                    neck_size = uniform(.06, .1)
                    neck_ratio = uniform(.4, .5)
                    self.x_anchors = [0, 1, 1, (neck_ratio + 1) / 2 + (1 - neck_ratio) / 2 * self.x_cap,
                        neck_ratio + (1 - neck_ratio) * self.x_cap, self.x_cap, self.x_cap, 0]
                    self.z_anchors = [0, 0, self.z_neck, self.z_neck + uniform(.6, .7) * neck_size,
                        self.z_neck + neck_size, 1 - self.z_cap, 1, 1]
                    self.is_vector = [0, 1, 1, 0, 1, 1, 1, 0]
                case 'bordeaux':
                    self.z_neck = uniform(.6, .7)
                    self.z_cap = uniform(.1, .15)
                    neck_size = uniform(.1, .15)
                    self.x_anchors = 0, 1, 1, (1 + self.x_cap) / 2, self.x_cap, self.x_cap, 0
                    self.z_anchors = [0, 0, self.z_neck, self.z_neck + uniform(.6, .7) * neck_size,
                        self.z_neck + neck_size, 1, 1]
                    self.is_vector = [0, 1, 1, 0, 1, 1, 0]
                case 'champagne':
                    self.z_neck = uniform(.4, .5)
                    self.z_cap = uniform(.05, .08)
                    self.x_anchors = [0, 1, 1, 1, (1 + self.x_cap) / 2, self.x_cap, self.x_cap, 0]
                    self.z_anchors = [0, 0, self.z_neck, self.z_neck + uniform(.08, .1),
                        self.z_neck + uniform(.15, .18), 1 - self.z_cap, 1, 1]
                    self.is_vector = [0, 1, 1, 0, 0, 1, 1, 0]
                case 'coke':
                    self.z_waist = uniform(.4, .5)
                    self.z_neck = self.z_waist + uniform(.2, .25)
                    self.z_cap = uniform(.05, .08)
                    self.x_anchors = [0, uniform(.85, .95), 1, uniform(.85, .95), 1, 1, self.x_cap, self.x_cap,
                        0]
                    self.z_anchors = [0, 0, uniform(.08, .12), uniform(.18, .25), self.z_waist, self.z_neck,
                        1 - self.z_cap, 1, 1]
                    self.is_vector = [0, 1, 0, 0, 1, 1, 1, 1, 0]
                case 'vintage':
                    self.z_waist = uniform(.1, .15)
                    self.z_neck = uniform(.7, .75)
                    self.z_cap = uniform(.0, .08)
                    x_lower = uniform(.85, .95)
                    self.x_anchors = [0, x_lower, (x_lower + 1) / 2, 1, 1, (self.x_cap + 1) / 2, self.x_cap,
                        self.x_cap, 0]
                    self.z_anchors = [0, 0, self.z_waist - uniform(.1, .15), self.z_waist, self.z_neck,
                        self.z_neck + uniform(.1, .2), 1 - self.z_cap, 1, 1]
                    self.is_vector = [0, 1, 0, 1, 1, 0, 1, 1, 0]


            self.texture_shared = uniform() < .2
            self.cap_subsurf = uniform() < .5

    def create_asset(self, **params) -> bpy.types.Object:
        bottle = self.make_bottle()
        wrap = self.make_wrap(bottle)
        cap = self.make_cap()
        obj = join_objects([bottle, wrap, cap])

        return obj

    def finalize_assets(self, assets):
            self.scratch.apply(assets)
            self.edge_wear.apply(assets)

    def make_bottle(self):
        x_anchors = np.array(self.x_anchors) * self.x_length
        z_anchors = np.array(self.z_anchors) * self.z_length
        anchors = x_anchors, 0, z_anchors
        obj = spin(anchors, np.nonzero(self.is_vector)[0])
        subsurf(obj, 1, True)
        subsurf(obj, 1)
        if self.bottle_width > 0:
            butil.modify_mesh(obj, 'SOLIDIFY', thickness=self.bottle_width)
        self.surface.apply(obj, translucent=True)
        return obj

    def make_wrap(self, bottle):
        obj = new_cylinder(vertices=128)
        with butil.ViewportMode(obj, 'EDIT'):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [f for f in bm.faces if len(f.verts) > 4]
            bmesh.ops.delete(bm, geom=geom, context='FACES_ONLY')
            bmesh.update_edit_mesh(obj.data)
        subdivide_edge_ring(obj, 16)
        z_max = self.z_neck - uniform(.02, self.z_neck_offset) * (self.z_neck - self.z_waist)
        z_min = self.z_waist + uniform(.02, self.z_waist_offset) * (self.z_neck - self.z_waist)
        radius = np.max(read_co(bottle)[:, 0]) + 2e-3
        obj.scale = radius, radius, (z_max - z_min) * self.z_length
        obj.location[-1] = z_min * self.z_length
        butil.apply_transform(obj, True)
        wrap_front_back(obj, self.wrap_surface, self.texture_shared)
        return obj

    def make_cap(self):
        obj = new_cylinder(vertices=128)
        obj.scale = [(self.x_cap + .1) * self.x_length, (self.x_cap + .1) * self.x_length,
            (self.z_cap + .01) * self.z_length]
        obj.location[-1] = (1 - self.z_cap) * self.z_length
        butil.apply_transform(obj, loc=True)
        subsurf(obj, 1, self.cap_subsurf)
        self.cap_surface.apply(obj)
        return obj
