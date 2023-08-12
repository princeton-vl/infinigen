# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import colorsys

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.monocot.growth import MonocotGrowthFactory
from infinigen.assets.utils.decorate import add_distance_to_boundary, assign_material, join_objects, write_attribute, \
    write_material_index
from infinigen.assets.utils.draw import leaf, spin
from infinigen.assets.utils.misc import log_uniform
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core import surface
from infinigen.core.surface import shaderfunc_to_material
from infinigen.core.util.math import FixedSeed
from infinigen.core.util import blender as butil
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class VeratrumMonocotFactory(MonocotGrowthFactory):

    def __init__(self, factory_seed, coarse=False):
        super(VeratrumMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.stem_offset = uniform(1., 1.5)
            self.angle = uniform(np.pi / 4, np.pi / 3)
            self.z_drag = uniform(.4, .5)
            self.bend_angle = np.pi / 2
            self.min_y_angle = uniform(np.pi * .25, np.pi * .35)
            self.max_y_angle = uniform(np.pi * .6, np.pi * .7)
            self.count = int(log_uniform(32, 64))
            self.scale_curve = (0, uniform(.8, 1.)), (.4, .6), (.8, uniform(0, .1)), (1, 0)
            self.leaf_range = 0, uniform(.7, .8)
            self.bud_angle = uniform(np.pi / 15, np.pi / 12)
            self.freq = uniform(25, 50)
            self.branches_factory = VeratrumBranchMonocotFactory(factory_seed, coarse)
            self.branch_material = shaderfunc_to_material(self.shader_ear)

    @staticmethod
    def build_base_hue():
        return uniform(.12, .32)

    @staticmethod
    def shader_ear(nw: NodeWrangler):
        color = *colorsys.hsv_to_rgb(uniform(.1, .35), uniform(.1, .5), log_uniform(.2, .5)), 1
        specular = uniform(.0, .2)
        clearcoat = 0 if uniform(0, 1) < .8 else uniform(.2, .5)
        noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Scale': 50})
        roughness = nw.build_float_curve(noise_texture, [(0, .5), (1, .8)])
        bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
            'Base Color': color,
            'Roughness': roughness,
            'Specular': specular,
            'Clearcoat': clearcoat,
            'Subsurface': .01,
            'Subsurface Radius': (.01, .01, .01),
        })
        return bsdf

    def build_leaf(self, face_size):
        x_anchors = 0, .2 * np.cos(self.bud_angle), uniform(.6, .7), .8
        y_anchors = 0, .2 * np.sin(self.bud_angle), uniform(.06, .1), 0
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        distance = add_distance_to_boundary(obj)

        vg = obj.vertex_groups['distance']
        weights = np.cos(self.freq * distance) ** 4
        for i, w in enumerate(weights):
            vg.add([i], w, 'REPLACE')
        butil.modify_mesh(obj, 'DISPLACE', strength=-uniform(5e-3, 8e-3), mid_level=0, vertex_group='distance')
        self.decorate_leaf(obj, 8, np.pi / 2)
        return obj

    def create_asset(self, **params):
        obj = super().create_raw(**params)
        branches = self.branches_factory.create_asset(**params)
        branches.location[-1] = self.stem_offset - .02
        obj = join_objects([obj, branches])

        self.decorate_monocot(obj)
        assign_material(obj, [self.material, self.branch_material])
        write_material_index(obj, surface.read_attr_data(obj, 'ear', 'FACE').astype(int)[:, 0])
        tag_object(obj, 'veratrum')
        return obj


class VeratrumBranchMonocotFactory(AssetFactory):
    max_branches = 6

    def __init__(self, factory_seed, coarse=False):
        super(VeratrumBranchMonocotFactory, self).__init__(factory_seed, coarse)
        self.branch_factories = [VeratrumEarMonocotFactory(self.factory_seed * self.max_branches + i, coarse)
            for i in range(np.random.randint(3, self.max_branches) + 1)]
        self.primary_stem_offset = uniform(.4, .8)

        for i, f in enumerate(self.branch_factories):
            scale = log_uniform(.3, .6) if i > 0 else 1
            f.stem_offset = scale * self.primary_stem_offset
            f.count = int(log_uniform(64, 238) * scale)

    def create_asset(self, **params) -> bpy.types.Object:
        branches = [f.create_asset(**params) for f in self.branch_factories]
        for i, branch in enumerate(branches):
            if i > 0:
                branch.location[-1] = self.primary_stem_offset * uniform(0, .6)
                branch.rotation_euler = uniform(np.pi * .25, np.pi * .4), 0, uniform(0, np.pi * 2)
        obj = join_objects(branches)
        tag_object(obj, 'veratrum_branch')
        return obj


class VeratrumEarMonocotFactory(MonocotGrowthFactory):

    def __init__(self, factory_seed, coarse=False):
        super(VeratrumEarMonocotFactory, self).__init__(factory_seed, coarse)
        self.angle = uniform(np.pi / 4, np.pi / 3)
        self.min_y_angle = uniform(np.pi * .25, np.pi * .3)
        self.max_y_angle = uniform(np.pi * .3, np.pi * .35)
        self.count = np.random.randint(64, 128)
        self.leaf_prob = uniform(.6, .8)
        self.leaf_range = 0, .98

    def build_leaf(self, face_size):
        x_achors = 0, .04, .06, .04, 0
        y_anchors = 0, .01, 0, -.01, 0
        z_anchors = 0, - .01, -.01, -.006, 0
        anchors = [x_achors, y_anchors, z_anchors]
        obj = spin(anchors, [0, 2, 4], dupli=True, loop=True, resolution=np.random.randint(3, 5),
                   axis=(1, 0, 0))
        butil.modify_mesh(obj, 'WELD', merge_threshold=face_size / 2)
        write_attribute(obj, 1, 'ear', 'FACE')
        tag_object(obj, 'veratrum_ear')
        return obj
