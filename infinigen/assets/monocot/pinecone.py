# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import colorsys

import bpy
import numpy as np
from numpy.random import uniform

import infinigen.core.util.blender as butil
from infinigen.assets.monocot.growth import MonocotGrowthFactory
from infinigen.assets.utils.object import new_circle
from infinigen.assets.utils.draw import shape_by_angles, shape_by_xs
from infinigen.assets.utils.misc import build_color_ramp, log_uniform
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.detail import remesh_with_attrs
from infinigen.core.surface import shaderfunc_to_material
from infinigen.core.util.math import FixedSeed
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class PineconeFactory(MonocotGrowthFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.angle = 2 * np.pi / (np.random.randint(4, 8) + .5)
            self.max_y_angle = uniform(.7, .8) * np.pi / 2
            self.leaf_prob = uniform(.9, .95)
            self.count = int(log_uniform(64, 96))
            self.stem_offset = uniform(.2, .4)
            self.perturb = 0
            self.scale_curve = [(0, .5), (.5, uniform(.6, 1.)), (1, uniform(.1, .2))]
            self.bright_color = *colorsys.hsv_to_rgb(uniform(.02, .06), uniform(.8, 1.), .01), 1
            self.dark_color = *colorsys.hsv_to_rgb(uniform(.02, .06), uniform(.8, 1.), .005), 1
            self.material = shaderfunc_to_material(self.shader_monocot, self.dark_color, self.bright_color,
                                                   self.use_distance)

    def build_leaf(self, face_size):
        obj = new_circle(vertices=128)
        with butil.ViewportMode(obj, 'EDIT'):
            bpy.ops.mesh.fill_grid()
        angles = np.array([-1, -.8, -.5, 0, .5, .8, 1]) * self.angle / 2
        scale = uniform(.9, .95)
        scales = [0, .7, scale, 1, scale, .7, 0]
        displacement = [0, 0, 0, -uniform(.2, .3), 0, 0, 0]
        shape_by_angles(obj, angles, scales, displacement)

        with butil.ViewportMode(obj, 'EDIT'):
            bpy.ops.mesh.convex_hull()

        xs = [0, 1, 2]
        displacement = [0, 0, .5]
        shape_by_xs(obj, xs, displacement)

        obj.scale = [.1] * 3
        obj.rotation_euler[1] -= uniform(np.pi / 18, np.pi / 12)
        butil.apply_transform(obj)
        remesh_with_attrs(obj, face_size)

        texture = bpy.data.textures.new(name='pinecone', type='STUCCI')
        texture.noise_scale = log_uniform(.002, .005)
        butil.modify_mesh(obj, 'DISPLACE', True, strength=.001, mid_level=0, texture=texture)

        tag_object(obj, 'pinecone')
        return obj

    @staticmethod
    def shader_monocot(nw: NodeWrangler, dark_color, bright_color, use_distance):
        specular = uniform(.2, .4)
        color = build_color_ramp(nw, nw.musgrave(10), [.0, .3, .7, 1.],
                                 [bright_color, bright_color, dark_color, dark_color], )
        noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Scale': 50})
        roughness = nw.build_float_curve(noise_texture, [(0, .5), (1, .8)])
        bsdf = nw.new_node(Nodes.PrincipledBSDF,
                           input_kwargs={'Base Color': color, 'Roughness': roughness, 'Specular': specular})
        return bsdf
