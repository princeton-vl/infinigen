# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
from numpy.random import normal, uniform

import infinigen.core.util.blender as butil
from infinigen.assets.creatures.util.animation.driver_repeated import repeated_driver
from infinigen.assets.mollusk.base import BaseMolluskFactory
from infinigen.assets.utils.object import mesh2obj, data2mesh, new_circle
from infinigen.assets.utils.draw import shape_by_angles
from infinigen.assets.utils.misc import log_uniform
from infinigen.assets.utils.decorate import displace_vertices, join_objects
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface
from infinigen.core.util.math import FixedSeed
from infinigen.assets.utils.tag import tag_object, tag_nodegroup


class ShellBaseFactory(BaseMolluskFactory):

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.makers = [self.scallop_make, self.clam_make, self.mussel_make]
            self.maker = np.random.choice(self.makers)
            self.z_scale = log_uniform(2, 10)

    def build_ellipse(self, viewpoint=(0., 0, 1.), softness=.3):
        viewpoint = np.array(viewpoint)
        obj = new_circle(vertices=1024)
        with butil.ViewportMode(obj, 'EDIT'):
            bpy.ops.mesh.fill_grid()
        surface.add_geomod(obj, self.geo_shader_vector, apply=True, attributes=['vector'])
        butil.apply_transform(obj, loc=True)

        def displace(x, y, z):
            r = np.sqrt((x - 1) ** 2 + y ** 2 + z ** 2)
            t = 1 - softness + softness * r ** 4
            return ((1 - t)[:, np.newaxis] * (viewpoint[np.newaxis, :] - np.stack([x, y, z], -1))).T

        displace_vertices(obj, displace)
        return obj

    @staticmethod
    def geo_shader_vector(nw: NodeWrangler):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        pos = nw.new_node(Nodes.InputPosition)
        x, y, z = nw.separate(pos)
        vector = nw.combine(x, y, nw.vector_math('DISTANCE', pos, [1, 0, 0]))
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry, 'Vector': vector})
        return geometry

    def scallop_make(self):
        obj = self.build_ellipse()
        obj.scale = 1, 1.2, 1
        butil.apply_transform(obj)
        boundary = .42
        outer = uniform(.28, .32)
        inner = uniform(.18, .22)
        s = uniform(.6, .7)
        angles = [-boundary, -outer, -inner, inner, outer, boundary]
        scales = [0, s, 1, 1, s, 0]
        shape_by_angles(obj, np.array(angles) * np.pi, scales)
        self.add_radial_groove(obj)
        obj = self.add_hinge(obj)
        tag_object(obj, 'scallop')
        return obj

    def clam_make(self):
        obj = self.build_ellipse(softness=.5)
        obj.scale = 1, 1.2, 1
        butil.apply_transform(obj)
        s = uniform(.6, .7)
        angles = [-uniform(.4, .5), -uniform(.3, .35), uniform(-.25, .25), uniform(.3, .35), uniform(.4, .5)]
        scales = [0, s, 1, s, 0]
        shape_by_angles(obj, np.array(angles) * np.pi, scales)
        tag_object(obj, 'clam')
        return obj

    def mussel_make(self):
        obj = self.build_ellipse(softness=.5)
        obj.scale = 1, 3, 1
        butil.apply_transform(obj)
        s = uniform(.6, .8)
        angles = [-.5, -uniform(.1, .15), uniform(0., .25), .5]
        scales = [0, s, 1, uniform(.6, .8)]
        shape_by_angles(obj, np.array(angles) * np.pi, scales)
        tag_object(obj, 'mussel')
        return obj

    @staticmethod
    def add_radial_groove(obj):
        frequency = 45
        scale = 0.02

        def displace(x, y, z):
            a = np.arctan(y / (x + 1e-6 * (x >= 0)))
            r = np.sqrt(x * x + y * y + z * z)
            return scale * np.cos(a * frequency) * np.clip(r - .25, 0, None)

        displace_vertices(obj, displace)
        return obj

    @staticmethod
    def add_hinge(obj):
        length = .4
        width = .1
        x = uniform(.8, 1.)
        vertices = [[0, -length, 0], [width, -length * x, 0], [width, length * x, 0], [0, length, 0]]
        o = mesh2obj(data2mesh(vertices, [], [[0, 1, 2, 3]], 'trap'))
        butil.modify_mesh(o, 'SUBSURF', render_levels=2, levels=2, subdivision_type='SIMPLE')
        butil.modify_mesh(o, 'DISPLACE', strength=.2,
                          texture=bpy.data.textures.new(name='hinge', type='STUCCI'))
        obj = join_objects([obj, o])
        return obj

    def create_asset(self, **params):
        upper = self.maker()
        dim = np.sqrt(upper.dimensions[0] * upper.dimensions[1] + 0.01)
        upper.scale = [1 / dim] * 3
        upper.location[-1] += .005
        butil.apply_transform(upper, loc=True)
        lower = butil.deep_clone_obj(upper)
        lower.scale[-1] = -1
        butil.apply_transform(lower)

        base = uniform(0, np.pi / 4)
        lower.rotation_euler[1] = - base
        upper.rotation_euler[1] = - base - uniform(np.pi / 6, np.pi / 3)
        obj = join_objects([lower, upper])
        return obj


class ScallopBaseFactory(ShellBaseFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.maker = self.scallop_make


class ClamBaseFactory(ShellBaseFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.maker = self.clam_make


class MusselBaseFactory(ShellBaseFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.maker = self.mussel_make
