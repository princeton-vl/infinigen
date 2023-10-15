# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
from numpy.random import uniform

import infinigen.core.util.blender as butil
from infinigen.assets.mollusk.base import BaseMolluskFactory
from infinigen.assets.utils.object import center, mesh2obj, data2mesh, new_empty
from infinigen.assets.utils.misc import log_uniform
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface
from infinigen.core.util.math import FixedSeed
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class SnailBaseFactory(BaseMolluskFactory):
    freq = 256

    def __init__(self, factory_seed, coarse=False):
        super(SnailBaseFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.makers = [self.volute_make, self.nautilus_make, self.snail_make, self.conch_make]
            self.maker = np.random.choice(self.makers)
            self.ratio = uniform(0, .3) if uniform(0, 1) < .5 else uniform(.7, 1.)
            self.z_scale = log_uniform(.2, 1)
            self.distortion = log_uniform(2, 20)

    @staticmethod
    def build_cross_section(n=64, affine=1, spike=0., concave=2.2):
        perturb = 1 / (5 * n)
        angles = (np.arange(n) / n + uniform(-perturb, perturb, n)) * 2 * np.pi
        radius = np.abs(np.cos(angles)) ** concave + np.abs(np.sin(angles)) ** concave
        radius *= 1 + uniform(0, spike, n) * (uniform(0, 1, n) < .2)
        vertices = np.stack(
            [np.cos(angles) * radius, np.sin(angles) * radius * affine, np.zeros_like(angles)]).T
        edges = np.stack([np.arange(n), np.roll(np.arange(n), -1)]).T
        obj = mesh2obj(data2mesh(vertices, edges, [], 'circle'))
        obj.rotation_euler = 0, 0, uniform(0, np.pi / 12)
        butil.apply_transform(obj)
        return obj

    def snail_make(self, lateral=.15, longitudinal=0.04, freq=28, scale=.99, loop=8, affine=1, spike=0.):
        n = 40
        resolution = loop * freq
        concave = uniform(1.9, 2.1)
        obj = self.build_cross_section(n, affine, spike, concave)
        empty = new_empty(location=(longitudinal * np.random.choice([-1, 1]), 0, 0),
                          rotation=(2 * np.pi / freq, 0, 0), scale=[scale] * 3)
        butil.modify_mesh(obj, 'ARRAY', apply=True, use_relative_offset=False, use_constant_offset=True,
                          use_object_offset=True, constant_offset_displace=(0, 0, lateral), count=resolution,
                          offset_object=empty)
        butil.delete(empty)
        surface.add_geomod(obj, self.geo_shader_vector, apply=True, input_args=[n, lateral],
                           attributes=['vector'])

        with butil.ViewportMode(obj, 'EDIT'):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.bridge_edge_loops()

        return obj

    @staticmethod
    def geo_shader_vector(nw: NodeWrangler, n, interval):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        id = nw.new_node(Nodes.InputID)
        angle = nw.scalar_multiply(nw.math('MODULO', id, n), 2 * np.pi / n)
        height = nw.scalar_multiply(nw.math('FLOOR', nw.scalar_divide(id, n)), interval)
        vector = nw.combine(nw.math('COSINE', angle), nw.math('SINE', angle), height)
        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry, 'Vector': vector})

    @staticmethod
    def solve_longitude(ratio, freq, scale):
        return ratio * (1 + scale ** freq) / freq

    @staticmethod
    def solve_lateral(ratio, freq, scale):
        return ratio / (np.sin(2 * np.pi / freq * np.arange(freq)) * scale ** np.arange(freq)).sum()

    @staticmethod
    def solve_scale(shrink, freq):
        return shrink ** (1 / freq)

    def conch_make(self):
        scale = self.solve_scale(uniform(.7, .8), self.freq)
        lateral = self.solve_lateral(uniform(.3, .4), self.freq, scale)
        longitude = self.solve_longitude(uniform(.7, .8), self.freq, scale)
        loop = np.random.randint(8, 10)
        obj = self.snail_make(lateral, longitude, self.freq, scale, loop, affine=uniform(.8, .9), spike=.1)
        tag_object(obj, 'conch')
        return obj

    def auger_make(self):
        scale = self.solve_scale(uniform(.7, .8), self.freq)
        lateral = self.solve_lateral(uniform(.1, .15), self.freq, scale)
        longitude = self.solve_longitude(uniform(.9, 1.), self.freq, scale)
        loop = np.random.randint(8, 12)
        obj = self.snail_make(lateral, longitude, self.freq, scale, loop, affine=uniform(.5, .6))
        tag_object(obj, 'auger')
        return obj

    def volute_make(self):
        scale = self.solve_scale(uniform(.5, .6), self.freq)
        lateral = self.solve_lateral(uniform(.4, .5), self.freq, scale)
        longitude = self.solve_longitude(uniform(.6, .7), self.freq, scale)
        loop = np.random.randint(4, 5)
        obj = self.snail_make(lateral, longitude, self.freq, scale, loop)
        tag_object(obj, 'volute')
        return obj

    def nautilus_make(self):
        scale = self.solve_scale(uniform(.4, .5), self.freq)
        lateral = self.solve_lateral(uniform(1.2, 1.4), self.freq, scale)
        longitude = self.solve_longitude(uniform(.2, .3), self.freq, scale)
        loop = np.random.randint(4, 5)
        obj = self.snail_make(lateral, longitude, self.freq, scale, loop)
        tag_object(obj, 'nautilus')
        return obj

    @staticmethod
    def geo_affine(nw: NodeWrangler):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        affine = nw.new_node(Nodes.SetPosition, input_kwargs={
            'Geometry': geometry,
            'Offset': nw.combine(
                *[nw.vector_math('DOT_PRODUCT', uniform(-.1, .1, 3), nw.new_node(Nodes.InputPosition)) for _ in
                    range(3)])})
        return affine

    def create_asset(self, **params):
        obj = self.maker()
        obj.scale = [1 / max(obj.dimensions)] * 3
        obj.rotation_euler = uniform(0, np.pi * 2, 3)
        butil.apply_transform(obj)
        obj.location = -center(obj)
        obj.location[-1] += obj.dimensions[-1] * .4
        butil.apply_transform(obj, loc=True)
        surface.add_geomod(obj, self.geo_affine, apply=True)
        tag_object(obj, 'snail')
        return obj


class VoluteBaseFactory(SnailBaseFactory):
    def __init__(self, factory_seed, coarse=False):
        super(VoluteBaseFactory, self).__init__(factory_seed, coarse)
        self.maker = self.volute_make


class NautilusBaseFactory(SnailBaseFactory):
    def __init__(self, factory_seed, coarse=False):
        super(NautilusBaseFactory, self).__init__(factory_seed, coarse)
        self.maker = self.nautilus_make


class ConchBaseFactory(SnailBaseFactory):
    def __init__(self, factory_seed, coarse=False):
        super(ConchBaseFactory, self).__init__(factory_seed, coarse)
        self.maker = self.conch_make


class AugerBaseFactory(SnailBaseFactory):
    def __init__(self, factory_seed, coarse=False):
        super(AugerBaseFactory, self).__init__(factory_seed, coarse)
        self.maker = self.auger_make
