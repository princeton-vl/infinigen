# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np

from infinigen.assets.corals.base import BaseCoralFactory
from infinigen.assets.corals.tentacles import make_radius_points_fn
import infinigen.core.util.blender as butil
from infinigen.assets.utils.object import new_icosphere
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core import surface
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class TubeBaseCoralFactory(BaseCoralFactory):
    default_scale = [.7] * 3

    def __init__(self, factory_seed, coarse=False):
        super(TubeBaseCoralFactory, self).__init__(factory_seed, coarse)
        self.points_fn = make_radius_points_fn(.05, .4)

    def create_asset(self, face_size=0.01, **params) -> bpy.types.Object:
        obj = new_icosphere(subdivisions=2)
        obj.name = 'tube_coral'
        surface.add_geomod(obj, self.geo_coral_tube, apply=True)
        butil.modify_mesh(obj, 'BEVEL', True, offset_type='PERCENT', width_pct=10, segments=1)
        butil.modify_mesh(obj, 'SOLIDIFY', True, thickness=.05)
        butil.modify_mesh(obj, 'SUBSURF', True, levels=2, render_levels=2)
        butil.modify_mesh(obj, 'DISPLACE', True, strength=0.1,
                          texture=bpy.data.textures.new(name='tube_coral', type='STUCCI'), mid_level=0)
        tag_object(obj, 'tube_coral')
        return obj

    @staticmethod
    def geo_coral_tube(nw: NodeWrangler):
        ico_sphere_perturb = .2
        growth_z = 1
        short_length_range = .2, .4
        long_length_range = .4, 1.2
        angles = np.linspace(np.pi * 2 / 5, np.pi / 10, 6)
        scales = np.linspace(1, .9, 6)
        face_perturb = .4
        growth_prob = .75
        seed = np.random.randint(1e3)
        ico_sphere = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        perturbed_ico_sphere = nw.new_node(Nodes.SetPosition, input_kwargs={
            'Geometry': ico_sphere,
            'Offset': nw.uniform([-ico_sphere_perturb] * 3, [ico_sphere_perturb] * 3, seed)
        })
        mesh = nw.new_node(Nodes.DualMesh, input_kwargs={'Mesh': perturbed_ico_sphere})
        normal = nw.new_node(Nodes.InputNormal)
        top = nw.boolean_math('AND', nw.compare_direction('LESS_THAN', normal, (0, 0, 1), angles[0]),
                              nw.bernoulli(growth_prob, seed))

        for i, (angle, scale) in enumerate(zip(angles, scales)):
            direction = nw.vector_math('NORMALIZE', nw.add(
                nw.add(normal, nw.combine(0, 0, nw.uniform(0, growth_z, seed + i))),
                nw.uniform([face_perturb] * 3, [-face_perturb] * 3, seed + i)))
            length = nw.switch(nw.compare_direction('LESS_THAN', normal, (0, 0, 1), angle),
                               nw.uniform(*long_length_range, seed + i),
                               nw.uniform(*short_length_range, seed + i))
            mesh, top = nw.new_node(Nodes.ExtrudeMesh, input_kwargs={
                'Mesh': mesh,
                'Selection': top,
                'Offset': direction,
                'Offset Scale': length
            }).outputs[:2]
            mesh = nw.new_node(Nodes.ScaleElements,
                               input_kwargs={'Geometry': mesh, 'Selection': top, 'Scale': scale})

        geometry_without_top = nw.new_node(Nodes.DeleteGeometry,
                                           input_kwargs={'Geometry': mesh, 'Selection': top},
                                           attrs={'domain': 'FACE'})

        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry_without_top})
