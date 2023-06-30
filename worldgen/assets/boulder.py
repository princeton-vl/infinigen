from functools import reduce

import bpy
import numpy as np
import trimesh.convex
from numpy.random import uniform

from surfaces.scatters import ivy
from util import blender as butil
from util.math import FixedSeed
from nodes.node_wrangler import Nodes, NodeWrangler
from surfaces import surface
from assets.utils.object import trimesh2obj
from assets.utils.decorate import geo_extension, write_attribute
from assets.utils.misc import log_uniform
from placement.factory import AssetFactory


class BoulderFactory(AssetFactory):

    config_mappings = {'boulder': [True, False], 'slab': [False, True]}
        super(BoulderFactory, self).__init__(factory_seed, coarse)
        self.weights = [.8, .2]
        self.configs = ['boulder', 'slab']
        with FixedSeed(factory_seed):
            method = np.random.choice(self.configs, p=self.weights)
            self.has_horizontal_cut, self.is_slab = self.config_mappings[method]
        vertices = np.random.uniform(-1, 1, (32, 3))
        obj = trimesh2obj(trimesh.convex.convex_hull(vertices))
        surface.add_geomod(obj, self.geo_extrusion, apply=True)
        butil.modify_mesh(obj, 'SUBSURF', render_levels=2, levels=2, subdivision_type='SIMPLE')

        obj.location[-1] += obj.dimensions[-1] * .2
        butil.apply_transform(obj, loc=True)
        if self.is_slab:
            obj.scale = *log_uniform(.5, 2., 2), log_uniform(.1, .15)
        else:
            obj.scale = *log_uniform(.4, 1.2, 2), log_uniform(.4, .8)
        butil.apply_transform(obj)
        obj.rotation_euler[0] = uniform(-np.pi / 24, np.pi / 24)
        butil.apply_transform(obj)
        obj.rotation_euler[-1] = uniform(0, np.pi * 2)
        butil.apply_transform(obj)


    @staticmethod
    def geo_extrusion(nw: NodeWrangler, extrude_scale=1):
        geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        face_area = nw.new_node(Nodes.InputMeshFaceArea)

        tops = []
        extrude_configs = [(uniform(.2, .3), .8, .4), (.6, .2, .6)]
        top_facing = nw.compare_direction('LESS_THAN', nw.new_node(Nodes.InputNormal), (0, 0, 1), np.pi * 2 / 3)
        for prob, extrude, scale in extrude_configs:
            extrude = extrude * extrude_scale
            face_area_stats = nw.new_node(Nodes.AttributeStatistic, [geometry, None, face_area],
                                          attrs={'domain': 'FACE'}).outputs
            selection = reduce(lambda *xs: nw.boolean_math('AND', *xs), [top_facing, nw.bernoulli(prob),
                nw.compare('GREATER_THAN', face_area, face_area_stats['Mean'])])
            geometry, top, side = nw.new_node(Nodes.ExtrudeMesh, [geometry, selection, None,
                nw.uniform(extrude * .5, extrude)]).outputs
            geometry = nw.new_node(Nodes.ScaleElements, [geometry, top, nw.uniform(scale * .5, scale)])
            tops.append(top)

        geometry = nw.new_node(Nodes.StoreNamedAttribute,
                               [geometry, 'top', None, reduce(lambda *xs: nw.boolean_math('OR', *xs), tops)])
        nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geometry})

    def create_asset(self, i, placeholder, face_size=0.01, distance=0, **params):

