# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei



import logging
from functools import reduce

import bpy
import numpy as np
import trimesh.convex
from numpy.random import uniform
import gin

from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core import surface
from infinigen.assets.utils.object import trimesh2obj
from infinigen.assets.utils.decorate import geo_extension, write_attribute
from infinigen.assets.utils.misc import log_uniform
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.detail import remesh_with_attrs
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.placement.split_in_view import split_inview
from infinigen.core.placement import detail

logger = logging.getLogger(__name__)

class BoulderFactory(AssetFactory):

    config_mappings = {'boulder': [True, False], 'slab': [False, True]}

    def __init__(
        self, factory_seed,
        meshing_camera=None, 
        adapt_mesh_method='remesh', 
        cam_meshing_max_dist=1e7,
        coarse=False, do_voronoi=True
    ):
        super(BoulderFactory, self).__init__(factory_seed, coarse)

        self.camera = meshing_camera
        self.cam_meshing_max_dist = cam_meshing_max_dist
        self.adapt_mesh_method = adapt_mesh_method

        self.octree_depth = 3
        self.do_voronoi = do_voronoi
        self.weights = [.8, .2]
        self.configs = ['boulder', 'slab']
        with FixedSeed(factory_seed):
            self.rock_surface = surface.registry('rock_collection')
            method = np.random.choice(self.configs, p=self.weights)
            self.has_horizontal_cut, self.is_slab = self.config_mappings[method]

    @gin.configurable
    def create_placeholder(self, boulder_scale = 1, **kwargs) -> bpy.types.Object:
        butil.select_none()

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

        obj.scale *= boulder_scale
        butil.apply_transform(obj)
        obj.rotation_euler[0] = uniform(-np.pi / 24, np.pi / 24)
        butil.apply_transform(obj)
        obj.rotation_euler[-1] = uniform(0, np.pi * 2)
        butil.apply_transform(obj)

        with butil.SelectObjects(obj):
            bpy.ops.geometry.attribute_convert(mode='VERTEX_GROUP')

        butil.modify_mesh(obj, 'BEVEL', limit_method='VGROUP', vertex_group='top', invert_vertex_group=True,
                            offset_type='PERCENT', width_pct=10)
        butil.modify_mesh(obj, 'REMESH', apply=True, mode='SHARP', octree_depth=self.octree_depth)
        surface.add_geomod(obj, geo_extension, apply=True)

        if self.do_voronoi:
            voronoi_texture = bpy.data.textures.new(name='boulder', type='VORONOI')
            voronoi_texture.noise_scale = log_uniform(.2, .5)
            voronoi_texture.distance_metric = 'DISTANCE'
            butil.modify_mesh(obj, 'DISPLACE', texture=voronoi_texture, strength=.01, mid_level=0)

            voronoi_texture = bpy.data.textures.new(name='boulder', type='VORONOI')
            voronoi_texture.noise_scale = log_uniform(.05, .1)
            voronoi_texture.distance_metric = 'DISTANCE'
            butil.modify_mesh(obj, 'DISPLACE', texture=voronoi_texture, strength=.01, mid_level=0)

        return obj

    def finalize_placeholders(self, placeholders):
        with FixedSeed(self.factory_seed):
            self.rock_surface.apply(placeholders, is_rock=True)

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
            input_kwargs={'Geometry': geometry, 'Name': 'top', 'Value': reduce(lambda *xs: nw.boolean_math('OR', *xs), tops)})
        nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geometry})

    def create_asset(self, i, placeholder, face_size=0.01, distance=0, **params):

        if self.camera is not None and distance < self.cam_meshing_max_dist:
            assert self.adapt_mesh_method != 'remesh'
            skin_obj, outofview, vert_dists, _ = split_inview(placeholder, cam=self.camera, vis_margin=0.15)
            butil.parent_to(outofview, skin_obj, no_inverse=True, no_transform=True)
            face_size = detail.target_face_size(vert_dists.min())
        else:
            skin_obj = deep_clone_obj(placeholder, keep_modifiers=True, keep_materials=True)
        
        butil.parent_to(skin_obj, placeholder, no_inverse=True, no_transform=True)

        with butil.DisableModifiers(skin_obj):
            detail.adapt_mesh_resolution(skin_obj, face_size, method=self.adapt_mesh_method, apply=True)

        tag_object(skin_obj, 'boulder')
        return skin_obj