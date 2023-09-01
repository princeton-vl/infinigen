# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import bpy
import mathutils
import numpy as np
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category, hsv2rgba
from infinigen.core import surface

from infinigen.core.util.math import FixedSeed
from infinigen.core.util import blender as butil
from infinigen.core.placement.factory import AssetFactory

from infinigen.assets.fruits.fruit_utils import nodegroup_shape_quadratic, nodegroup_align_top_to_horizon
from infinigen.assets.fruits.cross_section_lib import nodegroup_circle_cross_section, nodegroup_star_cross_section, nodegroup_coconut_cross_section
from infinigen.assets.fruits.stem_lib import nodegroup_basic_stem, nodegroup_pineapple_stem, nodegroup_calyx_stem, nodegroup_empty_stem, nodegroup_coconut_stem

from infinigen.assets.fruits.surfaces.apple_surface import nodegroup_apple_surface
from infinigen.assets.fruits.surfaces.pineapple_surface import nodegroup_pineapple_surface
from infinigen.assets.fruits.surfaces.starfruit_surface import nodegroup_starfruit_surface
from infinigen.assets.fruits.surfaces.strawberry_surface import nodegroup_strawberry_surface
from infinigen.assets.fruits.surfaces.blackberry_surface import nodegroup_blackberry_surface
from infinigen.assets.fruits.surfaces.coconuthairy_surface import nodegroup_coconuthairy_surface
from infinigen.assets.fruits.surfaces.coconutgreen_surface import nodegroup_coconutgreen_surface
from infinigen.assets.fruits.surfaces.durian_surface import nodegroup_durian_surface
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

crosssectionlib = {
    'circle_cross_section': nodegroup_circle_cross_section,
    'star_cross_section': nodegroup_star_cross_section,
    'coconut_cross_section': nodegroup_coconut_cross_section,
}

shapelib = {'shape_quadratic': nodegroup_shape_quadratic}

surfacelib = {
    'apple_surface': nodegroup_apple_surface,
    'pineapple_surface': nodegroup_pineapple_surface,
    'starfruit_surface': nodegroup_starfruit_surface,
    'strawberry_surface': nodegroup_strawberry_surface,
    'blackberry_surface': nodegroup_blackberry_surface,
    'coconuthairy_surface': nodegroup_coconuthairy_surface,
    'coconutgreen_surface': nodegroup_coconutgreen_surface,
    'durian_surface': nodegroup_durian_surface,
}

stemlib = {
    'basic_stem': nodegroup_basic_stem,
    'pineapple_stem': nodegroup_pineapple_stem,
    'calyx_stem': nodegroup_calyx_stem,
    'empty_stem': nodegroup_empty_stem,
    'coconut_stem': nodegroup_coconut_stem,
}

def parse_args(nodeinfo, dictionary):
    for k1, v1 in dictionary.items():
        if isinstance(v1, str) and v1.startswith('noderef'):
            _, nodename, outputname = v1.split('-')
            dictionary[k1] = nodeinfo[nodename].outputs[outputname]

    return dictionary
        
def general_fruit_geometry_nodes(nw: NodeWrangler, 
        cross_section_params, shape_params, surface_params, stem_params):
    nodeinfo = {}

    parse_args(nodeinfo, cross_section_params['cross_section_input_args'])
    crosssection = nw.new_node(crosssectionlib[cross_section_params['cross_section_name']](**cross_section_params['cross_section_func_args']).name,
        input_kwargs=cross_section_params['cross_section_input_args'])
    nodeinfo['crosssection'] = crosssection
    parse_args(nodeinfo, cross_section_params['cross_section_output_args'])

    parse_args(nodeinfo, shape_params['shape_input_args'])
    shapequadratic = nw.new_node(shapelib[shape_params['shape_name']](**shape_params['shape_func_args']).name,
        input_kwargs=shape_params['shape_input_args'])
    nodeinfo['shapequadratic'] = shapequadratic
    parse_args(nodeinfo, shape_params['shape_output_args'])

    parse_args(nodeinfo, surface_params['surface_input_args'])
    fruitsurface = nw.new_node(surfacelib[surface_params['surface_name']](**surface_params['surface_func_args']).name,
        input_kwargs=surface_params['surface_input_args'])
    nodeinfo['fruitsurface'] = fruitsurface
    parse_args(nodeinfo, surface_params['surface_output_args'])

    parse_args(nodeinfo, stem_params['stem_input_args'])
    stem = nw.new_node(stemlib[stem_params['stem_name']](**stem_params['stem_func_args']).name,
        input_kwargs=stem_params['stem_input_args'])
    nodeinfo['stem'] = stem
    parse_args(nodeinfo, stem_params['stem_output_args'])

    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [fruitsurface, stem]})

    realize_instances = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': join_geometry})

    align = nw.new_node(nodegroup_align_top_to_horizon().name,
        input_kwargs={'Geometry': realize_instances})

    output_dict = {'Geometry': align}
    output_dict.update(cross_section_params['cross_section_output_args'])
    output_dict.update(shape_params['shape_output_args'])
    output_dict.update(surface_params['surface_output_args'])
    output_dict.update(stem_params['stem_output_args'])

    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs=output_dict)

class FruitFactoryGeneralFruit(AssetFactory):
    def __init__(self, factory_seed, scale=1.0, coarse=False):
        super(FruitFactoryGeneralFruit, self).__init__(factory_seed, coarse=coarse)

        self.scale = scale

    def sample_cross_section_params(self, surface_resolution=256):
        raise NotImplementedError

    def sample_shape_params(self, surface_resolution=256):
        raise NotImplementedError

    def sample_surface_params(self):
        raise NotImplementedError

    def sample_stem_params(self):
        raise NotImplementedError

    def sample_geo_genome(self):
        surface_params = self.sample_surface_params()
        surface_resolution = surface_params['surface_resolution']

        cross_section_params = self.sample_cross_section_params(surface_resolution)
        shape_params = self.sample_shape_params(surface_resolution)
        stem_params = self.sample_stem_params()

        return cross_section_params, shape_params, surface_params, stem_params

    def create_asset(self, **params):

        bpy.ops.mesh.primitive_plane_add(
            size=4, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        obj = bpy.context.active_object

        with FixedSeed(self.factory_seed):
            cross_section_params, shape_params, surface_params, stem_params = self.sample_geo_genome()

        scale_multiplier = surface_params['scale_multiplier']

        output_list = []
        output_list.extend(cross_section_params['cross_section_output_args'].keys())
        output_list.extend(shape_params['shape_output_args'].keys())
        output_list.extend(surface_params['surface_output_args'].keys())
        output_list.extend(stem_params['stem_output_args'].keys())

        surface.add_geomod(obj, 
            general_fruit_geometry_nodes, 
            attributes=output_list, 
            apply=False, 
            input_args=[cross_section_params, shape_params, surface_params, stem_params])

        bpy.ops.object.convert(target='MESH')

        obj = bpy.context.object
        obj.scale *= normal(1, 0.1) * self.scale * scale_multiplier
        butil.apply_transform(obj)

        tag_object(obj, 'fruit_'+self.name)
        return obj


