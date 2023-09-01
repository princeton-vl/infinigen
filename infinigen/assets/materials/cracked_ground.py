# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Ankit Goyal, Zeyu Ma
# Acknowledgment: This file draws inspiration from https://www.youtube.com/watch?v=PIZ_wi3yFUM&list=PLsGl9GczcgBs6TtApKKK-L_0Nm6fovNPk&index=98 by Ryan King Art

import bpy
import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface
from infinigen.core.util.organization import SurfaceTypes

import gin
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import random_color_neighbour

type = SurfaceTypes.SDFPerturb
mod_name = "geo_cracked_ground"
name = "cracked_ground" 

@node_utils.to_nodegroup('nodegroup_apply_value_to_normal', singleton=False, type='GeometryNodeTree')
def nodegroup_apply_value_to_normal(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    normal = nw.new_node(Nodes.InputNormal)
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'value', 0.0000),
            ('NodeSocketFloat', 'displacement', 1.0000)])
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: normal, 'Scale': group_input.outputs["value"]},
        attrs={'operation': 'SCALE'})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 'Scale': group_input.outputs["displacement"]},
        attrs={'operation': 'SCALE'})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Vector': scale_1.outputs["Vector"]}, attrs={'is_active_output': True})

def shader_cracked_ground(nw: NodeWrangler, random_seed=0):
    # Code generated using version 2.6.4 of the node_transpiler
    with FixedSeed(random_seed):
        col_crac = random_color_neighbour((0.2016, 0.107, 0.0685, 1.0), 0.1, 0.1, 0.1)
        col_1 = random_color_neighbour((0.3005, 0.1119, 0.0284, 1.0), 0.1, 0.1, 0.1)
        col_2 = random_color_neighbour((0.6038, 0.4397, 0.2159, 1.0), 0.1, 0.1, 0.1)

    attribute_2 = nw.new_node(Nodes.Attribute, attrs={'attribute_name': 'bump'})
    
    attribute = nw.new_node(Nodes.Attribute, attrs={'attribute_name': 'crack'})
    
    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"], 'Scale': 15.0000, 'Detail': 10.0000})
    
    separate_color = nw.new_node(Nodes.SeparateColor, input_kwargs={'Color': noise_texture.outputs["Color"]})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_color.outputs["Red"], 1: 0.4000, 2: 0.7000, 3: 0.4900, 4: 0.5100})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_color.outputs["Green"], 1: 0.4000, 2: 0.7200, 3: 0.4000, 4: 1.1000})
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Hue': map_range.outputs["Result"], 'Value': map_range_1.outputs["Result"], 'Color': col_1})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': attribute.outputs["Fac"], 'Color1': hue_saturation_value, 'Color2': col_crac})
    
    mix_2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': attribute_2.outputs["Fac"], 'Color1': mix, 'Color2': col_2})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={'Base Color': mix_2, 'Specular': 0.2000, 'Roughness': 0.9000})
    
    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': principled_bsdf}, attrs={'is_active_output': True})
    return principled_bsdf

@gin.configurable
def geo_cracked_ground(nw: NodeWrangler, selection=None, random_seed=0):
    # Code generated using version 2.6.4 of the node_transpiler

    with FixedSeed(random_seed):
        # control the coordinate that noise textures evaluated on
        noise_rnd_seed = uniform(-10000, 10000)

        # scale of cracks; smaller means larger cracks
        sca_crac = nw.new_value(uniform(1, 3), "sca_crac")

        # scale of masks; mask=0 means no crack in that area
        sca_mask = nw.new_value(uniform(1, 3), "sca_mask")

        # scale of bumpy surface noise
        sca_noise = nw.new_value(uniform(2, 4), "sca_mask")

        # percentage of area with crac, 0.5 means in half of area
        crack_density =  nw.new_value(uniform(0.4, 0.55), "crack_density")

        # width of the crack
        wid_crac = nw.new_value(uniform(0.01, 0.04), "wid_crac")

        # scale of the grains, smaller means larger grains
        sca_gra = nw.new_value(uniform(20, 100), "sca_gra")

        # depth of crack
        dep_crac = nw.new_value(uniform(-0.1, -0.3), "dep_crac")

        # total displacement
        dep_landscape = nw.new_value(uniform(0.3, 0.7), "dep_landscape")


    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
    
    position = nw.new_node(Nodes.InputPosition)
    
    seed = nw.new_value(noise_rnd_seed, "seed")
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': position, 'W': seed, 'Scale': sca_noise, 'Detail': 15.0000, 'Roughness': 0.5375},
        attrs={'noise_dimensions': '4D'})
    
    store_named_attribute = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Name': 'noise', 'Value': noise_texture.outputs["Fac"]})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': position, 'W': seed, 'Scale': sca_crac, 'Detail': 15.0000},
        attrs={'noise_dimensions': '4D'})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': noise_texture_1.outputs["Color"], 'W': seed, 'Scale': 2.3000},
        attrs={'feature': 'DISTANCE_TO_EDGE', 'voronoi_dimensions': '4D'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture.outputs["Distance"], 2: wid_crac, 3: 1.0000, 4: 0.0000})
    
    noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': position, 'W': seed, 'Scale': sca_mask, 'Detail': 15.0000},
        attrs={'noise_dimensions': '4D'})
        
    subtract = nw.new_node(Nodes.Math, input_kwargs={0: 1.0000, 1: crack_density}, attrs={'operation': 'SUBTRACT'})
    
    subtract_1 = nw.new_node(Nodes.Math, input_kwargs={0: subtract, 1: 0.0200}, attrs={'operation': 'SUBTRACT'})
    
    add = nw.new_node(Nodes.Math, input_kwargs={0: subtract, 1: 0.0200})
    
    map_range_1 = nw.new_node(Nodes.MapRange, input_kwargs={'Value': noise_texture_2.outputs["Fac"], 1: subtract_1, 2: add})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: map_range_1.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    store_named_attribute_1 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': store_named_attribute, 'Name': 'crack', 'Value': multiply})
    
    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': position, 'W': seed, 'Scale': sca_gra},
        attrs={'voronoi_dimensions': '4D'})
    
    map_range_2 = nw.new_node(Nodes.MapRange, input_kwargs={'Value': voronoi_texture_1.outputs["Distance"], 1: 0.9000})
    
    store_named_attribute_2 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': store_named_attribute_1, 'Name': 'bump', 'Value': map_range_2.outputs["Result"]})
    
    applyvaluetonormal = nw.new_node(nodegroup_apply_value_to_normal().name,
        input_kwargs={'value': noise_texture.outputs["Fac"], 'displacement': 0.3000})
    
    applyvaluetonormal_1 = nw.new_node(nodegroup_apply_value_to_normal().name, input_kwargs={'value': multiply, 'displacement': dep_crac})
    
    add_1 = nw.new_node(Nodes.VectorMath, input_kwargs={0: applyvaluetonormal, 1: applyvaluetonormal_1})
    
    applyvaluetonormal_2 = nw.new_node(nodegroup_apply_value_to_normal().name,
        input_kwargs={'value': map_range_2.outputs["Result"], 'displacement': 0.0200})
    
    add_2 = nw.new_node(Nodes.VectorMath, input_kwargs={0: add_1.outputs["Vector"], 1: applyvaluetonormal_2})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add_2.outputs["Vector"], 'Scale': dep_landscape},
        attrs={'operation': 'SCALE'})

    offset = scale
    if selection is not None:
        offset = nw.multiply(offset, surface.eval_argument(nw, selection))
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': store_named_attribute_2, 'Offset': offset})
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position}, attrs={'is_active_output': True})

def apply(obj, selection=None, **kwargs):
    # seed = randint(10000000)
    surface.add_geomod(obj, geo_cracked_ground, selection=selection) #, input_kwargs={'random_seed': seed})
    surface.add_material(obj, shader_cracked_ground, selection=selection) #, input_kwargs={'random_seed': seed})