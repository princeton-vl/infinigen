# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang
# Acknowledgment: This file draws inspiration from https://www.youtube.com/watch?v=mJVuodaPHTQ and https://www.youtube.com/watch?v=v7a4ouBLIow by Lance Phan


import os, sys, random
import numpy as np
import math as ma
from infinigen.assets.materials.utils.surface_utils import clip, sample_range, sample_ratio, sample_color, geo_voronoi_noise
import bpy
import mathutils
from numpy.random import uniform as U, normal as N, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface


@node_utils.to_nodegroup('nodegroup_node_grid', singleton=False, type='GeometryNodeTree')
def nodegroup_node_grid(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Value', 0.5)])
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"], 1: 2.0},
        attrs={'operation': 'MULTIPLY'})
    
    floor = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply},
        attrs={'operation': 'FLOOR'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: floor},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_1})
    
    trunc = nw.new_node(Nodes.Math,
        input_kwargs={0: add},
        attrs={'operation': 'TRUNC'})
    
    trunc_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_1},
        attrs={'operation': 'TRUNC'})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: trunc_1})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'floor1': trunc, 'floor2': add_1})


@node_utils.to_nodegroup('nodegroup_node_group', singleton=False, type='GeometryNodeTree')
def nodegroup_UV(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Vector', (0.0, 0.0, 0.0))])
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["Vector"]})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: 0.75, 1: separate_xyz.outputs["X"]},
        attrs={'operation': 'SUBTRACT'})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract},
        attrs={'operation': 'ABSOLUTE'})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={1: absolute},
        attrs={'operation': 'SUBTRACT'})
    
    absolute_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_1},
        attrs={'operation': 'ABSOLUTE'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: absolute_1, 1: 2.0},
        attrs={'operation': 'MULTIPLY'})
    
    subtract_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: multiply},
        attrs={'operation': 'SUBTRACT', 'use_clamp': True})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': subtract_2, 'Y': separate_xyz.outputs["Y"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Vector': combine_xyz})

@node_utils.to_nodegroup('nodegroup_scales', singleton=False, type='GeometryNodeTree')
def nodegroup_scales(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Mesh', None),
            ('NodeSocketVector', 'Vector', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Scale', 40.0),
            ('NodeSocketFloat', 'Xscale', 1.0),
            ('NodeSocketFloat', 'Yscale', 1.0),
            ('NodeSocketFloat', 'Xnoise', 0.02),
            ('NodeSocketFloat', 'Ynoise', 0.02),
            ('NodeSocketFloat', 'Offset', 0.0002)])
    
    #subdivide_mesh = nw.new_node(Nodes.SubdivideMesh,
    #    input_kwargs={'Mesh': group_input.outputs["Mesh"]})
    
    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["Vector"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: 10.0},
        attrs={'operation': 'MULTIPLY'})
    
    reroute_2 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': multiply})
    
    angle = nw.new_node(Nodes.Value,
        label='Angle')
    angle.outputs[0].default_value = 0.0
    
    cosine = nw.new_node(Nodes.Math,
        input_kwargs={0: angle},
        attrs={'operation': 'COSINE'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute_2, 1: cosine},
        attrs={'operation': 'MULTIPLY'})
    
    sine = nw.new_node(Nodes.Math,
        input_kwargs={0: angle},
        attrs={'operation': 'SINE'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: sine},
        attrs={'operation': 'MULTIPLY'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_1, 1: multiply_2},
        attrs={'operation': 'SUBTRACT'})
    
    group_input_2 = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Mesh', None),
            ('NodeSocketVector', 'Vector', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Scale', 40.0),
            ('NodeSocketFloat', 'Xscale', 1.0),
            ('NodeSocketFloat', 'Yscale', 1.0),
            ('NodeSocketFloat', 'Xnoise', 0.02),
            ('NodeSocketFloat', 'Ynoise', 0.02),
            ('NodeSocketFloat', 'Offset', 0.0002)])
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: group_input_2.outputs["Xscale"]},
        attrs={'operation': 'MULTIPLY'})
    
    noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'W': 0.8, 'Scale': 10.0},
        attrs={'noise_dimensions': '4D'})
    
    multiply_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture_2.outputs["Fac"], 1: group_input_2.outputs["Xnoise"]},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_3, 1: multiply_4})
    
    multiply_5 = nw.new_node(Nodes.Math,
        input_kwargs={0: reroute_2, 1: sine},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_6 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: cosine},
        attrs={'operation': 'MULTIPLY'})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_5, 1: multiply_6})
    
    multiply_7 = nw.new_node(Nodes.Math,
        input_kwargs={0: add_1, 1: group_input_2.outputs["Yscale"]},
        attrs={'operation': 'MULTIPLY'})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'W': 0.8, 'Scale': 10.0},
        attrs={'noise_dimensions': '4D'})
    
    multiply_8 = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture_1.outputs["Fac"], 1: group_input_2.outputs["Ynoise"]},
        attrs={'operation': 'MULTIPLY'})
    
    add_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_7, 1: multiply_8})
    
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': add, 'Y': add_2})
    
    group_input_1 = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Mesh', None),
            ('NodeSocketVector', 'Vector', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'Scale', 40.0),
            ('NodeSocketFloat', 'Xscale', 1.0),
            ('NodeSocketFloat', 'Yscale', 1.0),
            ('NodeSocketFloat', 'Xnoise', 0.02),
            ('NodeSocketFloat', 'Ynoise', 0.02),
            ('NodeSocketFloat', 'Offset', 0.0002)])
    
    multiply_8 = nw.new_node(Nodes.VectorMath, input_kwargs={0: combine_xyz_2, 1: group_input_1.outputs["Scale"]}, attrs={'operation': 'MULTIPLY'})

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': multiply_8})
    
    nodegrid = nw.new_node(nodegroup_node_grid().name, input_kwargs={'Value': separate_xyz.outputs["Y"]})
    
    greater_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: nodegrid.outputs["floor1"], 1: separate_xyz.outputs["Y"]},
        attrs={'operation': 'LESS_THAN'})

    less_than = nw.new_node(Nodes.Compare, input_kwargs={0: nodegrid.outputs["floor1"], 1: separate_xyz.outputs["Y"]})

    nodegrid_1 = nw.new_node(nodegroup_node_grid().name, input_kwargs={'Value': separate_xyz.outputs["X"]})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': nodegrid_1.outputs["floor2"], 'Y': nodegrid.outputs["floor1"]})
    
    multiply_9 = nw.new_node(Nodes.VectorMath, input_kwargs={0: less_than, 1: combine_xyz}, attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': nodegrid_1.outputs["floor1"], 'Y': nodegrid.outputs["floor2"]})
    
    multiply_10 = nw.new_node(Nodes.VectorMath, input_kwargs={0: greater_than, 1: combine_xyz_1}, attrs={'operation': 'MULTIPLY'})
    
    add_3 = nw.new_node(Nodes.VectorMath, input_kwargs={0: multiply_9.outputs["Vector"], 1: multiply_10.outputs["Vector"]})
    
    subtract_1 = nw.new_node(Nodes.VectorMath, input_kwargs={0: multiply_8, 1: add_3}, attrs={'operation': 'SUBTRACT'})
    
    distance = nw.new_node(Nodes.VectorMath, input_kwargs={0: multiply_8, 1: add_3}, attrs={'operation': 'DISTANCE'})
    
    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: distance.outputs["Value"], 1: 0.0100})
    
    less_than_1 = nw.new_node(Nodes.Compare, input_kwargs={0: add_4, 1: 0.5000}, attrs={'operation': 'LESS_THAN'})
    
    greater_than_1 = nw.new_node(Nodes.Compare, input_kwargs={0: add_4, 1: 0.5000})
    
    multiply_11 = nw.new_node(Nodes.VectorMath, input_kwargs={0: less_than, 1: combine_xyz_1}, attrs={'operation': 'MULTIPLY'})
    
    multiply_12 = nw.new_node(Nodes.VectorMath, input_kwargs={0: greater_than, 1: combine_xyz}, attrs={'operation': 'MULTIPLY'})
    
    add_5 = nw.new_node(Nodes.VectorMath, input_kwargs={0: multiply_11.outputs["Vector"], 1: multiply_12.outputs["Vector"]})
    
    subtract_2 = nw.new_node(Nodes.VectorMath, input_kwargs={0: multiply_8, 1: add_5}, attrs={'operation': 'SUBTRACT'})
    
    multiply_13 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: greater_than_1, 1: subtract_2.outputs["Vector"]},
        attrs={'operation': 'MULTIPLY'})
    
    _multiply_add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract_1.outputs["Vector"], 1: less_than_1, 2: multiply_13.outputs["Vector"]},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    multiply_add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: _multiply_add, 1: (1, -1, 1)},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_14 = nw.new_node(Nodes.VectorMath, input_kwargs={0: greater_than_1, 1: add_5}, attrs={'operation': 'MULTIPLY'})
    
    multiply_add_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add_3, 1: less_than_1, 2: multiply_14.outputs["Vector"]},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': multiply_add_1, 'W': sample_range(-10, 10), 'Scale': 33.0000},
        attrs={'noise_dimensions': '4D'})
    
    subtract_3 = nw.new_node(Nodes.MapRange,
        input_kwargs={0: noise_texture.outputs["Fac"], 1: 0.26, 2: 0.74, 3: -0.5, 4: 0.5},
        attrs={'clamp': True}
        )

    sine_1 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_3}, attrs={'operation': 'SINE'})
    
    cosine_1 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_3}, attrs={'operation': 'COSINE'})
    
    combine_xyz_color = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': sine_1, 'Y': cosine_1, 'Z': 0.0000})

    add_6 = nw.new_node(Nodes.VectorMath, input_kwargs={0: combine_xyz_color.outputs["Vector"], 1: multiply_add}, attrs={'operation': 'DOT_PRODUCT'})
    
    distance_1 = nw.new_node(Nodes.VectorMath, input_kwargs={0: multiply_8, 1: add_5}, attrs={'operation': 'DISTANCE'})
    
    add_7 = nw.new_node(Nodes.Math, input_kwargs={0: distance_1.outputs["Value"], 1: 0.0100})
    
    multiply_17 = nw.new_node(Nodes.Math, input_kwargs={0: greater_than_1, 1: add_7}, attrs={'operation': 'MULTIPLY'})
    
    multiply_add_2 = nw.new_node(Nodes.Math, input_kwargs={0: add_4, 1: less_than_1, 2: multiply_17}, attrs={'operation': 'MULTIPLY_ADD'})
    
    multiply_18 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_add_2, 1: 2.0000}, attrs={'operation': 'MULTIPLY'})
    
    multiply_19 = nw.new_node(Nodes.MapRange, 
        input_kwargs={0: multiply_18, 1: 0.9156, 2: 1.0000, 3: 0.0000, 4: 0.5},
        attrs={'clamp': True}
    )

    subtract_4 = nw.new_node(Nodes.Math, input_kwargs={0: add_6, 1: multiply_19}, attrs={'operation': 'SUBTRACT'})
    
    subtract_5 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_4, 1: 0.0000}, attrs={'operation': 'SUBTRACT'})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    multiply_20 = nw.new_node(Nodes.VectorMath, input_kwargs={0: subtract_5, 1: normal}, attrs={'operation': 'MULTIPLY'})
    
    multiply_21 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply_20.outputs["Vector"], 1: group_input.outputs["Offset"]},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Mesh"], 'Offset': multiply_21.outputs["Vector"]})
    
    capture_attribute_1 = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': set_position, 1: multiply_add_1},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    capture_attribute_4 = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': capture_attribute_1.outputs["Geometry"], 1: multiply_19},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    separate_xyz_7 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': capture_attribute_1.outputs["Attribute"]})
    
    attribute_statistic = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': capture_attribute_1.outputs["Geometry"], 2: separate_xyz_7.outputs["X"]})
    
    subtract_8 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_7.outputs["X"], 1: attribute_statistic.outputs["Min"]},
        attrs={'operation': 'SUBTRACT'})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_8, 1: attribute_statistic.outputs["Range"]},
        attrs={'operation': 'DIVIDE'})
    
    attribute_statistic_1 = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': capture_attribute_1.outputs["Geometry"], 2: separate_xyz_7.outputs["Y"]})
    
    subtract_9 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_7.outputs["Y"], 1: attribute_statistic_1.outputs["Min"]},
        attrs={'operation': 'SUBTRACT'})
    
    divide_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_9, 1: attribute_statistic_1.outputs["Range"]},
        attrs={'operation': 'DIVIDE'})
    
    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': divide, 'Y': divide_1})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': capture_attribute_4.outputs["Geometry"], 
                'attr2': combine_xyz_3, 
                'attr5': capture_attribute_4.outputs["Attribute"]})

def shader_fish_body_regular(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate_1 = nw.new_node(Nodes.TextureCoord)
    
    mapping_1 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': texture_coordinate_1.outputs["Generated"]})
    
    noise_texture_6 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping_1, 'W': 0.8, 'Scale': 50.0},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture_6.inputs['W'].default_value = sample_range(-2, 2)

    colorramp_15 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_6.outputs["Fac"]})
    colorramp_15.color_ramp.elements[0].position = 0.3523
    colorramp_15.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_15.color_ramp.elements[1].position = 0.3727
    colorramp_15.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    attribute_3 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'offset2'})
    
    greater_than = nw.new_node(Nodes.Math,
        input_kwargs={0: attribute_3.outputs["Vector"], 1: 0.01},
        attrs={'operation': 'GREATER_THAN'})
    
    texture_coordinate_5 = nw.new_node(Nodes.TextureCoord)
    
    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': texture_coordinate_5.outputs["Normal"]})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: 0.5})
    if rand:
        add.inputs[1].default_value = sample_range(0.45, 0.6)
    
    colorramp_14 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': add})
    colorramp_14.color_ramp.elements[0].position = 0.0
    colorramp_14.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_14.color_ramp.elements[1].position = 0.2341
    colorramp_14.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    attribute_5 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'Color variations'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': attribute_5.outputs["Vector"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"]},
        attrs={'operation': 'MULTIPLY'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: multiply},
        attrs={'operation': 'SUBTRACT'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': subtract, 1: -0.2})
    
    colorramp_12 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': map_range.outputs["Result"]})
    colorramp_12.color_ramp.elements[0].position = 0.0
    colorramp_12.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_12.color_ramp.elements[1].position = 0.2518
    colorramp_12.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    texture_coordinate_3 = nw.new_node(Nodes.TextureCoord)
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': texture_coordinate_3.outputs["Generated"]})
    
    invert_1 = nw.new_node(Nodes.Invert,
        input_kwargs={'Color': separate_xyz_1.outputs["Z"]})
    
    subtract_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: 0.57},
        attrs={'operation': 'SUBTRACT'})
    
    absolute = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_1},
        attrs={'operation': 'ABSOLUTE'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: absolute, 1: 0.4},
        attrs={'operation': 'MULTIPLY'})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': multiply_1})
    
    subtract_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: invert_1, 1: map_range_1.outputs["Result"]},
        attrs={'operation': 'SUBTRACT'})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_2, 1: 0.1})
    
    colorramp_13 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': map_range.outputs["Result"]})
    colorramp_13.color_ramp.elements[0].position = 0.0
    colorramp_13.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp_13.color_ramp.elements[1].position = 0.6727
    colorramp_13.color_ramp.elements[1].color = (0.0685, 0.0685, 0.0685, 1.0)
    
    texture_coordinate_2 = nw.new_node(Nodes.TextureCoord)
    
    noise_texture_9 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate_2.outputs["Generated"]})
    
    mix_7 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.9042, 'Color1': noise_texture_9.outputs["Color"], 'Color2': texture_coordinate_2.outputs["Generated"]})
    
    wave_texture = nw.new_node(Nodes.WaveTexture,
        input_kwargs={'Vector': mix_7, 'Scale': 2.5, 'Distortion': 1.3, 'Detail': 0.0, 'Detail Roughness': 0.0, 'Phase Offset': 0.2})
    if rand:
        wave_texture.inputs['Scale'].default_value = sample_ratio(2, 0.5, 2)
        wave_texture.inputs['Phase Offset'].default_value = sample_range(0, 10)
        wave_texture.inputs['Distortion'].default_value = sample_range(0, 3)

    colorramp_8 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': wave_texture.outputs["Color"]})
    colorramp_8.color_ramp.elements[0].position = 0.0795
    colorramp_8.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_8.color_ramp.elements[1].position = 1.0
    colorramp_8.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    if rand:
        colorramp_8.color_ramp.elements[0].position = sample_range(0, 0.2)

    add_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: colorramp_8.outputs["Color"], 1: -0.5})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: add_2, 1: 2.0},
        attrs={'operation': 'DIVIDE'})
    
    invert = nw.new_node(Nodes.Invert,
        input_kwargs={'Color': separate_xyz_1.outputs["Z"]})
    if rand:
        invert.inputs['Fac'].default_value = sample_range(0.5, 1)

    add_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: divide, 1: invert})
    
    mix_10 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 1.0, 'Color1': colorramp_13.outputs["Color"], 'Color2': add_3},
        attrs={'blend_type': 'MULTIPLY'})
    
    colorramp_9 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': mix_10})
    colorramp_9.color_ramp.elements.new(0)
    colorramp_9.color_ramp.elements[0].position = 0.0
    colorramp_9.color_ramp.elements[0].color = (0.0179, 0.0119, 0.0, 1.0)
    colorramp_9.color_ramp.elements[1].position = 0.2318
    colorramp_9.color_ramp.elements[1].color = (0.1972, 0.1213, 0.0148, 1.0)
    colorramp_9.color_ramp.elements[2].position = 1.0
    colorramp_9.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)
    if rand:
        sample_color(colorramp_9.color_ramp.elements[1].color)
        colorramp_9.color_ramp.elements[1].position = sample_range(0.1, 0.9)

    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 3.0})
    
    colorramp_3 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture.outputs["Fac"]})
    colorramp_3.color_ramp.elements[0].position = 0.2614
    colorramp_3.color_ramp.elements[0].color = (0.0059, 0.0028, 0.0002, 1.0)
    colorramp_3.color_ramp.elements[1].position = 0.5795
    colorramp_3.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    mix_5 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 1.0, 'Color1': colorramp_9.outputs["Color"], 'Color2': colorramp_3.outputs["Color"]},
        attrs={'blend_type': 'MULTIPLY'})
    
    mix_9 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': add_1, 'Color1': (0.021, 0.0158, 0.0026, 1.0), 'Color2': mix_5})
    
    mix_8 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_12.outputs["Color"], 'Color1': (1.0, 1.0, 1.0, 1.0), 'Color2': mix_9})
    
    colorramp_4 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_6.outputs["Fac"]})
    colorramp_4.color_ramp.elements[0].position = 0.2455
    colorramp_4.color_ramp.elements[0].color = (0.0642, 0.0339, 0.006, 1.0)
    colorramp_4.color_ramp.elements[1].position = 0.4886
    colorramp_4.color_ramp.elements[1].color = (0.1224, 0.3306, 0.261, 1.0)
    
    mix_6 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.0, 'Color1': mix_8, 'Color2': colorramp_4.outputs["Color"]},
        attrs={'blend_type': 'ADD'})
    
    mix_11 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_14.outputs["Color"], 'Color1': (0.4072, 0.4072, 0.4072, 1.0), 'Color2': mix_5})
    
    colorramp_7 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': greater_than})
    colorramp_7.color_ramp.elements[0].position = 0.0
    colorramp_7.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp_7.color_ramp.elements[1].position = 0.7682
    colorramp_7.color_ramp.elements[1].color = (0.0228, 0.0165, 0.0, 1.0)
    
    mix_4 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': greater_than, 'Color1': mix_11, 'Color2': colorramp_7.outputs["Color"]})
    
    mix_12 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_15.outputs["Color"], 'Color1': (0.0119, 0.0078, 0.0086, 1.0), 'Color2': mix_4})
    if rand:
        sample_color(mix_12.inputs[6].default_value, keep_sum=True)

    principled_bsdf_1 = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_12, 'Subsurface Radius': (0.36, 0.46, 0.6), 'Subsurface Color': (1.0, 0.9405, 0.7747, 1.0), 'Metallic': 0.8, 'Specular': .9, 'Roughness': 0.3, 'IOR': 1.69},
        attrs={'subsurface_method': 'BURLEY'})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf_1})

def shader_fish_body_gold(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'Color variations'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': attribute.outputs["Vector"]})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': separate_xyz.outputs["X"]})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.4209
    colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    attribute_1 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'offset2'})
    
    greater_than = nw.new_node(Nodes.Math,
        input_kwargs={0: attribute_1.outputs["Vector"], 1: 0.01},
        attrs={'operation': 'GREATER_THAN'})
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': separate_xyz.outputs["X"]})
    colorramp_1.color_ramp.elements[0].position = 0.0
    colorramp_1.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.8659
    colorramp_1.color_ramp.elements[1].color = (0.0685, 0.0685, 0.0685, 1.0)
    
    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate.outputs["Generated"], 'W': 1.1861},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture.inputs['W'].default_value = sample_range(-2, 2)

    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.9042, 'Color1': noise_texture.outputs["Color"], 'Color2': texture_coordinate.outputs["Generated"]})
    
    wave_texture = nw.new_node(Nodes.WaveTexture,
        input_kwargs={'Vector': mix, 'Scale': 1.0062, 'Distortion': 2.7139, 'Detail': 0.0, 'Detail Roughness': 0.0, 'Phase Offset': 8.0064})
    if rand:
        wave_texture.inputs['Scale'].default_value = sample_ratio(2, 0.5, 2)
        wave_texture.inputs['Phase Offset'].default_value = sample_range(0, 10)
        wave_texture.inputs['Distortion'].default_value = sample_range(0, 3)

    colorramp_2 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': wave_texture.outputs["Color"]})
    colorramp_2.color_ramp.elements[0].position = 0.1254
    colorramp_2.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_2.color_ramp.elements[1].position = 1.0
    colorramp_2.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    if rand:
        colorramp_2.color_ramp.elements[0].position = sample_range(0, 0.2)

    add = nw.new_node(Nodes.Math,
        input_kwargs={0: colorramp_2.outputs["Color"], 1: -0.0})
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: add, 1: 2.0},
        attrs={'operation': 'DIVIDE'})
    
    texture_coordinate_1 = nw.new_node(Nodes.TextureCoord)
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': texture_coordinate_1.outputs["Generated"]})
    
    invert = nw.new_node(Nodes.Invert,
        input_kwargs={'Fac': 0.5593, 'Color': separate_xyz_1.outputs["Z"]})
    if rand:
        invert.inputs['Fac'].default_value = sample_range(0.5, 1)
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: divide, 1: invert})
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 1.0, 'Color1': colorramp_1.outputs["Color"], 'Color2': add_1},
        attrs={'blend_type': 'MULTIPLY'})
    
    colorramp_3 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': mix_1})
    colorramp_3.color_ramp.elements.new(0)
    colorramp_3.color_ramp.elements[0].position = 0.0091
    colorramp_3.color_ramp.elements[0].color = (0.9254, 0.0164, 0.0, 1.0)
    colorramp_3.color_ramp.elements[1].position = 0.5136
    colorramp_3.color_ramp.elements[1].color = (1.0, 0.2384, 0.0027, 1.0)
    colorramp_3.color_ramp.elements[2].position = 1.0
    colorramp_3.color_ramp.elements[2].color = (1.0, 0.1073, 0.0738, 1.0)
    if rand:
        for e in colorramp_3.color_ramp.elements:
            sample_color(e.color, offset=0.05)
        colorramp_3.color_ramp.elements[1].position = sample_range(0.2, 0.8)

    colorramp_4 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': greater_than})
    colorramp_4.color_ramp.elements[0].position = 0.0
    colorramp_4.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp_4.color_ramp.elements[1].position = 0.7682
    colorramp_4.color_ramp.elements[1].color = (0.0228, 0.0165, 0.0, 1.0)
    
    colorramp_5 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': colorramp_4.outputs["Color"]})
    colorramp_5.color_ramp.elements[0].position = 0.0
    colorramp_5.color_ramp.elements[0].color = (1.0, 0.0817, 0.0, 1.0)
    colorramp_5.color_ramp.elements[1].position = 1.0
    colorramp_5.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    mix_2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': greater_than, 'Color1': colorramp_3.outputs["Color"], 'Color2': colorramp_5.outputs["Color"]})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 20.0})
    
    colorramp_6 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_1.outputs["Fac"]})
    colorramp_6.color_ramp.elements[0].position = 0.2614
    colorramp_6.color_ramp.elements[0].color = (1.0, 0.2402, 0.0026, 1.0)
    colorramp_6.color_ramp.elements[1].position = 0.5795
    colorramp_6.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    mix_3 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.75, 'Color1': mix_2, 'Color2': colorramp_6.outputs["Color"]},
        attrs={'blend_type': 'MULTIPLY'})
    
    mix_4 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp.outputs["Color"], 'Color1': (1.0, 1.0, 1.0, 1.0), 'Color2': mix_3})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_4, 'Subsurface Radius': (0.36, 0.46, 0.6), 'Subsurface Color': (1.0, 0.9405, 0.7747, 1.0), 'Metallic': 0.5, 'Specular': 0.5273, 'Roughness': 0.1, 'IOR': 1.69},
        attrs={'subsurface_method': 'BURLEY'})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def geometry_fish_body(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler
    group_input = nw.new_node(Nodes.GroupInput)

    UV = nw.new_node(nodegroup_UV().name, 
        input_kwargs={'Vector': nw.expose_input('UVMap', attribute='UVMap', dtype='NodeSocketVector')})

    scales = nw.new_node(nodegroup_scales().name,
        input_kwargs={'Mesh': group_input, 'Vector': UV, 'Scale': 6, 'Xscale': 0.3, 'Yscale': 12.0, 'Offset': 0.002})
    if rand:
        scales.inputs['Scale'].default_value = sample_ratio(6, 2/3, 3/2)
        scales.inputs['Xscale'].default_value = sample_range(0.2, 0.3)
        scales.inputs['Yscale'].default_value = sample_range(8, 12)
        scales.inputs['Xnoise'].default_value = sample_range(0.1, 0.3)
        scales.inputs['Ynoise'].default_value = sample_range(0.1, 0.3)

    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 50.0},
        attrs={'noise_dimensions': '4D'})
    if rand:
        noise_texture.inputs['W'].default_value = sample_range(-2, 2)

    add = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Fac"], 1: -0.5})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add, 1: normal},
        attrs={'operation': 'MULTIPLY'})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.002
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply.outputs["Vector"], 1: value},
        attrs={'operation': 'MULTIPLY'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': scales.outputs["Geometry"], 'Offset': multiply_1.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position, 
        'attr2': scales.outputs['attr2'],
        'attr5': scales.outputs['attr5']})

@node_utils.to_nodegroup('nodegroup_gradient_color', singleton=False, type='ShaderNodeTree')
def nodegroup_gradient_color(nw: NodeWrangler, **input_args):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute_5 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'Color variations'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': attribute_5.outputs["Vector"]})
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Value', 0.5),
        #('NodeSocketColor', 'Color1', (1.0, 1.0, 1.0, 1.0)),
        #('NodeSocketColor', 'Color2', (0.5268, 0.6724, 0.5186, 1.0)),
        #('NodeSocketColor', 'Color3', (0.8055, 0.6284, 0.2728, 1.0)),
        #('NodeSocketColor', 'Color4', (0.838, 0.5269, 0.0338, 1.0)),
        #('NodeSocketColor', 'Color5', (0.0397, 0.0175, 0.0028, 1.0)),
        ('NodeSocketFloat', 'Value1', -0.2)])
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: group_input.outputs["Value"]},
        attrs={'operation': 'MULTIPLY'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: multiply},
        attrs={'operation': 'SUBTRACT'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': subtract, 1: group_input.outputs["Value1"]})

    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': map_range.outputs["Result"]})
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements.new(0)
    colorramp_1.color_ramp.elements[0].position = 0.0945
    colorramp_1.color_ramp.elements[0].color = input_args["Color1"]
    colorramp_1.color_ramp.elements[1].position = 0.2045
    colorramp_1.color_ramp.elements[1].color = input_args["Color2"]
    colorramp_1.color_ramp.elements[2].position = 0.3159
    colorramp_1.color_ramp.elements[2].color = input_args["Color3"]
    colorramp_1.color_ramp.elements[3].position = 0.4977
    colorramp_1.color_ramp.elements[3].color = input_args["Color4"]
    colorramp_1.color_ramp.elements[4].position = 0.7568
    colorramp_1.color_ramp.elements[4].color = input_args["Color5"]
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Color': colorramp_1.outputs["Color"]})


@node_utils.to_nodegroup('nodegroup_noise_color', singleton=False, type='ShaderNodeTree')
def nodegroup_noise_color(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate_3 = nw.new_node(Nodes.TextureCoord)
    
    mapping_3 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': texture_coordinate_3.outputs["Generated"]})
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Scale', 10.0),
            ('NodeSocketColor', 'Color1', (0.7379, 0.2623, 0.0648, 1.0)),
            ('NodeSocketColor', 'Color2', (0.5029, 0.4287, 0.1079, 1.0))])
    
    noise_texture_8 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping_3, 'W': U(-5, 5), 'Scale': group_input.outputs["Scale"]},
        attrs={'noise_dimensions': '4D'})
    
    colorramp_9 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_8.outputs["Fac"]})
    colorramp_9.color_ramp.elements[0].position = U(0, 0.5)
    colorramp_9.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_9.color_ramp.elements[1].position = 1.0
    colorramp_9.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    mix_12 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_9.outputs["Color"], 'Color1': group_input.outputs["Color1"], 'Color2': group_input.outputs["Color2"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={"Color": mix_12})


def shader_stripe_fish(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute_6 = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'offset2'})
    
    colorramp_11 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': attribute_6.outputs["Color"]})
    colorramp_11.color_ramp.elements[0].position = 0.0
    colorramp_11.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_11.color_ramp.elements[1].position = 0.2909
    colorramp_11.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    texture_coordinate_2 = nw.new_node(Nodes.TextureCoord)
    
    mapping_4 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': texture_coordinate_2.outputs["Generated"]})
    
    noise_texture_9 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping_4, 'Scale': N(10, 2)})
    
    colorramp_10 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_9.outputs["Fac"]})
    colorramp_10.color_ramp.elements[0].position = 0.3773
    colorramp_10.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_10.color_ramp.elements[1].position = 1.0
    x = U(0.1, 0.9)
    colorramp_10.color_ramp.elements[1].color = (x, x, x, 1.0)
    
    mapping_2 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': texture_coordinate_2.outputs["Generated"]})
    
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': mapping_2})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Z"]},
        attrs={'operation': 'SUBTRACT'})
    
    power = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: 4.0 if U()<0.7 else 3.0},
        attrs={'operation': 'POWER'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: power, 1: U(-2, 3)},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: multiply})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': add, 'Y': separate_xyz_1.outputs["Y"], 'Z': separate_xyz_1.outputs["Z"]})
    
    reroute = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': combine_xyz})
    
    scale = nw.new_node(Nodes.Value,
        label='scale')
    scale.outputs[0].default_value = max(N(2, 0.6), 0.8)
    
    distortion = nw.new_node(Nodes.Value,
        label='distortion')
    distortion.outputs[0].default_value = N(5, 1)
    
    detail = nw.new_node(Nodes.Value,
        label='detail')
    detail.outputs[0].default_value = 15.0
    
    detailscale = nw.new_node(Nodes.Value,
        label='detailscale')
    detailscale.outputs[0].default_value = U(0.1, 0.8)
    
    offset = nw.new_node(Nodes.Value,
        label='offset')
    offset.outputs[0].default_value = N(1.8, 0.2)
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: offset, 1: -1.0},
        attrs={'operation': 'MULTIPLY'})
    
    wave_texture_2 = nw.new_node(Nodes.WaveTexture,
        input_kwargs={'Vector': reroute, 'Scale': scale, 'Distortion': distortion, 'Detail': detail, 'Detail Scale': detailscale, 'Phase Offset': multiply_1})
    
    no_side_stripe = U() > 0.7
    side_stripe_width = U(0.02, 0.1)
    colorramp_8 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': wave_texture_2.outputs["Color"]})
    colorramp_8.color_ramp.elements[0].position = 0.0
    colorramp_8.color_ramp.elements[0].color = (no_side_stripe, no_side_stripe, no_side_stripe, 1.0)
    colorramp_8.color_ramp.elements[1].position = side_stripe_width
    colorramp_8.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    wave_texture_1 = nw.new_node(Nodes.WaveTexture,
        input_kwargs={'Vector': reroute, 'Scale': scale, 'Distortion': distortion, 'Detail': detail, 'Detail Scale': detailscale, 'Phase Offset': offset})
    
    colorramp_6 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': wave_texture_1.outputs["Color"]})
    colorramp_6.color_ramp.elements[0].position = 0.0
    colorramp_6.color_ramp.elements[0].color = (no_side_stripe, no_side_stripe, no_side_stripe, 1.0)
    colorramp_6.color_ramp.elements[1].position = side_stripe_width
    colorramp_6.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    wave_texture = nw.new_node(Nodes.WaveTexture,
        input_kwargs={'Vector': reroute, 'Scale': scale, 'Distortion': distortion, 'Detail': detail, 'Detail Scale': detailscale})
    
    colorramp_5 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': wave_texture.outputs["Color"]})
    colorramp_5.color_ramp.elements[0].position = U(0.2, 0.8)
    colorramp_5.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_5.color_ramp.elements[1].position = 0.9955
    colorramp_5.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    if U() < 0.5:
        group_1 = nw.new_node(nodegroup_noise_color().name,
            input_kwargs={"Scale": N(10, 2), "Color1":(U(), U(), U(), 1.0), "Color2": (U(), U(), U(), 1.0)})
    else:
        group_1 = nw.new_node(nodegroup_gradient_color(
                Color1=(U(0.8, 1), U(0.8, 1), U(0.8, 1), 1.0),
                Color2=(U(), U(), U(), 1.0),
                Color3=(U(), U(), U(), 1.0),
                Color4=(U(), U(), U(), 1.0),
                Color5=(U(0, 0.1), U(0, 0.1), U(0, 0.1), 1.0),).name,
            input_kwargs={
                "Value": N(0.5, 0.1),
                "Value1": N(-0.2, 0.1),
            })

    group = nw.new_node(nodegroup_gradient_color(
            Color1=(U(0.8, 1), U(0.8, 1), U(0.8, 1), 1.0),
            Color2=(U(), U(), U(), 1.0),
            Color3=(U(), U(), U(), 1.0),
            Color4=(U(), U(), U(), 1.0),
            Color5=(U(0, 0.1), U(0, 0.1), U(0, 0.1), 1.0),).name,
        input_kwargs={
            "Value": N(0.5, 0.1),
            "Value1": N(-0.2, 0.1),
        })
    
    mix_8 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_5.outputs["Color"], 'Color1': group_1, 'Color2': group})
    
    side_stripe_color = (U(), U(), U(), 1.0)
    mix_9 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_6.outputs["Color"], 'Color1': side_stripe_color, 'Color2': mix_8})
    
    mix_10 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_8.outputs["Color"], 'Color1': side_stripe_color, 'Color2': mix_9})
    
    mix_11 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_10.outputs["Color"], 'Color1': mix_10, 'Color2': (U(), U(), U(), 1.0)})
    
    mix_7 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_11.outputs["Color"], 'Color1': mix_11, 'Color2': (U(0, 0.5), U(0, 0.5), U(0, 0.5), 1.0)})
    
    noise_texture_7 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 10.0})
    
    roughness = U(0, 0.5)
    colorramp_14 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_7.outputs["Fac"]})
    colorramp_14.color_ramp.elements[0].position = 0.0
    colorramp_14.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_14.color_ramp.elements[1].position = 0.9955
    colorramp_14.color_ramp.elements[1].color = (roughness, roughness, roughness, 1.0)
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_7, 'Metallic': max(0, N(0.2, 0.1)), 'Roughness': colorramp_14.outputs["Color"]})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})


def apply(obj, geo_kwargs=None, shader_kwargs=None, **kwargs):

    attributes = [
        'Color variations',
        'offset2'
    ]

    x = random.random()
    if x < 0.2:
        shader = shader_fish_body_gold
    elif x < 0.5:
        shader = shader_stripe_fish
    else:
        shader = shader_fish_body_regular

    surface.add_geomod(obj, geometry_fish_body, input_kwargs=geo_kwargs, attributes=attributes)
    surface.add_material(obj, shader, input_kwargs=shader_kwargs)

if __name__ == "__main__":
    for i in range(1):
        bpy.ops.wm.open_mainfile(filepath='dev_scene_fish_nurb.blend')
        i = 0
        for obj in bpy.data.objects:
            if obj.name.find('Nurb') >= 0:
                apply(obj, geo_kwargs={'rand': True}, shader_kwargs={'rand': True, 'stripefish':True})
                i += 1
        fn = os.path.join(os.path.abspath(os.curdir), 'dev_scene_test_fish_nurb2.blend')
        bpy.ops.wm.save_as_mainfile(filepath=fn)
