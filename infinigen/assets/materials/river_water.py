# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import bpy
import mathutils
from numpy.random import normal as N
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

from infinigen.core.util.random import random_color_neighbour

def shader_river_water(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    light_path = nw.new_node(Nodes.LightPath)

    multiply = nw.new_node(Nodes.Math, input_kwargs={1: light_path.outputs["Is Camera Ray"]}, attrs={'operation': 'MULTIPLY'})

    transparent_bsdf = nw.new_node(Nodes.TransparentBSDF)

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={'Roughness': 0.0000, 'IOR': 1.3300, 'Transmission': 1.0000})

    mix_shader = nw.new_node(Nodes.MixShader, input_kwargs={'Fac': multiply, 1: transparent_bsdf, 2: principled_bsdf})

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': texture_coordinate.outputs["Object"]})

    map_range_1 = nw.new_node(Nodes.MapRange, input_kwargs={'Value': separate_xyz.outputs["Y"], 2: 20.0000, 3: -0.4000})

    colorramp = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': map_range_1.outputs["Result"]})
    colorramp.color_ramp.interpolation = "B_SPLINE"
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.0000
    colorramp.color_ramp.elements[0].color = random_color_neighbour([0.1982, 0.1841, 0.0513, 1.0000], 0.05, 0.05, 0.05)
    colorramp.color_ramp.elements[1].position = 0.3545 + 0.01 * N()
    colorramp.color_ramp.elements[1].color = random_color_neighbour([0.1278, 0.1384, 0.0615, 1.0000], 0.05, 0.05, 0.05)
    colorramp.color_ramp.elements[2].position = 0.6773 + 0.01 * N()
    colorramp.color_ramp.elements[2].color = random_color_neighbour([0.0563, 0.0897, 0.0347, 1.0000], 0.05, 0.05, 0.05)
    colorramp.color_ramp.elements[3].position = 1.0000
    colorramp.color_ramp.elements[3].color = random_color_neighbour([0.0256, 0.0123, 0.0000, 1.0000], 0.05, 0.05, 0.05)

    map_range_2 = nw.new_node(Nodes.MapRange, input_kwargs={'Value': separate_xyz.outputs["Y"], 2: 20.0000, 3: 1.0000, 4: 6.0000})

    volume_scatter = nw.new_node('ShaderNodeVolumeScatter',
        input_kwargs={'Color': colorramp.outputs["Color"], 'Density': map_range_2.outputs["Result"], 'Anisotropy': 0.1500})

    rgb = nw.new_node(Nodes.RGB)
    rgb.outputs[0].default_value = random_color_neighbour((0.0290, 0.2718, 0.6748, 1.0000), 0.05, 0.05, 0.05)

    geometry = nw.new_node(Nodes.NewGeometry)

    musgrave_texture = nw.new_node(Nodes.MusgraveTexture, input_kwargs={'Vector': geometry.outputs["Position"], 'Scale': 11.6400})

    map_range = nw.new_node(Nodes.MapRange, input_kwargs={'Value': musgrave_texture, 3: 0.0784, 4: 0.2000})

    principled_volume = nw.new_node(Nodes.PrincipledVolume,
        input_kwargs={'Color': rgb, 'Density': map_range.outputs["Result"], 'Anisotropy': 0.3909})

    mix_shader_2 = nw.new_node(Nodes.MixShader, input_kwargs={1: volume_scatter, 2: principled_volume})

    volume_absorption = nw.new_node('ShaderNodeVolumeAbsorption', input_kwargs={'Color': rgb, 'Density': 5.9000+ 0.1 * N()})

    mix_shader_1 = nw.new_node(Nodes.MixShader, input_kwargs={1: mix_shader_2, 2: volume_absorption})

    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': mix_shader, 'Volume': mix_shader_1},
        attrs={'is_active_output': True})

def geometry_river_water(nw: NodeWrangler):
    # Code generated using version 2.6.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])

    position = nw.new_node(Nodes.InputPosition)

    wave = nw.new_node(Nodes.Vector, label='wave')
    wave.vector = (581.0000, 380.0000, 982.0000)

    add = nw.new_node(Nodes.VectorMath, input_kwargs={0: position, 1: wave})

    add_1 = nw.new_node(Nodes.VectorMath, input_kwargs={0: (0.0000, 3.8168, 0.0000), 1: add.outputs["Vector"]})

    water_scale = nw.new_node(Nodes.Value, label='water_scale')
    water_scale.outputs[0].default_value = 4.8569

    water_detail = nw.new_node(Nodes.Value, label='water_detail')
    water_detail.outputs[0].default_value = 5.8690

    water_dimension = nw.new_node(Nodes.Value, label='water_dimension')
    water_dimension.outputs[0].default_value = 1.1885

    water_lacunarity = nw.new_node(Nodes.Value, label='water_lacunarity')
    water_lacunarity.outputs[0].default_value = 1.8505

    musgrave_texture_1 = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Vector': add_1.outputs["Vector"], 'Scale': water_scale, 'Detail': water_detail, 'Dimension': water_dimension, 'Lacunarity': water_lacunarity})

    water_height = nw.new_node(Nodes.Value, label='water_height')
    water_height.outputs[0].default_value = 0.0011

    position_1 = nw.new_node(Nodes.InputPosition)

    musgrave_texture = nw.new_node(Nodes.MusgraveTexture, input_kwargs={'Vector': position_1, 'Scale': 4.8811})

    add_2 = nw.new_node(Nodes.Math, input_kwargs={1: musgrave_texture})

    multiply = nw.new_node(Nodes.Math, input_kwargs={0: water_height, 1: add_2}, attrs={'operation': 'MULTIPLY'})

    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: musgrave_texture_1, 1: multiply}, attrs={'operation': 'MULTIPLY'})

    ripple0 = nw.new_node(Nodes.Vector, label='ripple0')
    ripple0.vector = (130.0000, 634.0000, 140.0000)

    add_3 = nw.new_node(Nodes.VectorMath, input_kwargs={0: ripple0, 1: position})

    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': add_3.outputs["Vector"], 'Scale': 0.1000},
        attrs={'feature': 'DISTANCE_TO_EDGE'})

    voronoi_texture = nw.new_node(Nodes.VoronoiTexture, input_kwargs={'Vector': add_3.outputs["Vector"], 'Scale': 0.1000})

    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add_3.outputs["Vector"], 1: voronoi_texture.outputs["Position"]},
        attrs={'operation': 'SUBTRACT'})

    wave_texture = nw.new_node(Nodes.WaveTexture,
        input_kwargs={'Vector': subtract.outputs["Vector"], 'Scale': 1.0000, 'Phase Offset': -79.3357},
        attrs={'wave_type': 'RINGS', 'rings_direction': 'SPHERICAL'})

    multiply_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: voronoi_texture_1.outputs["Distance"], 1: wave_texture.outputs["Color"]},
        attrs={'operation': 'MULTIPLY'})

    ripple1 = nw.new_node(Nodes.Vector, label='ripple1')
    ripple1.vector = (819.0000, 938.0000, 541.0000)

    add_4 = nw.new_node(Nodes.VectorMath, input_kwargs={0: ripple1, 1: position})

    voronoi_texture_3 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': add_4.outputs["Vector"], 'Scale': 0.1000},
        attrs={'feature': 'DISTANCE_TO_EDGE'})

    voronoi_texture_2 = nw.new_node(Nodes.VoronoiTexture, input_kwargs={'Vector': add_4.outputs["Vector"], 'Scale': 0.1000})

    subtract_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add_4.outputs["Vector"], 1: voronoi_texture_2.outputs["Position"]},
        attrs={'operation': 'SUBTRACT'})

    wave_texture_1 = nw.new_node(Nodes.WaveTexture,
        input_kwargs={'Vector': subtract_1.outputs["Vector"], 'Scale': 1.0000, 'Phase Offset': -46.3218},
        attrs={'wave_type': 'RINGS', 'rings_direction': 'SPHERICAL'})

    multiply_3 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: voronoi_texture_3.outputs["Distance"], 1: wave_texture_1.outputs["Color"]},
        attrs={'operation': 'MULTIPLY'})

    add_5 = nw.new_node(Nodes.VectorMath, input_kwargs={0: multiply_2.outputs["Vector"], 1: multiply_3.outputs["Vector"]})

    ripple_height = nw.new_node(Nodes.Value, label='ripple_height')
    ripple_height.outputs[0].default_value = 0.0109

    multiply_4 = nw.new_node(Nodes.Math, input_kwargs={0: add_5.outputs["Vector"], 1: ripple_height}, attrs={'operation': 'MULTIPLY'})

    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: multiply_4})

    multiply_5 = nw.new_node(Nodes.VectorMath, input_kwargs={0: add_6, 1: (0.0000, 0.0000, 1.0000)}, attrs={'operation': 'MULTIPLY'})

    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': multiply_5.outputs["Vector"]})

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position}, attrs={'is_active_output': True})



def apply(obj, selection=None, **kwargs):
    surface.add_geomod(obj, geometry_river_water, selection=selection, attributes=[])
    surface.add_material(obj, shader_river_water, selection=selection)