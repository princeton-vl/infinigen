# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hongyu Wen
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=MP7EZCFrXek by blenderbitesize and https://www.youtube.com/watch?v=VPI9xq41nOk by Ryan King


import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.assets.materials.utils.surface_utils import clip, sample_range, sample_ratio, sample_color, geo_voronoi_noise
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface
import random

def shader_black_white_snake(nw: NodeWrangler, rand=True):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'Position'})
    
    mapping = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': attribute})
    
    wave_texture = nw.new_node(Nodes.WaveTexture,
        input_kwargs={'Vector': mapping, 'Scale': 10.0, 'Distortion': 1.5, 'Detail': 5.0, 'Detail Roughness': 0.8})
    
    mapping_1 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': attribute})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping_1, 'Scale': 3.0, 'Detail': 5.0})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.8, 'Color1': noise_texture.outputs["Color"], 'Color2': mapping_1})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mix, 'Scale': 8.0},
        attrs={'feature': 'DISTANCE_TO_EDGE'})
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.2, 'Color1': wave_texture.outputs["Fac"], 'Color2': voronoi_texture.outputs["Distance"]})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': mix_1})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (1.0, 0.9647, 0.8308, 1.0)
    colorramp.color_ramp.elements[1].position = 0.0977
    colorramp.color_ramp.elements[1].color = (0.0003, 0.0003, 0.0009, 1.0)
    if rand:
        for e in colorramp.color_ramp.elements:
            sample_color(e.color, offset=0.05, keep_sum=True)
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp.outputs["Color"], 'Metallic': 0.6, 'Specular': 0.2, 'Roughness': 0.4},
        attrs={'subsurface_method': 'BURLEY'})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})
    
def shader_brown(nw: NodeWrangler, rand=False):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'Position'})
    
    mapping_2 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': attribute, 'Scale': (0.5, 1.0, 1.0)})
    
    noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping_2, 'Scale': 10.0, 'Detail': 20.0, 'Roughness': 0.4, 'Distortion': 0.1})
    
    colorramp_2 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_2.outputs["Fac"]})
    colorramp_2.color_ramp.elements[0].position = 0.4045
    colorramp_2.color_ramp.elements[0].color = (0.013, 0.0011, 0.0027, 1.0)
    colorramp_2.color_ramp.elements[1].position = 0.4568
    colorramp_2.color_ramp.elements[1].color = (0.159, 0.0254, 0.0134, 1.0)
    if rand:
        for e in colorramp_2.color_ramp.elements:
            sample_color(e.color, offset=0.05)
    
    principled_bsdf_1 = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp_2.outputs["Color"], 'Metallic': 0.4, 'Specular': 0.3, 'Roughness': 1},
        attrs={'subsurface_method': 'BURLEY'})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': attribute, 'Scale': 0.4, 'Detail': 15.0})
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.95, 'Color1': noise_texture_1.outputs["Fac"], 'Color2': attribute},
        attrs={'blend_type': 'LINEAR_LIGHT'})
    
    voronoi_texture_1 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mix_1, 'Scale': 4.0, 'Randomness': 3.0})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': voronoi_texture_1.outputs["Distance"]})
    colorramp.color_ramp.elements[0].position = 0.1614
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.3068
    colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    voronoi_texture_2 = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mix_1, 'Scale': 10.0, 'Randomness': 3.0})
    
    colorramp_3 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': voronoi_texture_2.outputs["Distance"]})
    colorramp_3.color_ramp.elements[0].position = 0.1682
    colorramp_3.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_3.color_ramp.elements[1].position = 0.2864
    colorramp_3.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    mix_2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.0, 'Color1': colorramp.outputs["Color"], 'Color2': colorramp_3.outputs["Color"]})
    
    colorramp_4 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': mix_2})
    colorramp_4.color_ramp.elements.new(0)
    colorramp_4.color_ramp.elements.new(0)
    colorramp_4.color_ramp.elements[0].position = 0.0
    colorramp_4.color_ramp.elements[0].color = (0.843, 0.4775, 0.0444, 1.0)
    colorramp_4.color_ramp.elements[1].position = 0.1545
    colorramp_4.color_ramp.elements[1].color = (0.1524, 0.09, 0.0114, 1.0)
    colorramp_4.color_ramp.elements[2].position = 0.4409
    colorramp_4.color_ramp.elements[2].color = (0.0503, 0.0338, 0.0072, 1.0)
    colorramp_4.color_ramp.elements[3].position = 1.0
    colorramp_4.color_ramp.elements[3].color = (0.159, 0.0369, 0.011, 1.0)
    if rand:
        for e in colorramp_4.color_ramp.elements:
            sample_color(e.color, offset=0.2)
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp_4.outputs["Color"], 'Metallic': 0.4, 'Roughness': 0.5},
        attrs={'subsurface_method': 'BURLEY'})
    
    mix_shader = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': 0.3, 1: principled_bsdf_1, 2: principled_bsdf})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': mix_shader})
    
def shader_golden(nw: NodeWrangler, rand=False):
    # Code generated using version 2.4.3 of the node_transpiler
    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'Position'})
    
    mapping = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': attribute, 'Scale': (0.5, 1.0, 1.0)})
    
    wave_texture = nw.new_node(Nodes.WaveTexture,
        input_kwargs={'Vector': mapping, 'Distortion': 1.0, 'Detail': 15.0, 'Detail Roughness': 0.8, 'Phase Offset': 2.0})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': wave_texture.outputs["Color"], 'Scale': 1.5, 'Detail': 5.0, 'Roughness': 0.4})
    
    mapping_1 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': attribute})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping_1, 'Scale': 3.0, 'Detail': 5.0})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.8, 'Color1': noise_texture.outputs["Color"], 'Color2': mapping_1})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mix, 'Scale': 8.0},
        attrs={'feature': 'DISTANCE_TO_EDGE'})
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.1, 'Color1': noise_texture_1.outputs["Color"], 'Color2': voronoi_texture.outputs["Distance"]})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': mix_1})
    colorramp.color_ramp.elements[0].position = 0.4682
    colorramp.color_ramp.elements[0].color = (0.017, 0.0094, 0.0033, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (0.4969, 0.2582, 0.0666, 1.0)
    if rand:
        for e in colorramp.color_ramp.elements:
            sample_color(e.color, offset=0.05, keep_sum=True)
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp.outputs["Color"], 'Metallic': 0.4, 'Specular': 0.2, 'Roughness': 0.4},
        attrs={'subsurface_method': 'BURLEY'})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def shader_green(nw: NodeWrangler, rand=True):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'Position'})
    
    mapping_3 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': attribute})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': mapping_3})
    
    colorramp_3 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': separate_xyz.outputs["Z"]})
    colorramp_3.color_ramp.elements[0].position = 0.3864
    colorramp_3.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_3.color_ramp.elements[1].position = 0.6682
    colorramp_3.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    mapping_2 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': attribute, 'Scale': (0.5, 1.0, 1.0)})
    
    noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping_2, 'Scale': 10.0, 'Detail': 10.0, 'Roughness': 0.40000000000000002, 'Distortion': 0.10000000000000001})
    
    colorramp_2 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_2.outputs["Fac"]})
    colorramp_2.color_ramp.elements.new(0)
    colorramp_2.color_ramp.elements[0].position = 0.2318
    colorramp_2.color_ramp.elements[0].color = (0.64449999999999996, 0.52710000000000001, 0.0011999999999999999, 1.0)
    colorramp_2.color_ramp.elements[1].position = 0.375
    colorramp_2.color_ramp.elements[1].color = (0.050299999999999997, 0.033799999999999997, 0.0071999999999999998, 1.0)
    colorramp_2.color_ramp.elements[2].position = 0.45
    colorramp_2.color_ramp.elements[2].color = (0.0172, 0.040599999999999997, 0.0, 1.0)
    if rand:
        for e in colorramp_2.color_ramp.elements:
            sample_color(e.color, offset=0.1, keep_sum=True)
    
    principled_bsdf_1 = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp_2.outputs["Color"], 'Metallic': 0.40000000000000002, 'Roughness': 0.27000000000000002},
        attrs={'subsurface_method': 'BURLEY'})
    
    mapping = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': attribute, 'Scale': (0.5, 1.0, 1.0)})
    
    wave_texture = nw.new_node(Nodes.WaveTexture,
        input_kwargs={'Vector': mapping, 'Distortion': 1.0, 'Detail': 15.0, 'Detail Roughness': 0.80000000000000004, 'Phase Offset': 2.0})
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': wave_texture.outputs["Color"], 'Scale': 1.3999999999999999, 'Detail': 5.0, 'Roughness': 0.40000000000000002})
    
    mapping_1 = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': attribute})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping_1, 'Scale': 3.0, 'Detail': 5.0})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.80000000000000004, 'Color1': noise_texture.outputs["Color"], 'Color2': mapping_1})
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': mix, 'Scale': 8.0},
        attrs={'feature': 'DISTANCE_TO_EDGE'})
    
    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.10000000000000001, 'Color1': noise_texture_1.outputs["Color"], 'Color2': voronoi_texture.outputs["Distance"]})
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': mix_1})
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.2818
    colorramp.color_ramp.elements[0].color = (0.76819999999999999, 0.78349999999999997, 0.76049999999999995, 1.0)
    colorramp.color_ramp.elements[1].position = 0.4295
    colorramp.color_ramp.elements[1].color = (0.0012999999999999999, 0.0012999999999999999, 0.0012999999999999999, 1.0)
    colorramp.color_ramp.elements[2].position = 0.5068
    colorramp.color_ramp.elements[2].color = (0.0095999999999999992, 0.0149, 0.0, 1.0)
    colorramp.color_ramp.elements[3].position = 0.6727
    colorramp.color_ramp.elements[3].color = (0.0872, 0.23549999999999999, 0.0, 1.0)
    if rand:
        for e in colorramp.color_ramp.elements:
            sample_color(e.color, keep_sum=True)
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp.outputs["Color"], 'Metallic': 0.40000000000000002, 'Roughness': 0.27000000000000002},
        attrs={'subsurface_method': 'BURLEY'})
    
    mix_shader = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': colorramp_3.outputs["Color"], 1: principled_bsdf_1, 2: principled_bsdf})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': mix_shader})

def shader_shining_golden(nw: NodeWrangler, rand=True):
    # Code generated using version 2.4.3 of the node_transpiler

    base_color = [0.8, 0.2227, 0.0326, 1.0]
    if rand:
        base_color = sample_color(base_color, keep_sum=True)
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': base_color, 'Metallic': 0.6, 'Roughness': 0.27},
        attrs={'subsurface_method': 'BURLEY'})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

class shaders:
    def choose():
        choices = [shader_black_white_snake, shader_shining_golden, shader_golden, shader_green]
        # choices = [shader_green]
        return random.choice(choices)

def apply(obj, selection=None, **kwargs):
    shader = shaders.choose()
    surface.add_material(obj, shader, selection=selection)
