# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang

import bpy
import mathutils
from numpy.random import uniform as U, normal as N, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface
from infinigen.core.util.organization import SurfaceTypes
from infinigen.core.util.math import FixedSeed
import gin

type = SurfaceTypes.SDFPerturb
mod_name = "geo_mud"
name = "mud"

def shader_mud(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    geometry_5 = nw.new_node(Nodes.NewGeometry)
    
    noise_texture_1_w = nw.new_node(Nodes.Value, label='noise_texture_1_w')
    noise_texture_1_w.outputs[0].default_value = 9.6366
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': geometry_5.outputs["Position"], 'W': noise_texture_1_w, 'Scale': N(5, 0.5)},
        attrs={'noise_dimensions': '4D'})
    
    color1 = [0.0216, 0.0145, 0.0113, 1.0000]
    color2 = [0.0424, 0.0308, 0.0142, 1.0000]    
    for i in range(3):
        color1[i] += N(0, 0.005)
        color2[i] += N(0, 0.005)
    colorramp_3 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': noise_texture_1.outputs["Fac"]})
    colorramp_3.color_ramp.elements[0].position = 0.0000
    colorramp_3.color_ramp.elements[0].color = color1
    colorramp_3.color_ramp.elements[1].position = 1.0000
    colorramp_3.color_ramp.elements[1].color = color2
    
    geometry_1 = nw.new_node(Nodes.NewGeometry)
    
    musgrave_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Vector': geometry_1.outputs["Position"], 'Scale': 0.2000, 'W': U(-10, 10)},
        attrs={'musgrave_dimensions': '4D', 'musgrave_type': 'RIDGED_MULTIFRACTAL'})
    
    colorramp_5 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': musgrave_texture})
    colorramp_5.color_ramp.elements[0].position = 0.0000
    colorramp_5.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    colorramp_5.color_ramp.elements[1].position = N(0.1045, 0.01)
    colorramp_5.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]
    
    x1 = U(0.85, 0.95)
    x2 = U(0.65, 0.75)
    colorramp_6 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': colorramp_5.outputs["Color"]})
    colorramp_6.color_ramp.elements[0].position = 0.0000
    colorramp_6.color_ramp.elements[0].color = [x1, x1, x1, 1.0000]
    colorramp_6.color_ramp.elements[1].position = 1.0000
    colorramp_6.color_ramp.elements[1].color = [x2, x2, x2, 1.0000]
    
    x1 = U(0.05, 0.15)
    x2 = U(0.45, 0.55)
    colorramp_4 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': noise_texture_1.outputs["Fac"]})
    colorramp_4.color_ramp.elements[0].position = 0.0000
    colorramp_4.color_ramp.elements[0].color = [x1, x1, x1, 1.0000]
    colorramp_4.color_ramp.elements[1].position = 1.0000
    colorramp_4.color_ramp.elements[1].color = [x2, x2, x2, 1.0000]
    
    mix_3 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_5.outputs["Color"], 'Color1': (0.0000, 0.0000, 0.0000, 1.0000), 'Color2': colorramp_4.outputs["Color"]})
    
    principled_bsdf_2 = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': colorramp_3.outputs["Color"], 'Specular': colorramp_6.outputs["Color"], 'Roughness': mix_3})
    
    material_output = nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': principled_bsdf_2}, attrs={'is_active_output': True})
    return principled_bsdf_2

@gin.configurable
def geo_mud(nw: NodeWrangler, random_seed=0, selection=None):
    # Code generated using version 2.6.4 of the node_transpiler
    with FixedSeed(random_seed):

        group_input = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])
        
        position_5 = nw.new_node(Nodes.InputPosition)
        
        noise_texture_3 = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Vector': position_5})
        
        mix_2 = nw.new_node(Nodes.MixRGB,
            input_kwargs={'Fac': nw.new_value(N(0.6, 0.1), "mix_2_fac"), 'Color1': noise_texture_3.outputs["Color"], 'Color2': position_5})
        
        noise_texture_4 = nw.new_node(Nodes.NoiseTexture, input_kwargs={'Vector': mix_2, 'Scale': nw.new_value(N(50, 5), "noise_texture_4_scale")})
        
        voronoi_texture_2 = nw.new_node(Nodes.VoronoiTexture, input_kwargs={'Vector': mix_2, 'Scale': nw.new_value(N(3.0000, 0.5), "voronoi_texture_2_scale")})
        
        colorramp_1 = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': voronoi_texture_2.outputs["Distance"]})
        colorramp_1.color_ramp.elements[0].position = 0.0000
        colorramp_1.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
        colorramp_1.color_ramp.elements[1].position = 1.0000
        colorramp_1.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]
        
        float_curve_1 = nw.new_node(Nodes.FloatCurve, 
        input_kwargs={
            'Value': colorramp_1.outputs["Color"]})

        node_utils.assign_curve(
            float_curve_1.mapping.curves[0], 
            [(0.0000, 0.0000), (0.3386, 0.0844), (0.8114, 0.6312), (1.0000, 0.7656)]
        )
        # node_utils.assign_curve(
        #     float_curve_1.mapping.curves[0], 
        #     [(0.0000, 0.0000), (0.3386+N(0, 0.05), 0.0844), (0.8114+N(0, 0.05), 0.6312), (1.0000, 0.7656)]
        # )
        
        value_6 = nw.new_node(Nodes.Value)
        value_6.outputs[0].default_value = N(2, 0.2)
        
        multiply = nw.new_node(Nodes.VectorMath, input_kwargs={0: float_curve_1, 1: value_6}, attrs={'operation': 'MULTIPLY'})
        
        add = nw.new_node(Nodes.VectorMath, input_kwargs={0: noise_texture_4.outputs["Fac"], 1: multiply.outputs["Vector"]})
        
        normal = nw.new_node(Nodes.InputNormal)
        
        multiply_1 = nw.new_node(Nodes.VectorMath, input_kwargs={0: add.outputs["Vector"], 1: normal}, attrs={'operation': 'MULTIPLY'})
        
        value_5 = nw.new_node(Nodes.Value)
        value_5.outputs[0].default_value = N(0.04, 0.005)
        
        multiply_2 = nw.new_node(Nodes.VectorMath,
            input_kwargs={0: multiply_1.outputs["Vector"], 1: value_5},
            attrs={'operation': 'MULTIPLY'})
        
        offset = multiply_2.outputs["Vector"]
        if selection is not None:
            offset = nw.multiply(offset, surface.eval_argument(nw, selection))
        
        set_position = nw.new_node(Nodes.SetPosition,
            input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': offset})
        
        group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position}, attrs={'is_active_output': True})

def apply(obj, selection=None, **kwargs):
    surface.add_geomod(obj, geo_mud, selection=selection)
    surface.add_material(obj, shader_mud, selection=selection)