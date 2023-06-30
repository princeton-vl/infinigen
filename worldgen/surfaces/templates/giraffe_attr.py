import os, sys
import numpy as np
import math as ma
from surfaces.surface_utils import clip, sample_range, sample_ratio, sample_color, geo_voronoi_noise
import bpy
import mathutils
from nodes.node_wrangler import Nodes, NodeWrangler
from nodes import node_utils
from surfaces import surface


    # Code generated using version 2.4.3 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute,
        attrs={'attribute_name': 'local_pos'})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': attribute.outputs["Color"]})
    
    mix = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': 0.9, 'Color1': noise_texture.outputs["Color"], 'Color2': attribute.outputs["Color"]})
    
    mapping = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': mix})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 10.0
    if rand:
        value.outputs[0].default_value = sample_ratio(value.outputs[0].default_value, 0.5, 2)
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        attrs={'voronoi_dimensions': '2D'})
    
    voronoi_texture_4 = nw.new_node(Nodes.VoronoiTexture,
        attrs={'voronoi_dimensions': '2D', 'feature': 'SMOOTH_F1'})
    
    subtract = nw.new_node(Nodes.Math,
        attrs={'operation': 'SUBTRACT'})
    
    less_than = nw.new_node(Nodes.Math,
        attrs={'operation': 'LESS_THAN'})
    
    colorramp_1 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': less_than})
    colorramp_1.color_ramp.elements[0].position = 0.2545
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.2886
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    group = nw.new_node(nodegroup_color_mask().name)
    
    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': group})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.9301, 0.5647, 0.3372, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (0.9755, 1.0, 0.9096, 1.0)
    if rand:

    mix_1 = nw.new_node(Nodes.MixRGB,
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_1},
        attrs={'subsurface_method': 'BURLEY'})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

def apply(obj, geo_kwargs=None, shader_kwargs=None, **kwargs):
    surface.add_material(obj, shader_giraffe_attr, reuse=False, input_kwargs=shader_kwargs)

if __name__ == "__main__":
    for i in range(1):
        bpy.ops.wm.open_mainfile(filepath='dev_scene_1019.blend')
        #creature(73349, 0).parts(0, factory=QuadrupedBody)
        apply(bpy.data.objects['creature(73349, 0).parts(0, factory=QuadrupedBody)'], geo_kwargs={'rand': True}, shader_kwargs={'rand': True})
        fn = os.path.join(os.path.abspath(os.curdir), 'dev_scene_test_giraffe_attr.blend')
        bpy.ops.wm.save_as_mainfile(filepath=fn)
        #bpy.context.scene.render.filepath = os.path.join('surfaces/surface_thumbnails', 'bone%d.jpg'%(i))
        #bpy.context.scene.render.image_settings.file_format='JPEG'
        #bpy.ops.render.render(write_still=True)