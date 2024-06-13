# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core import surface

def waterfall_shader(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    light_path = nw.new_node(Nodes.LightPath)

    rgb = nw.new_node(Nodes.RGB)
    rgb.outputs[0].default_value = (0.6866, 0.9357, 1.0, 1.0)

    transparent_bsdf = nw.new_node(Nodes.TransparentBSDF,
        input_kwargs={'Color': rgb})

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': rgb, 'Roughness': 0.0, 'IOR': 1.33, 'Transmission Weight': 1.0})

    mix_shader = nw.new_node(Nodes.MixShader,
        input_kwargs={'Fac': light_path.outputs["Is Camera Ray"], 'Shader': transparent_bsdf, 'Shader_001': principled_bsdf})

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    mapping = nw.new_node(Nodes.Mapping,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"]})

    musgrave_texture_1 = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Vector': mapping, 'Scale': 3.3, 'Detail': 13.0, 'Roughness': 0.3})

    colorramp = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': musgrave_texture_1})
    colorramp.color_ramp.interpolation = "EASE"
    colorramp.color_ramp.elements[0].position = 0.325  
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.6727
    colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    principled_bsdf_1 = nw.new_node(Nodes.PrincipledBSDF,  
        input_kwargs={'Metallic': 0.2636, 'Roughness': 0.0, 'IOR': 1.333, 'Transmission Weight': 0.8205, 'Alpha': colorramp.outputs["Color"]})

    mix_shader_1 = nw.new_node(Nodes.MixShader,
        input_kwargs={'Shader': mix_shader, 'Shader_001': principled_bsdf_1})

    volume_scatter = nw.new_node(Nodes.VolumeScatter, 
        input_kwargs={'Color': (0.5841, 0.7339, 0.8, 1.0)})

    material_output = nw.new_node(Nodes.MaterialOutput,  
        input_kwargs={'Surface': mix_shader_1, 'Volume': volume_scatter})

def geometry_geo_water(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler
    
    group_input = nw.expose_input('Geometry', dtype='NodeSocketGeometry')

    position = nw.new_node(Nodes.InputPosition)

    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={'Vector': position, 'Vector_001': (630.0, 564.0, 374.0)})

    musgrave_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Vector': add.outputs["Vector"], 'Scale': 4.1307, 'Detail': 9.7953, 'Roughness': 1.34, 'Lacunarity': 1.8087})

    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={'Vector': musgrave_texture, 'Vector_001': (0.0, 0.0, 0.0128)},
        attrs={'operation': 'MULTIPLY'})

    value = nw.new_node(Nodes.Value)  
    value.outputs[0].default_value = 1.0

    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={'Vector': multiply.outputs["Vector"], 'Vector_001': value},
        attrs={'operation': 'MULTIPLY'})

    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': group_input, 'Offset': multiply_1.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position.outputs['Geometry']})



def apply(obj, selection=None, **kwargs):
    surface.add_geomod(obj, geometry_geo_water, selection=selection, attributes=[])
    surface.add_material(obj, waterfall_shader, selection=selection)