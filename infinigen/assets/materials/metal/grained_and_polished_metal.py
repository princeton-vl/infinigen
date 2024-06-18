# Authors: Yiming Zuo

import bpy
import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

@node_utils.to_nodegroup('nodegroup_grained_metal', singleton=False, type='ShaderNodeTree')
def nodegroup_grained_metal(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketColor', 'Base Color', (0.8000, 0.8000, 0.8000, 1.0000)),
            ('NodeSocketFloat', 'Scale', 5.0000),
            ('NodeSocketFloat', 'Seed', 0.0000),
            ('NodeSocketFloat', 'Roughness', 0.0000)])
    map_range = nw.new_node(Nodes.MapRange, input_kwargs={'Value': group_input.outputs["Roughness"], 3: 0.0500, 4: 0.2500})
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': group_input.outputs["Base Color"], 'Metallic': 1.0000, 'Specular': 0.0000, 'Roughness': map_range.outputs["Result"]})
    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Scale"], 1: 2000.0000},
        attrs={'operation': 'MULTIPLY'})
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"], 'W': group_input.outputs["Seed"], 'Scale': multiply, 'Detail': 15.0000, 'Distortion': 2.0000},
        attrs={'noise_dimensions': '4D'})
    multiply_1 = nw.new_node(Nodes.Math, input_kwargs={0: map_range.outputs["Result"], 1: 0.4000}, attrs={'operation': 'MULTIPLY'})
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Fac"], 1: multiply_1},
        attrs={'operation': 'MULTIPLY'})
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'BSDF': principled_bsdf, 'Displacement': multiply_3},
        attrs={'is_active_output': True})


def shader_grained_metal(nw: NodeWrangler, scale=1.0, base_color=None, roughness=None, seed=None, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler
    if roughness is None:
        roughness = uniform(0.0, 1.0)
    if seed is None:
        seed = uniform(-1000.0, 1000.0)
    if base_color is None:


        input_kwargs={'Base Color': base_color,
                      'Scale': scale,
                      'Seed': seed,
                      'Roughness': roughness,
                      })
    displacement = nw.new_node('ShaderNodeDisplacement', input_kwargs={'Height': group.outputs["Displacement"], 'Midlevel': 0.0000})
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': group.outputs["BSDF"], 'Displacement': displacement},
        attrs={'is_active_output': True})

def apply(obj, selection=None, **kwargs):
