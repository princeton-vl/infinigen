# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import bpy
import mathutils
from numpy.random import uniform, normal, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

from infinigen.assets.fruits.fruit_utils import nodegroup_point_on_mesh, nodegroup_random_rotation_scale, nodegroup_hair, nodegroup_instance_on_points

def shader_hair_shader(nw: NodeWrangler, basic_color):
    # Code generated using version 2.4.3 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': texture_coordinate.outputs["Object"], 'Scale': 0.5, 'Detail': 10.0, 'Roughness': 0.7})
    
    separate_rgb = nw.new_node(Nodes.SeparateColor,
        input_kwargs={'Color': noise_texture.outputs["Color"]})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["Green"], 1: 0.4, 2: 0.7, 3: 0.48, 4: 0.55},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_rgb.outputs["Blue"], 1: 0.4, 2: 0.7, 3: 0.4},
        attrs={'interpolation_type': 'SMOOTHSTEP'})
    
    hue_saturation_value = nw.new_node('ShaderNodeHueSaturation',
        input_kwargs={'Hue': map_range_1.outputs["Result"], 'Value': map_range_2.outputs["Result"], 'Color': basic_color})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': hue_saturation_value})
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': principled_bsdf})

@node_utils.to_nodegroup('nodegroup_coconuthairy_surface', singleton=False, type='GeometryNodeTree')
def nodegroup_coconuthairy_surface(nw: NodeWrangler, basic_color=(0.9473, 0.552, 0.2623, 1.0)):
    # Code generated using version 2.4.3 of the node_transpiler
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketGeometry', 'Geometry', None),
            ('NodeSocketFloat', 'spline parameter', 0.0)])

    material = nw.new_node('GeometryNodeInputMaterial')
    material.material = surface.shaderfunc_to_material(shader_hair_shader, basic_color)

    set_material = nw.new_node(Nodes.SetMaterial,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Material': material})
    
    pointonmesh = nw.new_node(nodegroup_point_on_mesh().name,
        input_kwargs={'Mesh': group_input.outputs["Geometry"], 'spline parameter': group_input.outputs["spline parameter"], 'Distance Min': 0.03, 'noise amount': 0.0, 'noise scale': 0.0})
    
    randomrotationscale = nw.new_node(nodegroup_random_rotation_scale().name,
        input_kwargs={'noise scale': 100.0, 'rot mean': (0.47, 0.0, 4.8), ' rot std z': 100.0, 'scale mean': 0.2, 'scale std': 0.0})
    
    hair = nw.new_node(nodegroup_hair().name,
        input_kwargs={'length resolution': 1, 'cross section resolution': 1, 'scale': 0.3, 'Radius': 0.03, 'Material': material, 'Middle': (0.0, 0.3, 1.0), 'End': (0.0, -1.4, 2.0)})
    
    instanceonpoints = nw.new_node(nodegroup_instance_on_points().name,
        input_kwargs={'rotation base': pointonmesh.outputs["Rotation"], 'rotation delta': randomrotationscale.outputs["Vector"], 'translation': (0.0, 0.0, 0.0), 'scale': randomrotationscale.outputs["Value"], 'Points': pointonmesh.outputs["Geometry"], 'Instance': hair})
    
    pointonmesh_1 = nw.new_node(nodegroup_point_on_mesh().name,
        input_kwargs={'Mesh': group_input.outputs["Geometry"], 'spline parameter': group_input.outputs["spline parameter"], 'Distance Min': 0.06, 'parameter min': 0.2, 'noise amount': 0.5, 'noise scale': 2.0})
    
    randomrotationscale_1 = nw.new_node(nodegroup_random_rotation_scale().name,
        input_kwargs={'rot mean': (1.3, 0.0, 0.0), ' rot std z': 3.0, 'scale mean': 0.3, 'scale std': 0.5})
    
    hair_1 = nw.new_node(nodegroup_hair().name,
        input_kwargs={'scale': 1.0, 'Material': material, 'Middle': (0.0, 0.5, 1.0), 'End': (0.0, -1.9, 2.0)})
    
    instanceonpoints_1 = nw.new_node(nodegroup_instance_on_points().name,
        input_kwargs={'rotation base': pointonmesh_1.outputs["Rotation"], 'rotation delta': randomrotationscale_1.outputs["Vector"], 'translation': (0.0, 0.0, 0.0), 'scale': randomrotationscale_1.outputs["Value"], 'Points': pointonmesh_1.outputs["Geometry"], 'Instance': hair_1})
    
    join_geometry_2 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [set_material, instanceonpoints, instanceonpoints_1]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry_2})