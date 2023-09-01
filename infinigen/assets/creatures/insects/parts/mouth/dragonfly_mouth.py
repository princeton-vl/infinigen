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

from infinigen.assets.creatures.insects.utils.geom_utils import nodegroup_simple_tube_v2

@node_utils.to_nodegroup('nodegroup_dragonfly_mouth', singleton=False, type='GeometryNodeTree')
def nodegroup_dragonfly_mouth(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 1.5
    
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': (9.5, 9.36, 5.54), 'proportions': (1.0, 1.0, 1.0), 'aspect': value, 'do_bezier': False, 'fullness': 7.9})
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': simple_tube_v2.outputs["Geometry"], 'Translation': (0.0, 0.0, -9.1), 'Rotation': (0.0, 1.7645, 0.0), 'Scale': (1.0, 1.2, 1.0)})
    
    simple_tube_v2_1 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': (9.64, 5.46, 9.04), 'proportions': (1.0, 1.0, 1.0), 'aspect': value, 'do_bezier': False, 'fullness': 7.9})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': simple_tube_v2_1.outputs["Geometry"], 'Rotation': (0.0, 1.5708, 0.0), 'Scale': (1.0, 1.2, 1.0)})
    
    simple_tube_v2_2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': (8.4, 6.16, 4.7), 'proportions': (1.0, 1.0, 1.0), 'aspect': value, 'do_bezier': False, 'fullness': 7.9})
    
    transform_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': simple_tube_v2_2.outputs["Geometry"], 'Translation': (-1.1, 0.0, -17.2), 'Rotation': (0.0, 2.6005, 0.0), 'Scale': (1.0, 1.2, 1.0)})
    
    simple_tube_v2_3 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': (10.1, 4.28, 6.7), 'angles_deg': (4.64, 0.0, 0.0), 'proportions': (1.0, 1.0, 1.0), 'aspect': 2.1, 'do_bezier': False, 'fullness': 7.9})
    
    transform_4 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': simple_tube_v2_3.outputs["Geometry"], 'Translation': (-6.56, 0.0, 5.34), 'Rotation': (0.0, 0.8126, 0.0), 'Scale': (1.0, 1.2, 1.0)})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [transform_1, transform, transform_2, transform_4]})
    
    transform_3 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': join_geometry})
    
    normal = nw.new_node(Nodes.InputNormal)
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Scale': 0.5})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': noise_texture.outputs["Fac"], 4: 0.3})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: normal, 'Scale': map_range.outputs["Result"]},
        attrs={'operation': 'SCALE'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': transform_3, 'Offset': scale.outputs["Vector"]})
    
    subdivision_surface = nw.new_node(Nodes.SubdivisionSurface,
        input_kwargs={'Mesh': set_position, 'Level': 2})
    
    group_output_1 = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': subdivision_surface})