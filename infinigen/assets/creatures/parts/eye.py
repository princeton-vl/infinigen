# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy
import mathutils

import numpy as np
from numpy.random import uniform, normal as N, randint

from infinigen.core.util.math import clip_gaussian
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface
from infinigen.assets.materials.utils.surface_utils import nodegroup_norm_value, nodegroup_norm_vec

from infinigen.assets.creatures.util.nodegroups.curve import nodegroup_simple_tube, nodegroup_warped_circle_curve, nodegroup_smooth_taper, nodegroup_profile_part
from infinigen.assets.creatures.util.nodegroups.math import nodegroup_aspect_to_dim


from infinigen.assets.creatures.util.creature import PartFactory
from infinigen.assets.creatures.util import part_util
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('nodegroup_eyelid', singleton=True, type='GeometryNodeTree')
def nodegroup_eyelid(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'Eyeball Radius', 1.0),
            ('NodeSocketFloat', 'Aspect Ratio', 0.34999999999999998),
            ('NodeSocketFloat', 'fullness', 2.0),
            ('NodeSocketVector', 'TearDuctCoord', (0.0, -1.5, -0.20000000000000001)),
            ('NodeSocketVector', 'PeakCoord', (1.2, -0.20000000000000001, 2.0)),
            ('NodeSocketVector', 'EyelidEndCoord', (0.0, 1.5, 0.29999999999999999)),
            ('NodeSocketFloat', 'StartRadPct', 0.5),
            ('NodeSocketFloat', 'EndRadPct', 0.5),
            ('NodeSocketFloatAngle', 'Tilt', -0.34910000000000002)])
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["TearDuctCoord"], 'Scale': group_input.outputs["Eyeball Radius"]},
        attrs={'operation': 'SCALE'})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["PeakCoord"], 'Scale': group_input.outputs["Eyeball Radius"]},
        attrs={'operation': 'SCALE'})
    
    scale_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["EyelidEndCoord"], 'Scale': group_input.outputs["Eyeball Radius"]},
        attrs={'operation': 'SCALE'})
    
    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Start': scale.outputs["Vector"], 'Middle': scale_1.outputs["Vector"], 'End': scale_2.outputs["Vector"]})
    
    set_curve_tilt = nw.new_node(Nodes.SetCurveTilt,
        input_kwargs={'Curve': quadratic_bezier, 'Tilt': group_input.outputs["Tilt"]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    aspect_to_dim = nw.new_node(nodegroup_aspect_to_dim().name,
        input_kwargs={'Aspect Ratio': group_input.outputs["Aspect Ratio"]})
    
    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: aspect_to_dim},
        attrs={'operation': 'MULTIPLY'})
    
    warped_circle_curve = nw.new_node(nodegroup_warped_circle_curve().name,
        input_kwargs={'Position': multiply.outputs["Vector"]})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Eyeball Radius"], 1: group_input.outputs["StartRadPct"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Eyeball Radius"], 1: group_input.outputs["EndRadPct"]},
        attrs={'operation': 'MULTIPLY'})
    
    smoothtaper = nw.new_node(nodegroup_smooth_taper().name,
        input_kwargs={'start_rad': multiply_1, 'end_rad': multiply_2, 'fullness': group_input.outputs["fullness"]})
    
    profilepart = nw.new_node(nodegroup_profile_part().name,
        input_kwargs={'Skeleton Curve': set_curve_tilt, 'Profile Curve': warped_circle_curve, 'Radius Func': smoothtaper})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': tag_nodegroup(nw, profilepart, 'eyelid')})

@node_utils.to_nodegroup('nodegroup_mammal_eye', singleton=True, type='GeometryNodeTree')
def nodegroup_mammal_eye(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloatDistance', 'Radius', 0.050000000000000003),
            ('NodeSocketFloat', 'Eyelid Thickness Ratio', 0.34999999999999998),
            ('NodeSocketFloat', 'Eyelid Fullness', 2.0),
            ('NodeSocketBool', 'Eyelids', True)])
    
    eyelid = nw.new_node(nodegroup_eyelid().name,
        input_kwargs={'Eyeball Radius': group_input.outputs["Radius"], 'Aspect Ratio': group_input.outputs["Eyelid Thickness Ratio"], 'fullness': group_input.outputs["Eyelid Fullness"], 'TearDuctCoord': (0.0, -1.2, -0.20000000000000001), 'PeakCoord': (1.2, 0.40000000000000002, -1.7), 'EyelidEndCoord': (0.0, 1.2, 0.31), 'Tilt': 0.69810000000000005})
    
    eyelid_1 = nw.new_node(nodegroup_eyelid().name,
        input_kwargs={'Eyeball Radius': group_input.outputs["Radius"], 'Aspect Ratio': group_input.outputs["Eyelid Thickness Ratio"], 'fullness': group_input.outputs["Eyelid Fullness"], 'PeakCoord': (1.2, -0.20000000000000001, 1.8)})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [eyelid, eyelid_1]})
    
    switch = nw.new_node(Nodes.Switch,
        input_kwargs={1: group_input.outputs["Eyelids"], 15: join_geometry})
    
    uv_sphere = nw.new_node(Nodes.MeshUVSphere,
        input_kwargs={'Radius': group_input.outputs["Radius"]})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (0.10000000000000001, 0.0, 0.0), 'Scale': group_input.outputs["Radius"]},
        attrs={'operation': 'SCALE'})
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': uv_sphere, 'Translation': scale.outputs["Vector"], 'Rotation': (0.0, 1.5708, 0.0), 'Scale': (1.0, 1.0, 0.69999999999999996)})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (-1.7, 0.0, 0.0), 'Scale': group_input.outputs["Radius"]},
        attrs={'operation': 'SCALE'})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 6.0},
        attrs={'operation': 'MULTIPLY'})
    
    scale_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (0.33000000000000002, 0.33000000000000002, 0.33000000000000002), 'Scale': multiply},
        attrs={'operation': 'SCALE'})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 3.0},
        attrs={'operation': 'MULTIPLY'})
    
    simple_tube = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Origin': scale_1.outputs["Vector"], 'Angles Deg': (0.0, 0.0, 0.0), 'Seg Lengths': scale_2.outputs["Vector"], 'Start Radius': group_input.outputs["Radius"], 'End Radius': multiply_1, 'Fullness': 0.29999999999999999, 'Do Bezier': False, 'Aspect Ratio': 1.1000000000000001})
    
    transform_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': simple_tube.outputs["Geometry"], 'Rotation': (0.0, 0.0, 0.34910000000000002)})
    
    eyeball = nw.new_node(Nodes.SubdivisionSurface,
        input_kwargs={'Mesh': transform_1, 'Level': 2})
    
    position_2 = nw.new_node(Nodes.InputPosition)

    normvec = nw.new_node(nodegroup_norm_vec().name, input_kwargs={'Geometry': eyeball, 'Name': 'EyeballPosition', 'Vector': position_2})

    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': None, 'BodyExtra_Lid': switch.outputs[6], 'Eyeballl': normvec, 'ParentCutter': transform_2})

class MammalEye(PartFactory):

    tags = ['head_detail', 'eye_socket']

    def sample_params(self):
        return {
            'Radius': 0.03 * N(1, 0.1),
            'Eyelid Thickness Ratio': 0.35 * N(1, 0.05),
            'Eyelid Fullness': 2.0 * N(1, 0.1)
        }

    def make_part(self, params):
        part = part_util.nodegroup_to_part(nodegroup_mammal_eye, params)
        tag_object(part.obj, 'mammal_eye')
        return part
