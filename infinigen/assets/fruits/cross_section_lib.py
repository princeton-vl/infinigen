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

from infinigen.assets.fruits.fruit_utils import nodegroup_rot_semmetry

@node_utils.to_nodegroup('nodegroup_circle_cross_section', singleton=False, type='GeometryNodeTree')
def nodegroup_circle_cross_section(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'random seed', 0.0),
            ('NodeSocketFloat', 'noise scale', 0.5),
            ('NodeSocketFloat', 'noise amount', 0.1),
            ('NodeSocketInt', 'Resolution', 256),
            ('NodeSocketFloat', 'radius', 0.0)])

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.5
    
    curve_circle = nw.new_node(Nodes.CurveCircle,
        input_kwargs={'Resolution': group_input.outputs["Resolution"]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: group_input.outputs["random seed"]})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': add.outputs["Vector"], 'Scale': group_input.outputs["noise scale"]})
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture.outputs["Color"], 1: (0.5, 0.5, 0.5)},
        attrs={'operation': 'SUBTRACT'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': subtract.outputs["Vector"]})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"]})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 'Scale': group_input.outputs["noise amount"]},
        attrs={'operation': 'SCALE'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': curve_circle.outputs["Curve"], 'Offset': scale.outputs["Vector"]})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': set_position, 'Scale': group_input.outputs["radius"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': transform, 'curve parameters': value})

@node_utils.to_nodegroup('nodegroup_star_cross_section', singleton=False, type='GeometryNodeTree')
def nodegroup_star_cross_section(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'random seed', 0.0),
            ('NodeSocketFloat', 'noise scale', 2.4),
            ('NodeSocketFloat', 'noise amount', 0.2),
            ('NodeSocketInt', 'Resolution', 256),
            ('NodeSocketFloat', 'radius', 1.0)])

    curve_circle = nw.new_node(Nodes.CurveCircle,
        input_kwargs={'Resolution': group_input.outputs["Resolution"]})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    rotsemmetry = nw.new_node(nodegroup_rot_semmetry().name,
        input_kwargs={'N': 5, 'spline parameter': spline_parameter.outputs["Factor"]})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': curve_circle.outputs["Curve"], 2: rotsemmetry.outputs["Result"]})
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': rotsemmetry.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 0.4156), (0.65, 0.8125), (1.0, 1.0)])

    position = nw.new_node(Nodes.InputPosition)

    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 'Scale': float_curve},
        attrs={'operation': 'SCALE'})
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: group_input.outputs["random seed"]})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': add.outputs["Vector"], 'Scale': group_input.outputs["noise scale"]})
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture.outputs["Color"], 1: (0.5, 0.5, 0.5)},
        attrs={'operation': 'SUBTRACT'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': subtract.outputs["Vector"]})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"]})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 'Scale': group_input.outputs["noise amount"]},
        attrs={'operation': 'SCALE'})
    
    add_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 1: scale_1.outputs["Vector"]})
    
    scale_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add_1.outputs["Vector"], 'Scale': group_input.outputs["radius"]},
        attrs={'operation': 'SCALE'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Position': scale_2.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position, 'curve parameters': capture_attribute.outputs[2]})

@node_utils.to_nodegroup('nodegroup_cylax_cross_section', singleton=False, type='GeometryNodeTree')
def nodegroup_cylax_cross_section(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketInt', 'fork number', 10),
            ('NodeSocketFloat', 'bottom radius', 0.0),
            ('NodeSocketFloatDistance', 'noise random seed', 0.0),
            ('NodeSocketFloat', 'noise amount', 0.4),
            ('NodeSocketFloatDistance', 'radius', 1.0)])
    
    curve_circle = nw.new_node(Nodes.CurveCircle,
        input_kwargs={'Resolution': 256, 'Radius': group_input.outputs["radius"]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    rotsemmetry = nw.new_node(nodegroup_rot_semmetry().name,
        input_kwargs={'N': group_input.outputs["fork number"], 'spline parameter': spline_parameter.outputs["Factor"]})

    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': rotsemmetry.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 0.0), (0.65, 0.8125), (1.0, 1.0)])
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': float_curve, 3: group_input.outputs["bottom radius"]})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 'Scale': map_range_1.outputs["Result"]},
        attrs={'operation': 'SCALE'})
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: group_input.outputs["noise random seed"]})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': add.outputs["Vector"], 'Scale': 2.4})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.5
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture.outputs["Color"], 1: value},
        attrs={'operation': 'SUBTRACT'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': subtract.outputs["Vector"]})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"]})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 'Scale': group_input.outputs["noise amount"]},
        attrs={'operation': 'SCALE'})
    
    add_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 1: scale_1.outputs["Vector"]})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': curve_circle.outputs["Curve"], 'Position': add_1.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position})

@node_utils.to_nodegroup('nodegroup_coconut_cross_section', singleton=False, type='GeometryNodeTree')
def nodegroup_coconut_cross_section(nw: NodeWrangler, control_points=[(0.0, 0.7156), (0.1023, 0.7156), (1.0, 0.7594)]):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketFloat', 'random seed', 0.0),
            ('NodeSocketFloat', 'noise scale', 2.4),
            ('NodeSocketFloat', 'noise amount', 0.2),
            ('NodeSocketInt', 'Resolution', 256),
            ('NodeSocketFloat', 'radius', 1.0)])
    
    curve_circle = nw.new_node(Nodes.CurveCircle,
        input_kwargs={'Resolution': group_input.outputs["Resolution"]})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    rot_semmetry = nw.new_node(nodegroup_rot_semmetry().name,
        input_kwargs={'N': 3, 'spline parameter': spline_parameter.outputs["Factor"]})
    
    capture_attribute = nw.new_node(Nodes.CaptureAttribute,
        input_kwargs={'Geometry': curve_circle.outputs["Curve"], 2: rot_semmetry})
    
    position = nw.new_node(Nodes.InputPosition)
    
    float_curve_1 = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': rot_semmetry})
    node_utils.assign_curve(float_curve_1.mapping.curves[0], control_points)
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 'Scale': float_curve_1},
        attrs={'operation': 'SCALE'})
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: position, 1: group_input.outputs["random seed"]})
    
    noise_texture = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': add.outputs["Vector"], 'Scale': group_input.outputs["noise scale"]})
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: noise_texture.outputs["Color"], 1: (0.5, 0.5, 0.5)},
        attrs={'operation': 'SUBTRACT'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': subtract.outputs["Vector"]})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz.outputs["X"], 'Y': separate_xyz.outputs["Y"]})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 'Scale': group_input.outputs["noise amount"]},
        attrs={'operation': 'SCALE'})
    
    add_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 1: scale_1.outputs["Vector"]})
    
    scale_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add_1.outputs["Vector"], 'Scale': group_input.outputs["radius"]},
        attrs={'operation': 'SCALE'})
    
    set_position = nw.new_node(Nodes.SetPosition,
        input_kwargs={'Geometry': capture_attribute.outputs["Geometry"], 'Position': scale_2.outputs["Vector"]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': set_position, 'curve parameters': capture_attribute.outputs[2]})

