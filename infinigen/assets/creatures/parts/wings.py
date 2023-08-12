# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: 
# - Alexander Raistrick: base version
# - Beining Han: flying variant


import bpy

import numpy as np
from numpy.random import uniform as U, normal as N

from infinigen.core.util.math import clip_gaussian

from infinigen.assets.creatures.util.genome import Joint, IKParams

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.assets.creatures.util.nodegroups.curve import nodegroup_simple_tube, nodegroup_simple_tube_v2
from infinigen.assets.creatures.util.nodegroups.attach import nodegroup_surface_muscle
from infinigen.assets.creatures.util.nodegroups.math import nodegroup_deg2_rad
from infinigen.assets.creatures.util.nodegroups.geometry import nodegroup_symmetric_clone

from infinigen.assets.creatures.util.creature import PartFactory
from infinigen.assets.creatures.util.part_util import nodegroup_to_part
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('nodegroup_feather', singleton=False, type='GeometryNodeTree')
def nodegroup_feather(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Length Rad1 Rad2', (0.5, 0.1, 0.1))])
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["Length Rad1 Rad2"]})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (1.0, 0.0, 0.0), 'Scale': separate_xyz.outputs["X"]},
        attrs={'operation': 'SCALE'})
    
    curve_line = nw.new_node(Nodes.CurveLine,
        input_kwargs={'End': scale.outputs["Vector"]})
    
    subdivide_curve = nw.new_node(Nodes.SubdivideCurve,
        input_kwargs={'Curve': curve_line, 'Cuts': 30})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': spline_parameter.outputs["Factor"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 0.0), (0.2327, 0.985), (0.8909, 0.6), (1.0, 0.0)])
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': spline_parameter.outputs["Factor"], 3: separate_xyz.outputs["Y"], 4: separate_xyz.outputs["Z"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: float_curve, 1: map_range.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': subdivide_curve, 'Radius': multiply})
    
    curve_line_1 = nw.new_node(Nodes.CurveLine,
        input_kwargs={'Start': (0.0, -1.0, 0.0), 'End': (0.0, 1.0, 0.0)})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_radius, 'Profile Curve': curve_line_1})
    
    subdivide_curve_1 = nw.new_node(Nodes.SubdivideCurve,
        input_kwargs={'Curve': curve_line, 'Cuts': 4})
    
    trim_curve = nw.new_node(Nodes.TrimCurve,
        input_kwargs={'Curve': subdivide_curve_1, 'End': 0.8742})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': spline_parameter.outputs["Factor"], 3: 0.15, 4: 0.05})
    
    multiply_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_1.outputs["Result"], 1: separate_xyz.outputs["Y"]},
        attrs={'operation': 'MULTIPLY'})
    
    set_curve_radius_1 = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': trim_curve, 'Radius': multiply_1})
    
    curve_circle = nw.new_node(Nodes.CurveCircle,
        input_kwargs={'Resolution': 6})
    
    curve_to_mesh_1 = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_radius_1, 'Profile Curve': curve_circle.outputs["Curve"]})
    
    #join_geometry = nw.new_node(Nodes.JoinGeometry,
    #    input_kwargs={'Geometry': [curve_to_mesh, curve_to_mesh_1]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Mesh': tag_nodegroup(nw, curve_to_mesh, 'feather')})

@node_utils.to_nodegroup('nodegroup_bird_tail', singleton=False, type='GeometryNodeTree')
def nodegroup_bird_tail(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    simple_tube = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Angles Deg': (0.0, 0.0, 0.0), 'Seg Lengths': (0.11, 0.11, 0.11), 'Start Radius': 0.07, 'End Radius': 0.02, 'Fullness': 3.0})
    
    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'Feather Length Rad1 Rad2', (0.5, 0.08, 0.1)),
            ('NodeSocketVector', 'Feather Rot Extent', (136.51, -11.8, 34.0)),
            ('NodeSocketVector', 'Feather Rot Rand Bounds', (5.0, 5.0, 5.0)),
            ('NodeSocketIntUnsigned', 'N Feathers', 16)])
    
    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': group_input.outputs["N Feathers"], 'Start': (0.0, 0.0, -0.1), 'Middle': (0.0, 0.15, -0.05), 'End': (0.0, 0.15, 0.11)})
    
    feather = nw.new_node(nodegroup_feather().name,
        input_kwargs={'Length Rad1 Rad2': group_input.outputs["Feather Length Rad1 Rad2"]})
    
    index = nw.new_node(Nodes.Index)
    
    divide = nw.new_node(Nodes.Math,
        input_kwargs={0: index, 1: group_input.outputs["N Feathers"]},
        attrs={'operation': 'DIVIDE'})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Vector': divide, 9: (-90.0, -14.88, 4.01), 10: group_input.outputs["Feather Rot Extent"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Feather Rot Rand Bounds"], 'Scale': -1.0},
        attrs={'operation': 'SCALE'})
    
    random_value = nw.new_node(Nodes.RandomValue,
        input_kwargs={0: scale.outputs["Vector"], 1: group_input.outputs["Feather Rot Rand Bounds"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: map_range.outputs["Vector"], 1: random_value.outputs["Value"]})
    
    deg2rad = nw.new_node(nodegroup_deg2_rad().name,
        input_kwargs={'Deg': add.outputs["Vector"]})
    
    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': quadratic_bezier, 'Instance': feather, 'Rotation': deg2rad})
    
    realize_instances = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': instance_on_points})
    
    symmetric_clone = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': realize_instances})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': simple_tube.outputs["Geometry"], 'Skeleton Curve': simple_tube.outputs["Skeleton Curve"], 'TailFeathers': symmetric_clone.outputs["Both"]})

@node_utils.to_nodegroup('nodegroup_bird_wing', singleton=False, type='GeometryNodeTree')
def nodegroup_bird_wing(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (1.0, 0.26, 0.0)),
            ('NodeSocketFloat', 'feather_density', 18.7),
            ('NodeSocketFloat', 'aspect', 1.0),
            ('NodeSocketFloat', 'fullness', 4.0),
            ('NodeSocketFloatFactor', 'Wing Shape Sculpting', 1.0),
            ('NodeSocketVector', 'Feather length_rad1_rad2', (0.6, 0.04, 0.04)),
            ('NodeSocketFloat', 'Extension', 1.68)])
    
    map_range_3 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Vector': group_input.outputs["Extension"], 9: (-83.46, 154.85, -155.38), 10: (-15.04, 60.5, -41.1)},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"], 'angles_deg': map_range_3.outputs["Vector"], 'proportions': (0.2, 0.27, 0.3), 'aspect': group_input.outputs["aspect"], 'do_bezier': False, 'fullness': group_input.outputs["fullness"]})
    
    curve_length = nw.new_node(Nodes.CurveLength,
        input_kwargs={'Curve': simple_tube_v2.outputs["Skeleton Curve"]})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: curve_length, 1: group_input.outputs["feather_density"]},
        attrs={'operation': 'MULTIPLY'})
    
    resample_curve = nw.new_node(Nodes.ResampleCurve,
        input_kwargs={'Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Count': multiply})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': resample_curve})
    
    reroute = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': curve_to_mesh})
    
    feather = nw.new_node(nodegroup_feather().name,
        input_kwargs={'Length Rad1 Rad2': group_input.outputs["Feather length_rad1_rad2"]})
    
    index = nw.new_node(Nodes.Index)
    
    attribute_statistic = nw.new_node(Nodes.AttributeStatistic,
        input_kwargs={'Geometry': curve_to_mesh, 2: index})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': index, 1: attribute_statistic.outputs["Min"], 2: attribute_statistic.outputs["Max"]})
    
    transfer_attribute_index = nw.new_node(Nodes.SampleNearest,
        input_kwargs={'Geometry': curve_to_mesh, 'Sample Position': map_range_1.outputs["Result"]})

    transfer_attribute = nw.new_node(Nodes.SampleIndex,
        input_kwargs={'Geometry': curve_to_mesh, 'Index': transfer_attribute_index})

    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Factor': group_input.outputs["Wing Shape Sculpting"], 'Value': (transfer_attribute, 'Value')})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 0.0), (0.5164, 0.245), (0.7564, 0.625), (1.0, 1.0)])
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': group_input.outputs["Extension"], 3: 115.65, 4: 0.0})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'Y': map_range_2.outputs["Result"]})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Vector': float_curve, 9: (0.0, 80.0, 0.0), 10: combine_xyz},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: map_range.outputs["Vector"], 1: (-5.0, 0.0, -1.0)})
    
    deg2rad = nw.new_node(nodegroup_deg2_rad().name,
        input_kwargs={'Deg': add.outputs["Vector"]})
    
    vector_curves = nw.new_node(Nodes.VectorCurve,
        input_kwargs={'Fac': group_input.outputs["Wing Shape Sculpting"], 'Vector': transfer_attribute})
    node_utils.assign_curve(vector_curves.mapping.curves[0], [(-1.0, -0.0), (0.0036, 0.0), (0.0473, 0.6), (0.3527, 0.54), (0.6, 0.9), (0.8836, 0.92), (1.0, 0.58)], handles=['AUTO', 'VECTOR', 'AUTO', 'AUTO', 'VECTOR', 'AUTO', 'AUTO'])
    node_utils.assign_curve(vector_curves.mapping.curves[1], [(-1.0, 1.0), (1.0, 1.0)])
    node_utils.assign_curve(vector_curves.mapping.curves[2], [(-1.0, 1.0), (1.0, 1.0)])
    
    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': reroute, 'Instance': feather, 'Rotation': deg2rad, 'Scale': vector_curves})
    
    add_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: map_range.outputs["Vector"], 1: (-5.0, 0.0, 0.0)})
    
    deg2rad_1 = nw.new_node(nodegroup_deg2_rad().name,
        input_kwargs={'Deg': add_1.outputs["Vector"]})
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vector_curves, 1: (0.75, 1.0, 1.0)},
        attrs={'operation': 'MULTIPLY'})
    
    instance_on_points_1 = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': reroute, 'Instance': feather, 'Rotation': deg2rad_1, 'Scale': multiply_1.outputs["Vector"]})
    
    add_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: map_range.outputs["Vector"], 1: (-10.3, 0.0, 1.0)})
    
    deg2rad_2 = nw.new_node(nodegroup_deg2_rad().name,
        input_kwargs={'Deg': add_2.outputs["Vector"]})
    
    multiply_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vector_curves, 1: (0.45, 1.0, 1.0)},
        attrs={'operation': 'MULTIPLY'})
    
    instance_on_points_2 = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': reroute, 'Instance': feather, 'Rotation': deg2rad_2, 'Scale': multiply_2.outputs["Vector"]})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [instance_on_points, instance_on_points_1, instance_on_points_2]})
    
    realize_instances = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': join_geometry_1})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Feathers': realize_instances})

class BirdTail(PartFactory):

    tags = ['tail', 'wing']

    def sample_params(self):
        return {
            'Feather Length Rad1 Rad2': np.array((0.4, 0.06, 0.04)) * N(1, 0.1) * N(1, 0.1, 3),
            'Feather Rot Extent': np.array((25, -10, -16)) * N(1, 0.1, 3),
            'Feather Rot Rand Bounds': np.array((5.0, 5.0, 5.0)) * N(1, 0.1) * N(1, 0.05, 3),
            'N Feathers': int(N(16, 3))
        }

    def make_part(self, params):
        part = nodegroup_to_part(nodegroup_bird_tail, params)
        return part

class BirdWing(PartFactory):

    tags = ['limb', 'wing']

    def sample_params(self):
        return {
            'length_rad1_rad2': np.array((clip_gaussian(1.2, 0.7, 0.4, 2), 0.1, 0.02)),
            'feather_density': 30,
            'aspect': N(0.4, 0.05),
            'fullness': N(4, 0.1),
            'Wing Shape Sculpting': U(0.6, 1),
            'Feather length_rad1_rad2': np.array((0.7 * N(1, 0.2), 0.04, 0.04)),
            'Extension': U(0, 0.05) if U() < 0.8 else U(0.7, 1)
        }

    def make_part(self, params):
        # split extras is essential to make automatic rigging work. We will join them back together later
        part = nodegroup_to_part(nodegroup_bird_wing, params, split_extras=True)
        part.joints = {
            0: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])), # shoulder
            0.27: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])),
            0.65: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])) # elbow
        } 
        part.iks = {1.0: IKParams(name='wingtip', chain_parts=1)}
        tag_object(part.obj, 'bird_wing')
        part.settings['parent_extras_rigid'] = True
        return part


@node_utils.to_nodegroup('nodegroup_flying_feather', singleton=False, type='GeometryNodeTree')
def nodegroup_flying_feather(nw: NodeWrangler):
    # Code generated using version 2.5.1 of the node_transpiler

    vector = nw.new_node(Nodes.Vector)
    vector.vector = (0.5000, 0.0500, 0.0000)

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketVector', 'Length Rad1 Rad2', (0.5000, 0.1000, 0.1000))])

    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
                               input_kwargs={'Vector': group_input.outputs["Length Rad1 Rad2"]})

    scale = nw.new_node(Nodes.VectorMath,
                        input_kwargs={0: vector, 'Scale': separate_xyz.outputs["X"]},
                        attrs={'operation': 'SCALE'})

    scale_1 = nw.new_node(Nodes.VectorMath,
                          input_kwargs={0: (1.0000, 0.0000, 0.0000), 'Scale': separate_xyz.outputs["X"]},
                          attrs={'operation': 'SCALE'})

    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
                                   input_kwargs={'Resolution': 32, 'Start': (0.0000, 0.0000, 0.0000),
                                                 'Middle': scale.outputs["Vector"], 'End': scale_1.outputs["Vector"]})

    set_position = nw.new_node(Nodes.SetPosition,
                               input_kwargs={'Geometry': quadratic_bezier})

    subdivide_curve_1 = nw.new_node(Nodes.SubdivideCurve,
                                    input_kwargs={'Curve': set_position, 'Cuts': 4})

    trim_curve = nw.new_node(Nodes.TrimCurve,
                             input_kwargs={'Curve': subdivide_curve_1, 'End': 0.8742})

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    map_range_1 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': spline_parameter.outputs["Factor"], 3: 0.1500, 4: 0.0100})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: map_range_1.outputs["Result"], 1: separate_xyz.outputs["Y"]},
                           attrs={'operation': 'MULTIPLY'})

    set_curve_radius_1 = nw.new_node(Nodes.SetCurveRadius,
                                     input_kwargs={'Curve': trim_curve, 'Radius': multiply})

    curve_circle = nw.new_node(Nodes.CurveCircle,
                               input_kwargs={'Resolution': 6})

    curve_to_mesh_1 = nw.new_node(Nodes.CurveToMesh,
                                  input_kwargs={'Curve': set_curve_radius_1,
                                                'Profile Curve': curve_circle.outputs["Curve"]})

    subdivide_curve = nw.new_node(Nodes.SubdivideCurve,
                                  input_kwargs={'Curve': set_position, 'Cuts': 30})

    float_curve = nw.new_node(Nodes.FloatCurve,
                              input_kwargs={'Value': spline_parameter.outputs["Factor"]})
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0000, 0.0000), (0.3373, 0.8188), (0.7182, 0.7375), (1.0000, 0.0000)])

    white_noise_texture = nw.new_node(Nodes.WhiteNoiseTexture)

    multiply_1 = nw.new_node(Nodes.Math,
                             input_kwargs={0: white_noise_texture.outputs["Value"], 1: 0.1000},
                             attrs={'operation': 'MULTIPLY'})

    add = nw.new_node(Nodes.Math,
                      input_kwargs={0: float_curve, 1: multiply_1})

    map_range = nw.new_node(Nodes.MapRange,
                            input_kwargs={'Value': spline_parameter.outputs["Factor"], 3: separate_xyz.outputs["Y"],
                                          4: separate_xyz.outputs["Z"]})

    multiply_2 = nw.new_node(Nodes.Math,
                             input_kwargs={0: add, 1: map_range.outputs["Result"]},
                             attrs={'operation': 'MULTIPLY'})

    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
                                   input_kwargs={'Curve': subdivide_curve, 'Radius': multiply_2})

    curve_line_1 = nw.new_node(Nodes.CurveLine,
                               input_kwargs={'Start': (0.0000, -1.0000, 0.1000), 'End': (0.0000, 1.0000, 0.0000)})

    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
                                input_kwargs={'Curve': set_curve_radius, 'Profile Curve': curve_line_1,
                                              'Fill Caps': True})

    join_geometry = nw.new_node(Nodes.JoinGeometry,
                                input_kwargs={'Geometry': [curve_to_mesh_1, curve_to_mesh]})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Mesh': join_geometry})

@node_utils.to_nodegroup('nodegroup_flying_bird_tail', singleton=False, type='GeometryNodeTree')
def nodegroup_flying_bird_tail(nw: NodeWrangler):
    # Code generated using version 2.5.1 of the node_transpiler

    simple_tube = nw.new_node(nodegroup_simple_tube().name,
                              input_kwargs={'Angles Deg': (0.0000, 0.0000, 0.0000),
                                            'Seg Lengths': (0.00, 0.00, 0.00), 'Start Radius': 0.000,
                                            'End Radius': 0.000, 'Fullness': 3.0000})

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketVector', 'Feather Length Rad1 Rad2', (0.5000, 0.0800, 0.1000)),
                                            ('NodeSocketVector', 'Feather Rot Extent', (136.5100, -11.8000, 34.0000)),
                                            ('NodeSocketVector', 'Feather Rot Rand Bounds', (5.0000, 5.0000, 5.0000)),
                                            ('NodeSocketIntUnsigned', 'N Feathers', 16)])

    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
                                   input_kwargs={'Resolution': group_input.outputs["N Feathers"],
                                                 'Start': (0.0000, 0.0000, 0.0000), 'Middle': (0.0000, 0.0500, 0.0000),
                                                 'End': (-0.0500, 0.1000, 0.0300)})

    feather = nw.new_node(nodegroup_flying_feather().name,
                          input_kwargs={'Length Rad1 Rad2': group_input.outputs["Feather Length Rad1 Rad2"]})

    curve_tangent = nw.new_node(Nodes.CurveTangent)

    align_euler_to_vector = nw.new_node(Nodes.AlignEulerToVector,
                                        input_kwargs={'Vector': curve_tangent},
                                        attrs={'axis': 'Y'})

    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
                                     input_kwargs={'Points': quadratic_bezier, 'Instance': feather,
                                                   'Rotation': align_euler_to_vector})

    rotate_instances = nw.new_node(Nodes.RotateInstances,
                                   input_kwargs={'Instances': instance_on_points, 'Rotation': (1.5708, 0.0000, 0.0000)})

    random_value_1 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.1000, 3: 0.1000})

    random_value_2 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.1000, 3: 0.1000, 'Seed': 1})

    random_value_3 = nw.new_node(Nodes.RandomValue,
                                 input_kwargs={2: -0.1000, 3: 0.1000})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'X': random_value_1.outputs[1], 'Y': random_value_2.outputs[1],
                                            'Z': random_value_3.outputs[1]})

    rotate_instances_1 = nw.new_node(Nodes.RotateInstances,
                                     input_kwargs={'Instances': rotate_instances, 'Rotation': combine_xyz})

    index_1 = nw.new_node(Nodes.Index)

    map_range_1 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': index_1, 2: group_input.outputs["N Feathers"]})

    float_curve = nw.new_node(Nodes.FloatCurve,
                              input_kwargs={'Value': map_range_1.outputs["Result"]})

    if U(0, 1) < 0.5:
        control_points = [0.2, 0.3, 0.45, 0.9]
    else:
        control_points = [0.25, 0.3, 0.35, 0.4]
    node_utils.assign_curve(float_curve.mapping.curves[0],
                            [(0.0136, control_points[0] + N(0., 0.02)), (0.3273, control_points[1] + N(0., 0.02)),
                             (0.7500, control_points[2] + N(0., 0.03)), (1.0000, control_points[3] + N(0., 0.04))])

    multiply_add = nw.new_node(Nodes.Math,
                               input_kwargs={0: float_curve, 1: 1.2000},
                               attrs={'operation': 'MULTIPLY_ADD'})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
                                input_kwargs={'X': multiply_add, 'Y': 1.0000, 'Z': 1.0000})

    scale_instances = nw.new_node(Nodes.ScaleInstances,
                                  input_kwargs={'Instances': rotate_instances_1, 'Scale': combine_xyz_1})

    realize_instances = nw.new_node(Nodes.RealizeInstances,
                                    input_kwargs={'Geometry': scale_instances})

    symmetric_clone = nw.new_node(nodegroup_symmetric_clone().name,
                                  input_kwargs={'Geometry': realize_instances})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': simple_tube.outputs["Geometry"],
                                             'Skeleton Curve': simple_tube.outputs["Skeleton Curve"],
                                             'Feathers': symmetric_clone.outputs["Both"]})



@node_utils.to_nodegroup('nodegroup_flying_bird_wing', singleton=False, type='GeometryNodeTree')
def nodegroup_flying_bird_wing(nw: NodeWrangler):
    # Code generated using version 2.5.1 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
                              expose_input=[('NodeSocketVector', 'length_rad1_rad2', (1.0000, 0.2600, 0.0000)),
                                            ('NodeSocketFloat', 'feather_density', 18.7000),
                                            ('NodeSocketFloat', 'aspect', 1.0000),
                                            ('NodeSocketFloat', 'fullness', 4.0000),
                                            ('NodeSocketFloatFactor', 'Wing Shape Sculpting', 1.0000),
                                            ('NodeSocketVector', 'Length Rad1 Rad2', (0.6000, 0.0400, 0.0400)),
                                            ('NodeSocketFloat', 'Extension', 1.6800)])

    map_range_3 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Vector': group_input.outputs["Extension"],
                                            9: (-76.2600, 170.9500, -144.3800), 10: (10.0000, -10.0000, 0.0000)},
                              attrs={'data_type': 'FLOAT_VECTOR'})

    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
                                 input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"],
                                               'angles_deg': map_range_3.outputs["Vector"],
                                               'proportions': (0.2000, 0.2700, 0.5000),
                                               'aspect': group_input.outputs["aspect"], 'do_bezier': False,
                                               'fullness': group_input.outputs["fullness"]})

    curve_length = nw.new_node(Nodes.CurveLength,
                               input_kwargs={'Curve': simple_tube_v2.outputs["Skeleton Curve"]})

    multiply = nw.new_node(Nodes.Math,
                           input_kwargs={0: curve_length, 1: group_input.outputs["feather_density"]},
                           attrs={'operation': 'MULTIPLY'})

    resample_curve = nw.new_node(Nodes.ResampleCurve,
                                 input_kwargs={'Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Count': multiply})

    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
                                input_kwargs={'Curve': resample_curve})

    reroute = nw.new_node(Nodes.Reroute,
                          input_kwargs={'Input': curve_to_mesh})

    feather = nw.new_node(nodegroup_flying_feather().name,
                          input_kwargs={'Length Rad1 Rad2': group_input.outputs["Length Rad1 Rad2"]})

    index = nw.new_node(Nodes.Index)

    attribute_statistic = nw.new_node(Nodes.AttributeStatistic,
                                      input_kwargs={'Geometry': curve_to_mesh, 2: index})

    map_range_1 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': index, 1: attribute_statistic.outputs["Min"],
                                            2: attribute_statistic.outputs["Max"]})
      
    transfer_attribute_index = nw.new_node(Nodes.SampleNearest,
        input_kwargs={'Geometry': curve_to_mesh, 'Sample Position': map_range_1.outputs["Result"]})

    transfer_attribute = nw.new_node(Nodes.SampleIndex,
        input_kwargs={'Geometry': curve_to_mesh, 'Index': transfer_attribute_index})

    map_range_2 = nw.new_node(Nodes.MapRange,
                              input_kwargs={'Value': group_input.outputs["Extension"], 3: 115.6500, 4: 0.0000})

    combine_xyz = nw.new_node(Nodes.CombineXYZ,
                              input_kwargs={'Y': map_range_2.outputs["Result"]})

    wing_feathers = []

    for i in range(3):
        float_curve = nw.new_node(Nodes.FloatCurve,
                                  input_kwargs={'Factor': group_input.outputs["Wing Shape Sculpting"],
                                                'Value': (transfer_attribute, 'Value')})
        node_utils.assign_curve(float_curve.mapping.curves[0],
                                [(0.0000, 0.0000), (0.25, 0.2), (0.50, 0.4),
                                 (0.75, 0.6), (1.0000, 0.8 - i * 0.02 + N(0., 0.02))])

        map_range = nw.new_node(Nodes.MapRange,
                                input_kwargs={'Vector': float_curve, 9: (0.0000, 80.0000, 0.0000), 10: combine_xyz},
                                attrs={'data_type': 'FLOAT_VECTOR'})

        add = nw.new_node(Nodes.VectorMath,
                          input_kwargs={0: map_range.outputs["Vector"], 1: (0., -5 + 5 * i, (i - 1) * 8.)})

        deg2rad = nw.new_node(nodegroup_deg2_rad().name,
                              input_kwargs={'Deg': add.outputs["Vector"]})

        vector_curves = nw.new_node(Nodes.VectorCurve,
                                    input_kwargs={'Fac': group_input.outputs["Wing Shape Sculpting"],
                                                  'Vector': (transfer_attribute, 'Value')})
        node_utils.assign_curve(vector_curves.mapping.curves[0],
                                [(-1.0000, -0.0000), (0.0218, 0.4), (0.20, 0.45),
                                 (0.5, 0.5), (0.65000, 0.6), (0.80, 0.7), (1.0000, 0.78 + N(0., 0.02))],
                                handles=['AUTO', 'VECTOR', 'AUTO', 'AUTO', 'VECTOR', 'AUTO', 'AUTO'])
        node_utils.assign_curve(vector_curves.mapping.curves[1], [(-1.0000, 1.0000), (1.0000, 1.0000)])
        node_utils.assign_curve(vector_curves.mapping.curves[2], [(-1.0000, 1.0000), (1.0000, 1.0000)])

        scale = nw.new_node(Nodes.VectorMath,
                            input_kwargs={0: vector_curves, 'Scale': U(1.6, 2.0) - i * 0.65},
                            attrs={'operation': 'SCALE'})

        instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
                                         input_kwargs={'Points': reroute, 'Instance': feather, 'Rotation': deg2rad,
                                                       'Scale': scale.outputs["Vector"]})

        random_value_1 = nw.new_node(Nodes.RandomValue,
                                     input_kwargs={2: -0.01, 3: 0.01})

        random_value_2 = nw.new_node(Nodes.RandomValue,
                                     input_kwargs={2: -0.03, 3: 0.03, 'Seed': 1})

        random_value_3 = nw.new_node(Nodes.RandomValue,
                                     input_kwargs={2: -0.01, 3: 0.01, 'Seed': 2})

        combine_xyz = nw.new_node(Nodes.CombineXYZ,
                                  input_kwargs={'X': random_value_1.outputs[1], 'Y': random_value_2.outputs[1],
                                                'Z': random_value_3.outputs[1]})

        rotate_instances_1 = nw.new_node(Nodes.RotateInstances,
                                         input_kwargs={'Instances': instance_on_points, 'Rotation': combine_xyz})
        wing_feathers.append(rotate_instances_1)

    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
                                  input_kwargs={'Geometry': wing_feathers})

    realize_instances = nw.new_node(Nodes.RealizeInstances,
                                    input_kwargs={'Geometry': join_geometry_1})

    group_output = nw.new_node(Nodes.GroupOutput,
                               input_kwargs={'Geometry': simple_tube_v2.outputs["Geometry"],
                                             'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"],
                                             'Feathers': realize_instances})


class FlyingBirdTail(PartFactory):

    tags = ['tail', 'wing']

    def sample_params(self):
        return {
            'Feather Length Rad1 Rad2': np.array((0.4, 0.06, 0.04)) * N(1, 0.1) * N(1, 0.1, 3),
            'Feather Rot Extent': np.array((25, -10, -16)) * N(1, 0.1, 3),
            'Feather Rot Rand Bounds': np.array((5.0, 5.0, 5.0)) * N(1, 0.1) * N(1, 0.05, 3),
            'N Feathers': int(N(16, 3))
        }

    def make_part(self, params):
        part = nodegroup_to_part(nodegroup_flying_bird_tail, params)
        return part


class FlyingBirdWing(PartFactory):

    tags = ['limb', 'wing']

    def sample_params(self):
        return {
            'length_rad1_rad2': np.array((clip_gaussian(1.2, 0.7, 0.4, 2), U(0.08, 0.13), 0.02)),
            'feather_density': 40,
            'aspect': N(0.35, 0.04),
            'fullness': N(4, 0.1),
            'Wing Shape Sculpting': U(0.6, 1),
            'Length Rad1 Rad2': np.array((0.6 * N(1, 0.2), 0.04, 0.04)),
            'Extension': U(0, 0.05) if U() < 0.8 else U(0.7, 1)
        }

    def make_part(self, params):
        # split extras is essential to make automatic rigging work. We will join them back together later
        part = nodegroup_to_part(nodegroup_flying_bird_wing, params, split_extras=True)
        part.joints = {
            0: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])), # shoulder
            0.27: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])),
            0.65: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])) # elbow
        } 
        part.iks = {1.0: IKParams(name='wingtip', chain_length=3)}
        part.settings['parent_extras_rigid'] = True
        return part
