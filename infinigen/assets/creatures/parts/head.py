# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy
import numpy as np
from numpy.random import uniform, normal as N, randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

from infinigen.core.util.math import clip_gaussian
from infinigen.assets.creatures.util.genome import Joint, IKParams

from infinigen.assets.creatures.util.nodegroups.curve import nodegroup_simple_tube, nodegroup_polar_bezier, nodegroup_simple_tube_v2, nodegroup_warped_circle_curve
from infinigen.assets.creatures.util.nodegroups.attach import nodegroup_surface_muscle, nodegroup_part_surface_simple, nodegroup_attach_part, nodegroup_smooth_taper, nodegroup_profile_part
from infinigen.assets.creatures.util.nodegroups.geometry import nodegroup_solidify, nodegroup_symmetric_clone
from infinigen.assets.creatures.util.nodegroups.math import nodegroup_deg2_rad

from infinigen.assets.creatures.util.creature import PartFactory
from infinigen.assets.creatures.util import part_util
from infinigen.assets.creatures.parts.eye import nodegroup_mammal_eye
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

@node_utils.to_nodegroup('nodegroup_carnivore_jaw', singleton=True, type='GeometryNodeTree')
def nodegroup_carnivore_jaw(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (0.0, 0.0, 0.0)),
            ('NodeSocketFloatFactor', 'Width Shaping', 0.6764),
            ('NodeSocketFloat', 'Canine Length', 0.050000000000000003),
            ('NodeSocketFloat', 'Incisor Size', 0.01),
            ('NodeSocketFloat', 'Tooth Crookedness', 0.0),
            ('NodeSocketFloatFactor', 'Tongue Shaping', 1.0),
            ('NodeSocketFloat', 'Tongue X Scale', 0.90000000000000002)])
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["length_rad1_rad2"]})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (0.33000000000000002, 0.33000000000000002, 0.33000000000000002), 'Scale': separate_xyz.outputs["X"]},
        attrs={'operation': 'SCALE'})
    
    polarbezier = nw.new_node(nodegroup_polar_bezier().name,
        input_kwargs={'angles_deg': (0.0, 0.0, 13.0), 'Seg Lengths': scale.outputs["Vector"]})
    
    position = nw.new_node(Nodes.InputPosition)
    
    vector_curves = nw.new_node(Nodes.VectorCurve,
        input_kwargs={'Vector': position})
    node_utils.assign_curve(vector_curves.mapping.curves[0], [(-1.0, -1.0), (0.0035999999999999999, 0.0), (0.24360000000000001, 0.20999999999999999), (1.0, 1.0)])
    node_utils.assign_curve(vector_curves.mapping.curves[1], [(-1.0, 0.12), (-0.77449999999999997, 0.059999999999999998), (-0.65090000000000003, -0.44), (-0.36730000000000002, -0.40000000000000002), (-0.0545, -0.01), (0.1055, 0.02), (0.52729999999999999, 0.5), (0.7964, 0.64000000000000001), (1.0, 1.0)], handles=['AUTO', 'AUTO', 'AUTO', 'AUTO_CLAMPED', 'AUTO', 'AUTO', 'VECTOR', 'AUTO', 'AUTO'])
    node_utils.assign_curve(vector_curves.mapping.curves[2], [(-1.0, -1.0), (1.0, 1.0)])
    
    warped_circle_curve = nw.new_node(nodegroup_warped_circle_curve().name,
        input_kwargs={'Position': vector_curves})
    
    spline_parameter = nw.new_node(Nodes.SplineParameter)
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Factor': group_input.outputs["Width Shaping"], 'Value': spline_parameter.outputs["Factor"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 0.95499999999999996), (0.42549999999999999, 0.78500000000000003), (0.65449999999999997, 0.53500000000000003), (0.94910000000000005, 0.75), (1.0, 0.59499999999999997)], handles=['AUTO', 'AUTO', 'AUTO', 'AUTO_CLAMPED', 'AUTO'])
    
    smoothtaper = nw.new_node(nodegroup_smooth_taper().name,
        input_kwargs={'start_rad': separate_xyz.outputs["Y"], 'end_rad': separate_xyz.outputs["Z"], 'fullness': 2.6000000000000001})
    
    multiply = nw.new_node(Nodes.Math,
        input_kwargs={0: float_curve, 1: smoothtaper},
        attrs={'operation': 'MULTIPLY'})
    
    profilepart = nw.new_node(nodegroup_profile_part().name,
        input_kwargs={'Skeleton Curve': polarbezier.outputs["Curve"], 'Profile Curve': warped_circle_curve, 'Radius Func': multiply})
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': profilepart, 'Scale': (1.0, 1.7, 1.0)})
    
    greater_than = nw.new_node(Nodes.Compare,
        input_kwargs={0: group_input.outputs["Canine Length"]})
    
    scale_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (0.33000000000000002, 0.33000000000000002, 0.33000000000000002), 'Scale': group_input.outputs["Canine Length"]},
        attrs={'operation': 'SCALE'})
    
    canine_tooth = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Seg Lengths': scale_1.outputs["Vector"], 'Start Radius': 0.014999999999999999, 'End Radius': 0.0030000000000000001},
        label='Canine Tooth')
    
    attach_part = nw.new_node(nodegroup_attach_part().name,
        input_kwargs={'Skin Mesh': transform, 'Skeleton Curve': polarbezier.outputs["Curve"], 'Geometry': canine_tooth.outputs["Geometry"], 'Length Fac': 0.90000000000000002, 'Ray Rot': (1.5708, 0.12039999999999999, 1.5708), 'Rad': 1.0, 'Part Rot': (-17.600000000000001, -53.490000000000002, 0.0)})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': attach_part.outputs["Geometry"]})
    
    symmetric_clone = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': join_geometry})
    
    switch_1 = nw.new_node(Nodes.Switch,
        input_kwargs={1: greater_than, 15: symmetric_clone.outputs["Both"]})
    
    greater_than_1 = nw.new_node(Nodes.Compare,
        input_kwargs={0: group_input.outputs["Incisor Size"]})
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: attach_part.outputs["Position"], 1: (0.014999999999999999, -0.050000000000000003, 0.0)})
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add.outputs["Vector"], 1: (1.0, -1.0, 1.0)},
        attrs={'operation': 'MULTIPLY'})
    
    add_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add.outputs["Vector"], 1: multiply_1.outputs["Vector"]})
    
    multiply_add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: add_1.outputs["Vector"], 1: (0.5, 0.5, 0.5), 2: (-0.02, 0.0, 0.0)},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    quadratic_bezier = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 6, 'Start': add.outputs["Vector"], 'Middle': multiply_add.outputs["Vector"], 'End': multiply_1.outputs["Vector"]})
    
    curve_to_mesh = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': quadratic_bezier})
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': curve_to_mesh})
    
    scale_2 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (3.0, 1.0, 0.59999999999999998), 'Scale': group_input.outputs["Incisor Size"]},
        attrs={'operation': 'SCALE'})
    
    cube = nw.new_node(Nodes.MeshCube,
        input_kwargs={'Size': scale_2.outputs["Vector"]})
    
    subdivision_surface = nw.new_node(Nodes.SubdivisionSurface,
        input_kwargs={'Mesh': cube, 'Level': 3})
    
    transform_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': subdivision_surface})
    
    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': transform_1, 'Instance': transform_2, 'Rotation': (0.0, -1.5708, 0.0)})
    
    subtract = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (2.0, 2.0, 2.0), 1: group_input.outputs["Tooth Crookedness"]},
        attrs={'operation': 'SUBTRACT'})
    
    random_value = nw.new_node(Nodes.RandomValue,
        input_kwargs={0: subtract.outputs["Vector"], 1: group_input.outputs["Tooth Crookedness"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    scale_instances = nw.new_node(Nodes.ScaleInstances,
        input_kwargs={'Instances': instance_on_points, 'Scale': random_value.outputs["Value"]})
    
    scale_3 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (-3.0, -3.0, -3.0), 'Scale': group_input.outputs["Tooth Crookedness"]},
        attrs={'operation': 'SCALE'})
    
    scale_4 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (3.0, 3.0, 3.0), 'Scale': group_input.outputs["Tooth Crookedness"]},
        attrs={'operation': 'SCALE'})
    
    random_value_1 = nw.new_node(Nodes.RandomValue,
        input_kwargs={0: scale_3.outputs["Vector"], 1: scale_4.outputs["Vector"]},
        attrs={'data_type': 'FLOAT_VECTOR'})
    
    deg2rad = nw.new_node(nodegroup_deg2_rad().name,
        input_kwargs={'Deg': random_value_1.outputs["Value"]})
    
    rotate_instances = nw.new_node(Nodes.RotateInstances,
        input_kwargs={'Instances': scale_instances, 'Rotation': deg2rad})
    
    realize_instances = nw.new_node(Nodes.RealizeInstances,
        input_kwargs={'Geometry': rotate_instances})
    
    switch = nw.new_node(Nodes.Switch,
        input_kwargs={1: greater_than_1, 15: realize_instances})
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [switch_1.outputs[6], switch.outputs[6]]})
    
    resample_curve = nw.new_node(Nodes.ResampleCurve,
        input_kwargs={'Curve': polarbezier.outputs["Curve"]})
    
    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)
    
    float_curve_1 = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Factor': group_input.outputs["Tongue Shaping"], 'Value': spline_parameter_1.outputs["Factor"]})
    node_utils.assign_curve(float_curve_1.mapping.curves[0], [(0.0, 1.0), (0.69820000000000004, 0.55000000000000004), (0.97450000000000003, 0.34999999999999998), (1.0, 0.17499999999999999)])
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={3: separate_xyz.outputs["Y"], 4: separate_xyz.outputs["Z"]},
        attrs={'clamp': False})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: float_curve_1, 1: map_range.outputs["Result"]},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_2, 1: 1.0},
        attrs={'operation': 'MULTIPLY'})
    
    set_curve_radius = nw.new_node(Nodes.SetCurveRadius,
        input_kwargs={'Curve': resample_curve, 'Radius': multiply_3})
    
    quadratic_bezier_1 = nw.new_node(Nodes.QuadraticBezier,
        input_kwargs={'Resolution': 3, 'Middle': (0.0, 0.69999999999999996, 0.0)})
    
    curve_to_mesh_1 = nw.new_node(Nodes.CurveToMesh,
        input_kwargs={'Curve': set_curve_radius, 'Profile Curve': quadratic_bezier_1, 'Fill Caps': True})
    
    solidify = nw.new_node(nodegroup_solidify().name,
        input_kwargs={'Mesh': curve_to_mesh_1, 'Distance': 0.02})
    
    set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth,
        input_kwargs={'Geometry': solidify, 'Shade Smooth': False})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["Tongue X Scale"], 'Y': 1.0, 'Z': 1.0})
    
    transform_3 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': set_shade_smooth, 'Rotation': (0.0, -0.015900000000000001, 0.0), 'Scale': combine_xyz})
    
    subdivision_surface_1 = nw.new_node(Nodes.SubdivisionSurface,
        input_kwargs={'Mesh': transform_3, 'Level': 2})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': transform, 'Skeleton Curve': polarbezier.outputs["Curve"], 'Teeth': join_geometry_1, 'Tongue': subdivision_surface_1})

@node_utils.to_nodegroup('nodegroup_carnivore_head', singleton=False, type='GeometryNodeTree')
def nodegroup_carnivore_head(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (0.0, 0.0, 0.0)),
            ('NodeSocketVector', 'snout_length_rad1_rad2', (0.0, 0.0, 0.0)),
            ('NodeSocketFloat', 'snout_y_scale', 0.62),
            ('NodeSocketVectorXYZ', 'Nose Bridge Scale', (1.0, 0.35, 0.9)),
            ('NodeSocketVector', 'Jaw Muscle Middle Coord', (0.24, 0.41, 1.3)),
            ('NodeSocketVector', 'Jaw StartRad, EndRad, Fullness', (0.06, 0.11, 1.5)),
            ('NodeSocketVector', 'Jaw ProfileHeight, StartTilt, EndTilt', (0.8, 33.1, 0.0)),
            ('NodeSocketVector', 'Lip Muscle Middle Coord', (0.95, 0.0, 1.5)),
            ('NodeSocketVector', 'Lip StartRad, EndRad, Fullness', (0.05, 0.09, 1.48)),
            ('NodeSocketVector', 'Lip ProfileHeight, StartTilt, EndTilt', (0.8, 0.0, -17.2)),
            ('NodeSocketVector', 'Forehead Muscle Middle Coord', (0.7, -1.32, 1.31)),
            ('NodeSocketVector', 'Forehead StartRad, EndRad, Fullness', (0.06, 0.05, 2.5)),
            ('NodeSocketVector', 'Forehead ProfileHeight, StartTilt, EndTilt', (0.3, 60.6, 66.0)),
            ('NodeSocketFloat', 'aspect', 1.0),
            ('NodeSocketFloatDistance', 'EyeRad', 0.03),
            ('NodeSocketVector', 'EyeOffset', (-0.2, 0.5, 0.2))])
    
    vector = nw.new_node(Nodes.Vector)
    vector.vector = (-0.07, 0.0, 0.05)
    
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"], 'angles_deg': (-5.67, 0.0, 0.0), 'aspect': group_input.outputs["aspect"], 'fullness': 3.63, 'Origin': vector})
    
    snout_origin = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: simple_tube_v2.outputs["Endpoint"], 1: (-0.1, 0.0, 0.0)},
        label='Snout Origin')
    
    split_length_width1_width2 = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': group_input.outputs["snout_length_rad1_rad2"]},
        label='Split Length / Width1 / Width2')
    
    snout_seg_lengths = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (0.33, 0.33, 0.33), 'Scale': split_length_width1_width2.outputs["X"]},
        label='Snout Seg Lengths',
        attrs={'operation': 'SCALE'})
    
    bridge = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Origin': snout_origin.outputs["Vector"], 'Angles Deg': (-4.0, -4.5, -5.61), 'Seg Lengths': snout_seg_lengths.outputs["Vector"], 'Start Radius': 0.17, 'End Radius': 0.1, 'Fullness': 5.44},
        label='Bridge')
    
    transform = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': bridge.outputs["Geometry"], 'Translation': (0.0, 0.0, 0.03), 'Scale': group_input.outputs["Nose Bridge Scale"]})
    
    snout = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Origin': snout_origin.outputs["Vector"], 'Angles Deg': (-3.0, -4.5, -5.61), 'Seg Lengths': snout_seg_lengths.outputs["Vector"], 'Start Radius': split_length_width1_width2.outputs["Y"], 'End Radius': split_length_width1_width2.outputs["Z"], 'Fullness': 2.0},
        label='Snout')
    
    transform_1 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': snout.outputs["Geometry"], 'Translation': (0.0, 0.0, 0.03), 'Scale': (1.0, 0.7, 0.7)})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': 1.0, 'Y': group_input.outputs["snout_y_scale"], 'Z': 1.0})
    
    transform_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': transform_1, 'Scale': combine_xyz})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [transform, transform_2]})
    
    union = nw.new_node(Nodes.MeshBoolean,
        input_kwargs={'Mesh 2': [join_geometry, simple_tube_v2.outputs["Geometry"]], 'Self Intersection': True},
        attrs={'operation': 'UNION'})
    
    curve_line_1 = nw.new_node(Nodes.CurveLine,
        input_kwargs={'Start': vector, 'End': snout.outputs["Endpoint"]})
    
    scale = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: (0.33, 0.33, 0.33)},
        attrs={'operation': 'SCALE'})
    
    jaw_cutter = nw.new_node(nodegroup_simple_tube().name,
        input_kwargs={'Origin': (0.0, 0.0, 0.09), 'Angles Deg': (0.0, 0.0, 0.0), 'Seg Lengths': scale.outputs["Vector"], 'Start Radius': 0.13},
        label='Jaw Cutter')
    
    attach_part = nw.new_node(nodegroup_attach_part().name,
        input_kwargs={'Skin Mesh': union.outputs["Mesh"], 'Skeleton Curve': curve_line_1, 'Geometry': jaw_cutter.outputs["Geometry"], 'Length Fac': 0.2, 'Ray Rot': (0.0, 1.5708, 0.0), 'Rad': 1.25, 'Part Rot': (0.0, -8.5, 0.0), 'Do Tangent Rot': True})
    
    mammaleye = nw.new_node(nodegroup_mammal_eye().name,
        input_kwargs={'Radius': group_input.outputs["EyeRad"]})
    
    reroute_4 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': group_input.outputs["length_rad1_rad2"]})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': reroute_4})
    
    reroute_3 = nw.new_node(Nodes.Reroute,
        input_kwargs={'Input': simple_tube_v2.outputs["Endpoint"]})
    
    multiply_add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["EyeOffset"], 1: separate_xyz.outputs["Z"], 2: reroute_3},
        attrs={'operation': 'MULTIPLY_ADD'})
    
    transform_4 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': mammaleye.outputs["ParentCutter"], 'Translation': multiply_add.outputs["Vector"]})
    
    symmetric_clone = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': transform_4})
    
    difference = nw.new_node(Nodes.MeshBoolean,
        input_kwargs={'Mesh 1': union.outputs["Mesh"], 'Mesh 2': [attach_part.outputs["Geometry"], symmetric_clone.outputs["Both"]], 'Self Intersection': True})
    
    jaw_muscle = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': union.outputs["Mesh"], 'Skeleton Curve': curve_line_1, 'Coord 0': (0.19, -0.41, 0.78), 'Coord 1': group_input.outputs["Jaw Muscle Middle Coord"], 'Coord 2': (0.67, 1.26, 0.52), 'StartRad, EndRad, Fullness': group_input.outputs["Jaw StartRad, EndRad, Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input.outputs["Jaw ProfileHeight, StartTilt, EndTilt"]},
        label='Jaw Muscle')
    
    lip = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': union.outputs["Mesh"], 'Skeleton Curve': curve_line_1, 'Coord 0': (0.51, -0.13, 0.02), 'Coord 1': group_input.outputs["Lip Muscle Middle Coord"], 'Coord 2': (0.99, 10.57, 0.1), 'StartRad, EndRad, Fullness': group_input.outputs["Lip StartRad, EndRad, Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input.outputs["Lip ProfileHeight, StartTilt, EndTilt"]},
        label='Lip')
    
    forehead = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Coord 0': (0.31, -1.06, 0.97), 'Coord 1': group_input.outputs["Forehead Muscle Middle Coord"], 'Coord 2': (0.95, -1.52, 0.9), 'StartRad, EndRad, Fullness': group_input.outputs["Forehead StartRad, EndRad, Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input.outputs["Forehead ProfileHeight, StartTilt, EndTilt"]},
        label='Forehead')
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [jaw_muscle, lip, forehead]})
    
    symmetric_clone_1 = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': join_geometry_1})
    
    join_geometry_2 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [difference.outputs["Mesh"], symmetric_clone_1.outputs["Both"]]})
    
    subdivide_curve = nw.new_node(Nodes.SubdivideCurve,
        input_kwargs={'Curve': curve_line_1, 'Cuts': 10})
    
    transform_3 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': mammaleye.outputs["Eyeballl"], 'Translation': multiply_add.outputs["Vector"]})
    
    symmetric_clone_2 = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': transform_3})
    
    transform_5 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': mammaleye.outputs["BodyExtra_Lid"], 'Translation': multiply_add.outputs["Vector"]})
    
    symmetric_clone_3 = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': transform_5})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry_2, 'Skeleton Curve': subdivide_curve, 'Base Mesh': union.outputs["Mesh"], 'Eyeball_Left': symmetric_clone_2.outputs["Orig"], 'Eyeball_Right': symmetric_clone_2.outputs["Inverted"], 'BodyExtra_Lid': symmetric_clone_3.outputs["Both"]})

@node_utils.to_nodegroup('nodegroup_neck', singleton=True, type='GeometryNodeTree')
def nodegroup_neck(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (1.0, 0.5, 0.3)),
            ('NodeSocketVector', 'angles_deg', (0.0, 3.2, -18.11)),
            ('NodeSocketVector', 'Muscle StartRad, EndRad, Fullness', (0.17, 0.17, 2.5)),
            ('NodeSocketVector', 'ProfileHeight, StartTilt, EndTilt', (0.5, 0.0, 66.0)),
            ('NodeSocketFloat', 'fullness', 5.0),
            ('NodeSocketFloat', 'aspect', 1.0)])
    
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"], 'angles_deg': group_input.outputs["angles_deg"], 'aspect': group_input.outputs["aspect"], 'fullness': group_input.outputs["fullness"]})
    
    rear_top = nw.new_node(nodegroup_surface_muscle().name,
        input_kwargs={'Skin Mesh': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Coord 0': (0.1, 0.0, 0.9), 'Coord 1': (0.48, -0.77, 1.0), 'Coord 2': (0.87, -1.5708, 0.8), 'StartRad, EndRad, Fullness': group_input.outputs["Muscle StartRad, EndRad, Fullness"], 'ProfileHeight, StartTilt, EndTilt': group_input.outputs["ProfileHeight, StartTilt, EndTilt"]},
        label='Rear Top')
    
    join_geometry_1 = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': rear_top})
    
    symmetric_clone = nw.new_node(nodegroup_symmetric_clone().name,
        input_kwargs={'Geometry': join_geometry_1})
    
    join_geometry = nw.new_node(Nodes.JoinGeometry,
        input_kwargs={'Geometry': [symmetric_clone.outputs["Both"], simple_tube_v2.outputs["Geometry"]]})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': join_geometry, 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"], 'Base Mesh': simple_tube_v2.outputs["Geometry"]})

class Neck(PartFactory):

    tags = ['neck']

    def sample_params(self):
        return {
            'length_rad1_rad2': np.array((0.65, 0.35, 0.16)) * N(1, (0.2, 0, 0), 3),
            'angles_deg': np.array((0.0, 3.2, -18.11)) * N(1, 0.2, 3),
            'Muscle StartRad, EndRad, Fullness': (0.17, 0.17, 2.5),
            'ProfileHeight, StartTilt, EndTilt': (0.5, 0.0, 66.0),
            'fullness': 5.0,
            'aspect': 1.0 * N(1, 0.05)
        }

    def make_part(self, params):
        part = part_util.nodegroup_to_part(nodegroup_neck, params)
        part.joints = {
            i: Joint(rest=(0,0,0), bounds=np.array([[-30, 0, -30], [30, 0, 30]]))
            for i in np.linspace(0, 1, 4, endpoint=True)
        }
        tag_object(part.obj, 'neck')
        return part

class CarnivoreHead(PartFactory):

    tags = ['head']

    def sample_params(self):
        params = {
            'length_rad1_rad2': np.array((0.36, 0.20, 0.18)) * N(1, 0.2, 3),
            'snout_length_rad1_rad2': np.array((0.22, 0.15, 0.15)) * N(1, 0.2, 3),
            'aspect': N(1, 0.2),
        }

        muscle_params = {
            'Nose Bridge Scale': (1.0, 0.35, 0.9),
            'Jaw Muscle Middle Coord': (0.24, 0.41, 1.3),
            'Jaw StartRad, EndRad, Fullness': (0.06, 0.11, 1.5),
            'Jaw ProfileHeight, StartTilt, EndTilt': (0.8, 33.1, 0.0),
            'Lip Muscle Middle Coord': (0.95, 0.0, 1.5),
            'Lip StartRad, EndRad, Fullness': (0.05, 0.09, 1.48),
            'Lip ProfileHeight, StartTilt, EndTilt': (0.8, 0.0, -17.2),
            'Forehead Muscle Middle Coord': (0.7, -1.32, 1.31),
            'Forehead StartRad, EndRad, Fullness': (0.06, 0.05, 2.5),
            'Forehead ProfileHeight, StartTilt, EndTilt': (0.3, 60.6, 66.0)
        }

        for k, v in muscle_params.items():
            v = np.array(v)
            v *= N(1, 0.05, len(v))
            params[k] = v

        params.update(muscle_params)
        params['EyeRad'] = 0.023 * N(1, 0.3)
        params['EyeOffset'] = np.array((-0.25, 0.45, 0.3)) + N(0, (0, 0.02, 0.03))

        return params

    def make_part(self, params):
        part = part_util.nodegroup_to_part(nodegroup_carnivore_head, params)
        part.iks = {1.0: IKParams('head', rotation_weight=0.1, chain_length=1)}
        part.settings['rig_extras'] = True
        tag_object(part.obj, 'carnivore_head')
        return part

class CarnivoreJaw(PartFactory):

    tags = ['head', 'jaw']

    def sample_params(self):
        return {
            'length_rad1_rad2': np.array((0.4, 0.12, 0.08)) * N(1, 0.1, 3), 
            'Width Shaping': 1.0 * clip_gaussian(1, 0.1, 0.5, 1),
            'Canine Length': 0.05 * N(1, 0.2),
            'Incisor Size': 0.01 * N(1, 0.2),
            'Tooth Crookedness': 1.2 * N(1, 0.3),
            'Tongue Shaping': 1 * clip_gaussian(1, 0.1, 0.5, 1),
            'Tongue X Scale': 0.9 * clip_gaussian(1, 0.1, 0.5, 1)
        }

    def make_part(self, params):
        part = part_util.nodegroup_to_part(nodegroup_carnivore_jaw, params)
        tag_object(part.obj, 'carnivore_jaw')
        return part

@node_utils.to_nodegroup('nodegroup_flying_bird_head', singleton=True, type='GeometryNodeTree')
def nodegroup_flying_bird_head(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (0.34999999999999998, 0.11, 0.17000000000000001)),
            ('NodeSocketVector', 'angles_deg', (0.0, -24.0, -20.0)),
            ('NodeSocketVector', 'eye_coord', (0.5, 0.0, 1.0)),
            ('NodeSocketFloatDistance', 'Radius', 0.040000000000000001)])
    
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"], 'angles_deg': group_input.outputs["angles_deg"], 'aspect': N(0.9, 0.05), 'fullness': 0.9, 'Origin': (-0.13, 0.0, 0.1)})
    
    simple_tube_v2_1 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"], 'angles_deg': group_input.outputs["angles_deg"], 'aspect': 1.1899999999999999, 'fullness': 2.25, 'Origin': (-0.13, 0.0, 0.1-0.040000000000000001)})
    
    union = nw.new_node(Nodes.MeshBoolean,
        input_kwargs={'Mesh 2': [simple_tube_v2.outputs["Geometry"], simple_tube_v2_1.outputs["Geometry"]]},
        attrs={'operation': 'UNION'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': simple_tube_v2.outputs["Geometry"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"]})

@node_utils.to_nodegroup('nodegroup_bird_head', singleton=True, type='GeometryNodeTree')
def nodegroup_bird_head(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(Nodes.GroupInput,
        expose_input=[('NodeSocketVector', 'length_rad1_rad2', (0.34999999999999998, 0.11, 0.17000000000000001)),
            ('NodeSocketVector', 'angles_deg', (0.0, -24.0, -20.0)),
            ('NodeSocketVector', 'eye_coord', (0.5, 0.0, 1.0)),
            ('NodeSocketFloatDistance', 'Radius', 0.040000000000000001)])
    
    simple_tube_v2 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"], 'angles_deg': group_input.outputs["angles_deg"], 'aspect': 0.85999999999999999, 'fullness': 1.7, 'Origin': (-0.13, 0.0, 0.1)})
    
    simple_tube_v2_1 = nw.new_node(nodegroup_simple_tube_v2().name,
        input_kwargs={'length_rad1_rad2': group_input.outputs["length_rad1_rad2"], 'angles_deg': group_input.outputs["angles_deg"], 'aspect': 1.1899999999999999, 'fullness': 2.25, 'Origin': (-0.13, 0.0, 0.1-0.040000000000000001)})
    
    union = nw.new_node(Nodes.MeshBoolean,
        input_kwargs={'Mesh 2': [simple_tube_v2.outputs["Geometry"], simple_tube_v2_1.outputs["Geometry"]]},
        attrs={'operation': 'UNION'})
    
    group_output = nw.new_node(Nodes.GroupOutput,
        input_kwargs={'Geometry': union.outputs["Mesh"], 'Skeleton Curve': simple_tube_v2.outputs["Skeleton Curve"]})

class BirdHead(PartFactory):

    tags = ['head']

    def sample_params(self):
        return {
            'length_rad1_rad2': np.array((0.35, 0.11, 0.13)) * N(1, 0.05) * N(1, 0.1, 3),
            'angles_deg': N(0, 5, 3),
            'eye_coord': np.array((0.65, -0.32, 0.95)) * N(1, (0.1, 0.2, 0), 3),
            'Radius': 0.025 * N(1, 0.05)
        }

    def make_part(self, params):
        part = part_util.nodegroup_to_part(nodegroup_bird_head, params)
        part.iks = {1.0: IKParams('head', rotation_weight=0.1, chain_parts=2)}
        part.settings['rig_extras'] = True
        tag_object(part.obj, 'bird_head')
        return part

class FlyingBirdHead(PartFactory):

    tags = ['head']

    def sample_params(self):
        return {
            'length_rad1_rad2': np.array((0.3, 0.04, 0.12)) * N(1, 0.05, size=(3,)),
            'angles_deg': N(0, 0.1, 3),
            'eye_coord': np.array((0.65, -0.32, 0.95)) * N(1, (0.1, 0.2, 0), 3),
            'Radius': 0.03 * N(1, 0.05)
        }

    def make_part(self, params):
        part = part_util.nodegroup_to_part(nodegroup_flying_bird_head, params)
        part.iks = {1.0: IKParams('head', rotation_weight=0.1, chain_parts=2)}
        part.settings['rig_extras'] = True
        tag_object(part.obj, 'bird_head')
        return part