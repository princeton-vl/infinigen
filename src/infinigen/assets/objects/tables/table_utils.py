# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


@node_utils.to_nodegroup(
    "nodegroup_n_gon_profile", singleton=False, type="GeometryNodeTree"
)
def nodegroup_n_gon_profile(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "Profile N-gon", 4),
            ("NodeSocketFloat", "Profile Width", 1.0000),
            ("NodeSocketFloat", "Profile Aspect Ratio", 1.0000),
            ("NodeSocketFloat", "Profile Fillet Ratio", 0.2000),
        ],
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.5000

    curve_circle = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={
            "Resolution": group_input.outputs["Profile N-gon"],
            "Radius": value,
        },
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 3.1416, 1: group_input.outputs["Profile N-gon"]},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide})

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle.outputs["Curve"],
            "Rotation": combine_xyz_1,
        },
    )

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Rotation": (0.0000, 0.0000, -1.5708)},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Profile Aspect Ratio"],
            1: group_input.outputs["Profile Width"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Profile Width"],
            "Y": multiply,
            "Z": 1.0000,
        },
    )

    transform_1 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": transform_2, "Scale": combine_xyz}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Profile Width"],
            1: group_input.outputs["Profile Fillet Ratio"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    fillet_curve_1 = nw.new_node(
        "GeometryNodeFilletCurve",
        input_kwargs={
            "Curve": transform_1,
            "Count": 8,
            "Radius": multiply_1,
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Output": fillet_curve_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_n_gon_cylinder", singleton=False, type="GeometryNodeTree"
)
def nodegroup_n_gon_cylinder(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Radius Curve", None),
            ("NodeSocketFloat", "Height", 0.5000),
            ("NodeSocketInt", "N-gon", 0),
            ("NodeSocketFloat", "Profile Width", 0.5000),
            ("NodeSocketFloat", "Aspect Ratio", 0.5000),
            ("NodeSocketFloat", "Fillet Ratio", 0.2000),
            ("NodeSocketInt", "Profile Resolution", 64),
            ("NodeSocketInt", "Resolution", 128),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz_1})

    set_curve_tilt = nw.new_node(
        Nodes.SetCurveTilt, input_kwargs={"Curve": curve_line, "Tilt": 3.1416}
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve,
        input_kwargs={
            "Curve": set_curve_tilt,
            "Count": group_input.outputs["Resolution"],
        },
    )

    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)

    capture_attribute = nw.new_node(
        Nodes.CaptureAttribute,
        input_kwargs={
            "Geometry": resample_curve,
            2: spline_parameter_1.outputs["Factor"],
        },
    )

    ngonprofile = nw.new_node(
        nodegroup_n_gon_profile().name,
        input_kwargs={
            "Profile N-gon": group_input.outputs["N-gon"],
            "Profile Width": group_input.outputs["Profile Width"],
            "Profile Aspect Ratio": group_input.outputs["Aspect Ratio"],
            "Profile Fillet Ratio": group_input.outputs["Fillet Ratio"],
        },
    )

    resample_curve_1 = nw.new_node(
        Nodes.ResampleCurve,
        input_kwargs={
            "Curve": ngonprofile,
            "Count": group_input.outputs["Profile Resolution"],
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": capture_attribute.outputs["Geometry"],
            "Profile Curve": resample_curve_1,
            "Fill Caps": True,
        },
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_1})

    sample_curve = nw.new_node(
        Nodes.SampleCurve,
        input_kwargs={
            "Curves": group_input.outputs["Radius Curve"],
            "Factor": capture_attribute.outputs[1],
        },
        attrs={"use_all_curves": True},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": sample_curve.outputs["Position"]}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": separate_xyz.outputs["X"], "Y": separate_xyz.outputs["Y"]},
    )

    length = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: combine_xyz}, attrs={"operation": "LENGTH"}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: length.outputs["Value"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: length.outputs["Value"]},
        attrs={"operation": "MULTIPLY"},
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    attribute_statistic = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": group_input.outputs["Radius Curve"],
            2: separate_xyz_1.outputs["Z"],
        },
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": separate_xyz.outputs["Z"],
            1: attribute_statistic.outputs["Min"],
            2: attribute_statistic.outputs["Max"],
            3: multiply,
            4: 0.0000,
        },
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": multiply_1,
            "Y": multiply_2,
            "Z": map_range.outputs["Result"],
        },
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": curve_to_mesh, "Position": combine_xyz_2},
    )

    index = nw.new_node(Nodes.Index)

    domain_size = nw.new_node(
        Nodes.DomainSize, input_kwargs={"Geometry": curve_to_mesh}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: domain_size.outputs["Face Count"], 1: 2.0000},
        attrs={"operation": "SUBTRACT"},
    )

    less_than = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: subtract},
        attrs={"operation": "LESS_THAN", "data_type": "INT"},
    )

    delete_geometry = nw.new_node(
        Nodes.DeleteGeometry,
        input_kwargs={"Geometry": curve_to_mesh, "Selection": less_than},
        attrs={"domain": "FACE"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Mesh": set_position,
            "Profile Curve": resample_curve_1,
            "Caps": delete_geometry,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_generate_radius_curve", singleton=False, type="GeometryNodeTree"
)
def nodegroup_generate_radius_curve(nw: NodeWrangler, curve_control_points):
    # Code generated using version 2.6.4 of the node_transpiler

    curve_line = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (1.0000, 0.0000, 1.0000),
            "End": (1.0000, 0.0000, -1.0000),
        },
    )

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "Resolution", 128)]
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve,
        input_kwargs={"Curve": curve_line, "Count": group_input.outputs["Resolution"]},
    )

    position = nw.new_node(Nodes.InputPosition)

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    float_curve = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": spline_parameter.outputs["Factor"]}
    )
    node_utils.assign_curve(float_curve.mapping.curves[0], curve_control_points)

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": float_curve, "Y": 1.0000, "Z": 1.0000}
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: position, 1: combine_xyz_1},
        attrs={"operation": "MULTIPLY"},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": resample_curve,
            "Position": multiply.outputs["Vector"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_position},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_create_anchors", singleton=False, type="GeometryNodeTree"
)
def nodegroup_create_anchors(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "Profile N-gon", 0),
            ("NodeSocketFloat", "Profile Width", 0.5000),
            ("NodeSocketFloat", "Profile Aspect Ratio", 0.5000),
            ("NodeSocketFloat", "Profile Rotation", 0.0000),
        ],
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Profile N-gon"], 3: 1},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Profile N-gon"], 3: 2},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    ngonprofile = nw.new_node(
        nodegroup_n_gon_profile().name,
        input_kwargs={
            "Profile N-gon": group_input.outputs["Profile N-gon"],
            "Profile Width": group_input.outputs["Profile Width"],
            "Profile Aspect Ratio": group_input.outputs["Profile Aspect Ratio"],
            "Profile Fillet Ratio": 0.0000,
        },
    )

    curve_to_points = nw.new_node(
        Nodes.CurveToPoints,
        input_kwargs={"Curve": ngonprofile},
        attrs={"mode": "EVALUATED"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Profile Width"], 1: 0.3535},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Profile Width"], 1: -0.3535},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz, "End": combine_xyz_1}
    )

    curve_to_points_1 = nw.new_node(
        Nodes.CurveToPoints,
        input_kwargs={"Curve": curve_line},
        attrs={"mode": "EVALUATED"},
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            0: equal_1,
            1: curve_to_points.outputs["Points"],
            2: curve_to_points_1.outputs["Points"],
        },
    )

    points = nw.new_node("GeometryNodePoints")

    switch = nw.new_node(Nodes.Switch, input_kwargs={0: equal, 1: switch_1, 2: points})

    set_point_radius = nw.new_node(
        Nodes.SetPointRadius, input_kwargs={"Points": switch}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Profile Rotation"]}
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": set_point_radius, "Rotation": combine_xyz_2},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_create_legs_and_strechers", singleton=False, type="GeometryNodeTree"
)
def nodegroup_create_legs_and_strechers(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Anchors", None),
            ("NodeSocketBool", "Keep Legs", False),
            ("NodeSocketGeometry", "Leg Instance", None),
            ("NodeSocketFloat", "Table Height", 0.0000),
            ("NodeSocketFloat", "Leg Bottom Relative Scale", 0.0000),
            ("NodeSocketFloat", "Leg Bottom Relative Rotation", 0.0000),
            ("NodeSocketBool", "Keep Odd Strechers", True),
            ("NodeSocketBool", "Keep Even Strechers", True),
            ("NodeSocketGeometry", "Strecher Instance", None),
            ("NodeSocketInt", "Strecher Index Increment", 0),
            ("NodeSocketFloat", "Strecher Relative Position", 0.5000),
            ("NodeSocketFloat", "Leg Bottom Offset", 0.0000),
            ("NodeSocketBool", "Align Leg X rot", False),
        ],
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Table Height"]}
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": group_input.outputs["Anchors"],
            "Translation": combine_xyz,
        },
    )

    position = nw.new_node(Nodes.InputPosition)

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Leg Bottom Offset"]}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: combine_xyz_3},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: position, 1: subtract.outputs["Vector"]},
        attrs={"operation": "SUBTRACT"},
    )

    vector_rotate = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={
            "Vector": subtract_1.outputs["Vector"],
            "Angle": group_input.outputs["Leg Bottom Relative Rotation"],
        },
        attrs={"rotation_type": "Z_AXIS"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Leg Bottom Relative Scale"],
            "Y": group_input.outputs["Leg Bottom Relative Scale"],
            "Z": 1.0000,
        },
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: vector_rotate, 1: combine_xyz_4},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: position, 1: multiply.outputs["Vector"]},
        attrs={"operation": "SUBTRACT"},
    )

    align_euler_to_vector = nw.new_node(
        Nodes.AlignEulerToVector,
        input_kwargs={"Vector": subtract_2},
        attrs={"axis": "Z"},
    )

    align_euler_to_vector_3 = nw.new_node(
        Nodes.AlignEulerToVector,
        input_kwargs={"Rotation": align_euler_to_vector, "Vector": position},
        attrs={"pivot_axis": "Z"},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            0: group_input.outputs["Align Leg X rot"],
            1: align_euler_to_vector,
            2: align_euler_to_vector_3,
        },
        attrs={"input_type": "VECTOR"},
    )

    length = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: subtract_2}, attrs={"operation": "LENGTH"}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": 1.0000, "Y": 1.0000, "Z": length.outputs["Value"]},
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": transform,
            "Instance": group_input.outputs["Leg Instance"],
            "Rotation": switch,
            "Scale": combine_xyz_2,
        },
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points}
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={0: group_input.outputs["Keep Legs"], 2: realize_instances},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Strecher Relative Position"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: subtract_2, "Scale": multiply_1},
        attrs={"operation": "SCALE"},
    )

    position_2 = nw.new_node(Nodes.InputPosition)

    add = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: scale.outputs["Vector"], 1: position_2}
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": transform, "Position": add.outputs["Vector"]},
    )

    index = nw.new_node(Nodes.Index)

    modulo = nw.new_node(
        Nodes.Math, input_kwargs={0: index, 1: 2.0000}, attrs={"operation": "MODULO"}
    )

    op_and = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: modulo, 1: group_input.outputs["Keep Odd Strechers"]},
    )

    op_not = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: modulo}, attrs={"operation": "NOT"}
    )

    op_and_1 = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: group_input.outputs["Keep Even Strechers"], 1: op_not},
    )

    op_or = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: op_and, 1: op_and_1},
        attrs={"operation": "OR"},
    )

    domain_size = nw.new_node(
        Nodes.DomainSize,
        input_kwargs={"Geometry": transform},
        attrs={"component": "POINTCLOUD"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: domain_size.outputs["Point Count"],
            1: group_input.outputs["Strecher Index Increment"],
        },
        attrs={"operation": "DIVIDE"},
    )

    equal = nw.new_node(
        Nodes.Compare, input_kwargs={0: divide, 1: 2.0000}, attrs={"operation": "EQUAL"}
    )

    boolean = nw.new_node(Nodes.Boolean, attrs={"boolean": True})

    index_1 = nw.new_node(Nodes.Index)

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: domain_size.outputs["Point Count"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    less_than = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index_1, 3: divide_1},
        attrs={"operation": "LESS_THAN", "data_type": "INT"},
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={0: equal, 1: boolean, 2: less_than},
        attrs={"input_type": "BOOLEAN"},
    )

    op_and_2 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: op_or, 1: switch_2})

    position_1 = nw.new_node(Nodes.InputPosition)

    add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: index, 1: group_input.outputs["Strecher Index Increment"]},
    )

    modulo_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_1, 1: domain_size.outputs["Point Count"]},
        attrs={"operation": "MODULO"},
    )

    field_at_index = nw.new_node(
        Nodes.FieldAtIndex,
        input_kwargs={"Index": modulo_1, 1: position_1},
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    subtract_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: position_1, 1: field_at_index},
        attrs={"operation": "SUBTRACT"},
    )

    align_euler_to_vector_1 = nw.new_node(
        Nodes.AlignEulerToVector,
        input_kwargs={"Vector": subtract_3.outputs["Vector"]},
        attrs={"axis": "Z"},
    )

    align_euler_to_vector_2 = nw.new_node(
        Nodes.AlignEulerToVector,
        input_kwargs={"Rotation": align_euler_to_vector_1},
        attrs={"pivot_axis": "Z"},
    )

    length_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: subtract_3.outputs["Vector"]},
        attrs={"operation": "LENGTH"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": 1.0000, "Y": 1.0000, "Z": length_1.outputs["Value"]},
    )

    instance_on_points_1 = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": set_position,
            "Selection": op_and_2,
            "Instance": group_input.outputs["Strecher Instance"],
            "Rotation": align_euler_to_vector_2,
            "Scale": combine_xyz_1,
        },
    )

    realize_instances_1 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points_1}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [switch_1, realize_instances_1]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_create_cap", singleton=False, type="GeometryNodeTree"
)
def nodegroup_create_cap(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Radius", 1.0000),
            ("NodeSocketInt", "Resolution", 64),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 257.0000},
        attrs={"operation": "MULTIPLY"},
    )

    uv_sphere = nw.new_node(
        Nodes.MeshUVSphere,
        input_kwargs={
            "Segments": group_input.outputs["Resolution"],
            "Rings": multiply,
            "Radius": group_input.outputs["Radius"],
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": uv_sphere.outputs["Mesh"],
            "Name": "uv_map",
            3: uv_sphere.outputs["UV Map"],
        },
        attrs={"data_type": "FLOAT_VECTOR", "domain": "CORNER"},
    )

    power = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 2.0000},
        attrs={"operation": "POWER"},
    )

    subtract = nw.new_node(
        Nodes.Math, input_kwargs={0: power, 1: 1.0000}, attrs={"operation": "SUBTRACT"}
    )

    sqrt = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract}, attrs={"operation": "SQRT"}
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: sqrt, 1: -1.0000}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_1})

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": store_named_attribute, "Translation": combine_xyz},
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    less_than = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: separate_xyz.outputs["Z"]},
        attrs={"operation": "LESS_THAN"},
    )

    delete_geometry = nw.new_node(
        Nodes.DeleteGeometry,
        input_kwargs={"Geometry": transform, "Selection": less_than},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": delete_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_arc_top", singleton=False, type="GeometryNodeTree")
def nodegroup_arc_top(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Diameter", 1.0000),
            ("NodeSocketFloat", "Sweep Angle", 180.0000),
        ],
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Diameter"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Sweep Angle"], 2: -90.0000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_add, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    radians = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply}, attrs={"operation": "RADIANS"}
    )

    radians_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Sweep Angle"]},
        attrs={"operation": "RADIANS"},
    )

    arc = nw.new_node(
        "GeometryNodeCurveArc",
        input_kwargs={
            "Resolution": 32,
            "Radius": divide,
            "Start Angle": radians,
            "Sweep Angle": radians_1,
        },
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": arc.outputs["Curve"],
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_align_bottom_to_floor", singleton=False, type="GeometryNodeTree"
)
def nodegroup_align_bottom_to_floor(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": group_input.outputs["Geometry"]}
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Min"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Translation": combine_xyz,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_1, "Offset": multiply},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_bent", singleton=False, type="GeometryNodeTree")
def nodegroup_bent(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketFloat", "Amount", -0.1000),
        ],
    )

    position = nw.new_node(Nodes.InputPosition)

    length = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: position}, attrs={"operation": "LENGTH"}
    )

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: length.outputs["Value"], 1: separate_xyz.outputs["X"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: group_input.outputs["Amount"]},
        attrs={"operation": "MULTIPLY"},
    )

    vector_rotate = nw.new_node(
        Nodes.VectorRotate, input_kwargs={"Vector": position, "Angle": multiply_1}
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Position": vector_rotate,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_position},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_merge_curve", singleton=False, type="GeometryNodeTree"
)
def nodegroup_merge_curve(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Curve", None)]
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh, input_kwargs={"Curve": group_input.outputs["Curve"]}
    )

    merge_by_distance = nw.new_node(
        Nodes.MergeByDistance, input_kwargs={"Geometry": curve_to_mesh_1}
    )

    mesh_to_curve = nw.new_node(
        Nodes.MeshToCurve, input_kwargs={"Mesh": merge_by_distance}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Curve": mesh_to_curve},
        attrs={"is_active_output": True},
    )
