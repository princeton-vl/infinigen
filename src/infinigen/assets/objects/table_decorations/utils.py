# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


@node_utils.to_nodegroup(
    "nodegroup_star_profile", singleton=False, type="GeometryNodeTree"
)
def nodegroup_star_profile(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "Resolution", 64),
            ("NodeSocketInt", "Points", 64),
            ("NodeSocketFloat", "Inner Radius", 0.9000),
        ],
    )

    star = nw.new_node(
        "GeometryNodeCurveStar",
        input_kwargs={
            "Points": group_input.outputs["Points"],
            "Inner Radius": group_input.outputs["Inner Radius"],
            "Outer Radius": 1.0000,
        },
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve,
        input_kwargs={
            "Curve": star.outputs["Curve"],
            "Count": group_input.outputs["Resolution"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Curve": resample_curve},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_flip_index", singleton=False, type="GeometryNodeTree"
)
def nodegroup_flip_index(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "V Resolution", 0),
            ("NodeSocketInt", "U Resolution", 0),
        ],
    )

    modulo = nw.new_node(
        Nodes.Math,
        input_kwargs={0: index, 1: group_input.outputs["V Resolution"]},
        attrs={"operation": "MODULO"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: modulo, 1: group_input.outputs["U Resolution"]},
        attrs={"operation": "MULTIPLY"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: index, 1: group_input.outputs["V Resolution"]},
        attrs={"operation": "DIVIDE"},
    )

    floor = nw.new_node(
        Nodes.Math, input_kwargs={0: divide}, attrs={"operation": "FLOOR"}
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: floor})

    group_output = nw.new_node(
        Nodes.GroupOutput, input_kwargs={"Index": add}, attrs={"is_active_output": True}
    )


@node_utils.to_nodegroup(
    "nodegroup_cylinder_side", singleton=False, type="GeometryNodeTree"
)
def nodegroup_cylinder_side(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "U Resolution", 32),
            ("NodeSocketInt", "V Resolution", 0),
        ],
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["V Resolution"], 1: 1.0000},
        attrs={"operation": "SUBTRACT"},
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": group_input.outputs["U Resolution"],
            "Side Segments": subtract,
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Name": "uv_map",
            3: cylinder.outputs["UV Map"],
        },
        attrs={"data_type": "FLOAT_VECTOR", "domain": "CORNER"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": store_named_attribute,
            "Top": cylinder.outputs["Top"],
            "Side": cylinder.outputs["Side"],
            "Bottom": cylinder.outputs["Bottom"],
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_shifted_circle", singleton=False, type="GeometryNodeTree"
)
def nodegroup_shifted_circle(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "Resolution", 32),
            ("NodeSocketFloat", "Radius", 1.0000),
            ("NodeSocketFloat", "Z", 0.0000),
            ("NodeSocketFloat", "Rot Z", 0.0000),
        ],
    )

    curve_circle_3 = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={
            "Resolution": group_input.outputs["Resolution"],
            "Radius": group_input.outputs["Radius"],
        },
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Z"]}
    )

    radians = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Rot Z"]},
        attrs={"operation": "RADIANS"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": radians})

    transform_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle_3.outputs["Curve"],
            "Translation": combine_xyz,
            "Rotation": combine_xyz_1,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_3},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_lofting", singleton=False, type="GeometryNodeTree")
def nodegroup_lofting(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Profile Curves", None),
            ("NodeSocketInt", "U Resolution", 32),
            ("NodeSocketInt", "V Resolution", 32),
            ("NodeSocketBool", "Use Nurb", False),
        ],
    )

    cylinderside = nw.new_node(
        nodegroup_cylinder_side().name,
        input_kwargs={
            "U Resolution": group_input.outputs["U Resolution"],
            "V Resolution": group_input.outputs["V Resolution"],
        },
    )

    index = nw.new_node(Nodes.Index)

    evaluate_on_domain = nw.new_node(
        Nodes.EvaluateonDomain,
        input_kwargs={0: index},
        attrs={"data_type": "INT", "domain": "CURVE"},
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: evaluate_on_domain},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    curve_line = nw.new_node(Nodes.CurveLine)

    domain_size = nw.new_node(
        Nodes.DomainSize,
        input_kwargs={"Geometry": group_input.outputs["Profile Curves"]},
        attrs={"component": "CURVE"},
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve,
        input_kwargs={
            "Curve": curve_line,
            "Count": domain_size.outputs["Spline Count"],
        },
    )

    instance_on_points_1 = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": group_input.outputs["Profile Curves"],
            "Selection": equal,
            "Instance": resample_curve,
        },
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points_1}
    )

    position = nw.new_node(Nodes.InputPosition)

    flipindex = nw.new_node(
        nodegroup_flip_index().name,
        input_kwargs={
            "V Resolution": domain_size.outputs["Spline Count"],
            "U Resolution": group_input.outputs["U Resolution"],
        },
    )

    sample_index_2 = nw.new_node(
        Nodes.SampleIndex,
        input_kwargs={
            "Geometry": group_input.outputs["Profile Curves"],
            "Value": position,
            "Index": flipindex,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": realize_instances,
            "Position": sample_index_2,
        },
    )

    set_spline_type_1 = nw.new_node(
        Nodes.SplineType,
        input_kwargs={"Curve": set_position},
        attrs={"spline_type": "CATMULL_ROM"},
    )

    set_spline_type = nw.new_node(
        Nodes.SplineType,
        input_kwargs={"Curve": set_position},
        attrs={"spline_type": "NURBS"},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            0: group_input.outputs["Use Nurb"],
            1: set_spline_type_1,
            2: set_spline_type,
        },
    )

    resample_curve_1 = nw.new_node(
        Nodes.ResampleCurve,
        input_kwargs={
            "Curve": switch,
            "Count": group_input.outputs["V Resolution"],
        },
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    flipindex_1 = nw.new_node(
        nodegroup_flip_index().name,
        input_kwargs={
            "V Resolution": group_input.outputs["U Resolution"],
            "U Resolution": group_input.outputs["V Resolution"],
        },
    )

    sample_index_3 = nw.new_node(
        Nodes.SampleIndex,
        input_kwargs={
            "Geometry": resample_curve_1,
            "Value": position_1,
            "Index": flipindex_1,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": cylinderside.outputs["Geometry"],
            "Position": sample_index_3,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": set_position_1,
            "Top": cylinderside.outputs["Top"],
            "Side": cylinderside.outputs["Side"],
            "Bottom": cylinderside.outputs["Bottom"],
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_lofting_poly", singleton=False, type="GeometryNodeTree"
)
def nodegroup_lofting_poly(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Profile Curves", None),
            ("NodeSocketInt", "U Resolution", 32),
            ("NodeSocketInt", "V Resolution", 32),
            ("NodeSocketBool", "Use Nurb", False),
        ],
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["V Resolution"]}
    )

    cylinderside_001 = nw.new_node(
        nodegroup_cylinder_side().name,
        input_kwargs={
            "U Resolution": group_input.outputs["U Resolution"],
            "V Resolution": reroute_2,
        },
    )

    index = nw.new_node(Nodes.Index)

    evaluate_on_domain = nw.new_node(
        Nodes.EvaluateonDomain,
        input_kwargs={0: index},
        attrs={"domain": "CURVE", "data_type": "INT"},
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={"A": evaluate_on_domain},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    curve_line = nw.new_node(Nodes.CurveLine)

    domain_size = nw.new_node(
        Nodes.DomainSize,
        input_kwargs={"Geometry": group_input.outputs["Profile Curves"]},
        attrs={"component": "CURVE"},
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve,
        input_kwargs={
            "Curve": curve_line,
            "Count": domain_size.outputs["Spline Count"],
        },
    )

    instance_on_points_1 = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": group_input.outputs["Profile Curves"],
            "Selection": equal,
            "Instance": resample_curve,
        },
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points_1}
    )

    position = nw.new_node(Nodes.InputPosition)

    flipindex_001 = nw.new_node(
        nodegroup_flip_index().name,
        input_kwargs={
            "V Resolution": domain_size.outputs["Spline Count"],
            "U Resolution": group_input.outputs["U Resolution"],
        },
    )

    sample_index_2 = nw.new_node(
        Nodes.SampleIndex,
        input_kwargs={
            "Geometry": group_input.outputs["Profile Curves"],
            "Value": position,
            "Index": flipindex_001,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": realize_instances,
            "Position": sample_index_2,
        },
    )

    set_spline_type_1 = nw.new_node(
        Nodes.SplineType, input_kwargs={"Curve": set_position}
    )

    set_spline_type = nw.new_node(
        Nodes.SplineType,
        input_kwargs={"Curve": set_position},
        attrs={"spline_type": "NURBS"},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            0: group_input.outputs["Use Nurb"],
            1: set_spline_type_1,
            2: set_spline_type,
        },
    )

    resample_curve_1 = nw.new_node(
        Nodes.ResampleCurve,
        input_kwargs={"Curve": switch, "Count": reroute_2},
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    flipindex_001_1 = nw.new_node(
        nodegroup_flip_index().name,
        input_kwargs={
            "V Resolution": group_input.outputs["U Resolution"],
            "U Resolution": reroute_2,
        },
    )

    sample_index_3 = nw.new_node(
        Nodes.SampleIndex,
        input_kwargs={
            "Geometry": resample_curve_1,
            "Value": position_1,
            "Index": flipindex_001_1,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": cylinderside_001.outputs["Geometry"],
            "Position": sample_index_3,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": set_position_1,
            "Top": cylinderside_001.outputs["Top"],
            "Side": cylinderside_001.outputs["Side"],
            "Bottom": cylinderside_001.outputs["Bottom"],
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_warp_around_curve", singleton=False, type="GeometryNodeTree"
)
def nodegroup_warp_around_curve(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketGeometry", "Curve", None),
            ("NodeSocketInt", "Curve Resolution", 1024),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Curve Resolution"], 1: 1.0000}
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve,
        input_kwargs={"Curve": group_input.outputs["Curve"], "Count": add},
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    position_2 = nw.new_node(Nodes.InputPosition)

    separate_xyz_3 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_2})

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": group_input.outputs["Geometry"]}
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Min"]}
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Max"]}
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": separate_xyz_3.outputs["Z"],
            1: separate_xyz_1.outputs["Z"],
            2: separate_xyz_2.outputs["Z"],
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Curve Resolution"],
            1: map_range.outputs["Result"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    round = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply}, attrs={"operation": "ROUND"}
    )

    sample_index_3 = nw.new_node(
        Nodes.SampleIndex,
        input_kwargs={"Geometry": resample_curve, "Value": position_1, "Index": round},
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    normal = nw.new_node(Nodes.InputNormal)

    sample_index_5 = nw.new_node(
        Nodes.SampleIndex,
        input_kwargs={"Geometry": resample_curve, "Value": normal, "Index": round},
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: sample_index_5, "Scale": separate_xyz.outputs["X"]},
        attrs={"operation": "SCALE"},
    )

    curve_tangent = nw.new_node(Nodes.CurveTangent)

    sample_index_4 = nw.new_node(
        Nodes.SampleIndex,
        input_kwargs={
            "Geometry": resample_curve,
            "Value": curve_tangent,
            "Index": round,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    cross_product = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: sample_index_4, 1: sample_index_5},
        attrs={"operation": "CROSS_PRODUCT"},
    )

    scale_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: cross_product.outputs["Vector"],
            "Scale": separate_xyz.outputs["Y"],
        },
        attrs={"operation": "SCALE"},
    )

    add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 1: scale_1.outputs["Vector"]},
    )

    add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: sample_index_3, 1: add_1.outputs["Vector"]},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Position": add_2.outputs["Vector"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_position},
        attrs={"is_active_output": True},
    )
