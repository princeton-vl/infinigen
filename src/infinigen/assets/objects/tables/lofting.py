# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import bpy

from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


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
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
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


@node_utils.to_nodegroup(
    "nodegroup_shifted_square", singleton=False, type="GeometryNodeTree"
)
def nodegroup_shifted_square(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "Resolution", 10),
            ("NodeSocketFloat", "Width", 1.0000),
            ("NodeSocketFloat", "Z", 0.0000),
            ("NodeSocketFloat", "Rot Z", 0.5000),
        ],
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Width": group_input.outputs["Width"],
            "Height": group_input.outputs["Width"],
        },
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve,
        input_kwargs={
            "Curve": quadrilateral,
            "Count": group_input.outputs["Resolution"],
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

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": resample_curve,
            "Translation": combine_xyz,
            "Rotation": combine_xyz_1,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Curve": transform_geometry},
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
            "V Resolution": group_input,
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
        input_kwargs={"A": evaluate_on_domain.outputs[1]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    curve_line = nw.new_node(Nodes.CurveLine)

    domain_size = nw.new_node(
        Nodes.DomainSize,
        input_kwargs={"Geometry": group_input},
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
            "Points": group_input,
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
        input_kwargs={"Geometry": group_input, "Value": position, "Index": flipindex},
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
        input_kwargs={"Curve": switch, "Count": group_input},
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    flipindex_1 = nw.new_node(
        nodegroup_flip_index().name,
        input_kwargs={
            "V Resolution": group_input.outputs["U Resolution"],
            "U Resolution": group_input,
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
    "nodegroup_warp_around_curve", singleton=False, type="GeometryNodeTree"
)
def nodegroup_warp_around_curve(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketGeometry", "Curve", None),
            ("NodeSocketInt", "U Resolution", 32),
            ("NodeSocketInt", "V Resolution", 32),
            ("NodeSocketFloat", "Radius", 1.0000),
        ],
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve,
        input_kwargs={
            "Curve": group_input.outputs["Curve"],
            "Count": group_input.outputs["V Resolution"],
        },
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    index = nw.new_node(Nodes.Index)

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: index, 1: group_input.outputs["U Resolution"]},
        attrs={"operation": "DIVIDE"},
    )

    floor = nw.new_node(
        Nodes.Math, input_kwargs={0: divide}, attrs={"operation": "FLOOR"}
    )

    sample_index_3 = nw.new_node(
        Nodes.SampleIndex,
        input_kwargs={"Geometry": resample_curve, "Value": position_1, "Index": floor},
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    normal = nw.new_node(Nodes.InputNormal)

    sample_index_5 = nw.new_node(
        Nodes.SampleIndex,
        input_kwargs={"Geometry": resample_curve, "Value": normal, "Index": floor},
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
            "Index": floor,
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

    add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: scale.outputs["Vector"], 1: scale_1.outputs["Vector"]},
    )

    scale_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add.outputs["Vector"], "Scale": group_input.outputs["Radius"]},
        attrs={"operation": "SCALE"},
    )

    add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: sample_index_3, 1: scale_2.outputs["Vector"]},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Position": add_1.outputs["Vector"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_position},
        attrs={"is_active_output": True},
    )


def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    integer = nw.new_node(Nodes.Integer)
    integer.integer = 32

    shiftedsquare = nw.new_node(
        nodegroup_shifted_square().name, input_kwargs={"Resolution": integer}
    )

    shiftedcircle = nw.new_node(
        nodegroup_shifted_circle().name,
        input_kwargs={"Resolution": integer, "Radius": 0.9200, "Z": 2.5600},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": shiftedcircle, "Rotation": (0.0000, 0.0000, 0.7854)},
    )

    shiftedsquare_1 = nw.new_node(
        nodegroup_shifted_square().name,
        input_kwargs={"Resolution": integer, "Z": 10.0000},
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: integer, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    star = nw.new_node(
        "GeometryNodeCurveStar",
        input_kwargs={"Points": divide, "Inner Radius": 0.5000, "Outer Radius": 0.6600},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": star.outputs["Curve"],
            "Translation": (0.0000, 0.0000, 7.6000),
            "Rotation": (0.0000, 0.0000, 0.7854),
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                shiftedsquare,
                transform_geometry,
                shiftedsquare_1,
                transform_geometry_1,
            ]
        },
    )

    v_resolution = nw.new_node(Nodes.Integer, label="V Resolution")
    v_resolution.integer = 64

    lofting = nw.new_node(
        nodegroup_lofting().name,
        input_kwargs={
            "Profile Curves": join_geometry,
            "U Resolution": integer,
            "V Resolution": v_resolution,
        },
    )

    object_info = nw.new_node(
        Nodes.ObjectInfo, input_kwargs={"Object": bpy.data.objects["BezierCurve"]}
    )

    warparoundcurve = nw.new_node(
        nodegroup_warp_around_curve().name,
        input_kwargs={
            "Geometry": lofting.outputs["Geometry"],
            "Curve": object_info.outputs["Geometry"],
            "U Resolution": integer,
            "V Resolution": v_resolution,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": warparoundcurve},
        attrs={"is_active_output": True},
    )


def apply(obj, selection=None, **kwargs):
    surface.add_geomod(obj, geometry_nodes, selection=selection, attributes=[])


apply(bpy.context.active_object)
