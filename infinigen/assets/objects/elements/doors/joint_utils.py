# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Yiming Zuo: primary author
# Acknowledgment: This file draws inspiration
# from https://www.youtube.com/watch?v=o50FE2W1m8Y
# by Open Class

import bpy

from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


@node_utils.to_nodegroup(
    "nodegroup_add_geometry_metadata", singleton=False, type="GeometryNodeTree"
)
def nodegroup_add_geometry_metadata(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
            ("NodeSocketInt", "Value", 1),
        ],
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Name": group_input.outputs["Label"],
            "Value": group_input.outputs["Value"],
        },
        attrs={"data_type": "INT"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": store_named_attribute},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_symmetry_along_y", singleton=False, type="GeometryNodeTree"
)
def nodegroup_symmetry_along_y(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": group_input, "Scale": (1.0000, -1.0000, 1.0000)},
    )

    flip_faces = nw.new_node(
        Nodes.FlipFaces, input_kwargs={"Mesh": transform_geometry_2}
    )

    handle_1 = nw.new_node(
        nodegroup_add_geometry_metadata().name,
        input_kwargs={"Geometry": group_input, "Label": "handle", "Value": 1},
    )

    handle_2 = nw.new_node(
        nodegroup_add_geometry_metadata().name,
        input_kwargs={"Geometry": flip_faces, "Label": "handle", "Value": 2},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [handle_1, handle_2]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_1},
        attrs={"is_active_output": True},
    )


def nodegroup_arc_on_door_warper(
    nw,
    door_width,
    door_depth,
):
    arc_on_door = nw.new_node(
        nodegroup_arc_on_door().name,
        input_kwargs={"door_width": door_width, "door_depth": door_depth},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput, input_kwargs={"Geometry": arc_on_door.outputs["Geometry"]}
    )


@node_utils.to_nodegroup(
    "nodegroup_arc_on_door", singleton=False, type="GeometryNodeTree"
)
def nodegroup_arc_on_door(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "door_width", 0.0000),
            ("NodeSocketFloat", "door_depth", 0.0000),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["door_depth"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["door_depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_1})

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_1, "End": combine_xyz}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["door_width"]},
        attrs={"operation": "MULTIPLY"},
    )

    arc = nw.new_node(
        "GeometryNodeCurveArc",
        input_kwargs={
            "Resolution": 32,
            "Radius": multiply_2,
            "Sweep Angle": 3.1416,
            "Connect Center": True,
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": arc.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": curve_to_mesh, "Rotation": (1.5708, 0.0000, 0.0000)},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry},
        attrs={"is_active_output": True},
    )


def nodegroup_door_frame_warper(
    nw, full_frame, top_dome, door_width, door_height, door_depth, frame_width
):
    door_frame = nw.new_node(
        nodegroup_door_frame().name,
        input_kwargs={
            "full_frame": full_frame,
            "top_dome": top_dome,
            "door_width": door_width,
            "door_height": door_height,
            "door_depth": door_depth,
            "frame_width": frame_width,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput, input_kwargs={"Geometry": door_frame.outputs["Geometry"]}
    )


@node_utils.to_nodegroup(
    "nodegroup_door_frame", singleton=False, type="GeometryNodeTree"
)
def nodegroup_door_frame(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketBool", "full_frame", False),
            ("NodeSocketBool", "top_dome", False),
            ("NodeSocketFloat", "door_width", 0.0000),
            ("NodeSocketFloat", "door_height", 0.0000),
            ("NodeSocketFloat", "door_depth", 0.0000),
            ("NodeSocketFloat", "frame_width", 0.0000),
        ],
    )

    added_height = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["full_frame"],
            1: group_input.outputs["frame_width"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: added_height, 1: group_input.outputs["door_height"]},
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_1})

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz_1})

    quadrilateral_1 = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Width": group_input.outputs["frame_width"],
            "Height": group_input.outputs["door_depth"],
        },
    )

    curve_to_mesh_2 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": quadrilateral_1,
            "Fill Caps": True,
        },
    )

    add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["frame_width"],
            1: group_input.outputs["door_width"],
        },
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add_1})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply})

    curve_line_1 = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_1, "Z": reroute_1}
    )

    curve_line_2 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz, "End": combine_xyz_2}
    )

    curve_line_3 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_1, "End": combine_xyz_2}
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute}, attrs={"operation": "MULTIPLY"}
    )

    arc = nw.new_node(
        "GeometryNodeCurveArc",
        input_kwargs={
            "Radius": multiply_2,
            "Start Angle": 1.5708,
            "Sweep Angle": 3.1416,
        },
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_2, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_3, "Z": reroute_1}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": arc.outputs["Curve"],
            "Translation": combine_xyz_3,
            "Rotation": (1.5708, 1.5708, 0.0000),
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["top_dome"],
            "False": curve_line_3,
            "True": transform_geometry,
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [curve_line_1, curve_line, curve_line_2, switch]},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh, input_kwargs={"Curve": join_geometry}
    )

    merge_by_distance = nw.new_node(
        Nodes.MergeByDistance, input_kwargs={"Geometry": curve_to_mesh}
    )

    mesh_to_curve = nw.new_node(
        Nodes.MeshToCurve, input_kwargs={"Mesh": merge_by_distance}
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["frame_width"], 1: 1.4142},
        attrs={"operation": "MULTIPLY"},
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": group_input.outputs["door_depth"], "Height": multiply_4},
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": mesh_to_curve,
            "Profile Curve": quadrilateral,
            "Fill Caps": True,
        },
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["full_frame"],
            "False": curve_to_mesh_2,
            "True": curve_to_mesh_1,
        },
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["frame_width"], 1: 1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["frame_width"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_5, "Z": multiply_6}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": switch_6, "Translation": combine_xyz_4},
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["full_frame"], 1: multiply_7},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_8, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["frame_width"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_9, 1: multiply_10})

    multiply_11 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute, 1: -0.2500},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_12 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["top_dome"], 1: multiply_11},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_13 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["full_frame"], 1: multiply_12},
        attrs={"operation": "MULTIPLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": transform_geometry_1,
            "center_x_offset": add_2,
            "center_z_offset": multiply_13,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("Hinge Joint", singleton=False, type="GeometryNodeTree")
def nodegroup_hinge_joint(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketString", "Joint ID (do not set)", ""),
            ("NodeSocketString", "Joint Label", ""),
            ("NodeSocketString", "Parent Label", ""),
            ("NodeSocketGeometry", "Parent", None),
            ("NodeSocketString", "Child Label", ""),
            ("NodeSocketGeometry", "Child", None),
            ("NodeSocketVector", "Position", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "Axis", (0.0000, 0.0000, 1.0000)),
            ("NodeSocketFloat", "Value", 0.0000),
            ("NodeSocketFloat", "Min", 0.0000),
            ("NodeSocketFloat", "Max", 0.0000),
            ("NodeSocketBool", "Show Center of Parent", False),
            ("NodeSocketBool", "Show Center of Child", False),
            ("NodeSocketBool", "Show Joint", False),
        ],
    )

    named_attribute_4 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    integer = nw.new_node(Nodes.Integer)
    integer.integer = 0

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": named_attribute_4.outputs["Exists"],
            "False": integer,
            "True": named_attribute_4.outputs["Attribute"],
        },
        attrs={"input_type": "INT"},
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Parent"],
            "Name": "part_id",
            "Value": switch_2,
        },
        attrs={"data_type": "INT"},
    )

    named_attribute_1 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    attribute_statistic = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": store_named_attribute_1,
            "Attribute": named_attribute_1.outputs["Attribute"],
        },
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_1.outputs["Attribute"],
            3: attribute_statistic.outputs["Min"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_2 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": store_named_attribute_1, "Selection": equal},
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                separate_geometry_2.outputs["Selection"],
                separate_geometry_2.outputs["Inverted"],
            ]
        },
    )

    named_attribute_3 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    integer_1 = nw.new_node(Nodes.Integer)
    integer_1.integer = 1

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: named_attribute_3.outputs["Attribute"], 1: 1.0000}
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": named_attribute_3.outputs["Exists"],
            "False": integer_1,
            "True": add,
        },
        attrs={"input_type": "INT"},
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Child"],
            "Name": "part_id",
            "Value": switch_3,
        },
        attrs={"data_type": "INT"},
    )

    named_attribute_2 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    attribute_statistic_1 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": store_named_attribute,
            "Attribute": named_attribute_2.outputs["Attribute"],
        },
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_2.outputs["Attribute"],
            3: attribute_statistic_1.outputs["Min"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_3 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": store_named_attribute, "Selection": equal_1},
    )

    named_attribute_11 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "is_jointed"},
        attrs={"data_type": "BOOLEAN"},
    )

    attribute_statistic_7 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": separate_geometry_3.outputs["Selection"],
            "Attribute": named_attribute_11.outputs["Attribute"],
        },
    )

    greater_than = nw.new_node(
        Nodes.Compare,
        input_kwargs={1: 1.0000, 2: attribute_statistic_7.outputs["Sum"]},
        attrs={"data_type": "INT"},
    )

    combine_matrix = nw.new_node("FunctionNodeCombineMatrix")

    named_attribute_10 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": greater_than,
            "False": combine_matrix,
            "True": named_attribute_10.outputs["Attribute"],
        },
        attrs={"input_type": "MATRIX"},
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute,
            "Name": "body_transform",
            "Value": switch_1,
        },
        attrs={"data_type": "FLOAT4X4"},
    )

    named_attribute_7 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "is_jointed"},
        attrs={"data_type": "BOOLEAN"},
    )

    attribute_statistic_4 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": separate_geometry_3.outputs["Selection"],
            "Attribute": named_attribute_7.outputs["Attribute"],
        },
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: attribute_statistic_4.outputs["Sum"]},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    position_4 = nw.new_node(Nodes.InputPosition)

    position_1 = nw.new_node(Nodes.InputPosition)

    bounding_box = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry_2.outputs["Selection"]},
    )

    position = nw.new_node(Nodes.InputPosition)

    attribute_statistic_2 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": position,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: position_1, 1: attribute_statistic_2.outputs["Mean"]},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_2,
            "False": position_4,
            "True": add_1.outputs["Vector"],
        },
        attrs={"input_type": "VECTOR"},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": store_named_attribute_2, "Position": switch},
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_position, "Name": "is_jointed", "Value": True},
        attrs={"data_type": "BOOLEAN"},
    )

    position_3 = nw.new_node(Nodes.InputPosition)

    named_attribute_12 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    attribute_statistic_6 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": set_position,
            "Attribute": named_attribute_12.outputs["Attribute"],
        },
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_12.outputs["Attribute"],
            3: attribute_statistic_6.outputs["Min"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_4 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": set_position, "Selection": equal_3},
    )

    bounding_box_1 = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry_4.outputs["Selection"]},
    )

    position_2 = nw.new_node(Nodes.InputPosition)

    attribute_statistic_5 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box_1.outputs["Bounding Box"],
            "Attribute": position_2,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    named_attribute_5 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    transpose_matrix = nw.new_node(
        "FunctionNodeTransposeMatrix",
        input_kwargs={"Matrix": named_attribute_5.outputs["Attribute"]},
    )

    transform_point = nw.new_node(
        "FunctionNodeTransformPoint",
        input_kwargs={
            "Vector": group_input.outputs["Position"],
            "Transform": transpose_matrix,
        },
    )

    add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: attribute_statistic_5.outputs["Mean"], 1: transform_point},
    )

    transform_direction = nw.new_node(
        "FunctionNodeTransformDirection",
        input_kwargs={
            "Direction": group_input.outputs["Axis"],
            "Transform": transpose_matrix,
        },
    )

    equal_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: group_input.outputs["Min"], "Epsilon": 0.0000},
        attrs={"operation": "EQUAL"},
    )

    equal_5 = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: group_input.outputs["Max"], "Epsilon": 0.0000},
        attrs={"operation": "EQUAL"},
    )

    op_and = nw.new_node(Nodes.BooleanMath, input_kwargs={0: equal_4, 1: equal_5})

    clamp = nw.new_node(
        Nodes.Clamp,
        input_kwargs={
            "Value": group_input.outputs["Value"],
            "Min": group_input.outputs["Min"],
            "Max": group_input.outputs["Max"],
        },
    )

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": op_and,
            "False": clamp,
            "True": group_input.outputs["Value"],
        },
        attrs={"input_type": "FLOAT"},
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_5})

    vector_rotate = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={
            "Vector": position_3,
            "Center": add_2.outputs["Vector"],
            "Axis": transform_direction,
            "Angle": reroute,
        },
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": store_named_attribute_3, "Position": vector_rotate},
    )

    named_attribute_6 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    separate_matrix = nw.new_node(
        "FunctionNodeSeparateMatrix",
        input_kwargs={"Matrix": named_attribute_6.outputs["Attribute"]},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_matrix.outputs["Column 1 Row 1"],
            "Y": separate_matrix.outputs["Column 2 Row 1"],
            "Z": separate_matrix.outputs["Column 3 Row 1"],
        },
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_direction})

    vector_rotate_1 = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={"Vector": combine_xyz, "Axis": reroute_1, "Angle": reroute},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": vector_rotate_1}
    )

    named_attribute_8 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    separate_matrix_1 = nw.new_node(
        "FunctionNodeSeparateMatrix",
        input_kwargs={"Matrix": named_attribute_8.outputs["Attribute"]},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_matrix_1.outputs["Column 1 Row 2"],
            "Y": separate_matrix_1.outputs["Column 2 Row 2"],
            "Z": separate_matrix_1.outputs["Column 3 Row 2"],
        },
    )

    vector_rotate_2 = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={"Vector": combine_xyz_1, "Axis": reroute_1, "Angle": reroute},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": vector_rotate_2}
    )

    named_attribute_9 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    separate_matrix_2 = nw.new_node(
        "FunctionNodeSeparateMatrix",
        input_kwargs={"Matrix": named_attribute_9.outputs["Attribute"]},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_matrix_2.outputs["Column 1 Row 3"],
            "Y": separate_matrix_2.outputs["Column 2 Row 3"],
            "Z": separate_matrix_2.outputs["Column 3 Row 3"],
        },
    )

    vector_rotate_3 = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={"Vector": combine_xyz_2, "Axis": reroute_1, "Angle": reroute},
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": vector_rotate_3}
    )

    combine_matrix_1 = nw.new_node(
        "FunctionNodeCombineMatrix",
        input_kwargs={
            "Column 1 Row 1": separate_xyz.outputs["X"],
            "Column 1 Row 2": separate_xyz_1.outputs["X"],
            "Column 1 Row 3": separate_xyz_2.outputs["X"],
            "Column 2 Row 1": separate_xyz.outputs["Y"],
            "Column 2 Row 2": separate_xyz_1.outputs["Y"],
            "Column 2 Row 3": separate_xyz_2.outputs["Y"],
            "Column 3 Row 1": separate_xyz.outputs["Z"],
            "Column 3 Row 2": separate_xyz_1.outputs["Z"],
            "Column 3 Row 3": separate_xyz_2.outputs["Z"],
        },
    )

    store_named_attribute_4 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": set_position_1,
            "Name": "body_transform",
            "Value": combine_matrix_1,
        },
        attrs={"data_type": "FLOAT4X4"},
    )

    string_1 = nw.new_node("FunctionNodeInputString", attrs={"string": "pos"})

    reroute_4 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Joint ID (do not set)"]},
    )

    join_strings_1 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_1, reroute_4]},
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Position"]}
    )

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_4,
            "Name": join_strings_1,
            "Value": reroute_3,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    string_2 = nw.new_node("FunctionNodeInputString", attrs={"string": "axis"})

    join_strings_2 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_2, reroute_4]},
    )

    store_named_attribute_6 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_5,
            "Name": join_strings_2,
            "Value": group_input.outputs["Axis"],
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    string_3 = nw.new_node("FunctionNodeInputString", attrs={"string": "min"})

    join_strings_3 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_3, reroute_4]},
    )

    store_named_attribute_8 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_6,
            "Name": join_strings_3,
            "Value": group_input.outputs["Min"],
        },
    )

    string_4 = nw.new_node("FunctionNodeInputString", attrs={"string": "max"})

    join_strings_4 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_4, reroute_4]},
    )

    store_named_attribute_7 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_8,
            "Name": join_strings_4,
            "Value": group_input.outputs["Max"],
        },
    )

    uv_sphere = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Segments": 10, "Rings": 10, "Radius": 0.0500}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere.outputs["Mesh"],
            "Translation": attribute_statistic_2.outputs["Mean"],
        },
    )

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Center of Parent"],
            "True": transform_geometry,
        },
    )

    store_named_attribute_13 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_4, "Name": "part_id", "Value": 999999999},
        attrs={"data_type": "INT"},
    )

    uv_sphere_1 = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Segments": 10, "Rings": 10, "Radius": 0.0500}
    )

    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_position_1})

    named_attribute_13 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    attribute_statistic_10 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": reroute_2,
            "Attribute": named_attribute_13.outputs["Attribute"],
        },
    )

    equal_6 = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_13.outputs["Attribute"],
            3: attribute_statistic_10.outputs["Min"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_5 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": reroute_2, "Selection": equal_6},
    )

    bounding_box_3 = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry_5.outputs["Selection"]},
    )

    position_7 = nw.new_node(Nodes.InputPosition)

    attribute_statistic_9 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box_3.outputs["Bounding Box"],
            "Attribute": position_7,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere_1.outputs["Mesh"],
            "Translation": attribute_statistic_9.outputs["Mean"],
        },
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Center of Child"],
            "True": transform_geometry_1,
        },
    )

    store_named_attribute_14 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_6, "Name": "part_id", "Value": 999999999},
        attrs={"data_type": "INT"},
    )

    cone = nw.new_node(
        "GeometryNodeMeshCone", input_kwargs={"Radius Bottom": 0.0500, "Depth": 0.2000}
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cone.outputs["Mesh"],
            "Translation": (0.0000, 0.0000, -0.0500),
        },
    )

    bounding_box_4 = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry_4.outputs["Selection"]},
    )

    position_8 = nw.new_node(Nodes.InputPosition)

    attribute_statistic_11 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box_4.outputs["Bounding Box"],
            "Attribute": position_8,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    add_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Position"],
            1: attribute_statistic_11.outputs["Mean"],
        },
    )

    attribute_statistic_12 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": separate_geometry_5.outputs["Selection"],
            "Attribute": transform_direction,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    normalize = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: attribute_statistic_12.outputs["Mean"]},
        attrs={"operation": "NORMALIZE"},
    )

    align_rotation_to_vector_1 = nw.new_node(
        "FunctionNodeAlignRotationToVector",
        input_kwargs={"Vector": normalize.outputs["Vector"]},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_3,
            "Translation": add_3.outputs["Vector"],
            "Rotation": align_rotation_to_vector_1,
        },
    )

    switch_7 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Joint"],
            "True": transform_geometry_2,
        },
    )

    store_named_attribute_15 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_7, "Name": "part_id", "Value": 999999999},
        attrs={"data_type": "INT"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                join_geometry_2,
                store_named_attribute_7,
                store_named_attribute_13,
                store_named_attribute_14,
                store_named_attribute_15,
            ]
        },
    )

    store_named_attribute_9 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_4,
            "Name": join_strings_1,
            "Value": reroute_3,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute_10 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_9,
            "Name": join_strings_2,
            "Value": group_input.outputs["Axis"],
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute_12 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_10,
            "Name": join_strings_3,
            "Value": group_input.outputs["Min"],
        },
    )

    store_named_attribute_11 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_12,
            "Name": join_strings_4,
            "Value": group_input.outputs["Max"],
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                store_named_attribute_11,
                store_named_attribute_13,
                store_named_attribute_14,
                store_named_attribute_15,
            ]
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": join_geometry,
            "Parent": join_geometry_2,
            "Child": join_geometry_1,
        },
        attrs={"is_active_output": True},
    )


def geometry_node_join(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    object_info_2 = nw.new_node(
        Nodes.ObjectInfo, input_kwargs={"Object": bpy.data.objects["Cube.001"]}
    )

    object_info = nw.new_node(
        Nodes.ObjectInfo, input_kwargs={"Object": bpy.data.objects["Cube"]}
    )

    object_info_1 = nw.new_node(
        Nodes.ObjectInfo, input_kwargs={"Object": bpy.data.objects["Mesh"]}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                object_info.outputs["Geometry"],
                object_info_1.outputs["Geometry"],
            ]
        },
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry,
            "Translation": (0.0000, 0.0600, -1.1200),
        },
    )

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Parent": object_info_2.outputs["Geometry"],
            "Child": transform_geometry,
            "Position": (0.4000, 0.0000, 0.0000),
            "Value": 1.0000,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": hinge_joint.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("Sliding Joint", singleton=False, type="GeometryNodeTree")
def nodegroup_sliding_joint(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketString", "Joint ID (do not set)", ""),
            ("NodeSocketString", "Joint Label", ""),
            ("NodeSocketString", "Parent Label", ""),
            ("NodeSocketGeometry", "Parent", None),
            ("NodeSocketString", "Child Label", ""),
            ("NodeSocketGeometry", "Child", None),
            ("NodeSocketVector", "Position", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "Axis", (0.0000, 0.0000, 1.0000)),
            ("NodeSocketFloat", "Value", 0.0000),
            ("NodeSocketFloat", "Min", 0.0000),
            ("NodeSocketFloat", "Max", 0.0000),
            ("NodeSocketBool", "Show Center of Parent", False),
            ("NodeSocketBool", "Show Center of Child", False),
            ("NodeSocketBool", "Show Joint", False),
        ],
    )

    named_attribute_4 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    integer = nw.new_node(Nodes.Integer)
    integer.integer = 0

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": named_attribute_4.outputs["Exists"],
            "False": integer,
            "True": named_attribute_4.outputs["Attribute"],
        },
        attrs={"input_type": "INT"},
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Parent"],
            "Name": "part_id",
            "Value": switch_2,
        },
        attrs={"data_type": "INT"},
    )

    named_attribute_1 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    attribute_statistic = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": store_named_attribute_1,
            "Attribute": named_attribute_1.outputs["Attribute"],
        },
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_1.outputs["Attribute"],
            3: attribute_statistic.outputs["Min"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_2 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": store_named_attribute_1, "Selection": equal},
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                separate_geometry_2.outputs["Selection"],
                separate_geometry_2.outputs["Inverted"],
            ]
        },
    )

    cone = nw.new_node(
        "GeometryNodeMeshCone", input_kwargs={"Radius Bottom": 0.0500, "Depth": 0.2000}
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cone.outputs["Mesh"],
            "Translation": (0.0000, 0.0000, -0.0500),
        },
    )

    named_attribute_3 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    integer_1 = nw.new_node(Nodes.Integer)
    integer_1.integer = 1

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: named_attribute_3.outputs["Attribute"], 1: 1.0000}
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": named_attribute_3.outputs["Exists"],
            "False": integer_1,
            "True": add,
        },
        attrs={"input_type": "INT"},
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Child"],
            "Name": "part_id",
            "Value": switch_3,
        },
        attrs={"data_type": "INT"},
    )

    named_attribute_2 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    attribute_statistic_1 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": store_named_attribute,
            "Attribute": named_attribute_2.outputs["Attribute"],
        },
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_2.outputs["Attribute"],
            3: attribute_statistic_1.outputs["Min"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_3 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": store_named_attribute, "Selection": equal_1},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_geometry_3.outputs["Selection"]}
    )

    bounding_box_3 = nw.new_node(Nodes.BoundingBox, input_kwargs={"Geometry": reroute})

    position_7 = nw.new_node(Nodes.InputPosition)

    attribute_statistic_9 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box_3.outputs["Bounding Box"],
            "Attribute": position_7,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Position"],
            1: attribute_statistic_9.outputs["Mean"],
        },
    )

    named_attribute_5 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    transpose_matrix = nw.new_node(
        "FunctionNodeTransposeMatrix",
        input_kwargs={"Matrix": named_attribute_5.outputs["Attribute"]},
    )

    transform_direction = nw.new_node(
        "FunctionNodeTransformDirection",
        input_kwargs={
            "Direction": group_input.outputs["Axis"],
            "Transform": transpose_matrix,
        },
    )

    attribute_statistic_5 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={"Geometry": reroute, "Attribute": transform_direction},
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    normalize = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: attribute_statistic_5.outputs["Mean"]},
        attrs={"operation": "NORMALIZE"},
    )

    align_rotation_to_vector_1 = nw.new_node(
        "FunctionNodeAlignRotationToVector",
        input_kwargs={"Vector": normalize.outputs["Vector"]},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_3,
            "Translation": add_1.outputs["Vector"],
            "Rotation": align_rotation_to_vector_1,
        },
    )

    switch_7 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Joint"],
            "True": transform_geometry_2,
        },
    )

    store_named_attribute_15 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_7, "Name": "part_id", "Value": 999999999},
        attrs={"data_type": "INT"},
    )

    uv_sphere_1 = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Segments": 10, "Rings": 10, "Radius": 0.0500}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere_1.outputs["Mesh"],
            "Translation": attribute_statistic_9.outputs["Mean"],
        },
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Center of Child"],
            "True": transform_geometry_1,
        },
    )

    store_named_attribute_14 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_6, "Name": "part_id", "Value": 999999999},
        attrs={"data_type": "INT"},
    )

    uv_sphere = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Segments": 10, "Rings": 10, "Radius": 0.0500}
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry_2.outputs["Selection"]},
    )

    position = nw.new_node(Nodes.InputPosition)

    attribute_statistic_2 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": position,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere.outputs["Mesh"],
            "Translation": attribute_statistic_2.outputs["Mean"],
        },
    )

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Center of Parent"],
            "True": transform_geometry,
        },
    )

    store_named_attribute_13 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_4, "Name": "part_id", "Value": 999999999},
        attrs={"data_type": "INT"},
    )

    named_attribute_11 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "is_jointed"},
        attrs={"data_type": "BOOLEAN"},
    )

    attribute_statistic_7 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": separate_geometry_3.outputs["Selection"],
            "Attribute": named_attribute_11.outputs["Attribute"],
        },
    )

    greater_than = nw.new_node(
        Nodes.Compare,
        input_kwargs={1: 1.0000, 2: attribute_statistic_7.outputs["Sum"]},
        attrs={"data_type": "INT"},
    )

    combine_matrix = nw.new_node("FunctionNodeCombineMatrix")

    named_attribute_10 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": greater_than,
            "False": combine_matrix,
            "True": named_attribute_10.outputs["Attribute"],
        },
        attrs={"input_type": "MATRIX"},
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute,
            "Name": "body_transform",
            "Value": switch_1,
        },
        attrs={"data_type": "FLOAT4X4"},
    )

    named_attribute_7 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "is_jointed"},
        attrs={"data_type": "BOOLEAN"},
    )

    attribute_statistic_4 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": separate_geometry_3.outputs["Selection"],
            "Attribute": named_attribute_7.outputs["Attribute"],
        },
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: attribute_statistic_4.outputs["Sum"]},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    position_4 = nw.new_node(Nodes.InputPosition)

    position_1 = nw.new_node(Nodes.InputPosition)

    add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: position_1, 1: attribute_statistic_2.outputs["Mean"]},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_2,
            "False": position_4,
            "True": add_2.outputs["Vector"],
        },
        attrs={"input_type": "VECTOR"},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": store_named_attribute_2, "Position": switch},
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_position, "Name": "is_jointed", "Value": True},
        attrs={"data_type": "BOOLEAN"},
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: group_input.outputs["Min"], "Epsilon": 0.0000},
        attrs={"operation": "EQUAL"},
    )

    equal_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: group_input.outputs["Max"], "Epsilon": 0.0000},
        attrs={"operation": "EQUAL"},
    )

    op_and = nw.new_node(Nodes.BooleanMath, input_kwargs={0: equal_3, 1: equal_4})

    clamp = nw.new_node(
        Nodes.Clamp,
        input_kwargs={
            "Value": group_input.outputs["Value"],
            "Min": group_input.outputs["Min"],
            "Max": group_input.outputs["Max"],
        },
    )

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": op_and,
            "False": clamp,
            "True": group_input.outputs["Value"],
        },
        attrs={"input_type": "FLOAT"},
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: transform_direction, "Scale": switch_5},
        attrs={"operation": "SCALE"},
    )

    position_5 = nw.new_node(Nodes.InputPosition)

    add_3 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: scale.outputs["Vector"], 1: position_5}
    )

    set_position_2 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": store_named_attribute_3,
            "Position": add_3.outputs["Vector"],
        },
    )

    string = nw.new_node("FunctionNodeInputString", attrs={"string": "pos"})

    reroute_2 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Joint ID (do not set)"]},
    )

    join_strings = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string, reroute_2]},
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Position"]}
    )

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": set_position_2,
            "Name": join_strings,
            "Value": reroute_1,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    string_1 = nw.new_node("FunctionNodeInputString", attrs={"string": "axis"})

    join_strings_1 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_1, reroute_2]},
    )

    store_named_attribute_6 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_5,
            "Name": join_strings_1,
            "Value": group_input.outputs["Axis"],
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    string_2 = nw.new_node("FunctionNodeInputString", attrs={"string": "min"})

    join_strings_2 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_2, reroute_2]},
    )

    store_named_attribute_8 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_6,
            "Name": join_strings_2,
            "Value": group_input.outputs["Min"],
        },
    )

    string_3 = nw.new_node("FunctionNodeInputString", attrs={"string": "max"})

    join_strings_3 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_3, reroute_2]},
    )

    store_named_attribute_7 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_8,
            "Name": join_strings_3,
            "Value": group_input.outputs["Max"],
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                join_geometry_2,
                store_named_attribute_15,
                store_named_attribute_14,
                store_named_attribute_13,
                store_named_attribute_7,
            ]
        },
    )

    store_named_attribute_9 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": set_position_2,
            "Name": join_strings,
            "Value": reroute_1,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute_10 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_9,
            "Name": join_strings_1,
            "Value": group_input.outputs["Axis"],
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute_12 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_10,
            "Name": join_strings_2,
            "Value": group_input.outputs["Min"],
        },
    )

    store_named_attribute_11 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_12,
            "Name": join_strings_3,
            "Value": group_input.outputs["Max"],
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                store_named_attribute_15,
                store_named_attribute_14,
                store_named_attribute_13,
                store_named_attribute_11,
            ]
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": join_geometry,
            "Parent": join_geometry_2,
            "Child": join_geometry_1,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "Duplicate Joints on Parent", singleton=False, type="GeometryNodeTree"
)
def nodegroup_duplicate_joints_on_parent(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketString", "Duplicate ID (do not set)", ""),
            ("NodeSocketGeometry", "Parent", None),
            ("NodeSocketGeometry", "Child", None),
            ("NodeSocketGeometry", "Points", None),
        ],
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": group_input.outputs["Points"],
            "Instance": group_input.outputs["Child"],
        },
    )

    index = nw.new_node(Nodes.Index)

    add = nw.new_node(Nodes.Math, input_kwargs={0: index, 1: 1.0000})

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": instance_on_points,
            "Name": group_input.outputs["Duplicate ID (do not set)"],
            "Value": add,
        },
        attrs={"domain": "INSTANCE", "data_type": "INT"},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": store_named_attribute}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [group_input.outputs["Parent"], realize_instances]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )
