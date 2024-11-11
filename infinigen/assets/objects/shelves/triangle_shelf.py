# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han

import bpy
import numpy as np
from numpy.random import normal, uniform

from infinigen.assets.materials.shelf_shaders import get_shelf_material
from infinigen.core import surface, tagging
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory


@node_utils.to_nodegroup(
    "nodegroup_table_profile", singleton=False, type="GeometryNodeTree"
)
def nodegroup_table_profile(nw: NodeWrangler):
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
    value.outputs[0].default_value = 0.7071

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
            "Count": 4,
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
    "nodegroup_curve_to_board", singleton=False, type="GeometryNodeTree"
)
def nodegroup_curve_to_board(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Profile Curve", None),
            ("NodeSocketGeometry", "Shape Curve", None),
            ("NodeSocketFloat", "Height", 0.5000),
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
        input_kwargs={"Curve": set_curve_tilt, "Count": 128},
    )

    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)

    capture_attribute = nw.new_node(
        Nodes.CaptureAttribute,
        input_kwargs={
            "Geometry": resample_curve,
            2: spline_parameter_1.outputs["Factor"],
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": capture_attribute.outputs["Geometry"],
            "Profile Curve": group_input.outputs["Shape Curve"],
            "Fill Caps": True,
        },
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_1})

    sample_curve = nw.new_node(
        Nodes.SampleCurve,
        input_kwargs={
            "Curve": group_input.outputs["Profile Curve"],
            "Factor": capture_attribute.outputs[1],
        },
        attrs={"mode": "FACTOR"},
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
            "Geometry": group_input.outputs["Profile Curve"],
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

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": set_position},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_leg_straight", singleton=False, type="GeometryNodeTree"
)
def nodegroup_leg_straight(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Profile Curve", None),
            ("NodeSocketFloat", "Height", 0.5000),
            ("NodeSocketInt", "N-gon", 0),
            ("NodeSocketFloat", "Profile Width", 0.5000),
            ("NodeSocketFloat", "Aspect Ratio", 0.5000),
            ("NodeSocketFloat", "Fillet Ratio", 0.2000),
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

    tableprofile = nw.new_node(
        nodegroup_table_profile().name,
        input_kwargs={
            "Profile N-gon": group_input.outputs["N-gon"],
            "Profile Width": group_input.outputs["Profile Width"],
            "Profile Aspect Ratio": group_input.outputs["Aspect Ratio"],
            "Profile Fillet Ratio": group_input.outputs["Fillet Ratio"],
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": capture_attribute.outputs["Geometry"],
            "Profile Curve": tableprofile,
            "Fill Caps": True,
        },
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_1})

    sample_curve = nw.new_node(
        Nodes.SampleCurve,
        input_kwargs={
            "Curve": group_input.outputs["Profile Curve"],
            "Factor": capture_attribute.outputs[1],
        },
        attrs={"mode": "FACTOR"},
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
            "Geometry": group_input.outputs["Profile Curve"],
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

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": set_position, "Profile Curve": tableprofile},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_curve_board", singleton=False, type="GeometryNodeTree"
)
def nodegroup_curve_board(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    curve_line = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (1.0000, 0.0000, -1.0000),
            "End": (1.0000, 0.0000, 1.0000),
        },
    )

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Thickness", 0.5000),
            ("NodeSocketFloat", "Fillet Radius Vertical", 0.0000),
            ("NodeSocketFloat", "width", 0.0000),
            ("NodeSocketFloat", "extrude_length", 0.0000),
        ],
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input.outputs["width"]}
    )

    curve_line_1 = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz_3})

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": group_input.outputs["width"]}
    )

    curve_line_2 = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz_4})

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["width"],
            "Y": group_input.outputs["extrude_length"],
        },
    )

    curve_line_3 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_3, "End": combine_xyz_6}
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["extrude_length"],
            "Y": group_input.outputs["width"],
        },
    )

    curve_line_4 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_4, "End": combine_xyz_5}
    )

    curve_line_5 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_6, "End": combine_xyz_5}
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                curve_line_1,
                curve_line_2,
                curve_line_3,
                curve_line_4,
                curve_line_5,
            ]
        },
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh, input_kwargs={"Curve": join_geometry_1}
    )

    merge_by_distance_1 = nw.new_node(
        Nodes.MergeByDistance, input_kwargs={"Geometry": curve_to_mesh_1}
    )

    mesh_to_curve = nw.new_node(
        Nodes.MeshToCurve, input_kwargs={"Mesh": merge_by_distance_1}
    )

    curve_to_board = nw.new_node(
        nodegroup_curve_to_board().name,
        input_kwargs={
            "Profile Curve": curve_line,
            "Shape Curve": mesh_to_curve,
            "Height": group_input.outputs["Thickness"],
        },
    )

    arc = nw.new_node(
        "GeometryNodeCurveArc",
        input_kwargs={"Resolution": 4, "Radius": 0.7071, "Sweep Angle": 4.7124},
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": arc.outputs["Curve"],
            "Rotation": (0.0000, 0.0000, -0.7854),
        },
    )

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Rotation": (0.0000, 1.5708, 0.0000)},
    )

    transform_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_2, "Translation": (0.0000, 0.5000, 0.0000)},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 1.0000, "Y": group_input, "Z": 1.0000}
    )

    transform_4 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": transform_3, "Scale": combine_xyz}
    )

    fillet_curve = nw.new_node(
        "GeometryNodeFilletCurve",
        input_kwargs={
            "Curve": transform_4,
            "Count": 8,
            "Radius": group_input,
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    transform_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": fillet_curve,
            "Rotation": (1.5708, 1.5708, 0.0000),
            "Scale": group_input.outputs["Thickness"],
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh, input_kwargs={"Profile Curve": transform_6}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    transform_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": curve_to_mesh, "Translation": combine_xyz_1},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [curve_to_board.outputs["Mesh"], transform_5]},
    )

    merge_by_distance = nw.new_node(
        Nodes.MergeByDistance, input_kwargs={"Geometry": join_geometry}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Thickness"]}
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": merge_by_distance, "Translation": combine_xyz_2},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_side_leg", singleton=False, type="GeometryNodeTree")
def nodegroup_side_leg(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    curve_line = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (1.0000, 0.0000, -1.0000),
            "End": (1.0000, 0.0000, 1.0000),
        },
    )

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Thickness", 0.5000),
            ("NodeSocketInt", "N-gon", 0),
            ("NodeSocketFloat", "Profile Width", 0.5000),
            ("NodeSocketFloat", "Aspect Ratio", 0.5000),
            ("NodeSocketFloat", "Fillet Ratio", 0.2000),
            ("NodeSocketFloat", "Fillet Radius Vertical", 0.0000),
        ],
    )

    legstraight = nw.new_node(
        nodegroup_leg_straight().name,
        input_kwargs={
            "Profile Curve": curve_line,
            "Height": group_input.outputs["Thickness"],
            "N-gon": group_input.outputs["N-gon"],
            "Profile Width": group_input.outputs["Profile Width"],
            "Aspect Ratio": group_input.outputs["Aspect Ratio"],
            "Fillet Ratio": group_input.outputs["Fillet Ratio"],
        },
    )

    arc = nw.new_node(
        "GeometryNodeCurveArc",
        input_kwargs={"Resolution": 4, "Radius": 0.7071, "Sweep Angle": 4.7124},
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": arc.outputs["Curve"],
            "Rotation": (0.0000, 0.0000, -0.7854),
        },
    )

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Rotation": (0.0000, 1.5708, 0.0000)},
    )

    transform_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_2, "Translation": (0.0000, 0.5000, 0.0000)},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 1.0000, "Y": group_input, "Z": 1.0000}
    )

    transform_4 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": transform_3, "Scale": combine_xyz}
    )

    fillet_curve = nw.new_node(
        "GeometryNodeFilletCurve",
        input_kwargs={
            "Curve": transform_4,
            "Count": 8,
            "Radius": group_input,
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    transform_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": fillet_curve,
            "Rotation": (1.5708, 1.5708, 0.0000),
            "Scale": group_input.outputs["Thickness"],
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": legstraight.outputs["Profile Curve"],
            "Profile Curve": transform_6,
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    transform_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": curve_to_mesh, "Translation": combine_xyz_1},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_5, legstraight.outputs["Mesh"]]},
    )

    merge_by_distance = nw.new_node(
        Nodes.MergeByDistance, input_kwargs={"Geometry": join_geometry}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Thickness"]}
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": merge_by_distance, "Translation": combine_xyz_2},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_side_boards", singleton=False, type="GeometryNodeTree"
)
def nodegroup_side_boards(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Y", 0.0000),
            ("NodeSocketFloat", "Z", 0.0000),
            ("NodeSocketFloat", "x1", 0.5000),
            ("NodeSocketFloat", "x2", 0.5000),
            ("NodeSocketFloat", "x3", 0.0010),
            ("NodeSocketFloat", "x4", 0.5000),
            ("NodeSocketFloat", "x5", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["x5"], 1: 0.0000}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": add,
            "Y": group_input.outputs["Y"],
            "Z": group_input.outputs["Z"],
        },
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz,
            "Vertices X": 5,
            "Vertices Y": 5,
            "Vertices Z": 5,
        },
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: add}, attrs={"operation": "MULTIPLY"}
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply, 1: group_input.outputs["x3"]}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["x1"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["x2"], 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_1, "Z": subtract}
    )

    transform = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_1}
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["x4"], 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_1, "Z": subtract_1}
    )

    transform_1 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_2}
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform, transform_1]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_shelf_boards", singleton=False, type="GeometryNodeTree"
)
def nodegroup_shelf_boards(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Thickness", 0.0100),
            ("NodeSocketFloat", "Bottom_z", 0.0000),
            ("NodeSocketFloat", "Mid_z", 0.0000),
            ("NodeSocketFloat", "Top_z", 0.0000),
            ("NodeSocketFloat", "Board_width", 0.3000),
            ("NodeSocketFloat", "Leg_gap", 0.5000),
            ("NodeSocketFloat", "extrude_length", 0.5000),
        ],
    )

    curve_board = nw.new_node(
        nodegroup_curve_board().name,
        input_kwargs={
            "Thickness": group_input.outputs["Thickness"],
            "Fillet Radius Vertical": 0.0100,
            "width": group_input.outputs["Board_width"],
            "extrude_length": group_input.outputs["extrude_length"],
        },
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Leg_gap"], 1: 0.0000}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Z": group_input.outputs["Bottom_z"]}
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_board,
            "Translation": combine_xyz_1,
            "Rotation": (0.0000, 0.0000, -1.5708),
        },
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Z": group_input.outputs["Mid_z"]}
    )

    transform_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_board,
            "Translation": combine_xyz_4,
            "Rotation": (0.0000, 0.0000, -1.5708),
        },
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Z": group_input.outputs["Top_z"]}
    )

    transform_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_board,
            "Translation": combine_xyz_5,
            "Rotation": (0.0000, 0.0000, -1.5708),
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_1, transform_5, transform_6]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_screw_head", singleton=False, type="GeometryNodeTree"
)
def nodegroup_screw_head(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder", input_kwargs={"Radius": 0.004, "Depth": 0.0030}
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "leg_width", 0.5000),
            ("NodeSocketFloat", "board_thickness", 0.5000),
            ("NodeSocketFloat", "board_height", 0.5000),
            ("NodeSocketFloat", "leg_gap", 0.5000),
            ("NodeSocketFloat", "board_width", 0.5000),
            ("NodeSocketFloat", "leg_depth", 0.0000),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["leg_width"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["leg_depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 0.0000, 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["board_thickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["board_height"], 1: multiply_2}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply, "Y": subtract, "Z": add}
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Translation": combine_xyz},
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["board_width"], 1: 0.0000}
    )

    divide1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["leg_depth"], 1: 0.5},
        attrs={"operation": "MULTIPLY"},
    )

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: add_1, 1: divide1})

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply, "Y": add_2, "Z": add}
    )

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Translation": combine_xyz_1},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["leg_gap"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: add_1, 1: multiply_3})

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_3, 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract_1, "Y": subtract, "Z": add}
    )

    transform_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Translation": combine_xyz_2},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_1, transform_2, transform_3]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_shelf_legs", singleton=False, type="GeometryNodeTree"
)
def nodegroup_shelf_legs(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "leg_gap", 0.5000),
            ("NodeSocketFloat", "leg_curve_ratio", 0.5000),
            ("NodeSocketFloat", "leg_width", 0.5000),
            ("NodeSocketFloat", "leg_length", 0.5000),
            ("NodeSocketFloat", "board_width", 0.5000),
            ("NodeSocketFloat", "leg_depth", 0.0000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["leg_width"], 1: 0.0000}
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["leg_length"], 1: 0.0000}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["leg_depth"],
            1: group_input.outputs["leg_length"],
        },
        attrs={"operation": "DIVIDE"},
    )

    add_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["leg_curve_ratio"], 1: 0.0000}
    )

    side_leg = nw.new_node(
        nodegroup_side_leg().name,
        input_kwargs={
            "Thickness": add,
            "N-gon": 4,
            "Profile Width": add_1,
            "Aspect Ratio": divide,
            "Fillet Ratio": add_2,
            "Fillet Radius Vertical": add_2,
        },
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": side_leg,
            "Translation": combine_xyz,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    add_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["board_width"], 1: 0.0000}
    )

    subtract = nw.new_node(
        Nodes.Math, input_kwargs={0: add_3, 1: add}, attrs={"operation": "SUBTRACT"}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["leg_gap"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: subtract, 1: multiply_1})

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": add_4})

    transform_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Translation": combine_xyz_3},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": add_3})

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Translation": combine_xyz_2},
    )

    transform_3 = nw.new_node(Nodes.Transform, input_kwargs={"Geometry": transform})

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_4, transform_2, transform_3]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_2},
        attrs={"is_active_output": True},
    )


def geometry_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    leg_gap = nw.new_node(Nodes.Value, label="leg_gap")
    leg_gap.outputs[0].default_value = kwargs["leg_board_gap"]

    curvature_ratio = nw.new_node(Nodes.Value, label="curvature_ratio")
    curvature_ratio.outputs[0].default_value = kwargs["leg_curvature_ratio"]

    leg_width = nw.new_node(Nodes.Value, label="leg_width")
    leg_width.outputs[0].default_value = kwargs["leg_width"]

    leg_length = nw.new_node(Nodes.Value, label="leg_length")
    leg_length.outputs[0].default_value = kwargs["leg_length"]

    leg_depth = nw.new_node(Nodes.Value, label="leg_depth")
    leg_depth.outputs[0].default_value = kwargs["leg_depth"]

    board_width = nw.new_node(Nodes.Value, label="board_width")
    board_width.outputs[0].default_value = kwargs["board_width"]

    shelf_legs = nw.new_node(
        nodegroup_shelf_legs().name,
        input_kwargs={
            "leg_gap": leg_gap,
            "leg_curve_ratio": curvature_ratio,
            "leg_width": leg_width,
            "leg_length": leg_length,
            "board_width": board_width,
            "leg_depth": leg_depth,
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": shelf_legs, "Material": kwargs["leg_material"]},
    )

    board_thickness = nw.new_node(Nodes.Value, label="board_thickness")
    board_thickness.outputs[0].default_value = kwargs["board_thickness"]

    board_extrude_length = nw.new_node(Nodes.Value, label="board_extrude_length")
    board_extrude_length.outputs[0].default_value = kwargs["board_extrude_length"]

    bottom_layer_height = nw.new_node(Nodes.Value, label="bottom_layer_height")
    bottom_layer_height.outputs[0].default_value = kwargs["bottom_layer_height"]

    mid_layer_height = nw.new_node(Nodes.Value, label="mid_layer_height")
    mid_layer_height.outputs[0].default_value = kwargs["mid_layer_height"]

    top_layer_height = nw.new_node(Nodes.Value, label="top_layer_height")
    top_layer_height.outputs[0].default_value = kwargs["top_layer_height"]

    screwhead1 = nw.new_node(
        nodegroup_screw_head().name,
        input_kwargs={
            "leg_width": leg_width,
            "board_thickness": board_thickness,
            "board_height": bottom_layer_height,
            "leg_gap": leg_gap,
            "board_width": board_width,
            "leg_depth": leg_depth,
        },
    )

    screwhead2 = nw.new_node(
        nodegroup_screw_head().name,
        input_kwargs={
            "leg_width": leg_width,
            "board_thickness": board_thickness,
            "board_height": mid_layer_height,
            "leg_gap": leg_gap,
            "board_width": board_width,
            "leg_depth": leg_depth,
        },
    )

    screwhead3 = nw.new_node(
        nodegroup_screw_head().name,
        input_kwargs={
            "leg_width": leg_width,
            "board_thickness": board_thickness,
            "board_height": top_layer_height,
            "leg_gap": leg_gap,
            "board_width": board_width,
            "leg_depth": leg_depth,
        },
    )

    join_geometry2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [screwhead1, screwhead2, screwhead3]},
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": join_geometry2,
            "Material": get_shelf_material("metal"),
        },
    )

    shelf_boards = nw.new_node(
        nodegroup_shelf_boards().name,
        input_kwargs={
            "Thickness": board_thickness,
            "Bottom_z": bottom_layer_height,
            "Mid_z": mid_layer_height,
            "Top_z": top_layer_height,
            "Board_width": board_width,
            "Leg_gap": leg_gap,
            "extrude_length": board_extrude_length,
        },
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": shelf_boards, "Material": kwargs["board_material"]},
    )

    side_board_height = nw.new_node(Nodes.Value, label="side_board_height")
    side_board_height.outputs[0].default_value = kwargs["side_board_height"]

    side_boards = nw.new_node(
        nodegroup_side_boards().name,
        input_kwargs={
            "Y": leg_depth,
            "Z": side_board_height,
            "x1": side_board_height,
            "x2": bottom_layer_height,
            "x3": leg_gap,
            "x4": top_layer_height,
            "x5": board_width,
        },
    )

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": side_boards, "Material": kwargs["leg_material"]},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [set_material, set_material_2, set_material_1, set_material_3]
        },
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": join_geometry}
    )

    transform4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": realize_instances, "Scale": (-1, 1, 1)},
    )

    triangulate = nw.new_node(
        "GeometryNodeTriangulate", input_kwargs={"Mesh": transform4}
    )

    transform5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": triangulate, "Rotation": (0.0000, 0.0000, -1.5708)},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform5},
        attrs={"is_active_output": True},
    )


class TriangleShelfBaseFactory(AssetFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(TriangleShelfBaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.params = {}

    def sample_params(self):
        return self.params.copy()

    def get_asset_params(self, i=0):
        params = self.sample_params()
        if params.get("leg_board_gap", None) is None:
            params["leg_board_gap"] = uniform(0.002, 0.005)
        if params.get("leg_width", None) is None:
            params["leg_width"] = uniform(0.01, 0.03)
        if params.get("leg_depth", None) is None:
            params["leg_depth"] = uniform(0.01, 0.02)
        if params.get("leg_length", None) is None:
            params["leg_length"] = np.clip(normal(0.6, 0.05), 0.45, 0.75)
        if params.get("leg_curvature_ratio", None) is None:
            params["leg_curvature_ratio"] = uniform(0.0, 0.02)
        if params.get("board_thickness", None) is None:
            params["board_thickness"] = uniform(0.01, 0.025)
        if params.get("board_width", None) is None:
            params["board_width"] = np.clip(normal(0.3, 0.03), 0.2, 0.4)
        if params.get("board_extrude_length", None) is None:
            params["board_extrude_length"] = uniform(0.03, 0.07)
        if params.get("side_board_height", None) is None:
            params["side_board_height"] = uniform(0.02, 0.04)
        if params.get("bottom_layer_height", None) is None:
            params["bottom_layer_height"] = uniform(0.05, 0.1)
        if params.get("shelf_layer_height", None) is None:
            params["top_layer_height"] = params["leg_length"] - uniform(0.02, 0.07)
        if params.get("board_material", None) is None:
            params["board_material"] = np.random.choice(
                ["black_wood", "wood", "white"], p=[0.2, 0.6, 0.2]
            )
        if params.get("leg_material", None) is None:
            params["leg_material"] = np.random.choice(
                ["black_wood", "wood", "white"], p=[0.2, 0.6, 0.2]
            )
        params["mid_layer_height"] = (
            params["top_layer_height"] + params["bottom_layer_height"]
        ) / 2.0

        params = self.get_material_func(params)
        return params

    def get_material_func(self, params, randomness=True):
        params["board_material"] = get_shelf_material(params["board_material"])
        params["leg_material"] = get_shelf_material(
            params["leg_material"], z_axis_texture=True
        )
        return params

    def create_asset(self, i=0, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=1,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        obj = bpy.context.active_object

        obj_params = self.get_asset_params(i)
        surface.add_geomod(
            obj, geometry_nodes, attributes=[], input_kwargs=obj_params, apply=True
        )
        tagging.tag_system.relabel_obj(obj)

        return obj


class TriangleShelfFactory(TriangleShelfBaseFactory):
    def sample_params(self):
        params = dict()
        params["Dimensions"] = (
            uniform(0.25, 0.35),
            uniform(0.25, 0.35),
            uniform(0.5, 0.7),
        )
        params["leg_length"] = params["Dimensions"][2]
        params["board_width"] = params["Dimensions"][0]
        return params
