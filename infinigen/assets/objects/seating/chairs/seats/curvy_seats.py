# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


from infinigen.assets.objects.table_decorations.utils import (
    nodegroup_lofting,
    nodegroup_warp_around_curve,
)
from infinigen.assets.objects.tables.table_utils import nodegroup_bent
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

# TODO: set material automatically


@node_utils.to_nodegroup(
    "generate_curvy_seats", singleton=False, type="GeometryNodeTree"
)
def generate_curvy_seats(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "U Resolution", 256),
            ("NodeSocketInt", "V Resolution", 128),
            ("NodeSocketFloat", "Width", 0.5000),
            ("NodeSocketFloat", "Thickness", 0.0300),
            ("NodeSocketFloat", "Front Relative Width", 0.5000),
            ("NodeSocketFloat", "Front Bent", -0.3800),
            ("NodeSocketFloat", "Seat Bent", -0.5600),
            ("NodeSocketFloat", "Mid Relative Width", 0.5000),
            ("NodeSocketFloat", "Mid Bent", -0.7000),
            ("NodeSocketFloat", "Back Relative Width", 0.5000),
            ("NodeSocketFloat", "Back Bent", -0.2000),
            ("NodeSocketFloat", "Top Relative Width", 0.5000),
            ("NodeSocketFloat", "Top Bent", -0.2000),
            ("NodeSocketFloat", "Seat Height", 0.6000),
            ("NodeSocketFloat", "Mid Pos", 0.5000),
            ("NodeSocketMaterial", "SeatMaterial", None),
        ],
    )

    curve_circle_1 = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={
            "Resolution": group_input.outputs["U Resolution"],
            "Radius": 0.5000,
        },
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Width"],
            "Y": group_input.outputs["Thickness"],
            "Z": 1.0000,
        },
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle_1.outputs["Curve"],
            "Translation": (0.0000, 0.0000, 0.5000),
            "Scale": combine_xyz,
        },
    )

    bent = nw.new_node(
        nodegroup_bent().name,
        input_kwargs={
            "Geometry": transform_geometry_1,
            "Amount": group_input.outputs["Seat Bent"],
        },
    )

    curve_circle_2 = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={
            "Resolution": group_input.outputs["U Resolution"],
            "Radius": 0.5000,
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Width"],
            1: group_input.outputs["Mid Relative Width"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": multiply,
            "Y": group_input.outputs["Thickness"],
            "Z": 1.0000,
        },
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle_2.outputs["Curve"],
            "Translation": (0.0000, 0.0000, 1.0000),
            "Scale": combine_xyz_2,
        },
    )

    bent_1 = nw.new_node(
        nodegroup_bent().name,
        input_kwargs={
            "Geometry": transform_geometry_2,
            "Amount": group_input.outputs["Mid Bent"],
        },
    )

    curve_circle_3 = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={
            "Resolution": group_input.outputs["U Resolution"],
            "Radius": 0.5000,
        },
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle_3.outputs["Curve"],
            "Scale": (0.0000, 0.0050, 1.0000),
        },
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={
            "Resolution": group_input.outputs["U Resolution"],
            "Radius": 0.5000,
        },
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Width"],
            1: group_input.outputs["Front Relative Width"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_1, "Y": 0.0050, "Z": 1.0000}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle.outputs["Curve"],
            "Translation": (0.0000, 0.0000, 0.0600),
            "Scale": combine_xyz_1,
        },
    )

    bent_2 = nw.new_node(
        nodegroup_bent().name,
        input_kwargs={
            "Geometry": transform_geometry,
            "Amount": group_input.outputs["Front Bent"],
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [bent_1, bent, bent_2, transform_geometry_3]},
    )

    curve_circle_4 = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={
            "Resolution": group_input.outputs["U Resolution"],
            "Radius": 0.5000,
        },
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Width"],
            1: group_input.outputs["Back Relative Width"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": multiply_2,
            "Y": group_input.outputs["Thickness"],
            "Z": 1.0000,
        },
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle_4.outputs["Curve"],
            "Translation": (0.0000, 0.0000, 1.5000),
            "Scale": combine_xyz_3,
        },
    )

    bent_3 = nw.new_node(
        nodegroup_bent().name,
        input_kwargs={
            "Geometry": transform_geometry_4,
            "Amount": group_input.outputs["Back Bent"],
        },
    )

    curve_circle_5 = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={
            "Resolution": group_input.outputs["U Resolution"],
            "Radius": 0.5000,
        },
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Width"],
            1: group_input.outputs["Top Relative Width"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_3, "Y": 0.0050, "Z": 1.0000}
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle_5.outputs["Curve"],
            "Translation": (0.0000, 0.0000, 2.0200),
            "Scale": combine_xyz_4,
        },
    )

    bent_4 = nw.new_node(
        nodegroup_bent().name,
        input_kwargs={
            "Geometry": transform_geometry_5,
            "Amount": group_input.outputs["Top Bent"],
        },
    )

    curve_circle_6 = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={
            "Resolution": group_input.outputs["U Resolution"],
            "Radius": 0.5000,
        },
    )

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle_6.outputs["Curve"],
            "Translation": (0.0000, 0.0000, 2.1000),
            "Scale": (0.0000, 0.0050, 1.0000),
        },
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_6, bent_4, bent_3]},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [join_geometry_2, join_geometry]}
    )

    lofting_001 = nw.new_node(
        nodegroup_lofting().name,
        input_kwargs={
            "Profile Curves": join_geometry_1,
            "U Resolution": group_input.outputs["U Resolution"],
            "V Resolution": group_input.outputs["V Resolution"],
        },
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply_4, "Z": 0.0300}
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"Y": group_input.outputs["Mid Pos"], "Z": -0.0500},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"Y": multiply_5, "Z": group_input.outputs["Seat Height"]},
    )

    bezier_segment = nw.new_node(
        Nodes.CurveBezierSegment,
        input_kwargs={
            "Resolution": 128,
            "Start": combine_xyz_6,
            "Start Handle": combine_xyz_7,
            "End Handle": (0.0000, 0.1000, 0.1000),
            "End": combine_xyz_5,
        },
    )

    warparoundcurvealt = nw.new_node(
        nodegroup_warp_around_curve().name,
        input_kwargs={
            "Geometry": lofting_001.outputs["Geometry"],
            "Curve": bezier_segment,
        },
    )

    # material_func =np.random.choice([plastic.shader_rough_plastic, metal.get_shader(), wood_new.shader_wood, leather.shader_leather])

    warparoundcurvealt = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": warparoundcurvealt,
            "Material": group_input.outputs["SeatMaterial"],
        },
    )
    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": warparoundcurvealt},
        attrs={"is_active_output": True},
    )
