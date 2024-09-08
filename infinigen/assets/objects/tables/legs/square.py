# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


from infinigen.assets.objects.tables.table_utils import (
    nodegroup_merge_curve,
    nodegroup_n_gon_profile,
)
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


@node_utils.to_nodegroup(
    "nodegroup_generate_leg_square", singleton=False, type="GeometryNodeTree"
)
def nodegroup_generate_leg_square(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "Fillet Radius", 0.0300),
            ("NodeSocketBool", "Has Bottom Connector", True),
            ("NodeSocketInt", "Profile N-gon", 4),
            ("NodeSocketFloat", "Profile Width", 0.1000),
            ("NodeSocketFloat", "Profile Aspect Ratio", 0.5000),
            ("NodeSocketFloat", "Profile Fillet Ratio", 0.1000),
        ],
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Has Bottom Connector"], 1: 4.0000},
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": group_input.outputs["Has Bottom Connector"],
            3: 4.7124,
            4: 6.2832,
        },
    )

    arc = nw.new_node(
        "GeometryNodeCurveArc",
        input_kwargs={
            "Resolution": add,
            "Radius": 0.7071,
            "Sweep Angle": map_range.outputs["Result"],
        },
    )

    mergecurve = nw.new_node(
        nodegroup_merge_curve().name, input_kwargs={"Curve": arc.outputs["Curve"]}
    )

    map_range_1 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": group_input.outputs["Has Bottom Connector"],
            3: 1.5708,
            4: 3.1416,
        },
    )

    set_curve_tilt = nw.new_node(
        Nodes.SetCurveTilt,
        input_kwargs={"Curve": mergecurve, "Tilt": map_range_1.outputs["Result"]},
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": set_curve_tilt,
            "Rotation": (0.0000, 0.0000, -0.7854),
        },
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform,
            "Translation": (0.0000, 0.0000, -0.5000),
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Width"],
            "Y": 1.0000,
            "Z": group_input.outputs["Height"],
        },
    )

    transform_2 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": transform_1, "Scale": combine_xyz}
    )

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius, input_kwargs={"Curve": transform_2, "Radius": 1.0000}
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": set_curve_radius,
            "Count": 8,
            "Radius": group_input.outputs["Fillet Radius"],
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    ngonprofile = nw.new_node(
        nodegroup_n_gon_profile().name,
        input_kwargs={
            "Profile N-gon": group_input.outputs["Profile N-gon"],
            "Profile Width": group_input.outputs["Profile Width"],
            "Profile Aspect Ratio": group_input.outputs["Profile Aspect Ratio"],
            "Profile Fillet Ratio": group_input.outputs["Profile Fillet Ratio"],
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": fillet_curve,
            "Profile Curve": ngonprofile,
            "Fill Caps": True,
        },
    )

    transform_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": curve_to_mesh, "Rotation": (0.0000, 0.0000, 1.5708)},
    )

    set_shade_smooth = nw.new_node(
        Nodes.SetShadeSmooth,
        input_kwargs={"Geometry": transform_3, "Shade Smooth": False},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_shade_smooth},
        attrs={"is_active_output": True},
    )
