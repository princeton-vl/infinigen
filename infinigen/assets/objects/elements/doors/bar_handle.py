from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

@node_utils.to_nodegroup(
    "nodegroup_beveled_cylinder", singleton=False, type="GeometryNodeTree"
)
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "width", 0.0000),
            ("NodeSocketFloat", "aspect_ratio", 0.0000),
            ("NodeSocketFloat", "height", 0.0000),
        ],
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["height"]}
    )

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": (0.0, 0.0, 0.01), "End": combine_xyz}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": 1.0000, "Height": group_input.outputs["aspect_ratio"]},
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": quadrilateral, "Count": 32}
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": resample_curve,
            "Count": 8,
            "Radius": 1.0000,
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    resample_curve_3 = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": fillet_curve, "Count": 64}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": resample_curve_3,
            "Scale": group_input.outputs["width"],
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": transform_geometry,
            "Fill Caps": True,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": curve_to_mesh},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_handle_end", singleton=False, type="GeometryNodeTree"
)
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Bump", 0.0000),
            ("NodeSocketFloat", "aspect_ratio", 0.0000),
            ("NodeSocketFloat", "height", 0.0000),
            ("NodeSocketFloat", "width", 0.5000),
        ],
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["height"]}
    )

    curve_line_1 = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz_2})

    quadrilateral_1 = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": 3.0000, "Height": 1.0000},
    )

    resample_curve_1 = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": quadrilateral_1, "Count": 8}
    )


    greater_equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: 1},
        attrs={"data_type": "INT", "operation": "GREATER_EQUAL"},
    )

    less_equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: 2},
        attrs={"data_type": "INT", "operation": "LESS_EQUAL"},
    )

    op_and = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: greater_equal, 1: less_equal}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": group_input.outputs["Bump"]}
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": resample_curve_1,
            "Selection": op_and,
            "Offset": combine_xyz,
        },
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": 1.0000,
            "Y": group_input.outputs["aspect_ratio"],
            "Z": 1.0000,
        },
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": set_position, "Scale": combine_xyz_1}
    )

    resample_curve_2 = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": transform_geometry_1, "Count": 16}
    )

    fillet_curve_1 = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": resample_curve_2,
            "Count": 8,
            "Radius": 0.5000,
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    resample_curve_4 = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": fillet_curve_1, "Count": 64}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["width"], 1: 0.3333},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": resample_curve_4, "Scale": multiply}
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line_1,
            "Profile Curve": transform_geometry_2,
            "Fill Caps": True,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": curve_to_mesh_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_push_bar_handle", singleton=False, type="GeometryNodeTree"
)
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "total_length", 1.0000),
            ("NodeSocketFloat", "thickness", 0.1000),
            ("NodeSocketFloat", "bar_aspect_ratio", 0.5000),
            ("NodeSocketFloat", "bar_height_ratio", 0.5000),
            ("NodeSocketFloat", "bar_length_ratio", 0.5000),
            ("NodeSocketFloat", "end_length_ratio", 0.5000),
            ("NodeSocketFloat", "end_height_ratio", 0.5000),
            ("NodeSocketFloat", "overall_x", 0.0000),
            ("NodeSocketFloat", "overall_y", 0.0000),
            ("NodeSocketFloat", "overall_z", 0.0000),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["total_length"],
            1: group_input.outputs["end_length_ratio"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["end_height_ratio"],
            1: group_input.outputs["thickness"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    handleend = nw.new_node(
        nodegroup_handle_end().name,
        input_kwargs={
            "Bump": 0.7500,
            "aspect_ratio": group_input.outputs["bar_aspect_ratio"],
            "height": multiply,
            "width": multiply_1,
        },
    )

    beveledcylinder = nw.new_node(
        nodegroup_beveled_cylinder().name,
        input_kwargs={
            "width": group_input.outputs["thickness"],
            "aspect_ratio": group_input.outputs["bar_aspect_ratio"],
            "height": group_input.outputs["total_length"],
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [handleend, beveledcylinder]}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["thickness"],
            1: group_input.outputs["bar_height_ratio"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["total_length"],
            1: group_input.outputs["bar_length_ratio"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    beveledcylinder_1 = nw.new_node(
        nodegroup_beveled_cylinder().name,
        input_kwargs={
            "width": multiply_2,
            "aspect_ratio": group_input.outputs["bar_aspect_ratio"],
            "height": multiply_3,
        },
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["thickness"],
            1: group_input.outputs["bar_aspect_ratio"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_4, 1: 0.4000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_6 = nw.new_node(
        Nodes.Math,
        attrs={"operation": "MULTIPLY"},
    )


    multiply_7 = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 0.9500}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply_5, "Z": multiply_7}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": beveledcylinder_1, "Translation": combine_xyz},
    )

    multiply_max_1 = nw.new_node(
        Nodes.Math,
        attrs={"operation": "MULTIPLY"},
    )
    multiply_max_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_max_1, 1: 0.5},
        attrs={"operation": "MULTIPLY"},
    )

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Parent": join_geometry,
            "Child": transform_geometry,
            "Axis": (0.0000, -1.0000, 0.0000),
            "Max": multiply_max_2,
        },
    )

    multiply_8 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_4}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_8})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": sliding_joint.outputs["Geometry"],
            "Translation": combine_xyz_1,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["overall_x"],
            "Y": group_input.outputs["overall_y"],
            "Z": group_input.outputs["overall_z"],
        },
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_1.outputs["Geometry"],
            "Translation": combine_xyz_2,
        },
    )


    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_2},
        attrs={"is_active_output": True},
    )








#         input_kwargs={'total_length': bar_length,
#             'thickness': thickness,
#             'bar_height_ratio': bar_height_ratio,
#             'bar_length_ratio': bar_length_ratio,
#             'end_length_ratio': end_length_ratio,

#     group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': flip_faces}, attrs={'is_active_output': True})
