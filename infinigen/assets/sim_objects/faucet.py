import gin
from numpy.random import randint, uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.joints import (
    nodegroup_add_jointed_geometry_metadata,
    nodegroup_hinge_joint,
)
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import weighted_sample


@node_utils.to_nodegroup(
    "nodegroup_node_group_main_part_faucet", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_main_part_faucet(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    cylinder_3 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 39, "Radius": 0.0200, "Depth": 0.0300},
    )

    transform_geometry_15 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_3.outputs["Mesh"],
            "Translation": (0.5950, 0.0000, 0.3800),
        },
    )

    cylinder_2 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 39, "Radius": 0.0100, "Depth": 0.7000},
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition, input_kwargs={"Geometry": cylinder_2.outputs["Mesh"]}
    )

    transform_geometry_14 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": set_position_1,
            "Translation": (0.3000, 0.0000, 0.2500),
            "Rotation": (0.0000, -2.0420, 0.0000),
            "Scale": (1.7000, 3.1000, 1.0000),
        },
    )

    join_geometry_6 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_15, transform_geometry_14]},
    )

    transform_geometry_16 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_6, "Scale": (0.9000, 1.0000, 1.0000)},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_16},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_curved_handle", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_curved_handle(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    bezier_segment_1 = nw.new_node(
        Nodes.CurveBezierSegment,
        input_kwargs={
            "Resolution": 8,
            "Start": (0.0000, 0.0000, 0.0000),
            "Start Handle": (0.0000, 0.0000, 0.7000),
            "End Handle": (0.2000, 0.0000, 0.7000),
            "End": (1.0000, 0.0000, 0.9000),
        },
    )

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    float_curve = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": spline_parameter.outputs["Factor"]}
    )
    node_utils.assign_curve(
        float_curve.mapping.curves[0],
        [(0.0000, 0.9750), (0.6295, 0.4125), (1.0000, 0.1625)],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: float_curve, 1: 1.3000},
        attrs={"operation": "MULTIPLY"},
    )

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius,
        input_kwargs={"Curve": bezier_segment_1, "Radius": multiply},
    )

    curve_circle_5 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 17, "Radius": 0.1100}
    )

    curve_to_mesh_3 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius,
            "Profile Curve": curve_circle_5.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_1})

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": separate_xyz_1.outputs["X"],
            1: 0.2000,
            3: 1.0000,
            4: 2.5000,
        },
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Y"], 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz_1.outputs["X"],
            "Y": multiply_1,
            "Z": separate_xyz_1.outputs["Z"],
        },
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": curve_to_mesh_3, "Position": combine_xyz_5},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_position},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_cylinder_base", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_cylinder_base(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    cylinder_5 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 39, "Radius": 0.0650, "Depth": 0.1500},
    )

    transform_geometry_23 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_5.outputs["Mesh"],
            "Translation": (0.0000, 0.0000, 0.0800),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_23},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_001", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_001(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketFloat", "Y", 0.0000)]
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 0.2000, "Y": group_input.outputs["Y"]}
    )

    bezier_segment = nw.new_node(
        Nodes.CurveBezierSegment,
        input_kwargs={
            "Resolution": 12,
            "Start": (0.0000, 0.0000, 0.0000),
            "Start Handle": (0.0000, 1.2000, 0.0000),
            "End Handle": combine_xyz,
            "End": (-0.0500, 0.1000, 0.0000),
        },
    )

    trim_curve = nw.new_node(
        Nodes.TrimCurve, input_kwargs={"Curve": bezier_segment, 3: 0.6625}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": trim_curve,
            "Rotation": (1.5708, 0.0000, 2.4173),
            "Scale": (5.2000, 0.5000, 7.8000),
        },
    )

    curve_circle_3 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 19, "Radius": 0.0300}
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": transform_geometry_2,
            "Profile Curve": curve_circle_3.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": curve_to_mesh_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_002", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_002(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Curve", None)]
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": group_input.outputs["Curve"],
            "Translation": (0.0000, 0.0000, 0.6000),
        },
    )

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"End": (0.0000, 0.0000, 0.6000)}
    )

    curve_circle_1 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 19, "Radius": 0.0300}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": curve_circle_1.outputs["Curve"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": reroute_1,
            "Fill Caps": True,
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_3, curve_to_mesh]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_handle", singleton=False, type="GeometryNodeTree")
def nodegroup_handle(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    bezier_segment = nw.new_node(
        Nodes.CurveBezierSegment,
        input_kwargs={
            "Resolution": 8,
            "Start": (0.0000, 0.0000, 0.0000),
            "Start Handle": (0.0000, 0.0000, 0.7000),
            "End Handle": (0.2000, 0.0000, 0.7000),
            "End": (1.0000, 0.0000, 0.9000),
        },
    )

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    float_curve = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": spline_parameter.outputs["Factor"]}
    )
    node_utils.assign_curve(
        float_curve.mapping.curves[0], [(0.0000, 0.9750), (1.0000, 0.1625)]
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: float_curve, 1: 1.3000},
        attrs={"operation": "MULTIPLY"},
    )

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius, input_kwargs={"Curve": bezier_segment, "Radius": multiply}
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 3, "Radius": 0.4500}
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": separate_xyz.outputs["X"],
            1: 0.2000,
            3: 1.0000,
            4: 2.5000,
        },
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz.outputs["X"],
            "Y": multiply_1,
            "Z": separate_xyz.outputs["Z"],
        },
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": curve_to_mesh, "Position": combine_xyz},
    )

    subdivision_surface = nw.new_node(
        Nodes.SubdivisionSurface, input_kwargs={"Mesh": set_position, "Level": 2}
    )

    set_shade_smooth = nw.new_node(
        Nodes.SetShadeSmooth, input_kwargs={"Geometry": subdivision_surface}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_shade_smooth},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_009", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_009(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    cylinder_4 = nw.new_node(
        "GeometryNodeMeshCylinder", input_kwargs={"Radius": 0.0500, "Depth": 0.0500}
    )

    transform_geometry_22 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_4.outputs["Mesh"],
            "Translation": (0.0000, 0.0000, 0.0800),
            "Scale": (1.0000, 1.0000, 2.6000),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_22},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_quadrillateral_base", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_quadrillateral_base(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 2.4000),
            ("NodeSocketFloat", "Radius", 0.2500),
        ],
    )

    quadrilateral_1 = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": group_input.outputs["Width"], "Height": 0.7000},
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": quadrilateral_1})

    fillet_curve_1 = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": reroute,
            "Count": 7,
            "Radius": group_input.outputs["Radius"],
        },
        attrs={"mode": "POLY"},
    )

    fill_curve_4 = nw.new_node(
        Nodes.FillCurve, input_kwargs={"Curve": fillet_curve_1}, attrs={"mode": "NGONS"}
    )

    extrude_mesh_4 = nw.new_node(
        Nodes.ExtrudeMesh, input_kwargs={"Mesh": fill_curve_4, "Offset Scale": 0.0300}
    )

    convex_hull = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": extrude_mesh_4.outputs["Mesh"]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": convex_hull},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "base_width", 0.3500),
            ("NodeSocketFloat", "tap_roation_z", 4.3500),
            ("NodeSocketFloat", "tap_height", 0.6600),
            ("NodeSocketFloat", "base_radius", 0.0600),
            ("NodeSocketFloat", "curl_tap", -0.0700),
            ("NodeSocketInt", "pedestral_sink", 0),
            ("NodeSocketFloat", "hands_length_x", 1.0000),
            ("NodeSocketFloat", "hands_length_Y", 1.0000),
            ("NodeSocketInt", "one_side", 1),
            ("NodeSocketInt", "vessel_sink", 1),
            ("NodeSocketInt", "wide_one_side_handle", 0),
            ("NodeSocketMaterial", "Tap", None),
            ("NodeSocketFloat", "pedestral_hinge_handle_0", 0.0000),
            ("NodeSocketFloat", "pedestral_hingle_handle_1", 0.0000),
            ("NodeSocketFloat", "hinge_tap_rotation", 0.0000),
            ("NodeSocketFloat", "hinge_side_handle_0", 0.0000),
            ("NodeSocketFloat", "hinge_side_handle_1", 0.0000),
            ("NodeSocketFloat", "hinge_vessel_tap_0", 0.0000),
            ("NodeSocketFloat", "hinge_vessel_handle", 0.0000),
            ("NodeSocketFloat", "hinge_vessel_handle_y_rotation", 0.0000),
        ],
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["vessel_sink"]}
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={3: reroute_3},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["pedestral_sink"]}
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={3: reroute_2},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    reroute_30 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": equal_1})

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_30})

    quadrilateral_base = nw.new_node(
        nodegroup_node_group_quadrillateral_base().name,
        input_kwargs={
            "Width": group_input.outputs["base_width"],
            "Radius": group_input.outputs["base_radius"],
        },
        label="Quadrilateral_Base",
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": quadrilateral_base,
            "Rotation": (-0.0017, -0.0017, -0.0017),
            "Scale": (0.9900, 0.9900, 0.9000),
        },
    )

    cylinder_base = nw.new_node(nodegroup_node_group_009().name, label="CylinderBase")

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": cylinder_base})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_geometry, reroute_15]}
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": join_geometry, "Label": "base"},
    )

    handle_no_gc = nw.new_node(nodegroup_handle().name)

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": handle_no_gc,
            "Translation": (0.0000, 0.0000, -0.0600),
            "Scale": (0.3000, 0.3000, 0.3000),
        },
    )

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_5, "Label": "handle"},
    )

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "pedestral_left_side_handle",
            "Parent": add_jointed_geometry_metadata,
            "Child": add_jointed_geometry_metadata_1,
            "Position": (0.0000, -0.2000, 0.0000),
            "Min": -3.1416,
            "Max": 3.1416,
        },
    )

    add_jointed_geometry_metadata_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": hinge_joint.outputs["Geometry"], "Label": "base"},
    )

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": handle_no_gc})

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_17,
            "Translation": (0.0000, 0.0000, -0.0600),
            "Scale": (0.3000, 0.3000, 0.3000),
        },
    )

    add_jointed_geometry_metadata_3 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_6, "Label": "handle"},
    )

    hinge_joint_1 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "pedestral_right_side_handle",
            "Parent": add_jointed_geometry_metadata_2,
            "Child": add_jointed_geometry_metadata_3,
            "Position": (0.0000, 0.2000, 0.0000),
            "Min": -3.1416,
            "Max": 3.1416,
        },
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_geometry, reroute_15]}
    )

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_2})

    switch_11 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_30,
            "False": hinge_joint_1.outputs["Geometry"],
            "True": reroute_24,
        },
    )

    add_jointed_geometry_metadata_4 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": switch_11, "Label": "base"},
    )

    main_faucet_skeleton_top_part = nw.new_node(
        nodegroup_node_group_001().name,
        input_kwargs={"Y": group_input.outputs["curl_tap"]},
        label="MainFaucetSkeletonTopPart",
    )

    main_faucet_skeleton_bottom_part = nw.new_node(
        nodegroup_node_group_002().name,
        input_kwargs={"Curve": main_faucet_skeleton_top_part},
        label="MainFaucetSkeletonBottomPart",
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["tap_height"]}
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 1.0000, "Y": 1.0000, "Z": reroute_1}
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": main_faucet_skeleton_bottom_part,
            "Scale": combine_xyz_3,
        },
    )

    add_jointed_geometry_metadata_5 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_4, "Label": "spout"},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_5,
            "Rotation": (0.0000, 0.0000, 3.8676),
        },
    )

    reroute_29 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_2}
    )

    hinge_joint_2 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "spout",
            "Parent": add_jointed_geometry_metadata_4,
            "Child": reroute_29,
            "Min": -3.1416,
            "Max": 3.1416,
        },
    )

    reroute_35 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_2.outputs["Geometry"]}
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["one_side"]}
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 18, "Radius": 0.0300, "Depth": 0.1000},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["hands_length_x"],
            "Y": group_input.outputs["hands_length_Y"],
            "Z": 1.0000,
        },
    )

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": (0.0000, 0.0500, 0.1000),
            "Rotation": (1.5708, 0.0000, 0.0000),
            "Scale": combine_xyz_4,
        },
    )

    transform_geometry_8 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": (0.0000, -0.0500, 0.1000),
            "Rotation": (1.5708, 0.0000, 0.0000),
            "Scale": combine_xyz_4,
        },
    )

    join_geometry_6 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_7, transform_geometry_8]},
    )

    reroute_22 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_8}
    )

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_14,
            "False": join_geometry_6,
            "True": reroute_23,
        },
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": switch_3, "Scale": (1.0000, 0.9800, 1.0000)},
    )

    add_jointed_geometry_metadata_6 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_1, "Label": "handle"},
    )

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 16, "Radius": 0.0050, "Depth": 0.1000},
    )

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cylinder_1.outputs["Mesh"]}
    )

    transform_geometry_9 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_16,
            "Translation": (0.0000, 0.0700, 0.0500),
            "Rotation": (0.0000, 3.1416, 0.0000),
            "Scale": (1.0000, 1.0000, 1.1000),
        },
    )

    add_jointed_geometry_metadata_7 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_9, "Label": "handle"},
    )

    hinge_joint_3 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "right_side_handle",
            "Parent": add_jointed_geometry_metadata_6,
            "Child": add_jointed_geometry_metadata_7,
            "Axis": (0.0000, 1.0000, 0.0000),
            "Value": 0.7000,
            "Min": -0.8000,
            "Max": 0.8000,
        },
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_14, 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    transform_geometry_10 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": (0.0000, -0.0300, 0.0500),
            "Rotation": (0.0000, 3.1416, 0.0000),
        },
    )

    reroute_20 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_10}
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    reroute_4 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["wide_one_side_handle"]},
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_4, 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    transform_geometry_11 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_10,
            "Scale": (4.1000, 1.0000, 1.0000),
        },
    )

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_3,
            "False": reroute_21,
            "True": transform_geometry_11,
        },
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal_2, "False": reroute_25, "True": switch_5},
    )

    add_jointed_geometry_metadata_8 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": switch_6, "Label": "handle"},
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_8,
            "Translation": (0.0000, -0.0400, 0.0000),
        },
    )

    reroute_33 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_3}
    )

    hinge_joint_4 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "left_side_handle_second",
            "Parent": hinge_joint_3.outputs["Geometry"],
            "Child": reroute_33,
            "Axis": (0.0000, 1.0000, 0.0000),
            "Value": 0.3000,
            "Min": -0.8000,
            "Max": 0.8000,
        },
    )

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    add_jointed_geometry_metadata_9 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_26, "Label": "handle"},
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["hinge_side_handle_0"]},
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    hinge_joint_5 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "left_side_handle_only",
            "Parent": add_jointed_geometry_metadata_9,
            "Child": add_jointed_geometry_metadata_8,
            "Axis": (0.0000, 1.0000, 0.0000),
            "Value": reroute_8,
            "Min": -0.8000,
            "Max": 0.8000,
        },
    )

    reroute_32 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_5.outputs["Geometry"]}
    )

    switch_9 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_27,
            "False": hinge_joint_4.outputs["Geometry"],
            "True": reroute_32,
        },
    )

    reroute_36 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_9})

    join_geometry_7 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [hinge_joint_2.outputs["Geometry"], reroute_36]},
    )

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_31,
            "False": reroute_35,
            "True": join_geometry_7,
        },
    )

    cylinder_base_1 = nw.new_node(
        nodegroup_node_group_cylinder_base().name, label="Cylinder_Base"
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": quadrilateral_base})

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [cylinder_base_1, reroute_19]}
    )

    add_jointed_geometry_metadata_10 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": join_geometry_4, "Label": "base"},
    )

    reroute_28 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_10}
    )

    curved_handle = nw.new_node(
        nodegroup_node_group_curved_handle().name, label="Curved_Handle"
    )

    subdivision_surface = nw.new_node(
        Nodes.SubdivisionSurface, input_kwargs={"Mesh": curved_handle}
    )

    set_shade_smooth = nw.new_node(
        Nodes.SetShadeSmooth, input_kwargs={"Geometry": subdivision_surface}
    )

    transform_geometry_13 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": set_shade_smooth,
            "Translation": (0.0000, 0.0000, 0.0500),
            "Scale": (0.4000, 0.4000, 0.3000),
        },
    )

    add_jointed_geometry_metadata_11 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_13, "Label": "handle"},
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["hinge_vessel_handle"]},
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    hinge_joint_6 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "top_handle_z",
            "Parent": reroute_28,
            "Child": add_jointed_geometry_metadata_11,
            "Value": reroute_10,
            "Min": -0.9000,
            "Max": 0.9000,
        },
    )

    reroute_11 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["hinge_vessel_handle_y_rotation"]},
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    hinge_joint_7 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "top_handle_y",
            "Parent": hinge_joint_6.outputs["Parent"],
            "Child": hinge_joint_6.outputs["Child"],
            "Axis": (0.0000, -1.0000, 0.0000),
            "Value": reroute_12,
            "Max": 0.1000,
        },
    )

    add_jointed_geometry_metadata_12 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": hinge_joint_7.outputs["Geometry"], "Label": "handle"},
    )

    main_part_long_faucet = nw.new_node(
        nodegroup_node_group_main_part_faucet().name, label="Main_part_long_faucet"
    )

    add_jointed_geometry_metadata_13 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": main_part_long_faucet, "Label": "spout"},
    )

    hinge_joint_8 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "vessel_spout",
            "Parent": add_jointed_geometry_metadata_12,
            "Child": add_jointed_geometry_metadata_13,
            "Position": (0.0000, 0.0000, -0.1000),
            "Min": -0.8000,
            "Max": 0.8000,
        },
    )

    reroute_34 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_8.outputs["Geometry"]}
    )

    switch_8 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": switch_4, "True": reroute_34},
    )

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Tap"]}
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    set_material = nw.new_node(
        Nodes.SetMaterial, input_kwargs={"Geometry": switch_8, "Material": reroute_6}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_material},
        attrs={"is_active_output": True},
    )


class FaucetFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    @gin.configurable(module="FaucetFactory")
    def sample_joint_parameters(
        cls,
        vessel_spout_stiffness_min: float = 0.0,
        vessel_spout_stiffness_max: float = 0.0,
        vessel_spout_damping_min: float = 30.0,
        vessel_spout_damping_max: float = 50.0,
        right_side_handle_stiffness_min: float = 0.0,
        right_side_handle_stiffness_max: float = 0.0,
        right_side_handle_damping_min: float = 0.08,
        right_side_handle_damping_max: float = 0.15,
        top_handle_y_stiffness_min: float = -100.0,
        top_handle_y_stiffness_max: float = -100.0,
        top_handle_y_damping_min: float = 15.0,
        top_handle_y_damping_max: float = 30.0,
        pedestral_right_side_handle_stiffness_min: float = 0.0,
        pedestral_right_side_handle_stiffness_max: float = 0.0,
        pedestral_right_side_handle_damping_min: float = 15.0,
        pedestral_right_side_handle_damping_max: float = 30.0,
        spout_stiffness_min: float = 0.0,
        spout_stiffness_max: float = 0.0,
        spout_damping_min: float = 15.0,
        spout_damping_max: float = 30.0,
        pedestral_left_side_handle_stiffness_min: float = 0.0,
        pedestral_left_side_handle_stiffness_max: float = 0.0,
        pedestral_left_side_handle_damping_min: float = 15.0,
        pedestral_left_side_handle_damping_max: float = 30.0,
        left_side_handle_second_stiffness_min: float = 0.0,
        left_side_handle_second_stiffness_max: float = 0.0,
        left_side_handle_second_damping_min: float = 0.08,
        left_side_handle_second_damping_max: float = 0.15,
        top_handle_z_stiffness_min: float = 0.0,
        top_handle_z_stiffness_max: float = 0.0,
        top_handle_z_damping_min: float = 15.0,
        top_handle_z_damping_max: float = 30.0,
        left_side_handle_only_stiffness_min: float = 0.0,
        left_side_handle_only_stiffness_max: float = 0.0,
        left_side_handle_only_damping_min: float = 0.08,
        left_side_handle_only_damping_max: float = 0.15,
    ):
        return {
            "vessel_spout": {
                "stiffness": uniform(
                    vessel_spout_stiffness_min, vessel_spout_stiffness_max
                ),
                "damping": uniform(vessel_spout_damping_min, vessel_spout_damping_max),
            },
            "right_side_handle": {
                "stiffness": uniform(
                    right_side_handle_stiffness_min, right_side_handle_stiffness_max
                ),
                "damping": uniform(
                    right_side_handle_damping_min, right_side_handle_damping_max
                ),
            },
            "top_handle_y": {
                "stiffness": uniform(
                    top_handle_y_stiffness_min, top_handle_y_stiffness_max
                ),
                "damping": uniform(top_handle_y_damping_min, top_handle_y_damping_max),
            },
            "pedestral_right_side_handle": {
                "stiffness": uniform(
                    pedestral_right_side_handle_stiffness_min,
                    pedestral_right_side_handle_stiffness_max,
                ),
                "damping": uniform(
                    pedestral_right_side_handle_damping_min,
                    pedestral_right_side_handle_damping_max,
                ),
            },
            "spout": {
                "stiffness": uniform(spout_stiffness_min, spout_stiffness_max),
                "damping": uniform(spout_damping_min, spout_damping_max),
            },
            "pedestral_left_side_handle": {
                "stiffness": uniform(
                    pedestral_left_side_handle_stiffness_min,
                    pedestral_left_side_handle_stiffness_max,
                ),
                "damping": uniform(
                    pedestral_left_side_handle_damping_min,
                    pedestral_left_side_handle_damping_max,
                ),
            },
            "left_side_handle_second": {
                "stiffness": uniform(
                    left_side_handle_second_stiffness_min,
                    left_side_handle_second_stiffness_max,
                ),
                "damping": uniform(
                    left_side_handle_second_damping_min,
                    left_side_handle_second_damping_max,
                ),
            },
            "top_handle_z": {
                "stiffness": uniform(
                    top_handle_z_stiffness_min, top_handle_z_stiffness_max
                ),
                "damping": uniform(top_handle_z_damping_min, top_handle_z_damping_max),
            },
            "left_side_handle_only": {
                "stiffness": uniform(
                    left_side_handle_only_stiffness_min,
                    left_side_handle_only_stiffness_max,
                ),
                "damping": uniform(
                    left_side_handle_only_damping_min, left_side_handle_only_damping_max
                ),
            },
        }

    def sample_parameters(self):
        params = {
            "base_width": uniform(0.23, 0.63),
            "tap_roation_z": 0,
            "tap_height": uniform(0.5, 0.9),
            "base_radius": uniform(0.03, 0.1),
            "curl_tap": uniform(-0.2, 0.2),
            "pedestral_sink": randint(0, 2),
            "hands_length_x": uniform(0.75, 1.25),
            "hands_length_Y": uniform(0.75, 1.25),
            "one_side": randint(0, 2),
            "vessel_sink": randint(0, 2),
            "wide_one_side_handle": randint(0, 2),
            "Tap": weighted_sample(material_assignments.decorative_metal)()(),
            "pedestral_hinge_handle_0": 0,
            "pedestral_hingle_handle_1": 0,
            "hinge_tap_rotation": 0,
            "hinge_side_handle_0": 0,
            "hinge_side_handle_1": 0,
            "hinge_vessel_tap_0": 0,
            "hinge_vessel_handle": 0,
            "hinge_vessel_handle_y_rotation": 0,
        }
        return params

    def create_asset(self, asset_params=None, **kwargs):
        obj = butil.spawn_vert()
        butil.modify_mesh(
            obj,
            "NODES",
            apply=False,
            node_group=geometry_nodes(),
            ng_inputs=self.sample_parameters(),
        )

        return obj
