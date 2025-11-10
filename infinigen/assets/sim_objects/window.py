import gin
from numpy.random import choice, randint, uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.joints import (
    nodegroup_add_jointed_geometry_metadata,
    nodegroup_duplicate_joints_on_parent,
    nodegroup_hinge_joint,
    nodegroup_sliding_joint,
)
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import weighted_sample


@node_utils.to_nodegroup("nodegroup_handle", singleton=False, type="GeometryNodeTree")
def nodegroup_handle(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "TopSize", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "TopThickness", 0.1000),
            ("NodeSocketFloat", "TopRoundness", 0.0000),
            ("NodeSocketVector", "SupportSize", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "SupportMargin", -0.5000),
            ("NodeSocketMaterial", "Material", None),
        ],
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["TopSize"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply})

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": separate_xyz.outputs["Z"]}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    quadratic_b_zier = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Resolution": 2,
            "Start": combine_xyz,
            "Middle": combine_xyz_2,
            "End": combine_xyz_1,
        },
    )

    roundquad = nw.new_node(
        nodegroup_round_quad().name,
        input_kwargs={
            "Width": separate_xyz.outputs["Y"],
            "Height": group_input.outputs["TopThickness"],
            "Roundness": group_input.outputs["TopRoundness"],
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": quadratic_b_zier,
            "Profile Curve": roundquad,
            "Fill Caps": True,
        },
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["SupportSize"]}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["Z"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_2})

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": separate_xyz.outputs["Z"]}
    )

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_5, "End": combine_xyz_6}
    )

    roundquad_1 = nw.new_node(
        nodegroup_round_quad().name,
        input_kwargs={
            "Width": separate_xyz_3.outputs["X"],
            "Height": separate_xyz_3.outputs["Y"],
            "Roundness": 0.2000,
        },
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": roundquad_1,
            "Fill Caps": True,
        },
    )

    set_shade_smooth = nw.new_node(
        Nodes.SetShadeSmooth,
        input_kwargs={"Geometry": curve_to_mesh_1, "Shade Smooth": False},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["X"]}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={1: group_input.outputs["SupportMargin"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["X"]},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_4, 1: multiply_5})

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_3, 1: add})

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": add_1})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": set_shade_smooth, "Translation": combine_xyz_3},
    )

    multiply_6 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute}, attrs={"operation": "MULTIPLY"}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_6, 1: add},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": subtract})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": set_shade_smooth, "Translation": combine_xyz_4},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry, transform_geometry_1]},
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 1": join_geometry_1,
            "Mesh 2": curve_to_mesh,
            "Self Intersection": True,
            "Hole Tolerant": True,
        },
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    separate_xyz_4 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["TopSize"]}
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_4.outputs["X"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: multiply_7},
        attrs={"operation": "SUBTRACT"},
    )

    sample_curve = nw.new_node(
        Nodes.SampleCurve,
        input_kwargs={"Curves": quadratic_b_zier, "Length": subtract_1},
        attrs={"mode": "LENGTH"},
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": sample_curve.outputs["Position"]}
    )

    less_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: separate_xyz_1.outputs["Z"]},
        attrs={"operation": "LESS_THAN"},
    )

    delete_geometry = nw.new_node(
        Nodes.DeleteGeometry,
        input_kwargs={"Geometry": difference.outputs["Mesh"], "Selection": less_than},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [curve_to_mesh, delete_geometry]}
    )

    set_shade_smooth_1 = nw.new_node(
        Nodes.SetShadeSmooth,
        input_kwargs={"Geometry": join_geometry, "Shade Smooth": False},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": set_shade_smooth_1,
            "Material": group_input.outputs["Material"],
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material, "Label": "Handle"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": add_jointed_geometry_metadata},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_round_trap", singleton=False, type="GeometryNodeTree"
)
def nodegroup_round_trap(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Height", 0.1000),
            ("NodeSocketFloat", "Bottom Width", 0.1000),
            ("NodeSocketFloat", "Top Width", 0.0500),
            ("NodeSocketFloat", "Offset", 0.0000),
            ("NodeSocketFloat", "Roundness", 1.0000),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Bottom Width"]}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Top Width"]}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Height": reroute,
            "Bottom Width": reroute_1,
            "Top Width": reroute_2,
            "Offset": group_input.outputs["Offset"],
        },
        attrs={"mode": "TRAPEZOID"},
    )

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute, 1: reroute_1},
        attrs={"operation": "MINIMUM"},
    )

    minimum_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum, 1: reroute_2},
        attrs={"operation": "MINIMUM"},
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": group_input.outputs["Roundness"], 4: 0.5000},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum_1, 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={"Curve": quadrilateral, "Count": 2, "Radius": multiply},
        attrs={"mode": "POLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Curve": fillet_curve},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_round_quad", singleton=False, type="GeometryNodeTree"
)
def nodegroup_round_quad(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "Roundness", 1.0000),
        ],
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": reroute_1, "Height": reroute},
    )

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: reroute},
        attrs={"operation": "MINIMUM"},
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": group_input.outputs["Roundness"], 4: 0.5000},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum, 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={"Curve": quadrilateral, "Count": 2, "Radius": multiply},
        attrs={"mode": "POLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Curve": fillet_curve},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_handle_to_panel", singleton=False, type="GeometryNodeTree"
)
def nodegroup_add_handle_to_panel(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "HandleThickness", 0.0100),
            ("NodeSocketFloat", "HandleRoundness", 0.0000),
            ("NodeSocketMaterial", "HandleMaterial", None),
            ("NodeSocketGeometry", "Panel", None),
            ("NodeSocketFloat", "FrameWidth", 0.5000),
            ("NodeSocketFloat", "HandleLength", 0.5000),
            ("NodeSocketFloat", "Width", 0.5000),
            ("NodeSocketBool", "HangleOnRight", False),
            ("NodeSocketFloat", "FrameThickness", 0.0000),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["FrameWidth"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["HandleLength"],
            "Y": multiply,
            "Z": multiply,
        },
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 0.1000, "Y": multiply, "Z": multiply}
    )

    handle = nw.new_node(
        nodegroup_handle().name,
        input_kwargs={
            "TopSize": combine_xyz_6,
            "TopThickness": group_input.outputs["HandleThickness"],
            "TopRoundness": group_input.outputs["HandleRoundness"],
            "SupportSize": combine_xyz_7,
            "SupportMargin": multiply,
            "Material": group_input.outputs["HandleMaterial"],
        },
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": group_input.outputs["HangleOnRight"], 3: -1.0000},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: -0.5000, 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: multiply_1},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["FrameThickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    bounding_box = nw.new_node(Nodes.BoundingBox, input_kwargs={"Geometry": handle})

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"]},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_3, 1: multiply_4})

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_2, "Z": add}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": handle,
            "Translation": combine_xyz_8,
            "Rotation": (0.0000, 0.0000, 1.5708),
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry, group_input.outputs["Panel"]]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": join_geometry,
            "Handle": bounding_box.outputs["Bounding Box"],
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_line_seq", singleton=False, type="GeometryNodeTree")
def nodegroup_line_seq(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", -1.0000),
            ("NodeSocketFloat", "Height", 0.5000),
            ("NodeSocketFloat", "Amount", 0.5000),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply, "Y": multiply_1}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_2, "Y": multiply_1}
    )

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz, "End": combine_xyz_1}
    )

    geometry_to_instance = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": curve_line}
    )

    duplicate_elements = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={
            "Geometry": geometry_to_instance,
            "Amount": group_input.outputs["Amount"],
        },
        attrs={"domain": "INSTANCE"},
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: 1.0000},
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Amount"], 1: 1.0000}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: add_1},
        attrs={"operation": "DIVIDE"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: divide}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_3})

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": duplicate_elements.outputs["Geometry"],
            "Offset": combine_xyz_2,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Curve": set_position},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_window_shutter", singleton=False, type="GeometryNodeTree"
)
def nodegroup_window_shutter(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input_2 = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 2.0000),
            ("NodeSocketFloat", "Height", 2.0000),
            ("NodeSocketFloat", "FrameWidth", 0.1000),
            ("NodeSocketFloat", "FrameThickness", 0.1000),
            ("NodeSocketFloat", "PanelWidth", 0.1000),
            ("NodeSocketFloat", "PanelThickness", 0.1000),
            ("NodeSocketFloat", "ShutterWidth", 0.1000),
            ("NodeSocketFloat", "ShutterThickness", 0.1000),
            ("NodeSocketFloat", "ShutterInterval", 0.5000),
            ("NodeSocketFloat", "ShutterRotation", 0.0000),
            ("NodeSocketMaterial", "FrameMaterial", None),
            ("NodeSocketBool", "WithHandle", False),
            ("NodeSocketFloat", "HandleLength", 0.5000),
            ("NodeSocketFloat", "HandleThickness", 0.0100),
            ("NodeSocketFloat", "HandleRoundness", 0.0000),
            ("NodeSocketBool", "HangleOnRight", False),
        ],
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["WithHandle"]}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Width": group_input_2.outputs["Width"],
            "Height": group_input_2.outputs["Height"],
        },
    )

    sqrt = nw.new_node(
        Nodes.Math, input_kwargs={0: 2.0000}, attrs={"operation": "SQRT"}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_2.outputs["FrameWidth"], 1: sqrt},
        attrs={"operation": "MULTIPLY"},
    )

    quadrilateral_1 = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Width": multiply,
            "Height": group_input_2.outputs["FrameThickness"],
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={"Curve": quadrilateral, "Profile Curve": quadrilateral_1},
    )

    addhandletopanel = nw.new_node(
        nodegroup_add_handle_to_panel().name,
        input_kwargs={
            "HandleThickness": group_input_2.outputs["HandleThickness"],
            "HandleRoundness": group_input_2.outputs["HandleRoundness"],
            "HandleMaterial": group_input_2.outputs["FrameMaterial"],
            "Panel": curve_to_mesh,
            "FrameWidth": group_input_2.outputs["FrameWidth"],
            "HandleLength": group_input_2.outputs["HandleLength"],
            "Width": group_input_2.outputs["Width"],
            "HangleOnRight": group_input_2.outputs["HangleOnRight"],
            "FrameThickness": group_input_2.outputs["PanelThickness"],
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_1,
            "False": curve_to_mesh,
            "True": addhandletopanel.outputs["Geometry"],
        },
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input_2.outputs["Height"],
            1: group_input_2.outputs["FrameWidth"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: group_input_2.outputs["ShutterInterval"]},
        attrs={"operation": "DIVIDE"},
    )

    floor = nw.new_node(
        Nodes.Math, input_kwargs={0: divide}, attrs={"operation": "FLOOR"}
    )

    shutter_true_interval = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: floor},
        label="ShutterTrueInterval",
        attrs={"operation": "DIVIDE"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: shutter_true_interval, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input_2.outputs["PanelWidth"],
            "Y": subtract_1,
            "Z": group_input_2.outputs["PanelThickness"],
        },
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_2})

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_2.outputs["ShutterWidth"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_2})

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz_3})

    geometry_to_instance_1 = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": curve_line}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["ShutterRotation"]}
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": reroute})

    rotate_instances_1 = nw.new_node(
        Nodes.RotateInstances,
        input_kwargs={"Instances": geometry_to_instance_1, "Rotation": combine_xyz_4},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": rotate_instances_1}
    )

    sample_curve = nw.new_node(
        Nodes.SampleCurve, input_kwargs={"Curves": realize_instances, "Factor": 1.0000}
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Offset": sample_curve.outputs["Position"],
        },
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input_2.outputs["Width"],
            1: group_input_2.outputs["FrameWidth"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": subtract_2,
            "Y": group_input_2.outputs["ShutterWidth"],
            "Z": group_input_2.outputs["ShutterThickness"],
        },
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    geometry_to_instance = nw.new_node(
        "GeometryNodeGeometryToInstance",
        input_kwargs={"Geometry": cube.outputs["Mesh"]},
    )

    shutter_number = nw.new_node(
        Nodes.Math,
        input_kwargs={0: floor, 1: 1.0000},
        label="ShutterNumber",
        attrs={"operation": "SUBTRACT"},
    )

    duplicate_elements = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={"Geometry": geometry_to_instance, "Amount": shutter_number},
        attrs={"domain": "INSTANCE"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: duplicate_elements.outputs["Duplicate Index"],
            1: shutter_true_interval,
        },
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_4, 1: shutter_true_interval}
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_3, 1: add})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": add_1})

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": duplicate_elements.outputs["Geometry"],
            "Offset": combine_xyz_1,
        },
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": reroute})

    rotate_instances = nw.new_node(
        Nodes.RotateInstances,
        input_kwargs={"Instances": set_position, "Rotation": combine_xyz_5},
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [switch, set_position_1, rotate_instances]},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": join_geometry_2,
            "Material": group_input_2.outputs["FrameMaterial"],
        },
    )

    set_shade_smooth = nw.new_node(
        Nodes.SetShadeSmooth,
        input_kwargs={"Geometry": set_material, "Shade Smooth": False},
    )

    realize_instances_1 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": set_shade_smooth}
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": addhandletopanel.outputs["Handle"]}
    )

    subtract_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: subtract_3.outputs["Vector"], "Scale": reroute_1},
        attrs={"operation": "SCALE"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": realize_instances_1,
            "HandleSize": scale.outputs["Vector"],
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_lever", singleton=False, type="GeometryNodeTree")
def nodegroup_lever(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Thickness", 0.0100),
            ("NodeSocketFloat", "BottomWidth", 0.0300),
            ("NodeSocketFloat", "TopWidth", 0.0050),
            ("NodeSocketFloat", "Offset", 0.0100),
            ("NodeSocketFloat", "Roundness", 1.0000),
            ("NodeSocketFloat", "Length", 0.0800),
        ],
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Thickness"]}
    )

    curve_line_1 = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz})

    subdivide_curve_1 = nw.new_node(
        Nodes.SubdivideCurve, input_kwargs={"Curve": curve_line_1}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BottomWidth"]}
    )

    roundtrap = nw.new_node(
        nodegroup_round_trap().name,
        input_kwargs={
            "Height": group_input.outputs["Length"],
            "Bottom Width": reroute,
            "Top Width": group_input.outputs["TopWidth"],
            "Offset": group_input.outputs["Offset"],
            "Roundness": group_input.outputs["Roundness"],
        },
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": roundtrap, "Translation": (0.0000, 0.0400, 0.0000)},
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute}, attrs={"operation": "MULTIPLY"}
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 2, "Radius": multiply}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [transform_geometry_1, curve_circle.outputs["Curve"]]
        },
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": subdivide_curve_1,
            "Profile Curve": join_geometry,
            "Fill Caps": True,
        },
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_to_mesh_1,
            "Rotation": (0.0000, 0.0000, 1.5708),
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry, "Label": "LockLever"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": add_jointed_geometry_metadata},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_lock_base", singleton=False, type="GeometryNodeTree"
)
def nodegroup_lock_base(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Length", 0.1000),
            ("NodeSocketFloat", "Width", 0.0500),
            ("NodeSocketFloat", "Height", 0.0200),
            ("NodeSocketFloat", "Roundness", 1.0000),
        ],
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Height"]}
    )

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz})

    subdivide_curve = nw.new_node(
        Nodes.SubdivideCurve, input_kwargs={"Curve": curve_line, "Cuts": 2}
    )

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    float_curve = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": spline_parameter.outputs["Factor"]}
    )
    node_utils.assign_curve(
        float_curve.mapping.curves[0],
        [(0.0000, 1.0000), (0.3795, 1.0000), (0.5682, 0.8844), (1.0000, 0.8094)],
        handles=["AUTO_CLAMPED", "VECTOR", "VECTOR", "AUTO"],
    )

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius,
        input_kwargs={"Curve": subdivide_curve, "Radius": float_curve},
    )

    roundquad = nw.new_node(
        nodegroup_round_quad().name,
        input_kwargs={
            "Width": group_input.outputs["Width"],
            "Height": group_input.outputs["Length"],
            "Roundness": group_input.outputs["Roundness"],
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius,
            "Profile Curve": roundquad,
            "Fill Caps": True,
        },
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": curve_to_mesh, "Rotation": (0.0000, 0.0000, 1.5708)},
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry, "Label": "LockBase"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": add_jointed_geometry_metadata},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_window_pane", singleton=False, type="GeometryNodeTree"
)
def nodegroup_window_pane(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input_1 = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 2.0000),
            ("NodeSocketFloat", "Height", 2.0000),
            ("NodeSocketFloat", "FrameWidth", 0.1000),
            ("NodeSocketFloat", "FrameThickness", 0.1000),
            ("NodeSocketFloat", "PanelWidth", 0.1000),
            ("NodeSocketFloat", "PanelThickness", 0.1000),
            ("NodeSocketInt", "PanelHAmount", 0),
            ("NodeSocketInt", "PanelVAmount", 0),
            ("NodeSocketBool", "WithGlass", False),
            ("NodeSocketFloat", "GlassThickness", 0.0000),
            ("NodeSocketMaterial", "FrameMaterial", None),
            ("NodeSocketMaterial", "GlassMaterial", None),
            ("NodeSocketBool", "WithHandle", False),
            ("NodeSocketFloat", "HandleLength", 0.5000),
            ("NodeSocketFloat", "HandleThickness", 0.0100),
            ("NodeSocketFloat", "HandleRoundness", 0.0000),
            ("NodeSocketBool", "HangleOnRight", False),
        ],
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Width": group_input_1.outputs["Width"],
            "Height": group_input_1.outputs["Height"],
        },
    )

    sqrt = nw.new_node(
        Nodes.Math, input_kwargs={0: 2.0000}, attrs={"operation": "SQRT"}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_1.outputs["FrameWidth"], 1: sqrt},
        attrs={"operation": "MULTIPLY"},
    )

    quadrilateral_1 = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Width": multiply,
            "Height": group_input_1.outputs["FrameThickness"],
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={"Curve": quadrilateral, "Profile Curve": quadrilateral_1},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input_1.outputs["PanelHAmount"], 1: -1.0000}
    )

    lineseq = nw.new_node(
        nodegroup_line_seq().name,
        input_kwargs={
            "Width": group_input_1.outputs["Width"],
            "Height": group_input_1.outputs["Height"],
            "Amount": add,
        },
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["PanelWidth"]}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_1.outputs["PanelThickness"], 1: 0.0010},
        attrs={"operation": "SUBTRACT"},
    )

    quadrilateral_2 = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": reroute, "Height": subtract},
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={"Curve": lineseq, "Profile Curve": quadrilateral_2},
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input_1.outputs["PanelVAmount"], 1: -1.0000}
    )

    lineseq_1 = nw.new_node(
        nodegroup_line_seq().name,
        input_kwargs={
            "Width": group_input_1.outputs["Height"],
            "Height": group_input_1.outputs["Width"],
            "Amount": add_1,
        },
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": lineseq_1, "Rotation": (0.0000, 0.0000, 1.5708)},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: 0.0010},
        attrs={"operation": "SUBTRACT"},
    )

    quadrilateral_3 = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": reroute, "Height": subtract_1},
    )

    curve_to_mesh_2 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": transform,
            "Profile Curve": quadrilateral_3,
            "Fill Caps": True,
        },
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [curve_to_mesh_1, curve_to_mesh_2]},
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [curve_to_mesh, join_geometry_3]}
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": join_geometry_2,
            "Material": group_input_1.outputs["FrameMaterial"],
        },
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input_1.outputs["Width"],
            "Y": group_input_1.outputs["Height"],
            "Z": group_input_1.outputs["GlassThickness"],
        },
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Name": "uv_map",
            "Value": cube.outputs["UV Map"],
        },
        attrs={"data_type": "FLOAT_VECTOR", "domain": "CORNER"},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": store_named_attribute,
            "Material": group_input_1.outputs["GlassMaterial"],
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, set_material_1]}
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input_1.outputs["WithGlass"],
            "False": set_material_1,
            "True": join_geometry,
        },
    )

    set_shade_smooth = nw.new_node(
        Nodes.SetShadeSmooth, input_kwargs={"Geometry": switch, "Shade Smooth": False}
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": set_shade_smooth}
    )

    addhandletopanel = nw.new_node(
        nodegroup_add_handle_to_panel().name,
        input_kwargs={
            "HandleThickness": group_input_1.outputs["HandleThickness"],
            "HandleRoundness": group_input_1.outputs["HandleRoundness"],
            "HandleMaterial": group_input_1.outputs["FrameMaterial"],
            "Panel": realize_instances,
            "FrameWidth": group_input_1.outputs["PanelWidth"],
            "HandleLength": group_input_1.outputs["HandleLength"],
            "Width": group_input_1.outputs["Width"],
            "HangleOnRight": group_input_1.outputs["HangleOnRight"],
            "FrameThickness": group_input_1.outputs["PanelThickness"],
        },
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input_1.outputs["WithHandle"],
            "False": realize_instances,
            "True": addhandletopanel.outputs["Geometry"],
        },
    )

    realize_instances_1 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": switch_1}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": realize_instances_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input_4 = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 2.0000),
            ("NodeSocketFloat", "Height", 2.0000),
            ("NodeSocketFloat", "FrameWidth", 0.1000),
            ("NodeSocketFloat", "FrameThickness", 0.1000),
            ("NodeSocketInt", "PanelHAmount", 0),
            ("NodeSocketInt", "PanelVAmount", 0),
            ("NodeSocketFloat", "SubFrameWidth", 0.0500),
            ("NodeSocketFloat", "SubFrameThickness", 0.0500),
            ("NodeSocketInt", "SubPanelHAmount", 3),
            ("NodeSocketInt", "SubPanelVAmount", 2),
            ("NodeSocketFloat", "GlassThickness", 0.0100),
            ("NodeSocketBool", "Shutter", True),
            ("NodeSocketFloat", "ShutterPanelRadius", 0.0050),
            ("NodeSocketFloat", "ShutterWidth", 0.0500),
            ("NodeSocketFloat", "ShutterThickness", 0.0050),
            ("NodeSocketFloat", "ShutterRotation", 0.0000),
            ("NodeSocketFloat", "ShutterInterval", 0.0500),
            ("NodeSocketMaterial", "FrameMaterial", None),
            ("NodeSocketBool", "WithGlass", True),
            ("NodeSocketMaterial", "GlassMaterial", None),
            ("NodeSocketBool", "JointType", True),
            ("NodeSocketFloat", "JointValue", 0.1000),
            ("NodeSocketBool", "WithHandle", True),
            ("NodeSocketFloat", "HandleRoundness", 0.0000),
            ("NodeSocketFloat", "HandleLength", 0.5000),
            ("NodeSocketFloat", "HandleThickness", 0.0100),
            ("NodeSocketBool", "SameSideRotation", False),
            ("NodeSocketBool", "HasLock", False),
            ("NodeSocketFloat", "LockRoundness", 1.0000),
        ],
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": group_input_4.outputs["JointType"],
            3: group_input_4.outputs["FrameThickness"],
            4: group_input_4.outputs["SubFrameThickness"],
        },
    )

    windowpane = nw.new_node(
        nodegroup_window_pane().name,
        input_kwargs={
            "Width": group_input_4.outputs["Width"],
            "Height": group_input_4.outputs["Height"],
            "FrameWidth": group_input_4.outputs["FrameWidth"],
            "FrameThickness": group_input_4.outputs["FrameThickness"],
            "PanelWidth": group_input_4.outputs["FrameWidth"],
            "PanelThickness": map_range.outputs["Result"],
            "PanelHAmount": group_input_4.outputs["PanelVAmount"],
            "PanelVAmount": group_input_4.outputs["PanelHAmount"],
            "FrameMaterial": group_input_4.outputs["FrameMaterial"],
            "GlassMaterial": group_input_4.outputs["GlassMaterial"],
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": windowpane, "Label": "Frame"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_4.outputs["PanelHAmount"], 1: 1.0000},
        attrs={"operation": "SUBTRACT"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_4.outputs["Width"], 1: subtract},
        attrs={"operation": "MULTIPLY"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: group_input_4.outputs["PanelHAmount"]},
        attrs={"operation": "DIVIDE"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_4.outputs["PanelVAmount"], 1: 1.0000},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_4.outputs["Height"], 1: subtract_1},
        attrs={"operation": "MULTIPLY"},
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: group_input_4.outputs["PanelVAmount"]},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide, "Y": divide_1}
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_2,
            "Vertices X": group_input_4.outputs["PanelHAmount"],
            "Vertices Y": group_input_4.outputs["PanelVAmount"],
            "Vertices Z": 1,
        },
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_4.outputs["FrameWidth"], 1: 2.5000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_4.outputs["FrameWidth"], 1: 0.8000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_4.outputs["FrameThickness"], 1: 0.3000},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_15 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_4.outputs["LockRoundness"]}
    )

    lockbase = nw.new_node(
        nodegroup_lock_base().name,
        input_kwargs={
            "Length": multiply_2,
            "Width": multiply_3,
            "Height": multiply_4,
            "Roundness": reroute_15,
        },
    )

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_4.outputs["FrameMaterial"]}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial, input_kwargs={"Geometry": lockbase, "Material": reroute_14}
    )

    geometry_to_instance = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": set_material}
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": cube_1.outputs["Mesh"],
            "Instance": geometry_to_instance,
        },
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "Y": group_input_4.outputs["Height"],
            "Z": group_input_4.outputs["FrameThickness"],
        },
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_4.outputs["PanelVAmount"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: multiply_5},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": divide_2, "Z": 0.5000}
    )

    multiply_6 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_3, 1: combine_xyz_4},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": instance_on_points,
            "Translation": multiply_6.outputs["Vector"],
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [add_jointed_geometry_metadata, transform_geometry]},
    )

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": join_geometry, "Label": "Frame&LockBase"},
    )

    multiply_7 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_4}, attrs={"operation": "MULTIPLY"}
    )

    multiply_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_3, 1: 0.8000},
        attrs={"operation": "MULTIPLY"},
    )

    lever = nw.new_node(
        nodegroup_lever().name,
        input_kwargs={
            "Thickness": multiply_7,
            "BottomWidth": multiply_8,
            "Roundness": reroute_15,
            "Length": multiply_2,
        },
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial, input_kwargs={"Geometry": lever, "Material": reroute_14}
    )

    add_jointed_geometry_metadata_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_1, "Label": "Lever"},
    )

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_4})

    add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_6.outputs["Vector"], 1: combine_xyz_7},
    )

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "Lock",
            "Parent": add_jointed_geometry_metadata_1,
            "Child": add_jointed_geometry_metadata_2,
            "Position": add.outputs["Vector"],
            "Value": 0.0000,
            "Max": 1.0000,
        },
    )

    duplicate_joints_on_parent = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Parent": hinge_joint.outputs["Parent"],
            "Child": hinge_joint.outputs["Child"],
            "Points": cube_1.outputs["Mesh"],
        },
    )

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input_4.outputs["HasLock"],
            "False": add_jointed_geometry_metadata,
            "True": duplicate_joints_on_parent,
        },
    )

    add_jointed_geometry_metadata_3 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": switch_5, "Label": "Fame&Lock"},
    )

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_3}
    )

    multiply_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input_4.outputs["FrameWidth"],
            1: group_input_4.outputs["PanelHAmount"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_4.outputs["Width"], 1: multiply_9},
        attrs={"operation": "SUBTRACT"},
    )

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_2, 1: group_input_4.outputs["PanelHAmount"]},
        attrs={"operation": "DIVIDE"},
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_3, 1: group_input_4.outputs["SubFrameWidth"]},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input_4.outputs["FrameWidth"],
            1: group_input_4.outputs["PanelVAmount"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    subtract_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_4.outputs["Height"], 1: multiply_10},
        attrs={"operation": "SUBTRACT"},
    )

    divide_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_4, 1: group_input_4.outputs["PanelVAmount"]},
        attrs={"operation": "DIVIDE"},
    )

    subtract_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_4, 1: group_input_4.outputs["SubFrameWidth"]},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_4.outputs["WithHandle"]}
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_4.outputs["HandleLength"]}
    )

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_4.outputs["HandleThickness"]}
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_4.outputs["HandleRoundness"]}
    )

    windowpane_1 = nw.new_node(
        nodegroup_window_pane().name,
        input_kwargs={
            "Width": subtract_3,
            "Height": subtract_5,
            "FrameWidth": group_input_4.outputs["SubFrameWidth"],
            "FrameThickness": group_input_4.outputs["SubFrameThickness"],
            "PanelWidth": group_input_4.outputs["SubFrameWidth"],
            "PanelThickness": group_input_4.outputs["SubFrameThickness"],
            "PanelHAmount": group_input_4.outputs["SubPanelHAmount"],
            "PanelVAmount": group_input_4.outputs["SubPanelVAmount"],
            "WithGlass": group_input_4.outputs["WithGlass"],
            "GlassThickness": group_input_4.outputs["GlassThickness"],
            "FrameMaterial": group_input_4.outputs["FrameMaterial"],
            "GlassMaterial": group_input_4.outputs["GlassMaterial"],
            "WithHandle": reroute_2,
            "HandleLength": reroute_4,
            "HandleThickness": reroute_5,
            "HandleRoundness": reroute_6,
            "HangleOnRight": True,
        },
    )

    windowshutter = nw.new_node(
        nodegroup_window_shutter().name,
        input_kwargs={
            "Width": subtract_3,
            "Height": subtract_5,
            "FrameWidth": group_input_4.outputs["FrameWidth"],
            "FrameThickness": group_input_4.outputs["FrameThickness"],
            "PanelWidth": group_input_4.outputs["ShutterPanelRadius"],
            "PanelThickness": group_input_4.outputs["ShutterPanelRadius"],
            "ShutterWidth": group_input_4.outputs["ShutterWidth"],
            "ShutterThickness": group_input_4.outputs["ShutterThickness"],
            "ShutterInterval": group_input_4.outputs["ShutterInterval"],
            "ShutterRotation": group_input_4.outputs["ShutterRotation"],
            "FrameMaterial": group_input_4.outputs["FrameMaterial"],
            "WithHandle": reroute_2,
            "HandleLength": reroute_4,
            "HandleThickness": reroute_5,
            "HandleRoundness": reroute_6,
            "HangleOnRight": True,
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input_4.outputs["Shutter"],
            "False": windowpane_1,
            "True": windowshutter.outputs["Geometry"],
        },
    )

    divide_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input_4.outputs["Width"],
            1: group_input_4.outputs["PanelHAmount"],
        },
        attrs={"operation": "DIVIDE"},
    )

    divide_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input_4.outputs["Height"],
            1: group_input_4.outputs["PanelVAmount"],
        },
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_5, "Y": divide_6}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_1})

    multiply_11 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_11, 1: (-0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": switch, "Translation": multiply_11.outputs["Vector"]},
    )

    add_jointed_geometry_metadata_4 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_3, "Label": "Pane"},
    )

    multiply_12 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_11.outputs["Vector"], 1: (-1.0000, -1.0000, -1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_4.outputs["JointValue"]}
    )

    hinge_joint_1 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "HingePanelL",
            "Parent": reroute_10,
            "Child": add_jointed_geometry_metadata_4,
            "Position": multiply_12.outputs["Vector"],
            "Axis": (0.0000, 1.0000, 0.0000),
            "Value": reroute_12,
            "Max": 1.5000,
        },
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input_4.outputs["Width"],
            "Y": group_input_4.outputs["Height"],
        },
    )

    subtract_6 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: combine_xyz_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide_5})

    subtract_7 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: subtract_6.outputs["Vector"], 1: combine_xyz_6},
        attrs={"operation": "SUBTRACT"},
    )

    divide_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_4.outputs["PanelHAmount"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    round = nw.new_node(
        Nodes.Math, input_kwargs={0: divide_7}, attrs={"operation": "ROUND"}
    )

    cube_2 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": subtract_7.outputs["Vector"],
            "Vertices X": round,
            "Vertices Y": group_input_4.outputs["PanelVAmount"],
            "Vertices Z": 1,
        },
    )

    divide_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input_4.outputs["Width"],
            1: group_input_4.outputs["PanelHAmount"],
        },
        attrs={"operation": "DIVIDE"},
    )

    divide_9 = nw.new_node(
        Nodes.Math, input_kwargs={0: divide_8, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    multiply_13 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input_4.outputs["SubFrameThickness"],
            1: group_input_4.outputs["JointType"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    multiply_14 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_13, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_9, "Z": multiply_14}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube_2.outputs["Mesh"], "Translation": combine_xyz_5},
    )

    duplicate_joints_on_parent_1 = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Parent": hinge_joint_1.outputs["Parent"],
            "Child": hinge_joint_1.outputs["Child"],
            "Points": transform_geometry_1,
        },
    )

    add_jointed_geometry_metadata_5 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": duplicate_joints_on_parent_1, "Label": "Frame&PaneL"},
    )

    multiply_15 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input_4.outputs["FrameWidth"],
            1: group_input_4.outputs["PanelHAmount"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    subtract_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_4.outputs["Width"], 1: multiply_15},
        attrs={"operation": "SUBTRACT"},
    )

    divide_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_8, 1: group_input_4.outputs["PanelHAmount"]},
        attrs={"operation": "DIVIDE"},
    )

    subtract_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_10, 1: group_input_4.outputs["SubFrameWidth"]},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_16 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input_4.outputs["FrameWidth"],
            1: group_input_4.outputs["PanelVAmount"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    subtract_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_4.outputs["Height"], 1: multiply_16},
        attrs={"operation": "SUBTRACT"},
    )

    divide_11 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_10, 1: group_input_4.outputs["PanelVAmount"]},
        attrs={"operation": "DIVIDE"},
    )

    subtract_11 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_11, 1: group_input_4.outputs["SubFrameWidth"]},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_4.outputs["WithHandle"]}
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_4.outputs["HandleLength"]}
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_4.outputs["HandleThickness"]}
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_4.outputs["HandleRoundness"]}
    )

    windowpane_2 = nw.new_node(
        nodegroup_window_pane().name,
        input_kwargs={
            "Width": subtract_9,
            "Height": subtract_11,
            "FrameWidth": group_input_4.outputs["SubFrameWidth"],
            "FrameThickness": group_input_4.outputs["SubFrameThickness"],
            "PanelWidth": group_input_4.outputs["SubFrameWidth"],
            "PanelThickness": group_input_4.outputs["SubFrameThickness"],
            "PanelHAmount": group_input_4.outputs["SubPanelHAmount"],
            "PanelVAmount": group_input_4.outputs["SubPanelVAmount"],
            "WithGlass": group_input_4.outputs["WithGlass"],
            "GlassThickness": group_input_4.outputs["GlassThickness"],
            "FrameMaterial": group_input_4.outputs["FrameMaterial"],
            "GlassMaterial": group_input_4.outputs["GlassMaterial"],
            "WithHandle": reroute_3,
            "HandleLength": reroute_7,
            "HandleThickness": reroute_8,
            "HandleRoundness": reroute_9,
        },
    )

    windowshutter_1 = nw.new_node(
        nodegroup_window_shutter().name,
        input_kwargs={
            "Width": subtract_9,
            "Height": subtract_11,
            "FrameWidth": group_input_4.outputs["FrameWidth"],
            "FrameThickness": group_input_4.outputs["FrameThickness"],
            "PanelWidth": group_input_4.outputs["ShutterPanelRadius"],
            "PanelThickness": group_input_4.outputs["ShutterPanelRadius"],
            "ShutterWidth": group_input_4.outputs["ShutterWidth"],
            "ShutterThickness": group_input_4.outputs["ShutterThickness"],
            "ShutterInterval": group_input_4.outputs["ShutterInterval"],
            "ShutterRotation": group_input_4.outputs["ShutterRotation"],
            "FrameMaterial": group_input_4.outputs["FrameMaterial"],
            "WithHandle": reroute_3,
            "HandleLength": reroute_7,
            "HandleThickness": reroute_8,
            "HandleRoundness": reroute_9,
        },
    )

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input_4.outputs["Shutter"],
            "False": windowpane_2,
            "True": windowshutter_1.outputs["Geometry"],
        },
    )

    multiply_17 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_11, 1: (0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": switch_4,
            "Translation": multiply_17.outputs["Vector"],
        },
    )

    add_jointed_geometry_metadata_6 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_4, "Label": "Pane"},
    )

    multiply_18 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_17.outputs["Vector"], 1: (-1.0000, -1.0000, -1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    hinge_joint_2 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "HingePanelR",
            "Parent": add_jointed_geometry_metadata_5,
            "Child": add_jointed_geometry_metadata_6,
            "Position": multiply_18.outputs["Vector"],
            "Axis": (0.0000, -1.0000, 0.0000),
            "Value": reroute_12,
            "Max": 1.5000,
        },
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_5, "Scale": -1.0000},
        attrs={"operation": "SCALE"},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_2.outputs["Mesh"],
            "Translation": scale.outputs["Vector"],
        },
    )

    duplicate_joints_on_parent_2 = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Parent": hinge_joint_2.outputs["Parent"],
            "Child": hinge_joint_2.outputs["Child"],
            "Points": transform_geometry_2,
        },
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": subtract_6.outputs["Vector"],
            "Vertices X": group_input_4.outputs["PanelHAmount"],
            "Vertices Y": group_input_4.outputs["PanelVAmount"],
            "Vertices Z": 1,
        },
    )

    duplicate_joints_on_parent_3 = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Parent": hinge_joint_1.outputs["Parent"],
            "Child": hinge_joint_1.outputs["Child"],
            "Points": cube.outputs["Mesh"],
        },
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input_4.outputs["SameSideRotation"],
            "False": duplicate_joints_on_parent_2,
            "True": duplicate_joints_on_parent_3,
        },
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_4.outputs["JointValue"]}
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract_3})

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "SlidingPanelL",
            "Parent": reroute_10,
            "Child": add_jointed_geometry_metadata_4,
            "Position": multiply_12.outputs["Vector"],
            "Axis": (-1.0000, 0.0000, 0.0000),
            "Value": reroute_13,
            "Max": reroute,
        },
    )

    duplicate_joints_on_parent_4 = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Parent": sliding_joint.outputs["Parent"],
            "Child": sliding_joint.outputs["Child"],
            "Points": transform_geometry_1,
        },
    )

    add_jointed_geometry_metadata_7 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": duplicate_joints_on_parent_4, "Label": "Frame&PaneL"},
    )

    sliding_joint_1 = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "SlidingPanelR",
            "Parent": add_jointed_geometry_metadata_7,
            "Child": add_jointed_geometry_metadata_6,
            "Position": multiply_18.outputs["Vector"],
            "Axis": (1.0000, 0.0000, 0.0000),
            "Value": reroute_13,
            "Max": reroute,
        },
    )

    duplicate_joints_on_parent_5 = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Parent": sliding_joint_1.outputs["Parent"],
            "Child": sliding_joint_1.outputs["Child"],
            "Points": transform_geometry_2,
        },
    )

    duplicate_joints_on_parent_6 = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Parent": sliding_joint.outputs["Parent"],
            "Child": sliding_joint.outputs["Child"],
            "Points": cube.outputs["Mesh"],
        },
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input_4.outputs["SameSideRotation"],
            "False": duplicate_joints_on_parent_5,
            "True": duplicate_joints_on_parent_6,
        },
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input_4.outputs["JointType"],
            "False": switch_3,
            "True": switch_6,
        },
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": switch_1, "Rotation": (-1.5708, 0.0000, -1.5708)},
    )

    add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input_4.outputs["Height"],
            1: group_input_4.outputs["FrameWidth"],
        },
    )

    multiply_19 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_19})

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_5, "Translation": combine_xyz_8},
    )

    add_jointed_geometry_metadata_8 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_6, "Label": "Window"},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances,
        input_kwargs={"Geometry": add_jointed_geometry_metadata_8},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": realize_instances},
        attrs={"is_active_output": True},
    )


class WindowFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    @gin.configurable(module="WindowFactory")
    def sample_joint_parameters(
        cls,
        Lock_stiffness_min: float = 0.0,
        Lock_stiffness_max: float = 0.0,
        Lock_damping_min: float = 1.2,
        Lock_damping_max: float = 1.6,
        HingePanelL_stiffness_min: float = 0.0,
        HingePanelL_stiffness_max: float = 0.0,
        HingePanelL_damping_min: float = 150.0,
        HingePanelL_damping_max: float = 150.0,
        HingePanelR_stiffness_min: float = 0.0,
        HingePanelR_stiffness_max: float = 0.0,
        HingePanelR_damping_min: float = 150.0,
        HingePanelR_damping_max: float = 150.0,
        SlidingPanelL_stiffness_min: float = 0.0,
        SlidingPanelL_stiffness_max: float = 0.0,
        SlidingPanelL_damping_min: float = 4000.0,
        SlidingPanelL_damping_max: float = 5000.0,
        SlidingPanelR_stiffness_min: float = 0.0,
        SlidingPanelR_stiffness_max: float = 0.0,
        SlidingPanelR_damping_min: float = 4000.0,
        SlidingPanelR_damping_max: float = 5000.0,
    ):
        return {
            "Lock": {
                "stiffness": uniform(Lock_stiffness_min, Lock_stiffness_max),
                "damping": uniform(Lock_damping_min, Lock_damping_max),
                "friction": 10000,
            },
                    HingePanelL_stiffness_min, HingePanelL_stiffness_max
                "damping": uniform(HingePanelL_damping_min, HingePanelL_damping_max),
                "stiffness": uniform(
                    HingePanelR_stiffness_min, HingePanelR_stiffness_max
                ),
                "damping": uniform(HingePanelR_damping_min, HingePanelR_damping_max),
            },
            "SlidingPanelL": {
                "stiffness": uniform(
                    SlidingPanelL_stiffness_min, SlidingPanelL_stiffness_max
                ),
                "damping": uniform(
                    SlidingPanelL_damping_min, SlidingPanelL_damping_max
                ),
            },
            "SlidingPanelR": {
                "stiffness": uniform(
                    SlidingPanelR_stiffness_min, SlidingPanelR_stiffness_max
                ),
                "damping": uniform(
                    SlidingPanelR_damping_min, SlidingPanelR_damping_max
                ),
            },
        }

    def sample_parameters(self):
        # add code here to randomly sample from parameters

        joint_type = randint(0, 2)  # 0 for hinge, 1 for sliding

        width = uniform(1, 4)
        height = uniform(1, 2)
        frame_width = uniform(0.05, 0.1)
        frame_thickness = uniform(0.04, frame_width)
        if joint_type == 1:
            frame_thickness *= 3

        if joint_type == 1:
            panel_h_amount = choice([2, 4])
        else:
            panel_h_amount = choice([1, 2, 4])
        panel_v_amount = 1

        sub_frame_width = uniform(0.02, frame_width)
        sub_frame_thickness = uniform(0.02, frame_thickness)
        sub_frame_h_amount = randint(1, 4)
        sub_frame_v_amount = randint(1, 4)
        glass_thickness = uniform(0.01, sub_frame_thickness)

        is_shutter = randint(0, 2) if joint_type == 0 else False
        shutter_panel_radius = uniform(0.001, 0.003)
        shutter_width = uniform(0.03, 0.05)
        shutter_thickness = uniform(0.003, 0.007)
        shutter_rotation = uniform(0, 1)
        shutter_inverval = shutter_width + uniform(0.001, 0.003)

        with_glass = True
        with_handle = randint(0, 2) if joint_type == 0 else False
        handle_roundness = uniform(0, 1)
        handle_length = uniform(0.2, 0.7)
        handle_thickness = uniform(0.01, 0.03)

        same_side_rotation = randint(0, 2) if panel_h_amount > 1 else True

        has_lock = True
        lock_roundness = uniform(0, 1)

        shader_frame_material_choice = weighted_sample(material_assignments.woods)()()
        shader_glass_material_choice = weighted_sample(material_assignments.glasses)()()

        params = {
            "Width": width,
            "Height": height,
            "FrameWidth": frame_width,
            "FrameThickness": frame_thickness,
            "PanelHAmount": panel_h_amount,
            "PanelVAmount": panel_v_amount,
            "SubFrameWidth": sub_frame_width,
            "SubFrameThickness": sub_frame_thickness,
            "SubPanelHAmount": sub_frame_h_amount,
            "SubPanelVAmount": sub_frame_v_amount,
            "GlassThickness": glass_thickness,
            "Shutter": is_shutter,
            "ShutterPanelRadius": shutter_panel_radius,
            "ShutterWidth": shutter_width,
            "ShutterThickness": shutter_thickness,
            "ShutterRotation": shutter_rotation,
            "ShutterInterval": shutter_inverval,
            "WithGlass": with_glass,
            "FrameMaterial": shader_frame_material_choice,
            "GlassMaterial": shader_glass_material_choice,
            "JointType": joint_type,
            "JointValue": joint_value,
            "WithHandle": with_handle,
            "HandleRoundness": handle_roundness,
            "HandleLength": handle_length,
            "HandleThickness": handle_thickness,
            "SameSideRotation": same_side_rotation,
            "HasLock": has_lock,
            "LockRoundness": lock_roundness,
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
