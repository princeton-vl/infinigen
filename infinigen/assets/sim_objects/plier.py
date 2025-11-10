import gin
from numpy.random import randint, uniform

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
    "nodegroup_plier_handle_001", singleton=False, type="GeometryNodeTree"
)
def nodegroup_plier_handle_001(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ)

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Length", 0.0000),
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "joint_radius", 0.0000),
            ("NodeSocketFloat", "handle_head_length_ratio", 0.0000),
            ("NodeSocketFloat", "Handle_curvature", 0.0000),
            ("NodeSocketFloat", "handle_thickness", 0.0000),
            ("NodeSocketFloat", "handle shape", 0.0000),
            ("NodeSocketInt", "handle_patterned", 0),
            ("NodeSocketMaterial", "handle_material", None),
            ("NodeSocketMaterial", "metal_material", None),
            ("NodeSocketFloat", "Value", 0.1000),
        ],
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle_curvature"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Length"],
            1: group_input.outputs["handle_head_length_ratio"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_5, 1: multiply_1},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_2})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["handle shape"]}
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_8},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["handle shape"], 3: 1},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 1.0000

    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 0.8000

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal_1, "False": value_1, "True": value_2},
        attrs={"input_type": "FLOAT"},
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.5000

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": switch_1, "True": value},
        attrs={"input_type": "FLOAT"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_14, 1: switch},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_17, "Y": multiply_3}
    )

    multiply_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_1}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_1, "Y": multiply_4}
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_1})

    quadratic_b_zier = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={"Start": combine_xyz_3, "Middle": combine_xyz, "End": reroute_16},
    )

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Value"]}
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: 0.0000, 1: reroute_12})

    trim_curve = nw.new_node(
        Nodes.TrimCurve, input_kwargs={"Curve": quadratic_b_zier, 2: add}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    multiply_5 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_15}, attrs={"operation": "MULTIPLY"}
    )

    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={"Radius": multiply_5})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["joint_radius"],
            1: group_input.outputs["Height"],
        },
        attrs={"operation": "DIVIDE"},
    )

    multiply_6 = nw.new_node(
        Nodes.Math, input_kwargs={0: divide, 1: 1.5000}, attrs={"operation": "MULTIPLY"}
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["handle_thickness"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_6, 1: reroute_7},
        attrs={"operation": "MINIMUM"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": minimum, "Y": 1.0000, "Z": 1.0000}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle.outputs["Curve"],
            "Scale": combine_xyz_5,
        },
    )

    reroute_20 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_2}
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": trim_curve,
            "Profile Curve": reroute_20,
            "Fill Caps": True,
        },
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["handle_material"]}
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": curve_to_mesh, "Material": reroute_10},
    )

    trim_curve_1 = nw.new_node(
        Nodes.TrimCurve, input_kwargs={"Curve": quadratic_b_zier, 3: add}
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_5, 1: 0.8000},
        attrs={"operation": "MULTIPLY"},
    )

    curve_circle_1 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 4, "Radius": multiply_7}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle_1.outputs["Curve"],
            "Rotation": (0.0000, 0.0000, 0.7854),
        },
    )

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_5})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry, "Scale": reroute_19},
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": trim_curve_1,
            "Profile Curve": transform_geometry_3,
            "Fill Caps": True,
        },
    )

    set_shade_smooth = nw.new_node(
        Nodes.SetShadeSmooth,
        input_kwargs={"Geometry": curve_to_mesh_1, "Shade Smooth": False},
    )

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["metal_material"]}
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": set_shade_smooth, "Material": reroute_11},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, set_material_1]}
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_5})

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_18, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": divide_1})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry,
            "Translation": combine_xyz_4,
            "Rotation": (0.0000, 0.0000, -0.1000),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_plier_head_001", singleton=False, type="GeometryNodeTree"
)
def nodegroup_plier_head_001(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Length", 0.5000),
            ("NodeSocketInt", "jagged_edge", 0),
            ("NodeSocketFloat", "Width", 0.5000),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketInt", "head_type", 0),
            ("NodeSocketFloat", "head_width", 0.0000),
            ("NodeSocketFloat", "handle_head_length_ratio", 0.0000),
            ("NodeSocketFloat", "joint_radius", 0.0000),
            ("NodeSocketMaterial", "Material", None),
            ("NodeSocketFloat", "head curvature", 0.0150),
            ("NodeSocketFloat", "number of jagged edges", 30.0000),
        ],
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["head_type"]}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_4},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Length"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: group_input.outputs["handle_head_length_ratio"]},
        attrs={"operation": "SUBTRACT"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: subtract},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply})

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": reroute_21})

    reroute_34 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_7})

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_34})

    reroute_38 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_35})

    quadratic_b_zier_4 = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Start": (0.0000, 0.0000, 0.0000),
            "Middle": (0.0010, 0.0000, 0.0000),
            "End": reroute_38,
        },
    )

    curve_to_mesh_2 = nw.new_node(
        Nodes.CurveToMesh, input_kwargs={"Curve": quadratic_b_zier_4}
    )

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["head_width"]}
    )

    reroute_15 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["joint_radius"]}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_14, 1: reroute_15},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_1})

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_6, 1: (0.0000, -1.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    quadratic_b_zier_5 = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Start": (0.0000, 0.0000, 0.0000),
            "Middle": (0.0000, -0.0010, 0.0000),
            "End": multiply_2.outputs["Vector"],
        },
    )

    curve_to_mesh_4 = nw.new_node(
        Nodes.CurveToMesh, input_kwargs={"Curve": quadratic_b_zier_5}
    )

    combine_xyz_10 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": group_input.outputs["joint_radius"]}
    )

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_10, 1: (-0.2000, -0.2000, -0.2000)},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_3.outputs["Vector"], 1: combine_xyz_7},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": add.outputs["Vector"]}
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: 0.7713},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: -1.0000, 1: multiply_1},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_11 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_4, "Y": multiply_5}
    )

    reroute_37 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add.outputs["Vector"]}
    )

    quadratic_b_zier_6 = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Start": multiply_2.outputs["Vector"],
            "Middle": combine_xyz_11,
            "End": reroute_37,
        },
    )

    curve_to_mesh_7 = nw.new_node(
        Nodes.CurveToMesh, input_kwargs={"Curve": quadratic_b_zier_6}
    )

    quadratic_b_zier_7 = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Start": reroute_35,
            "Middle": add.outputs["Vector"],
            "End": add.outputs["Vector"],
        },
    )

    curve_to_mesh_8 = nw.new_node(
        Nodes.CurveToMesh, input_kwargs={"Curve": quadratic_b_zier_7}
    )

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                curve_to_mesh_2,
                curve_to_mesh_4,
                curve_to_mesh_7,
                curve_to_mesh_8,
            ]
        },
    )

    merge_by_distance_2 = nw.new_node(
        Nodes.MergeByDistance, input_kwargs={"Geometry": join_geometry_4}
    )

    mesh_to_curve_2 = nw.new_node(
        Nodes.MeshToCurve, input_kwargs={"Mesh": merge_by_distance_2}
    )

    fill_curve_1 = nw.new_node(
        Nodes.FillCurve,
        input_kwargs={"Curve": mesh_to_curve_2},
        attrs={"mode": "NGONS"},
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_17})

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_18, 1: 0.9500},
        attrs={"operation": "MULTIPLY"},
    )

    extrude_mesh_3 = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": fill_curve_1,
            "Offset Scale": multiply_6,
            "Individual": False,
        },
    )

    reroute_47 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": extrude_mesh_3.outputs["Mesh"]}
    )

    reroute_48 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_47})

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (0.0500, 0.0150, 0.3000)})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Translation": (0.0000, 0.0075, 0.0000),
            "Rotation": (3.1416, 1.5708, 0.0000),
        },
    )

    combine_xyz_14 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": 0.5530})

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry, "Rotation": combine_xyz_14},
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": reroute_48, "Mesh 2": transform_geometry_5},
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_18, 1: 0.9500},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_12 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_7})

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": extrude_mesh_3.outputs["Mesh"],
            "Translation": combine_xyz_12,
            "Scale": (1.0000, 1.0000, -1.0000),
        },
    )

    difference_1 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": transform_geometry_6, "Mesh 2": transform_geometry_5},
    )

    join_geometry_6 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [difference.outputs["Mesh"], difference_1.outputs["Mesh"]]
        },
    )

    convex_hull = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": join_geometry_6}
    )

    reroute_44 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    multiply_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_44, 1: -0.4750},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_8})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": convex_hull,
            "Translation": combine_xyz_9,
            "Rotation": (0.0000, 0.0000, -0.1000),
        },
    )

    multiply_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["head_width"],
            1: group_input.outputs["joint_radius"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_9})

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_1})

    multiply_10 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_19, 1: (0.0000, -1.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_26 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply_10.outputs["Vector"]}
    )

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_26})

    quadratic_b_zier_1 = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Start": (0.0000, 0.0000, 0.0000),
            "Middle": (0.0000, -0.0010, 0.0000),
            "End": reroute_27,
        },
    )

    curve_to_mesh_3 = nw.new_node(
        Nodes.CurveToMesh, input_kwargs={"Curve": quadratic_b_zier_1}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["jagged_edge"]}
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: 1, 3: reroute_2},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply})

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_2})

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    reroute_32 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_32})

    quadratic_b_zier = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Start": (0.0000, 0.0000, 0.0000),
            "Middle": (0.0010, 0.0000, 0.0000),
            "End": reroute_33,
        },
    )

    curve_to_points = nw.new_node(
        Nodes.CurveToPoints,
        input_kwargs={"Curve": quadratic_b_zier, "Length": 0.0050},
        attrs={"mode": "LENGTH"},
    )

    curve_line_1 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={"Start": combine_xyz_2, "End": (0.0000, 0.0000, 0.0000)},
    )

    reroute_12 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["number of jagged edges"]},
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    curve_to_points_8 = nw.new_node(
        Nodes.CurveToPoints, input_kwargs={"Curve": curve_line_1, "Count": reroute_13}
    )

    index_1 = nw.new_node(Nodes.Index)

    modulo = nw.new_node(
        Nodes.Math, input_kwargs={0: index_1, 1: 2.0000}, attrs={"operation": "MODULO"}
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": curve_to_points_8.outputs["Points"],
            "Selection": modulo,
            "Offset": (0.0000, -0.0010, 0.0000),
        },
    )

    points_to_curves_2 = nw.new_node(
        "GeometryNodePointsToCurves", input_kwargs={"Points": set_position_1}
    )

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    reroute_30 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_29})

    curve_to_points_9 = nw.new_node(
        Nodes.CurveToPoints,
        input_kwargs={"Curve": points_to_curves_2, "Count": reroute_30},
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_1,
            "False": curve_to_points.outputs["Points"],
            "True": curve_to_points_9.outputs["Points"],
        },
    )

    points_to_curves_6 = nw.new_node(
        "GeometryNodePointsToCurves", input_kwargs={"Points": switch_3}
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh, input_kwargs={"Curve": points_to_curves_6}
    )

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["joint_radius"]}
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": reroute_5})

    multiply_11 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_5, 1: (-0.2000, -0.2000, -0.2000)},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_11.outputs["Vector"], 1: combine_xyz_2},
    )

    add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_1.outputs["Vector"], 1: multiply_10.outputs["Vector"]},
    )

    divide = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_2.outputs["Vector"], 1: (2.0000, 2.0000, 2.0000)},
        attrs={"operation": "DIVIDE"},
    )

    cross_product = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: divide.outputs["Vector"], 1: (0.0000, 0.0000, 1.0000)},
        attrs={"operation": "CROSS_PRODUCT"},
    )

    normalize = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: cross_product.outputs["Vector"]},
        attrs={"operation": "NORMALIZE"},
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["head curvature"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_7, 1: 0.0000})

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_3, "Y": add_3, "Z": add_3}
    )

    multiply_12 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: normalize.outputs["Vector"], 1: combine_xyz_3},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_31 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_2.outputs["Vector"]}
    )

    divide_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_31, 1: (3.0000, 3.0000, 3.0000)},
        attrs={"operation": "DIVIDE"},
    )

    add_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_12.outputs["Vector"], 1: divide_1.outputs["Vector"]},
    )

    reroute_28 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_1.outputs["Vector"]}
    )

    quadratic_b_zier_2 = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Start": reroute_27,
            "Middle": add_4.outputs["Vector"],
            "End": reroute_28,
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh, input_kwargs={"Curve": quadratic_b_zier_2}
    )

    quadratic_b_zier_3 = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Start": reroute_23,
            "Middle": add_1.outputs["Vector"],
            "End": add_1.outputs["Vector"],
        },
    )

    curve_to_mesh_6 = nw.new_node(
        Nodes.CurveToMesh, input_kwargs={"Curve": quadratic_b_zier_3}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                curve_to_mesh_3,
                curve_to_mesh_1,
                curve_to_mesh,
                curve_to_mesh_6,
            ]
        },
    )

    merge_by_distance = nw.new_node(
        Nodes.MergeByDistance, input_kwargs={"Geometry": join_geometry}
    )

    mesh_to_curve = nw.new_node(
        Nodes.MeshToCurve, input_kwargs={"Mesh": merge_by_distance}
    )

    fill_curve = nw.new_node(
        Nodes.FillCurve, input_kwargs={"Curve": mesh_to_curve}, attrs={"mode": "NGONS"}
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_24})

    multiply_13 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_25, 1: 0.9500},
        attrs={"operation": "MULTIPLY"},
    )

    extrude_mesh_1 = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": fill_curve,
            "Offset Scale": multiply_13,
            "Individual": False,
        },
    )

    multiply_14 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_25, 1: -0.4750},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_14})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": extrude_mesh_1.outputs["Mesh"],
            "Translation": combine_xyz_4,
            "Rotation": (0.0000, 0.0000, -0.1000),
        },
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_3,
            "Scale": (1.0000, 1.0000, -1.0000),
        },
    )

    flip_faces = nw.new_node(
        Nodes.FlipFaces, input_kwargs={"Mesh": transform_geometry_2}
    )

    reroute_50 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_3}
    )

    join_geometry_10 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [flip_faces, reroute_50]}
    )

    merge_by_distance_3 = nw.new_node(
        Nodes.MergeByDistance, input_kwargs={"Geometry": join_geometry_10}
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal,
            "False": transform_geometry_4,
            "True": merge_by_distance_3,
        },
    )

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Material"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_49 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    set_material = nw.new_node(
        Nodes.SetMaterial, input_kwargs={"Geometry": switch, "Material": reroute_49}
    )

    curve_line_2 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={"Start": combine_xyz_2, "End": (0.0000, 0.0000, 0.0000)},
    )

    curve_to_points_11 = nw.new_node(
        Nodes.CurveToPoints, input_kwargs={"Curve": curve_line_2, "Count": reroute_13}
    )

    index_2 = nw.new_node(Nodes.Index)

    modulo_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: index_2, 1: 2.0000}, attrs={"operation": "MODULO"}
    )

    set_position_2 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": curve_to_points_11.outputs["Points"],
            "Selection": modulo_1,
            "Offset": (0.0000, 0.0010, 0.0000),
        },
    )

    points_to_curves_3 = nw.new_node(
        "GeometryNodePointsToCurves", input_kwargs={"Points": set_position_2}
    )

    curve_to_mesh_5 = nw.new_node(
        Nodes.CurveToMesh, input_kwargs={"Curve": points_to_curves_3}
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                curve_to_mesh_6,
                curve_to_mesh_5,
                curve_to_mesh,
                curve_to_mesh_3,
            ]
        },
    )

    merge_by_distance_1 = nw.new_node(
        Nodes.MergeByDistance, input_kwargs={"Geometry": join_geometry_1}
    )

    mesh_to_curve_1 = nw.new_node(
        Nodes.MeshToCurve, input_kwargs={"Mesh": merge_by_distance_1}
    )

    fill_curve_2 = nw.new_node(
        Nodes.FillCurve,
        input_kwargs={"Curve": mesh_to_curve_1},
        attrs={"mode": "NGONS"},
    )

    multiply_15 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_25, 1: 0.9500},
        attrs={"operation": "MULTIPLY"},
    )

    extrude_mesh_4 = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={"Mesh": fill_curve_2, "Offset Scale": multiply_15},
    )

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": extrude_mesh_4.outputs["Mesh"],
            "Translation": combine_xyz_4,
            "Rotation": (0.0000, 0.0000, -0.1000),
        },
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_7, "Material": reroute_11},
    )

    transform_geometry_8 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": set_material_1, "Scale": (1.0000, 1.0000, -1.0000)},
    )

    flip_faces_1 = nw.new_node(
        Nodes.FlipFaces, input_kwargs={"Mesh": transform_geometry_8}
    )

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_1})

    reroute_52 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_51})

    join_geometry_9 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [flip_faces_1, reroute_52]}
    )

    merge_by_distance_4 = nw.new_node(
        Nodes.MergeByDistance, input_kwargs={"Geometry": join_geometry_9}
    )

    reroute_53 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": merge_by_distance_4})

    flip_faces_2 = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": reroute_53})

    group_output_1 = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"non_jagged": set_material, "jagged_opposite": flip_faces_2},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_joint_details_001", singleton=False, type="GeometryNodeTree"
)
def nodegroup_joint_details_001(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input_1 = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "joint_radius", 0.0000),
            ("NodeSocketMaterial", "Material", None),
            ("NodeSocketFloat", "Value", 0.5000),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_1.outputs["Value"], 1: 0.4000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: group_input_1.outputs["Height"]},
        attrs={"operation": "MULTIPLY"},
    )

    maximum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_1.outputs["joint_radius"], 1: multiply_1},
        attrs={"operation": "MAXIMUM"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_1.outputs["Height"]},
        attrs={"operation": "MULTIPLY"},
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Radius": maximum, "Depth": multiply_2},
    )

    multiply_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_2}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_3})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cylinder.outputs["Mesh"], "Translation": combine_xyz},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry,
            "Material": group_input_1.outputs["Material"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_material},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input_1 = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Length", 0.0000),
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Height", 0.0100),
            ("NodeSocketInt", "head_type", 0),
            ("NodeSocketFloat", "head_width", 0.7500),
            ("NodeSocketFloat", "handle_curvature", 0.0000),
            ("NodeSocketFloat", "handle_thickness", 1.2000),
            ("NodeSocketFloat", "joint_radius", 0.0100),
            ("NodeSocketMaterial", "head_and_joint_material", None),
            ("NodeSocketMaterial", "handle_material", None),
            ("NodeSocketFloat", "Value", 0.0000),
            ("NodeSocketFloat", "head curvature", 0.0150),
            ("NodeSocketInt", "jagged_head", 0),
            ("NodeSocketInt", "number of jagged edges", 30),
            ("NodeSocketInt", "handle_roundness", 0),
            ("NodeSocketFloat", "plastic handle cover length", 0.1000),
            ("NodeSocketFloat", "cut length ratio", 0.7300),
            ("NodeSocketFloat", "pincer length ratio", 0.7500),
        ],
    )

    joint_details = nw.new_node(
        nodegroup_joint_details_001().name,
        input_kwargs={
            "Height": group_input_1.outputs["Height"],
            "joint_radius": group_input_1.outputs["joint_radius"],
            "Material": group_input_1.outputs["head_and_joint_material"],
            "Value": group_input_1.outputs["handle_thickness"],
        },
        label="joint_details",
    )

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": joint_details})

    reroute_36 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_35})

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_36})

    reroute_21 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["Length"]}
    )

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    reroute_33 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["jagged_head"]}
    )

    reroute_34 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_33})

    reroute_23 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["Width"]}
    )

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    reroute_25 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["Height"]}
    )

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_25})

    reroute_31 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["head_type"]}
    )

    reroute_32 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_31})

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["head_width"]}
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input_1.outputs["head_type"]},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    reroute_19 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_1.outputs["pincer length ratio"]},
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: equal, 1: reroute_20},
        attrs={"operation": "MULTIPLY"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input_1.outputs["head_type"], 3: 1},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["cut length ratio"]}
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_17})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: equal_1, 1: reroute_18},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: multiply_1})

    reroute_27 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["joint_radius"]}
    )

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_27})

    reroute_29 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_1.outputs["head_and_joint_material"]},
    )

    reroute_30 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_29})

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["head curvature"]}
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    reroute_13 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_1.outputs["number of jagged edges"]},
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    plier_head = nw.new_node(
        nodegroup_plier_head_001().name,
        input_kwargs={
            "Length": reroute_22,
            "jagged_edge": reroute_34,
            "Width": reroute_24,
            "Height": reroute_26,
            "head_type": reroute_32,
            "head_width": reroute_10,
            "handle_head_length_ratio": add,
            "joint_radius": reroute_28,
            "Material": reroute_30,
            "head curvature": reroute_12,
            "number of jagged edges": reroute_14,
        },
        label="Plier_head",
    )

    reroute_41 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": plier_head.outputs["non_jagged"]}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["handle_curvature"]}
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["handle_thickness"]}
    )

    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["handle_roundness"]}
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["handle_material"]}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    reroute_7 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_1.outputs["plastic handle cover length"]},
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    plier_handle = nw.new_node(
        nodegroup_plier_handle_001().name,
        input_kwargs={
            "Length": reroute_22,
            "Width": reroute_24,
            "Height": reroute_26,
            "joint_radius": reroute_28,
            "handle_head_length_ratio": add,
            "Handle_curvature": reroute,
            "handle_thickness": reroute_2,
            "handle shape": reroute_6,
            "handle_patterned": 1,
            "handle_material": reroute_4,
            "metal_material": reroute_30,
            "Value": reroute_8,
        },
        label="Plier_handle",
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [reroute_37, reroute_41, plier_handle]},
    )

    reroute_44 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_1})

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_44, "Label": "the fixed arm of the plier"},
    )

    reroute_40 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": plier_handle})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_40, "Scale": (1.0000, -1.0000, 1.0000)},
    )

    flip_faces_1 = nw.new_node(
        Nodes.FlipFaces, input_kwargs={"Mesh": transform_geometry}
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_34, 3: 1},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_32},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    op_and = nw.new_node(Nodes.BooleanMath, input_kwargs={0: equal_2, 1: equal_3})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": plier_head.outputs["non_jagged"],
            "Scale": (1.0000, -1.0000, 1.0000),
        },
    )

    flip_faces_2 = nw.new_node(
        Nodes.FlipFaces, input_kwargs={"Mesh": transform_geometry_2}
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": plier_head.outputs["jagged_opposite"],
            "Scale": (1.0000, -1.0000, 1.0000),
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": op_and,
            "False": flip_faces_2,
            "True": transform_geometry_3,
        },
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_36, "Scale": (1.0000, 1.0000, -1.0000)},
    )

    flip_faces = nw.new_node(
        Nodes.FlipFaces, input_kwargs={"Mesh": transform_geometry_1}
    )

    reroute_39 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": flip_faces})

    reroute_42 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_39})

    reroute_43 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_42})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [flip_faces_1, switch, reroute_43]},
    )

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": join_geometry,
            "Label": "the plier arm attached to the hinge",
        },
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": join_geometry_1}
    )

    add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Min"], 1: bounding_box.outputs["Max"]},
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_1.outputs["Vector"], "Scale": -0.5000},
        attrs={"operation": "SCALE"},
    )

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "plier_joint",
            "Parent": add_jointed_geometry_metadata,
            "Child": add_jointed_geometry_metadata_1,
            "Position": scale.outputs["Vector"],
            "Min": -0.1800,
            "Max": 0.1800,
        },
    )

    reroute_38 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_26})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_38, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": hinge_joint.outputs["Geometry"],
            "Translation": combine_xyz,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_4},
        attrs={"is_active_output": True},
    )


class PlierFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    @gin.configurable(module="PlierFactory")
    def sample_joint_parameters(
        cls,
        plier_joint_stiffness_min: float = 0.0,
        plier_joint_stiffness_max: float = 0.0,
        plier_joint_damping_min: float = 0.2,
        plier_joint_damping_max: float = 1.0,
    ):
        return {
            "plier_joint": {
                "stiffness": uniform(
                    plier_joint_stiffness_min, plier_joint_stiffness_max
                ),
                "damping": uniform(plier_joint_damping_min, plier_joint_damping_max),
            },
        }

    def sample_parameters(self):
        # add code here to randomly sample from parameters
        import numpy as np

        from infinigen.assets.materials import metal, plastic

        def sample_gray():
            """Generate a gray color variation"""
            # Silver colors are desaturated with high brightness
            h = np.random.uniform(0, 1)  # Hue doesn't matter much due to low saturation
            s = np.random.uniform(0, 0.1)  # Very low saturation
            v = np.random.uniform(0.25, 0.9)  # Any brightness

            return (h, s, v)

        def sample_mat(gray=False):
            gray = sample_gray()
            shader = weighted_sample(
                [
                    (metal.MetalBasic, 0.7),
                    (plastic.Plastic, 0.2),
                    (plastic.BlackPlastic, 0.1),
                ]
            )()
            if gray:
                return shader(color_hsv=gray)
            else:
                return shader()

        def sample_handle_mat():
            shader = weighted_sample(
                [
                    (metal.MetalBasic, 0.2),
                    (plastic.Plastic, 0.7),
                    (plastic.BlackPlastic, 0.1),
                ]
            )()
            return shader()

        metal_mat = sample_mat(gray=True)
        handle_mat = sample_handle_mat()  # np.random.choice([plastic.Plastic, metal.MetalBasic, plastic.BlackPlastic], p=[0.7, 0.2, 0.1])()
        return_dict = {
            "Length": uniform(0.15, 0.2),
            "Width": uniform(0.05, 0.1),
            "Height": uniform(0.006, 0.015),
            "head_type": randint(0, 2),
            "head_width": uniform(0.5, 1),
            "handle_curvature": uniform(0.3, 0.6),
            "handle_thickness": uniform(1, 1.5),
            "joint_radius": uniform(0.01, 0.015),
            "head_and_joint_material": metal_mat,
            "handle_material": handle_mat,
            "Value": 0,
            "head curvature": uniform(0, 0.01),
            "jagged_head": randint(0, 2),
            "number of jagged edges": randint(10, 80),
            "handle_roundness": randint(0, 3),
            "plastic handle cover length": uniform(0.1, 0.3),
            "cut length ratio": uniform(0.65, 0.83),
            "pincer length ratio": uniform(0.6, 0.75),
        }
        print(
            f"=======================================================\nDictionary of configs: {return_dict}\n\n"
        )
        return return_dict

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
