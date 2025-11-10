import gin
from numpy.random import randint, uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.joints import (
    nodegroup_add_jointed_geometry_metadata,
    nodegroup_duplicate_joints_on_parent,
    nodegroup_hinge_joint,
)
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import weighted_sample


@node_utils.to_nodegroup(
    "nodegroup_burner_grate_curve", singleton=False, type="GeometryNodeTree"
)
def nodegroup_burner_grate_curve(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={"Resolution": 4})

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "Count", 0),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "Bar Length", 0.0000),
        ],
    )

    curve_to_points = nw.new_node(
        Nodes.CurveToPoints,
        input_kwargs={
            "Curve": curve_circle.outputs["Curve"],
            "Count": group_input.outputs["Count"],
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: 0.7071},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Bar Length"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_1, "Z": multiply}
    )

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_2, "End": combine_xyz_1}
    )

    curve_line_2 = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz_2})

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [curve_line, curve_line_2]}
    )

    index = nw.new_node(Nodes.Index)

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 6.2832, 1: group_input.outputs["Count"]},
        attrs={"operation": "DIVIDE"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: index, 1: divide}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_2})

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": curve_to_points.outputs["Points"],
            "Instance": join_geometry_1,
            "Rotation": combine_xyz,
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [curve_circle.outputs["Curve"], instance_on_points]},
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

    sqrt = nw.new_node(
        Nodes.Math, input_kwargs={0: 2.0000}, attrs={"operation": "SQRT"}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": mesh_to_curve,
            "Rotation": (0.0000, 0.0000, 0.7854),
            "Scale": sqrt,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": transform_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_mesh_type2", singleton=False, type="GeometryNodeTree"
)
def nodegroup_mesh_type2(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    combine_xyz_12 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 1.0000, "Y": 1.0000}
    )

    combine_xyz_13 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": -1.0000, "Y": 1.0000}
    )

    curve_line_3 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_12, "End": combine_xyz_13}
    )

    combine_xyz_14 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": -1.0000, "Y": -1.0000}
    )

    curve_line_4 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_13, "End": combine_xyz_14}
    )

    combine_xyz_15 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 1.0000, "Y": -1.0000}
    )

    curve_line_5 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_14, "End": combine_xyz_15}
    )

    curve_line_6 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_15, "End": combine_xyz_12}
    )

    join_geometry_5 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [curve_line_3, curve_line_4, curve_line_5, curve_line_6]
        },
    )

    combine_xyz_18 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": 1.0000})

    combine_xyz_19 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": -1.0000})

    curve_line_8 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_18, "End": combine_xyz_19}
    )

    combine_xyz_20 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 0.5000, "Y": 1.0000}
    )

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "k", 0.0000),
            ("NodeSocketFloat", "height", 0.0000),
            ("NodeSocketFloat", "Thickness", 0.0000),
        ],
    )

    curve_line_9 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": combine_xyz_20,
            "Direction": (0.0000, -1.0000, 0.0000),
            "Length": group_input.outputs["k"],
        },
        attrs={"mode": "DIRECTION"},
    )

    combine_xyz_21 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": -0.5000, "Y": 1.0000}
    )

    curve_line_10 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": combine_xyz_21,
            "Direction": (0.0000, -1.0000, 0.0000),
            "Length": group_input.outputs["k"],
        },
        attrs={"mode": "DIRECTION"},
    )

    combine_xyz_22 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 0.5000, "Y": -1.0000}
    )

    curve_line_11 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": combine_xyz_22,
            "Direction": (0.0000, 1.0000, 0.0000),
            "Length": group_input.outputs["k"],
        },
        attrs={"mode": "DIRECTION"},
    )

    combine_xyz_23 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": -0.5000, "Y": -1.0000}
    )

    curve_line_12 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": combine_xyz_23,
            "Direction": (0.0000, 1.0000, 0.0000),
            "Length": group_input.outputs["k"],
        },
        attrs={"mode": "DIRECTION"},
    )

    combine_xyz_16 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": 1.0000})

    combine_xyz_17 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": -1.0000})

    curve_line_7 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_16, "End": combine_xyz_17}
    )

    join_geometry_6 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                curve_line_8,
                curve_line_9,
                curve_line_10,
                curve_line_11,
                curve_line_12,
                curve_line_7,
            ]
        },
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["height"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_6, "Translation": combine_xyz},
    )

    points_3 = nw.new_node(
        "GeometryNodePoints", input_kwargs={"Position": combine_xyz_23}
    )

    points_2 = nw.new_node(
        "GeometryNodePoints", input_kwargs={"Position": combine_xyz_22}
    )

    points_1 = nw.new_node(
        "GeometryNodePoints", input_kwargs={"Position": combine_xyz_21}
    )

    points = nw.new_node(
        "GeometryNodePoints", input_kwargs={"Position": combine_xyz_20}
    )

    points_4 = nw.new_node(
        "GeometryNodePoints", input_kwargs={"Position": combine_xyz_16}
    )

    points_5 = nw.new_node(
        "GeometryNodePoints", input_kwargs={"Position": combine_xyz_17}
    )

    points_6 = nw.new_node(
        "GeometryNodePoints", input_kwargs={"Position": combine_xyz_18}
    )

    points_7 = nw.new_node(
        "GeometryNodePoints", input_kwargs={"Position": combine_xyz_19}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                points_3,
                points_2,
                points_1,
                points,
                points_4,
                points_5,
                points_6,
                points_7,
            ]
        },
    )

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 1.5000},
        attrs={"operation": "MULTIPLY"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["height"]}
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: reroute})

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: reroute}, attrs={"operation": "DIVIDE"}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 2.0000, "Y": 2.0000, "Z": divide}
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": join_geometry,
            "Instance": curve_line,
            "Scale": combine_xyz_1,
        },
    )

    join_geometry_7 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [join_geometry_5, transform_geometry, instance_on_points]
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh, input_kwargs={"Curve": join_geometry_7}
    )

    merge_by_distance = nw.new_node(
        Nodes.MergeByDistance,
        input_kwargs={"Geometry": curve_to_mesh, "Distance": 0.0100},
    )

    mesh_to_curve = nw.new_node(
        Nodes.MeshToCurve, input_kwargs={"Mesh": merge_by_distance}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": mesh_to_curve},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_burner_concentric_circles", singleton=False, type="GeometryNodeTree"
)
def nodegroup_burner_concentric_circles(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Value", 0.5000),
            ("NodeSocketFloat", "Radius", 1.0000),
            ("NodeSocketFloat", "Depth", 0.0500),
            ("NodeSocketFloat", "Input", 0.0000),
            ("NodeSocketMaterial", "SecondaryMaterial", None),
            ("NodeSocketMaterial", "PrimaryMaterial", None),
        ],
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"], 1: 1.0000},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": subtract})

    curve_line_2 = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz_5})

    curve_to_points_2 = nw.new_node(
        Nodes.CurveToPoints,
        input_kwargs={"Curve": curve_line_2, "Count": group_input.outputs["Value"]},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 1.5000},
        attrs={"operation": "MULTIPLY"},
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0200

    beveledcylinder = nw.new_node(
        nodegroup_beveled_cylinder().name,
        input_kwargs={"r1": multiply, "r2_scale": 0.8000, "n": 0, "h": value},
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: value}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_10 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_1})

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": beveledcylinder, "Translation": combine_xyz_10},
    )

    set_material_5 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_6,
            "Material": group_input.outputs["SecondaryMaterial"],
        },
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Radius"]}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_9, 1: 0.6700},
        attrs={"operation": "MULTIPLY"},
    )

    curve_circle_3 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 16, "Radius": multiply_2}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_9, 1: 0.3300},
        attrs={"operation": "MULTIPLY"},
    )

    curve_circle_4 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 12, "Radius": multiply_3}
    )

    curve_circle_1 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 23, "Radius": reroute_9}
    )

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                curve_circle_3.outputs["Curve"],
                curve_circle_4.outputs["Curve"],
                curve_circle_1.outputs["Curve"],
            ]
        },
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Input"]}
    )

    curve_circle_2 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 4, "Radius": reroute_8}
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle_2.outputs["Curve"],
            "Rotation": (0.0000, 0.0000, 0.7854),
        },
    )

    curve_to_mesh_2 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={"Curve": join_geometry_4, "Profile Curve": transform_geometry_5},
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_8, 2: value},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": curve_to_mesh_2, "Translation": combine_xyz_6},
    )

    set_material_6 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_4,
            "Material": group_input.outputs["PrimaryMaterial"],
        },
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material_5, set_material_6]}
    )

    index_1 = nw.new_node(Nodes.Index)

    random_value_1 = nw.new_node(
        Nodes.RandomValue,
        input_kwargs={2: 0.5000, 3: 2.0000, "ID": index_1, "Seed": 42},
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": random_value_1.outputs["Value"],
            "Y": random_value_1.outputs["Value"],
            "Z": 1.0000,
        },
    )

    instance_on_points_3 = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": curve_to_points_2.outputs["Points"],
            "Instance": join_geometry_3,
            "Scale": combine_xyz_8,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Instances": instance_on_points_3},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_burner_circle", singleton=False, type="GeometryNodeTree"
)
def nodegroup_burner_circle(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Value", 0.5000),
            ("NodeSocketFloat", "Radius", 1.0000),
            ("NodeSocketFloat", "Depth", 0.0500),
            ("NodeSocketMaterial", "Material", None),
        ],
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"], 1: 1.0000},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": subtract})

    curve_line_1 = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz_3})

    curve_to_points_1 = nw.new_node(
        Nodes.CurveToPoints,
        input_kwargs={"Curve": curve_line_1, "Count": group_input.outputs["Value"]},
    )

    cylinder_3 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 23,
            "Radius": group_input.outputs["Radius"],
            "Depth": group_input.outputs["Depth"],
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_3.outputs["Mesh"],
            "Translation": combine_xyz_4,
        },
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": transform_geometry_2}
    )

    index_2 = nw.new_node(Nodes.Index)

    random_value_2 = nw.new_node(
        Nodes.RandomValue,
        input_kwargs={2: 0.5000, 3: 2.0000, "ID": index_2, "Seed": 42},
    )

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": random_value_2.outputs["Value"],
            "Y": random_value_2.outputs["Value"],
            "Z": 1.0000,
        },
    )

    instance_on_points_2 = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": curve_to_points_1.outputs["Points"],
            "Instance": join_geometry_2,
            "Scale": combine_xyz_9,
        },
    )

    set_material_4 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": instance_on_points_2,
            "Material": group_input.outputs["Material"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_material_4},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_burner_grate_row", singleton=False, type="GeometryNodeTree"
)
def nodegroup_burner_grate_row(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "n", 0),
            ("NodeSocketFloat", "Thickness", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "Burner radius", 0.0000),
            ("NodeSocketFloat", "Bar Length", 0.0000),
            ("NodeSocketInt", "kind", 0),
            ("NodeSocketMaterial", "BlackMetal", None),
            ("NodeSocketMaterial", "PrimaryMaterial", None),
        ],
    )

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["kind"]}
    )

    less_equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_10, 3: 1},
        attrs={"data_type": "INT", "operation": "LESS_EQUAL"},
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_10, 3: 2},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    burnercircle = nw.new_node(
        nodegroup_burner_circle().name,
        input_kwargs={
            "Value": group_input.outputs["n"],
            "Radius": group_input.outputs["Burner radius"],
            "Depth": group_input.outputs["Height"],
            "Material": group_input.outputs["BlackMetal"],
        },
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": burnercircle})

    burnerconcentriccircles = nw.new_node(
        nodegroup_burner_concentric_circles().name,
        input_kwargs={
            "Value": group_input.outputs["n"],
            "Radius": group_input.outputs["Burner radius"],
            "Depth": group_input.outputs["Height"],
            "Input": group_input.outputs["Thickness"],
            "SecondaryMaterial": group_input.outputs["PrimaryMaterial"],
            "PrimaryMaterial": group_input.outputs["BlackMetal"],
        },
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": burnerconcentriccircles}
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": reroute_3, "True": reroute_7},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["n"], 1: 1.0000},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": subtract})

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz})

    curve_to_points = nw.new_node(
        Nodes.CurveToPoints,
        input_kwargs={"Curve": curve_line, "Count": group_input.outputs["n"]},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: 0.0000},
        attrs={"operation": "SUBTRACT"},
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 12,
            "Radius": group_input.outputs["Burner radius"],
            "Depth": subtract_1,
        },
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_1,
        },
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_1,
            "Material": group_input.outputs["PrimaryMaterial"],
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Burner radius"], 1: 1.5000},
        attrs={"operation": "MULTIPLY"},
    )

    beveledcylinder = nw.new_node(
        nodegroup_beveled_cylinder().name,
        input_kwargs={"r1": multiply, "r2_scale": 0.8000, "n": 12, "h": 0.0500},
    )

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": beveledcylinder,
            "Material": group_input.outputs["PrimaryMaterial"],
        },
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Burner radius"], 1: 0.0300}
    )

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.0100

    beveledcylinder_1 = nw.new_node(
        nodegroup_beveled_cylinder().name,
        input_kwargs={"r1": add, "r2_scale": 0.9000, "n": 12, "h": value_1},
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: value_1, 2: subtract_1},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_11 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add})

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": beveledcylinder_1, "Translation": combine_xyz_11},
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_7,
            "Material": group_input.outputs["BlackMetal"],
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [set_material_2, set_material_3, set_material_1]},
    )

    index = nw.new_node(Nodes.Index)

    random_value = nw.new_node(
        Nodes.RandomValue, input_kwargs={2: 0.5000, 3: 2.0000, "ID": index, "Seed": 42}
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": random_value.outputs["Value"],
            "Y": random_value.outputs["Value"],
            "Z": 1.0000,
        },
    )

    instance_on_points_1 = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": curve_to_points.outputs["Points"],
            "Instance": join_geometry,
            "Scale": combine_xyz_7,
        },
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["kind"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math, input_kwargs={1: 0.0250}, attrs={"operation": "SUBTRACT"}
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: subtract_2},
        attrs={"operation": "DIVIDE"},
    )

    meshtype2 = nw.new_node(
        nodegroup_mesh_type2().name,
        input_kwargs={
            "k": group_input.outputs["Bar Length"],
            "height": divide_1,
            "Thickness": group_input.outputs["Thickness"],
        },
    )

    burner_grate_curve = nw.new_node(
        nodegroup_burner_grate_curve().name,
        input_kwargs={
            "Count": 8,
            "Height": divide_1,
            "Bar Length": group_input.outputs["Bar Length"],
        },
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_1,
            "False": meshtype2,
            "True": burner_grate_curve,
        },
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Thickness"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": switch_2,
            "Translation": combine_xyz_2,
            "Scale": subtract_2,
        },
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": curve_to_points.outputs["Points"],
            "Instance": transform_geometry,
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh, input_kwargs={"Curve": instance_on_points}
    )

    merge_by_distance = nw.new_node(
        Nodes.MergeByDistance, input_kwargs={"Geometry": curve_to_mesh}
    )

    mesh_to_curve = nw.new_node(
        Nodes.MeshToCurve, input_kwargs={"Mesh": merge_by_distance}
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={"Resolution": 4, "Radius": group_input.outputs["Thickness"]},
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle.outputs["Curve"],
            "Rotation": (0.0000, 0.0000, 0.7854),
        },
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": mesh_to_curve,
            "Profile Curve": transform_geometry_3,
            "Fill Caps": True,
        },
    )

    set_shade_smooth = nw.new_node(
        Nodes.SetShadeSmooth,
        input_kwargs={"Geometry": curve_to_mesh_1, "Shade Smooth": False},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": set_shade_smooth,
            "Material": group_input.outputs["BlackMetal"],
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [instance_on_points_1, set_material]},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": less_equal, "False": switch_1, "True": join_geometry_1},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": switch},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_beveled_cylinder", singleton=False, type="GeometryNodeTree"
)
def nodegroup_beveled_cylinder(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "r1", 0.0000),
            ("NodeSocketFloat", "r2_scale", 0.0000),
            ("NodeSocketInt", "n", 18),
            ("NodeSocketFloat", "h", 0.0000),
        ],
    )

    cylinder_2 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": group_input.outputs["n"],
            "Radius": group_input.outputs["r1"],
            "Depth": group_input.outputs["h"],
        },
    )

    index_3 = nw.new_node(Nodes.Index)

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index_3},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    scale_elements = nw.new_node(
        Nodes.ScaleElements,
        input_kwargs={
            "Geometry": cylinder_2.outputs["Mesh"],
            "Selection": equal,
            "Scale": group_input.outputs["r2_scale"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Socket": scale_elements},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_rounded_cube", singleton=False, type="GeometryNodeTree"
)
def nodegroup_rounded_cube(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Dim", (2.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "k", 0.3000),
        ],
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Dim"]}
    )

    clamp = nw.new_node(
        Nodes.Clamp,
        input_kwargs={
            "Value": group_input.outputs["k"],
            "Max": separate_xyz_1.outputs["Z"],
        },
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": clamp})

    arc = nw.new_node(
        "GeometryNodeCurveArc",
        input_kwargs={"Resolution": 3, "Radius": reroute, "Sweep Angle": 1.5708},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Dim"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: reroute},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: clamp},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": subtract_1}
    )

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": arc.outputs["Curve"], "Translation": combine_xyz_1},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": reroute})

    add = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: combine_xyz_1, 1: combine_xyz_3}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: reroute})

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_1, "Y": subtract_1}
    )

    add_2 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: combine_xyz_3, 1: combine_xyz_2}
    )

    curve_line = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={"Start": add.outputs["Vector"], "End": add_2.outputs["Vector"]},
    )

    arc_1 = nw.new_node(
        "GeometryNodeCurveArc",
        input_kwargs={
            "Resolution": 3,
            "Radius": reroute,
            "Start Angle": 1.5708,
            "Sweep Angle": 1.5708,
        },
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": arc_1.outputs["Curve"], "Translation": combine_xyz_2},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [transform_geometry_7, curve_line, transform_geometry_1]
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh, input_kwargs={"Curve": join_geometry_1}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": separate_xyz.outputs["X"], "Height": subtract_1},
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract_1}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_2})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": quadrilateral, "Translation": combine_xyz},
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh, input_kwargs={"Curve": transform_geometry_2}
    )

    index_2 = nw.new_node(Nodes.Index)

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index_2},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    delete_geometry_1 = nw.new_node(
        Nodes.DeleteGeometry,
        input_kwargs={"Geometry": curve_to_mesh_1, "Selection": equal},
        attrs={"domain": "EDGE"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [curve_to_mesh, delete_geometry_1]},
    )

    merge_by_distance = nw.new_node(
        Nodes.MergeByDistance, input_kwargs={"Geometry": join_geometry}
    )

    mesh_to_curve = nw.new_node(
        Nodes.MeshToCurve, input_kwargs={"Mesh": merge_by_distance}
    )

    fill_curve = nw.new_node(Nodes.FillCurve, input_kwargs={"Curve": mesh_to_curve})

    flip_faces = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": fill_curve})

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["Y"]}
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh, input_kwargs={"Mesh": fill_curve, "Offset Scale": reroute_1}
    )

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [flip_faces, extrude_mesh.outputs["Mesh"]]},
    )

    convex_hull = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": join_geometry_4}
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": reroute_1, "Z": separate_xyz.outputs["Z"]}
    )

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_4, 1: (0.5000, 0.5000, -0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": convex_hull,
            "Translation": multiply_3.outputs["Vector"],
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_rot_sym_buttons", singleton=False, type="GeometryNodeTree"
)
def nodegroup_rot_sym_buttons(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input_1 = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "k", 0.0000),
            ("NodeSocketFloat", "bezier_param", 0.0000),
            ("NodeSocketInt", "kind", 0),
            ("NodeSocketMaterial", "PrimaryMaterial", None),
            ("NodeSocketMaterial", "SecondaryMaterial", None),
        ],
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input_1.outputs["kind"], 3: 2},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input_1.outputs["kind"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={"Resolution": 27})

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["kind"]}
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_13, 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_6},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_1.outputs["k"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide})

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    add = nw.new_node(Nodes.Math, input_kwargs={0: 1.0000, 1: divide})

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_8, "Y": add}
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["k"]}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_4, "Y": 1.0000}
    )

    curve_line_3 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_3, "End": combine_xyz_4}
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": 1.0000})

    curve_line_2 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_2, "End": combine_xyz_3}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ)

    curve_line_1 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_1, "End": combine_xyz_2}
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [curve_line_3, curve_line_2, curve_line_1]},
    )

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = -1.0000

    transform_geometry_1 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": join_geometry_1, "Scale": value_1}
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input_1.outputs["k"],
            "Y": group_input_1.outputs["bezier_param"],
        },
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input_1.outputs["k"], "Y": 1.0000}
    )

    quadratic_b_zier = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Resolution": 4,
            "Start": (0.0000, 0.0000, 0.0000),
            "Middle": combine_xyz_6,
            "End": combine_xyz_5,
        },
    )

    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = -1.0000

    transform_geometry_2 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": quadratic_b_zier, "Scale": value_2}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry_2})

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_3,
            "False": transform_geometry_1,
            "True": reroute_9,
        },
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_12, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply, "Y": -1.0000}
    )

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz})

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal_2, "False": switch_1, "True": curve_line},
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": switch, "Count": 6}
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_circle.outputs["Curve"],
            "Profile Curve": resample_curve,
        },
    )

    flip_faces = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": curve_to_mesh})

    set_shade_smooth = nw.new_node(
        Nodes.SetShadeSmooth,
        input_kwargs={"Geometry": flip_faces, "Shade Smooth": False},
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    equal_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: 1.0000, "Epsilon": 0.2000},
        attrs={"operation": "EQUAL"},
    )

    separate_geometry = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": set_shade_smooth, "Selection": equal_4},
        attrs={"domain": "FACE"},
    )

    convex_hull = nw.new_node(
        Nodes.ConvexHull,
        input_kwargs={"Geometry": separate_geometry.outputs["Selection"]},
    )

    equal_5 = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: separate_xyz.outputs["Z"], "Epsilon": 0.2000},
        attrs={"operation": "EQUAL"},
    )

    separate_geometry_1 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": set_shade_smooth, "Selection": equal_5},
        attrs={"domain": "FACE"},
    )

    convex_hull_1 = nw.new_node(
        Nodes.ConvexHull,
        input_kwargs={"Geometry": separate_geometry_1.outputs["Selection"]},
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_geometry.outputs["Inverted"]}
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [convex_hull, convex_hull_1, reroute_17]},
    )

    convex_hull_3 = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": curve_to_mesh}
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": (2.0000, 2.0000, 3.0000)}
    )

    intersect = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 2": [convex_hull_3, cube_1.outputs["Mesh"]],
            "Self Intersection": True,
            "Hole Tolerant": True,
        },
        attrs={"operation": "INTERSECT"},
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_1,
            "False": join_geometry_3,
            "True": intersect.outputs["Mesh"],
        },
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_shade_smooth})

    cube_2 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": (2.0000, 2.0000, 1.0000)}
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_2.outputs["Mesh"],
            "Translation": (0.0000, 0.0000, 0.5000),
        },
    )

    intersect_1 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 2": [set_shade_smooth, transform_geometry_5]},
        attrs={"solver": "EXACT", "operation": "INTERSECT"},
    )

    convex_hull_2 = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": intersect_1.outputs["Mesh"]}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [reroute_16, convex_hull_2]}
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry})

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": switch_3, "True": reroute_18},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["PrimaryMaterial"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    set_material = nw.new_node(
        Nodes.SetMaterial, input_kwargs={"Geometry": switch_2, "Material": reroute_11}
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.5800

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": value, "Y": 0.3000, "Z": 0.3000}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_7})

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: -0.5000, 1: value, 2: 1.0500},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_add})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Translation": combine_xyz_8},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: 1.0000, 1: multiply})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": transform_geometry_3, "Scale": add_1}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_1.outputs["SecondaryMaterial"]},
    )

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_4, "Material": reroute_2},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": set_material_3,
            "Translation": (0.0000, 0.0000, 1.0000),
        },
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_15})

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, reroute_14]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_2},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_stove_burners", singleton=False, type="GeometryNodeTree"
)
def nodegroup_stove_burners(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "n1", 0),
            ("NodeSocketFloat", "Thickness", 0.0000),
            ("NodeSocketFloat", "Burner Height", 0.0000),
            ("NodeSocketFloat", "Burner radius", 0.0000),
            ("NodeSocketFloat", "Bar Length", 0.4600),
            ("NodeSocketInt", "kind", 0),
            ("NodeSocketInt", "n2", 0),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketMaterial", "BlackMetal", None),
            ("NodeSocketMaterial", "PrimaryMaterial", None),
        ],
    )

    maximum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["n1"], 1: group_input.outputs["n2"]},
        attrs={"operation": "MAXIMUM"},
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 2.0000, "Y": maximum, "Z": reroute_1}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_1}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Translation": combine_xyz_4},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_3,
            "Material": group_input.outputs["PrimaryMaterial"],
        },
    )

    burner_grate_row = nw.new_node(
        nodegroup_burner_grate_row().name,
        input_kwargs={
            "n": group_input.outputs["n2"],
            "Thickness": group_input.outputs["Thickness"],
            "Height": group_input.outputs["Burner Height"],
            "Burner radius": group_input.outputs["Burner radius"],
            "Bar Length": group_input.outputs["Bar Length"],
            "kind": group_input.outputs["kind"],
            "BlackMetal": group_input.outputs["BlackMetal"],
            "PrimaryMaterial": group_input.outputs["PrimaryMaterial"],
        },
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["n2"], 1: -0.5000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 0.5000, "Y": multiply_add}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": burner_grate_row, "Translation": combine_xyz_2},
    )

    burner_grate_row_1 = nw.new_node(
        nodegroup_burner_grate_row().name,
        input_kwargs={
            "n": group_input.outputs["n1"],
            "Thickness": group_input.outputs["Thickness"],
            "Height": group_input.outputs["Burner Height"],
            "Burner radius": group_input.outputs["Burner radius"],
            "Bar Length": group_input.outputs["Bar Length"],
            "kind": group_input.outputs["kind"],
            "BlackMetal": group_input.outputs["BlackMetal"],
            "PrimaryMaterial": group_input.outputs["PrimaryMaterial"],
        },
    )

    multiply_add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["n1"], 1: -0.5000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": -0.5000, "Y": multiply_add_1}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": burner_grate_row_1, "Translation": combine_xyz_1},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry, transform_geometry_1]},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_1})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Translation": combine_xyz_5},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [set_material, transform_geometry_2]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_clamped_random_offset", singleton=False, type="GeometryNodeTree"
)
def nodegroup_clamped_random_offset(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "sigma", 0.0000),
            ("NodeSocketFloat", "min", 0.0000),
            ("NodeSocketFloat", "max", 1.0000),
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketInt", "Seed", 0),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["sigma"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    random_value = nw.new_node(
        Nodes.RandomValue,
        input_kwargs={
            0: multiply,
            1: group_input.outputs["sigma"],
            "Seed": group_input.outputs["Seed"],
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": random_value.outputs["Value"]}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": separate_xyz.outputs["X"], "Y": separate_xyz.outputs["Y"]},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Offset": combine_xyz_1,
        },
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    clamp = nw.new_node(
        Nodes.Clamp,
        input_kwargs={
            "Value": separate_xyz_1.outputs["X"],
            "Min": group_input.outputs["min"],
            "Max": group_input.outputs["max"],
        },
    )

    clamp_1 = nw.new_node(
        Nodes.Clamp,
        input_kwargs={
            "Value": separate_xyz_1.outputs["Y"],
            "Min": group_input.outputs["min"],
            "Max": group_input.outputs["max"],
        },
    )

    clamp_2 = nw.new_node(
        Nodes.Clamp,
        input_kwargs={
            "Value": separate_xyz_1.outputs["Z"],
            "Min": group_input.outputs["min"],
            "Max": group_input.outputs["max"],
        },
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": clamp, "Y": clamp_1, "Z": clamp_2}
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": set_position, "Position": combine_xyz_2},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_position_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_coordinate_transform", singleton=False, type="GeometryNodeTree"
)
def nodegroup_coordinate_transform(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketVector", "origin", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "p1", (1.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "p2", (0.0000, 1.0000, 0.0000)),
        ],
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["p2"]}
    )

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["p1"]}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_17, 1: reroute_10},
        attrs={"operation": "SUBTRACT"},
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["origin"]}
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_1, 1: reroute_10},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: subtract_1.outputs["Vector"],
            1: separate_xyz_2.outputs["Y"],
            2: reroute_18,
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    multiply_add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: subtract.outputs["Vector"],
            1: separate_xyz_2.outputs["X"],
            2: multiply_add.outputs["Vector"],
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    set_position_3 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Position": multiply_add_1.outputs["Vector"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_position_3},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_knob", singleton=False, type="GeometryNodeTree")
def nodegroup_knob(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "radius", 0.0000),
            ("NodeSocketFloat", "height", 0.0000),
            ("NodeSocketInt", "kind", 0),
            ("NodeSocketFloat", "k", 0.0000),
            ("NodeSocketFloat", "bezier_param", 0.0000),
            ("NodeSocketFloat", "base_height", 0.0000),
            ("NodeSocketMaterial", "PrimaryMaterial", None),
            ("NodeSocketMaterial", "BlackMetal", None),
        ],
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["kind"]}
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    greater_than = nw.new_node(
        Nodes.Compare, input_kwargs={2: reroute_27, 3: 2}, attrs={"data_type": "INT"}
    )

    rotsymbuttons = nw.new_node(
        nodegroup_rot_sym_buttons().name,
        input_kwargs={
            "k": group_input.outputs["k"],
            "bezier_param": group_input.outputs["bezier_param"],
            "kind": group_input.outputs["kind"],
            "PrimaryMaterial": group_input.outputs["PrimaryMaterial"],
            "SecondaryMaterial": group_input.outputs["BlackMetal"],
        },
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["radius"],
            "Y": group_input.outputs["radius"],
            "Z": group_input.outputs["height"],
        },
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": rotsymbuttons, "Scale": combine_xyz_7},
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_3}
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_8, 3: 3},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["radius"],
            "Y": group_input.outputs["radius"],
            "Z": group_input.outputs["radius"],
        },
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_6})

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_12, 1: (0.3000, 0.3000, 0.3000)},
        attrs={"operation": "MULTIPLY"},
    )

    cube_3 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": multiply.outputs["Vector"]}
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BlackMetal"]}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": cube_3.outputs["Mesh"], "Material": reroute_4},
    )

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_3})

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["radius"]}
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["height"]}
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["height"],
            1: group_input.outputs["base_height"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_6, 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_15, "Y": reroute_15, "Z": subtract}
    )

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_5, 1: (1.9000, 0.4000, 1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": multiply_2.outputs["Vector"]}
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": multiply.outputs["Vector"]}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: -0.4000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 2: multiply_3},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract})

    multiply_add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_18, 2: multiply_3},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_add, "Z": multiply_add_1}
    )

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_23, "Translation": combine_xyz_8},
    )

    cube_2 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": multiply_2.outputs["Vector"]}
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["PrimaryMaterial"]}
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    set_material_4 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": cube_2.outputs["Mesh"], "Material": reroute_10},
    )

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_4})

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_6, reroute_25]},
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_1})

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    multiply_add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Z"], 2: reroute_19},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_add_2})

    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_26})

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_9})

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_3, "Translation": reroute_29},
    )

    reroute_20 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply_2.outputs["Vector"]}
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["k"]}
    )

    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": multiply_2.outputs["Vector"]}
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_2, 1: separate_xyz_2.outputs["Z"]},
        attrs={"operation": "MULTIPLY"},
    )

    roundedcube = nw.new_node(
        nodegroup_rounded_cube().name, input_kwargs={"Dim": reroute_21, "k": multiply_4}
    )

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    set_material_6 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": roundedcube, "Material": reroute_24},
    )

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": set_material_6, "Translation": combine_xyz_9},
    )

    reroute_30 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_7}
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal,
            "False": transform_geometry_5,
            "True": reroute_30,
        },
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["radius"], 1: 0.1000},
        attrs={"operation": "MULTIPLY"},
    )

    beveledcylinder = nw.new_node(
        nodegroup_beveled_cylinder().name,
        input_kwargs={"r1": reroute, "r2_scale": 0.9000, "n": 16, "h": multiply_5},
    )

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": beveledcylinder})

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_5, 1: multiply_1})

    multiply_add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add, 2: 0.0000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add_3})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_17, "Translation": combine_xyz},
    )

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 16, "Radius": reroute, "Depth": multiply_1},
    )

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cylinder_1.outputs["Mesh"]}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_geometry, reroute_16]}
    )

    multiply_6 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_11}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_6})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Translation": combine_xyz_4},
    )

    set_material_5 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_4, "Material": reroute_24},
    )

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_5})

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [switch_1, reroute_28]}
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": greater_than,
            "False": reroute_13,
            "True": join_geometry_2,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": switch},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_button_panel_v2", singleton=False, type="GeometryNodeTree"
)
def nodegroup_button_panel_v2(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "h1", 0.0000),
            ("NodeSocketFloat", "h2", 0.0000),
            ("NodeSocketFloat", "l", 0.0000),
            ("NodeSocketFloat", "d", 0.0000),
        ],
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Width": group_input.outputs["l"],
            "Height": group_input.outputs["h1"],
        },
    )

    index = nw.new_node(Nodes.Index)

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    vector = nw.new_node(Nodes.Vector)
    vector.vector = (0.0000, 0.0000, 0.0000)

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["h2"], 1: group_input.outputs["h1"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": subtract})

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": vector, "True": combine_xyz_7},
        attrs={"input_type": "VECTOR"},
    )

    set_position = nw.new_node(
        Nodes.SetPosition, input_kwargs={"Geometry": quadrilateral, "Offset": switch_3}
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": group_input.outputs["l"], "Y": group_input.outputs["h1"]},
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_8, 1: (0.5000, 0.5000, 0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": set_position,
            "Translation": multiply.outputs["Vector"],
        },
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": transform_geometry}
    )

    fill_curve = nw.new_node(
        Nodes.FillCurve, input_kwargs={"Curve": transform_geometry_2}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["d"]}
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_1})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": fill_curve,
            "Translation": combine_xyz,
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={"Mesh": transform_geometry_1, "Offset Scale": reroute},
    )

    flip_faces = nw.new_node(
        Nodes.FlipFaces, input_kwargs={"Mesh": transform_geometry_1}
    )

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [extrude_mesh.outputs["Mesh"], flip_faces]},
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    sample_index_4 = nw.new_node(
        Nodes.SampleIndex,
        input_kwargs={"Geometry": flip_faces, "Value": position_1},
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": join_geometry_4}
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={4: bounding_box.outputs["Min"], 5: bounding_box.outputs["Max"]},
        attrs={"data_type": "VECTOR"},
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: sample_index_4, 1: mix.outputs["Result"]},
        attrs={"operation": "SUBTRACT"},
    )

    sample_index_5 = nw.new_node(
        Nodes.SampleIndex,
        input_kwargs={"Geometry": flip_faces, "Value": position_1, "Index": 1},
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    subtract_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: sample_index_5, 1: mix.outputs["Result"]},
        attrs={"operation": "SUBTRACT"},
    )

    greater_equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: group_input.outputs["h2"], 1: group_input.outputs["h1"]},
        attrs={"operation": "GREATER_EQUAL"},
    )

    position_2 = nw.new_node(Nodes.InputPosition)

    sample_index_1 = nw.new_node(
        Nodes.SampleIndex,
        input_kwargs={
            "Geometry": extrude_mesh.outputs["Mesh"],
            "Value": position_2,
            "Index": 5,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    sample_index_2 = nw.new_node(
        Nodes.SampleIndex,
        input_kwargs={
            "Geometry": extrude_mesh.outputs["Mesh"],
            "Value": position_2,
            "Index": 8,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": greater_equal,
            "False": sample_index_1,
            "True": sample_index_2,
        },
        attrs={"input_type": "VECTOR"},
    )

    subtract_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: switch, 1: mix.outputs["Result"]},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["h1"], 1: group_input.outputs["h2"]},
        attrs={"operation": "SUBTRACT"},
    )

    arctan2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_4, 1: group_input.outputs["l"]},
        attrs={"operation": "ARCTAN2"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": join_geometry_4,
            "origin": subtract_1.outputs["Vector"],
            "p1": subtract_2.outputs["Vector"],
            "p2": subtract_3.outputs["Vector"],
            "theta": arctan2,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "Row 1", 0),
            ("NodeSocketInt", "Row 2", 0),
            ("NodeSocketInt", "ExtraKnobs", 0),
            ("NodeSocketFloat", "KnobGap", 0.0000),
            ("NodeSocketBool", "KnobGapExists", False),
            ("NodeSocketFloat", "KnobSigma", 0.0000),
            ("NodeSocketInt", "Seed", 0),
            ("NodeSocketFloat", "Thickness", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "DashboardHeight", 0.0000),
            ("NodeSocketInt", "DashboardLocation", 0),
            ("NodeSocketFloat", "Dashboard Depth", 0.0000),
            ("NodeSocketFloat", "GrateScale", 1.0000),
            ("NodeSocketBool", "TwoButtonRows", False),
            ("NodeSocketFloat", "Burner Height", 0.0000),
            ("NodeSocketInt", "BurnerKind", 0),
            ("NodeSocketFloat", "Burner Radius", 0.0000),
            ("NodeSocketFloat", "BurnerBarLength", 0.0000),
            ("NodeSocketFloat", "ButtonRadius", 0.0000),
            ("NodeSocketFloat", "ButtonHeight", 0.0000),
            ("NodeSocketInt", "ButtonKind", 0),
            ("NodeSocketFloat", "Buttonk", 0.0000),
            ("NodeSocketFloat", "ButtonBezierParam", 0.0000),
            ("NodeSocketFloat", "ButtonBaseHeight", 0.0000),
            ("NodeSocketMaterial", "BlackMetal", None),
            ("NodeSocketMaterial", "ButtonMaterial", None),
            ("NodeSocketMaterial", "PrimaryMaterial", None),
        ],
    )

    greater_than = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["DashboardLocation"], 3: 1},
        attrs={"data_type": "INT"},
    )

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": greater_than})

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_25})

    reroute_39 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_26})

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["DashboardLocation"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_7, 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    clamp = nw.new_node(
        Nodes.Clamp,
        input_kwargs={
            "Value": group_input.outputs["DashboardHeight"],
            "Max": group_input.outputs["Height"],
        },
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Dashboard Depth"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_3, "Y": clamp, "Z": reroute_5}
    )

    clamp_1 = nw.new_node(
        Nodes.Clamp,
        input_kwargs={
            "Value": group_input.outputs["DashboardHeight"],
            "Min": group_input.outputs["Height"],
            "Max": 1000.0000,
        },
    )

    combine_xyz_10 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": clamp_1, "Y": reroute_3, "Z": reroute_5}
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": combine_xyz_9, "True": combine_xyz_10},
        attrs={"input_type": "VECTOR"},
    )

    combine_xyz_12 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Height"],
            "Y": group_input.outputs["Height"],
            "Z": group_input.outputs["Dashboard Depth"],
        },
    )

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_12})

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_26, "False": switch_3, "True": reroute_24},
        attrs={"input_type": "VECTOR"},
    )

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": switch_5})

    maximum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Row 1"], 1: group_input.outputs["Row 2"]},
        attrs={"operation": "MAXIMUM"},
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 2.0000

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": greater_than, "False": maximum, "True": value},
        attrs={"input_type": "FLOAT"},
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["GrateScale"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: switch_6, 1: reroute_33},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_43 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply})

    buttonpanelv2 = nw.new_node(
        nodegroup_button_panel_v2().name,
        input_kwargs={
            "h1": separate_xyz.outputs["X"],
            "h2": separate_xyz.outputs["Y"],
            "l": separate_xyz.outputs["Z"],
            "d": reroute_43,
        },
    )

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["PrimaryMaterial"]}
    )

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": buttonpanelv2.outputs["Geometry"],
            "Material": reroute_17,
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_1, "Label": "knob_base"},
    )

    reroute_58 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata}
    )

    knob = nw.new_node(
        nodegroup_knob().name,
        input_kwargs={
            "radius": group_input.outputs["ButtonRadius"],
            "height": group_input.outputs["ButtonHeight"],
            "kind": group_input.outputs["ButtonKind"],
            "k": group_input.outputs["Buttonk"],
            "bezier_param": group_input.outputs["ButtonBezierParam"],
            "base_height": group_input.outputs["ButtonBaseHeight"],
            "PrimaryMaterial": group_input.outputs["ButtonMaterial"],
            "BlackMetal": group_input.outputs["BlackMetal"],
        },
    )

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": knob})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": buttonpanelv2.outputs["theta"]}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": reroute_27, "Rotation": combine_xyz}
    )

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_2, "Label": "knob"},
    )

    combine_xyz_14 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": 1.0000})

    reroute_56 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz})

    vector_rotate = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={"Vector": combine_xyz_14, "Rotation": reroute_56},
        attrs={"rotation_type": "EULER_XYZ"},
    )

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "knob",
            "Parent": reroute_58,
            "Child": add_jointed_geometry_metadata_1,
            "Axis": vector_rotate,
        },
    )

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["TwoButtonRows"]}
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["KnobGapExists"]}
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    reroute_18 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["KnobSigma"]}
    )

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    curve_line = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (0.5000, 0.1000, 0.0000),
            "End": (0.5000, 0.9000, 0.0000),
        },
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Row 2"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["ExtraKnobs"],
            1: group_input.outputs["Row 1"],
        },
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_1, 1: add})

    curve_to_points = nw.new_node(
        Nodes.CurveToPoints, input_kwargs={"Curve": curve_line, "Count": add_1}
    )

    reroute_20 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Seed"]}
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    clampedrandomoffset = nw.new_node(
        nodegroup_clamped_random_offset().name,
        input_kwargs={
            "sigma": reroute_19,
            "min": 0.1000,
            "max": 0.9000,
            "Geometry": curve_to_points.outputs["Points"],
            "Seed": reroute_21,
        },
    )

    reroute_44 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": clampedrandomoffset})

    reroute_41 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["KnobGap"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_34 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    multiply_add = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_34}, attrs={"operation": "MULTIPLY_ADD"}
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 0.5000, "Y": multiply_add}
    )

    curve_line_4 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={"Start": combine_xyz_6, "End": (0.5000, 0.9000, 0.0000)},
    )

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add_1})

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    floor = nw.new_node(
        Nodes.Math, input_kwargs={0: divide}, attrs={"operation": "FLOOR"}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_35, 1: floor},
        attrs={"operation": "SUBTRACT"},
    )

    curve_to_points_4 = nw.new_node(
        Nodes.CurveToPoints, input_kwargs={"Curve": curve_line_4, "Count": subtract}
    )

    multiply_add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_11, 1: -0.5000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 0.5000, "Y": multiply_add_1}
    )

    curve_line_3 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={"Start": (0.5000, 0.1000, 0.0000), "End": combine_xyz_5},
    )

    curve_to_points_3 = nw.new_node(
        Nodes.CurveToPoints, input_kwargs={"Curve": curve_line_3, "Count": floor}
    )

    reroute_47 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": curve_to_points_3.outputs["Points"]}
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [curve_to_points_4.outputs["Points"], reroute_47]},
    )

    reroute_42 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    clampedrandomoffset_1 = nw.new_node(
        nodegroup_clamped_random_offset().name,
        input_kwargs={
            "sigma": reroute_41,
            "min": 0.1000,
            "max": 0.9000,
            "Geometry": join_geometry_1,
            "Seed": reroute_42,
        },
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_13,
            "False": reroute_44,
            "True": clampedrandomoffset_1,
        },
    )

    reroute_59 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_2})

    reroute_55 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_41})

    add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Row 1"],
            1: group_input.outputs["ExtraKnobs"],
        },
    )

    subtract_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_2, 1: 1.0000}, attrs={"operation": "SUBTRACT"}
    )

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract_1})

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: 1.0000},
        attrs={"operation": "SUBTRACT"},
    )

    maximum_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: subtract_2},
        attrs={"operation": "MAXIMUM"},
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_28, 1: maximum_1},
        attrs={"operation": "DIVIDE"},
    )

    multiply_add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: -0.4000, 1: divide_1},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 0.6660, "Y": multiply_add_2}
    )

    multiply_add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 0.4000, 1: divide_1},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 0.6660, "Y": multiply_add_3}
    )

    curve_line_1 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_1, "End": combine_xyz_2}
    )

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add_2})

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    curve_to_points_1 = nw.new_node(
        Nodes.CurveToPoints, input_kwargs={"Curve": curve_line_1, "Count": reroute_23}
    )

    reroute_36 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract_2})

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_36, 1: maximum_1},
        attrs={"operation": "DIVIDE"},
    )

    multiply_add_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: -0.4000, 1: divide_2},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 0.3300, "Y": multiply_add_4}
    )

    multiply_add_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 0.4000, 1: divide_2},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 0.3300, "Y": multiply_add_5}
    )

    curve_line_2 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_3, "End": combine_xyz_4}
    )

    reroute_30 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    curve_to_points_2 = nw.new_node(
        Nodes.CurveToPoints, input_kwargs={"Curve": curve_line_2, "Count": reroute_30}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                curve_to_points_1.outputs["Points"],
                curve_to_points_2.outputs["Points"],
            ]
        },
    )

    clampedrandomoffset_2 = nw.new_node(
        nodegroup_clamped_random_offset().name,
        input_kwargs={
            "sigma": reroute_55,
            "min": 0.1000,
            "max": 0.9000,
            "Geometry": join_geometry,
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_15,
            "False": reroute_59,
            "True": clampedrandomoffset_2,
        },
    )

    reroute_54 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": buttonpanelv2.outputs["p2"]}
    )

    reroute_50 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": buttonpanelv2.outputs["origin"]}
    )

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_50})

    reroute_52 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": buttonpanelv2.outputs["p1"]}
    )

    reroute_53 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_52})

    coordinatetransform = nw.new_node(
        nodegroup_coordinate_transform().name,
        input_kwargs={
            "Geometry": switch,
            "origin": reroute_54,
            "p1": reroute_51,
            "p2": reroute_53,
        },
    )

    duplicate_joints_on_parent = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Parent": hinge_joint.outputs["Parent"],
            "Child": hinge_joint.outputs["Child"],
            "Points": coordinatetransform,
        },
    )

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": equal})

    reroute_38 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_37})

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": reroute_33})

    add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Dashboard Depth"],
            1: group_input.outputs["GrateScale"],
        },
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_3, 1: -1.0000}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_38,
            "False": combine_xyz_7,
            "True": combine_xyz_8,
        },
        attrs={"input_type": "VECTOR"},
    )

    multiply_2 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: switch_4}, attrs={"operation": "MULTIPLY"}
    )

    reroute_48 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply_2.outputs["Vector"]}
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": duplicate_joints_on_parent,
            "Translation": reroute_48,
        },
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": transform_geometry_4}
    )

    reroute_45 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_4})

    reroute_46 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_45})

    reroute_57 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_46})

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_3, "Translation": reroute_57},
    )

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": transform_geometry_5}
    )

    reroute_32 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    greater_than_1 = nw.new_node(
        Nodes.Compare, input_kwargs={2: reroute_32, 3: 2}, attrs={"data_type": "INT"}
    )

    maximum_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Row 2"], 1: group_input.outputs["Row 1"]},
        attrs={"operation": "MAXIMUM"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_9, 1: maximum_2},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_3}, attrs={"operation": "MULTIPLY"}
    )

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_31, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_4, 1: -1.0000, 2: multiply_5},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_40 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_4})

    switch_7 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": greater_than_1,
            "False": multiply_add_6,
            "True": reroute_40,
        },
        attrs={"input_type": "FLOAT"},
    )

    combine_xyz_13 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": switch_7})

    reroute_49 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_13})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_3,
            "Translation": reroute_49,
            "Rotation": (0.0000, 0.0000, 1.5708),
        },
    )

    reroute_62 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_1}
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_39,
            "False": join_geometry_4,
            "True": reroute_62,
        },
    )

    stove_burners = nw.new_node(
        nodegroup_stove_burners().name,
        input_kwargs={
            "n1": group_input.outputs["Row 1"],
            "Thickness": group_input.outputs["Thickness"],
            "Burner Height": group_input.outputs["Burner Height"],
            "Burner radius": group_input.outputs["Burner Radius"],
            "Bar Length": group_input.outputs["BurnerBarLength"],
            "kind": group_input.outputs["BurnerKind"],
            "n2": group_input.outputs["Row 2"],
            "Height": group_input.outputs["Height"],
            "BlackMetal": group_input.outputs["BlackMetal"],
            "PrimaryMaterial": group_input.outputs["PrimaryMaterial"],
        },
    )

    combine_xyz_11 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["GrateScale"],
            "Y": group_input.outputs["GrateScale"],
            "Z": 1.0000,
        },
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": stove_burners, "Scale": combine_xyz_11},
    )

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    reroute_60 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_29})

    reroute_61 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_60})

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [switch_1, reroute_61]}
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": join_geometry_2}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": realize_instances},
        attrs={"is_active_output": True},
    )


class StovetopFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    @gin.configurable(module="StovetopFactory")
    def sample_joint_parameters(
        cls,
        knob_stiffness_min: float = 0.0,
        knob_stiffness_max: float = 0.0,
        knob_damping_min: float = 2.0,
        knob_damping_max: float = 5.0,
    ):
        return {
            "knob": {
                "stiffness": uniform(knob_stiffness_min, knob_stiffness_max),
                "damping": uniform(knob_damping_min, knob_damping_max),
            },
        }

    def sample_parameters(self):
        # add code here to randomly sample from parameters
        from infinigen.assets.composition.material_assignments import metals

        button_type = weighted_sample(
            [(0, 0.15), (1, 0.15), (2, 0.4), (3, 0.2), (4, 0.2)]
        )

        match button_type:
            case 0 | 1:
                # button type 0
                r = uniform(0.06, 0.09)
                h = uniform(0.06, 0.09)
                k = uniform(0.25, 0.45)
                bezier = uniform(0.25, 0.45)
                button_params = {
                    "ButtonRadius": r,
                    "ButtonHeight": h,
                    "ButtonKind": button_type,
                    "Buttonk": k,
                    "ButtonBezierParam": bezier,
                }
            case 2:
                # button type 2
                r = uniform(0.06, 0.09)
                h = uniform(0.06, 0.09)
                k = uniform(0.2, 0.4)
                button_params = {
                    "ButtonRadius": r,
                    "ButtonHeight": h,
                    "ButtonKind": button_type,
                    "Buttonk": k,
                }
            case 3:
                r = uniform(0.06, 0.09)
                h = uniform(0.09, 0.15)
                k = uniform(0, r)
                button_base_height = uniform(0.15, 0.4)
                button_params = {
                    "ButtonRadius": r,
                    "ButtonHeight": h,
                    "ButtonKind": button_type,
                    "Buttonk": k,
                    "ButtonBaseHeight": button_base_height,
                }
            case 4:
                r = uniform(0.06, 0.09)
                h = uniform(0.09, 0.15)
                button_base_height = uniform(0.15, 0.4)
                button_params = {
                    "ButtonRadius": r,
                    "ButtonHeight": h,
                    "ButtonKind": button_type,
                    "ButtonBaseHeight": button_base_height,
                }

        l = [(1, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 5)]
        i = weighted_sample(
            [
                (0, 0.1),
                (1, 0.35),
                (2, 0.35),
                (3, 0.1),
                (4, 0.05),
                (5, 0.05),
            ]
        )
        ra, rb = l[i]
        b = uniform(0, 1) < 0.5
        row1 = rb if b else ra
        row2 = ra if b else rb
        if row2 == 1 and row1 == 1:
            if uniform(0, 1) < 0.5:
                row1 = 2
            else:
                row2 = 2

        extra_knobs = weighted_sample([(0, 0.7), (1, 0.2), (2, 0.1)])
        knob_gap_exists = uniform(0, 1) < 0.3 and extra_knobs < 2
        random_offset = uniform(0, 1) < 0.4
        if not random_offset:
            sigma = 0
        elif knob_gap_exists and random_offset:
            sigma = uniform(0, 0.025)
        else:
            sigma = uniform(0, 0.035)

        dial_placement_params = {
            "Seed": randint(0, 1000),
            "Row 1": row1,
            "Row 2": row2,
            "ExtraKnobs": extra_knobs,
            "KnobGap": uniform(0.2, 0.35),
            "KnobGapExists": knob_gap_exists,
            "KnobSigma": sigma,
            "TwoButtonRows": ((row1 + row2 + extra_knobs) % 2 == 1)
            and uniform(0, 1) < 0.4,
        }

        grate_params = {"Thickness": uniform(0.015, 0.03)}

        kind = weighted_sample([(0, 0.4), (1, 0.4), (2, 0.3), (3, 0.15)])
        match kind:
            case 0:
                h = uniform(0.05, 0.1)
                r = uniform(0.08, 0.15)
                l = uniform(0.5, 1)
                burner_params = {
                    "BurnerKind": kind,
                    "Burner Height": h,
                    "Burner Radius": r,
                    "BurnerBarLength": l,
                }

            case 1:
                h = uniform(0.05, 0.1)
                r = uniform(0.08, 0.15)
                l = uniform(0.3, 1)
                burner_params = {
                    "BurnerKind": kind,
                    "Burner Height": h,
                    "Burner Radius": r,
                    "BurnerBarLength": l,
                }

            case 2:
                r = uniform(0.12, 0.175)
                burner_params = {"BurnerKind": kind, "Burner Radius": r}

            case 3:
                r = uniform(0.15, 0.25)
                h = 0.01
                burner_params = {
                    "BurnerKind": kind,
                    "Burner Height": h,
                    "Burner Radius": r,
                }

        match burner_params["BurnerKind"]:
            case 3:
                if uniform() < 0.2:
                    primary_material = weighted_sample(metals)()()
                    button_material = weighted_sample(metals)()()
                else:
                    primary_material = weighted_sample(metals)()()
                    if uniform() < 0.3:
                        button_material = weighted_sample(metals)()()
                    else:
                        button_material = primary_material

                material_params = {
                    "BlackMetal": material_assignments.metal.BrushedBlackMetal()(),
                    "PrimaryMaterial": primary_material,
                    "ButtonMaterial": button_material,
                }
            case 2:
                if uniform() < 0.2:
                    primary_material = weighted_sample(metals)()()
                    button_material = weighted_sample(metals)()()
                else:
                    if uniform() < 0.5:
                        primary_material = weighted_sample(metals)()()
                        if uniform() < 0.3:
                            button_material = weighted_sample(metals)()()
                        else:
                            button_material = primary_material
                    else:
                        primary_material = weighted_sample(metals)()()
                        if uniform() < 0.3:
                            button_material = weighted_sample(metals)()()
                        else:
                            button_material = primary_material

                material_params = {
                    "BlackMetal": material_assignments.metal.BrushedBlackMetal()(),
                    "PrimaryMaterial": primary_material,
                    "ButtonMaterial": button_material,
                }
            case _:
                if uniform() < 0.2:
                    primary_material = weighted_sample(metals)()()
                    button_material = weighted_sample(metals)()()
                else:
                    primary_material = weighted_sample(metals)()()
                    if uniform() < 0.3:
                        button_material = weighted_sample(metals)()()
                    else:
                        button_material = primary_material
                material_params = {
                    "BlackMetal": material_assignments.metal.BrushedBlackMetal()(),
                    "PrimaryMaterial": primary_material,
                    "ButtonMaterial": button_material,
                }

        # TODO: FIX locations 2,3
        if row1 + row2 <= 5:
            location = weighted_sample([(0, 0.45), (1, 0.35), (2, 0.1), (3, 0.1)])
        else:
            location = weighted_sample([(0, 0.5), (1, 0.4)])

        match location:
            case 0:
                # front (location 0)
                h = uniform(0.06, 0.2)
                dashboard_h = weighted_sample([(h, 0.3), (uniform(0, h), 0.7)])
                m = 2 if dial_placement_params["TwoButtonRows"] else 1
                d = uniform(2 * button_params["ButtonRadius"] * m, 0.4)
                s = uniform(0.5, 0.9)
                dashboard_params = {
                    "DashboardLocation": location,
                    "Height": h,
                    "DashboardHeight": dashboard_h,
                    "Dashboard Depth": d,
                    "GrateScale": s,
                }
            case 1:
                # back (location 1)
                h = uniform(0.06, 0.2)
                dashboard_h = weighted_sample([(h, 0.2), (uniform(h, 3 * h), 0.8)])
                s = uniform(0.5, 0.9)
                m = 2 if dial_placement_params["TwoButtonRows"] else 1
                d = uniform(max(0.2, 2 * button_params["ButtonRadius"] * m), 0.4)
                dashboard_params = {
                    "DashboardLocation": location,
                    "Height": h,
                    "DashboardHeight": dashboard_h,
                    "Dashboard Depth": d,
                    "GrateScale": s,
                }
            case _:
                h = uniform(0.06, 0.2)
                m = 2 if dial_placement_params["TwoButtonRows"] else 1
                d = uniform(2 * button_params["ButtonRadius"] * m, 0.4)
                s = uniform(0.5, 0.9)
                dashboard_params = {
                    "DashboardLocation": location,
                    "Height": h,
                    "DashboardHeight": h,
                    "Dashboard Depth": d,
                    "GrateScale": s,
                }

        params = {
            **dial_placement_params,
            **burner_params,
            **grate_params,
            **dashboard_params,
            **material_params,
            **button_params,
        }
        print(params)

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
