import gin
from numpy.random import randint, uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.joints import nodegroup_hinge_joint
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import weighted_sample


@node_utils.to_nodegroup(
    "nodegroup_node_group_008", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_008(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Base Width", 0.0000),
            ("NodeSocketFloat", "Extrusion", 1.0000),
        ],
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Base Width"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    divide_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: divide, 1: 1.0000}, attrs={"operation": "DIVIDE"}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide, "Z": divide_1}
    )

    mesh_line_1 = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={"Count": 2, "Offset": combine_xyz_1},
        attrs={"mode": "END_POINTS"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input.outputs["Base Width"]}
    )

    mesh_line = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={"Count": 2, "Offset": combine_xyz},
        attrs={"mode": "END_POINTS"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [mesh_line_1, mesh_line]}
    )

    convex_hull = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": join_geometry}
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": convex_hull,
            "Offset Scale": group_input.outputs["Extrusion"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": extrude_mesh.outputs["Mesh"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_010", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_010(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Base Width", 0.0000),
            ("NodeSocketFloat", "Extrusion", 1.0000),
        ],
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Base Width"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    divide_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: divide, 1: 1.0000}, attrs={"operation": "DIVIDE"}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide, "Z": divide_1}
    )

    mesh_line_1 = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={"Count": 2, "Offset": combine_xyz_1},
        attrs={"mode": "END_POINTS"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input.outputs["Base Width"]}
    )

    mesh_line = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={"Count": 2, "Offset": combine_xyz},
        attrs={"mode": "END_POINTS"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [mesh_line_1, mesh_line]}
    )

    convex_hull = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": join_geometry}
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": convex_hull,
            "Offset Scale": group_input.outputs["Extrusion"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": extrude_mesh.outputs["Mesh"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_007", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_007(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "W", 0.5000),
            ("NodeSocketFloat", "T", 0.5000),
            ("NodeSocketFloat", "FFC", 0.0000),
            ("NodeSocketFloat", "Dist", 0.0000),
            ("NodeSocketFloat", "FIStrength", 0.0000),
            ("NodeSocketFloat", "FIStart", 0.5000),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["W"], 1: group_input.outputs["T"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: group_input.outputs["FIStart"]},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: subtract},
        attrs={"operation": "MULTIPLY"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["FFC"]}
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Dist"]}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_1, "Y": reroute, "Z": reroute_1}
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz})

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": reroute_8})

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": combine_xyz}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Translation": combine_xyz_2},
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["X"]}
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["Y"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["FIStrength"]}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: reroute_2},
        attrs={"operation": "MULTIPLY"},
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_2, 1: multiply_1},
        attrs={"operation": "DIVIDE"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_1, 1: reroute_7},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_9, 1: multiply_3},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["Z"]}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_4, "Y": subtract_1, "Z": reroute_5}
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": transform_geometry, "Position": combine_xyz_1},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_position},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_017", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_017(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Thickness", 0.0000),
            ("NodeSocketFloat", "Flap Fraction Cover", 0.0000),
            ("NodeSocketFloat", "Distance to Opposite Side", 0.0000),
            ("NodeSocketFloat", "Flap Inward Strength", 0.0000),
            ("NodeSocketFloat", "Flap Inward Start", 0.0000),
        ],
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Flap Inward Start"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Flap Fraction Cover"],
            1: group_input.outputs["Distance to Opposite Side"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_5, 1: multiply},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Thickness"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_1, "Y": reroute_3, "Z": reroute_7}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_1, 1: (-0.5000, -1.0000, -1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Translation": multiply_2.outputs["Vector"],
        },
    )

    prism = nw.new_node(
        nodegroup_node_group_007().name,
        input_kwargs={
            "W": group_input.outputs["Distance to Opposite Side"],
            "T": group_input.outputs["Flap Fraction Cover"],
            "FFC": group_input.outputs["Width"],
            "Dist": group_input.outputs["Thickness"],
            "FIStrength": group_input.outputs["Flap Inward Strength"],
            "FIStart": group_input.outputs["Flap Inward Start"],
        },
        label="Prism",
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": prism})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_geometry, reroute_6]}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Rotation": (0.0000, 0.0000, 1.5708)},
    )

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_1}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": transform_geometry_1}
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Min"]}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_3})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_11, "Translation": combine_xyz_2},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    pivot = nw.new_node(
        nodegroup_node_group_010().name,
        input_kwargs={"Base Width": reroute_1, "Extrusion": divide},
        label="Pivot",
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide_1})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": pivot,
            "Translation": combine_xyz_3,
            "Rotation": (0.0000, 4.7124, 1.5708),
        },
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry_2})

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_geometry_3, reroute_8]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_9, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": divide_2})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_1, "Translation": combine_xyz_4},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": transform_geometry_4},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_018", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_018(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Thickness", 0.0000),
            ("NodeSocketFloat", "Flap Fraction Cover", 0.0000),
            ("NodeSocketFloat", "Distance to Opposite Side", 0.0000),
            ("NodeSocketFloat", "Flap Inward Strength", 0.0000),
            ("NodeSocketFloat", "Flap Inward Start", 0.0000),
        ],
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Flap Inward Start"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Flap Fraction Cover"],
            1: group_input.outputs["Distance to Opposite Side"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_5, 1: multiply},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Thickness"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_1, "Y": reroute_3, "Z": reroute_7}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_1, 1: (-0.5000, -1.0000, -1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Translation": multiply_2.outputs["Vector"],
        },
    )

    prism = nw.new_node(
        nodegroup_node_group_007().name,
        input_kwargs={
            "W": group_input.outputs["Distance to Opposite Side"],
            "T": group_input.outputs["Flap Fraction Cover"],
            "FFC": group_input.outputs["Width"],
            "Dist": group_input.outputs["Thickness"],
            "FIStrength": group_input.outputs["Flap Inward Strength"],
            "FIStart": group_input.outputs["Flap Inward Start"],
        },
        label="Prism",
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": prism})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_geometry, reroute_6]}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Rotation": (0.0000, 0.0000, 1.5708)},
    )

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_1}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": transform_geometry_1}
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Min"]}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_3})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_11, "Translation": combine_xyz_2},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    pivot = nw.new_node(
        nodegroup_node_group_010().name,
        input_kwargs={"Base Width": reroute_1, "Extrusion": divide},
        label="Pivot",
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide_1})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": pivot,
            "Translation": combine_xyz_3,
            "Rotation": (0.0000, 4.7124, 1.5708),
        },
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry_2})

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_geometry_3, reroute_8]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_9, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": divide_2})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_1, "Translation": combine_xyz_4},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": transform_geometry_4},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_012", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_012(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Parent Width", 0.0000),
            ("NodeSocketFloat", "Height", 0.5000),
            ("NodeSocketFloat", "Thickness", 0.5000),
            ("NodeSocketFloat", "Parent Center", 0.0000),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Thickness"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: divide, 1: divide_1})

    pivot = nw.new_node(
        nodegroup_node_group_008().name,
        input_kwargs={"Base Width": reroute_9, "Extrusion": add},
        label="Pivot",
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Parent Center"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_3, 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_3, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": subtract, "Z": multiply_1}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": pivot,
            "Translation": combine_xyz_1,
            "Rotation": (0.0000, 0.0000, 1.5708),
        },
    )

    add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Parent Width"],
            1: group_input.outputs["Thickness"],
        },
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add_1})

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": reroute_5})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_2,
            "Translation": combine_xyz,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Parent Width"]}
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_4, 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide_1})

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_2, "Y": reroute_10}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_7, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_2})

    add_2 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: combine_xyz_2, 1: combine_xyz_3}
    )

    transform_geometry_10 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_3,
            "Translation": add_2.outputs["Vector"],
        },
    )

    reroute_19 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_10}
    )

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: divide, 1: divide_1})

    pivot_1 = nw.new_node(
        nodegroup_node_group_008().name,
        input_kwargs={"Base Width": reroute_9, "Extrusion": add_3},
        label="Pivot",
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": pivot_1, "Rotation": (0.0000, 0.0000, 1.5708)},
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_4,
            "Rotation": (0.0000, -1.5708, 0.0000),
        },
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 0.0000, 1: reroute_9},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": subtract_1})

    add_4 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: combine_xyz_2, 1: combine_xyz_4}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_5,
            "Translation": add_4.outputs["Vector"],
        },
    )

    reroute_18 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_1}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: divide_3}, attrs={"operation": "SUBTRACT"}
    )

    pivot_2 = nw.new_node(
        nodegroup_node_group_008().name,
        input_kwargs={"Base Width": reroute_11, "Extrusion": subtract_2},
        label="Pivot",
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_1})

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": pivot_2,
            "Translation": reroute_12,
            "Rotation": (0.0000, 0.0000, 1.5708),
        },
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz})

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_6,
            "Translation": reroute_14,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    add_5 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: combine_xyz_2, 1: combine_xyz_3}
    )

    reroute_15 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_5.outputs["Vector"]}
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_15})

    transform_geometry_11 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_7, "Translation": reroute_16},
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_3, 1: reroute_7},
        attrs={"operation": "SUBTRACT"},
    )

    pivot_3 = nw.new_node(
        nodegroup_node_group_008().name,
        input_kwargs={"Base Width": reroute_11, "Extrusion": subtract_3},
        label="Pivot",
    )

    transform_geometry_8 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": pivot_3, "Rotation": (0.0000, 0.0000, 1.5708)},
    )

    transform_geometry_9 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_8,
            "Rotation": (0.0000, -1.5708, 0.0000),
        },
    )

    add_6 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: combine_xyz_2, 1: combine_xyz_4}
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_6.outputs["Vector"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_9, "Translation": reroute_17},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Right Rotator Cut": reroute_19,
            "Left Rotator Cut": reroute_18,
            "Right Rotator": transform_geometry_11,
            "Left Rotator ": transform_geometry,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_009", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_009(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Depth", 0.5000),
            ("NodeSocketFloat", "Height", 0.5000),
            ("NodeSocketFloat", "Thickness", 0.5000),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Thickness"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    pivot = nw.new_node(
        nodegroup_node_group_008().name,
        input_kwargs={"Base Width": reroute_1, "Extrusion": divide},
        label="Pivot",
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": pivot, "Rotation": (0.0000, 0.0000, 1.5708)},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Depth"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_3, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide_2})

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_1, "Y": multiply, "Z": reroute_5}
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_2, "Translation": combine_xyz_3},
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_3}
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide, 1: divide_3},
        attrs={"operation": "SUBTRACT"},
    )

    pivot_1 = nw.new_node(
        nodegroup_node_group_008().name,
        input_kwargs={"Base Width": reroute_9, "Extrusion": subtract},
        label="Pivot",
    )

    transform_geometry_8 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": pivot_1, "Rotation": (0.0000, 0.0000, 1.5708)},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_2, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_1, "Z": multiply_1}
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_4})

    transform_geometry_9 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_8,
            "Translation": reroute_12,
            "Rotation": (-3.1416, 0.0000, 0.0000),
        },
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": transform_geometry_9}
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_1.outputs["Vector"]}
    )

    divide_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide_3})

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    add = nw.new_node(Nodes.Math, input_kwargs={0: divide_4, 1: reroute_7})

    subtract_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: 0.0000, 1: add}, attrs={"operation": "SUBTRACT"}
    )

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract_2, "Y": reroute_17}
    )

    transform_geometry_10 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_13, "Translation": combine_xyz},
    )

    pivot_2 = nw.new_node(
        nodegroup_node_group_008().name,
        input_kwargs={"Base Width": reroute_1, "Extrusion": divide},
        label="Pivot",
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": pivot_2, "Rotation": (0.0000, 0.0000, 1.5708)},
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_4,
            "Translation": combine_xyz_4,
            "Rotation": (-3.1416, 0.0000, 0.0000),
        },
    )

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_5}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_14, "Translation": combine_xyz},
    )

    divide_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    divide_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_5, 1: divide_6},
        attrs={"operation": "SUBTRACT"},
    )

    pivot_3 = nw.new_node(
        nodegroup_node_group_008().name,
        input_kwargs={"Base Width": reroute_9, "Extrusion": subtract_3},
        label="Pivot",
    )

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": pivot_3, "Rotation": (0.0000, 0.0000, 1.5708)},
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_3})

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_6, "Translation": reroute_11},
    )

    reroute_15 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_7}
    )

    transform_geometry_11 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_15, "Translation": combine_xyz},
    )

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_9}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_16, "Translation": combine_xyz},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Top Rotator Cut": transform_geometry_10,
            "Bottom Rotator Cut": transform_geometry_1,
            "Top Rotator": transform_geometry_11,
            "Bottom Rotator": transform_geometry,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_013", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_013(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Top", None),
            ("NodeSocketGeometry", "Right", None),
            ("NodeSocketGeometry", "Left", None),
            ("NodeSocketGeometry", "Bottom", None),
        ],
    )

    intersect = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 2": [group_input.outputs["Right"], group_input.outputs["Top"]]
        },
        attrs={"solver": "EXACT", "operation": "INTERSECT"},
    )

    intersect_1 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 2": [group_input.outputs["Left"], group_input.outputs["Top"]]
        },
        attrs={"solver": "EXACT", "operation": "INTERSECT"},
    )

    intersect_2 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 2": [group_input.outputs["Bottom"], group_input.outputs["Right"]]
        },
        attrs={"solver": "EXACT", "operation": "INTERSECT"},
    )

    intersect_3 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 2": [group_input.outputs["Bottom"], group_input.outputs["Left"]]
        },
        attrs={"solver": "EXACT", "operation": "INTERSECT"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                intersect.outputs["Mesh"],
                intersect_1.outputs["Mesh"],
                intersect_2.outputs["Mesh"],
                intersect_3.outputs["Mesh"],
            ]
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_003", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_003(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Depth", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "Thickness", 0.0000),
            ("NodeSocketFloat", "Distance Opposite Site", 0.0000),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Depth"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Thickness"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    rotators = nw.new_node(
        nodegroup_node_group_009().name,
        input_kwargs={"Depth": reroute_6, "Height": reroute_9, "Thickness": reroute_8},
        label="Rotators",
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": reroute_3, "Z": reroute_5}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    transform_geometry = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube.outputs["Mesh"]}
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": transform_geometry}
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_1.outputs["Vector"]}
    )

    rotators_1 = nw.new_node(
        nodegroup_node_group_012().name,
        input_kwargs={
            "Parent Width": reroute_7,
            "Height": reroute_9,
            "Thickness": reroute_8,
            "Parent Center": separate_xyz.outputs["Y"],
        },
        label="Rotators",
    )

    corners = nw.new_node(
        nodegroup_node_group_013().name,
        input_kwargs={
            "Top": rotators.outputs["Top Rotator Cut"],
            "Right": rotators_1.outputs["Right Rotator Cut"],
            "Left": rotators_1.outputs["Left Rotator Cut"],
            "Bottom": rotators.outputs["Bottom Rotator Cut"],
        },
        label="Corners",
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                rotators.outputs["Top Rotator"],
                rotators.outputs["Bottom Rotator"],
                rotators_1.outputs["Right Rotator"],
                rotators_1.outputs["Left Rotator "],
            ]
        },
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [corners, join_geometry_1, reroute_10]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Face 1": join_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_001", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_001(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Depth", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "Thickness", 0.0000),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Depth"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Thickness"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    rotators = nw.new_node(
        nodegroup_node_group_009().name,
        input_kwargs={"Depth": reroute_6, "Height": reroute_9, "Thickness": reroute_8},
        label="Rotators",
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: group_input.outputs["Thickness"]},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": reroute_3, "Z": reroute_5}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    transform_geometry = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube.outputs["Mesh"]}
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": transform_geometry}
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_1.outputs["Vector"]}
    )

    rotators_1 = nw.new_node(
        nodegroup_node_group_012().name,
        input_kwargs={
            "Parent Width": reroute_7,
            "Height": reroute_9,
            "Thickness": reroute_8,
            "Parent Center": separate_xyz.outputs["Y"],
        },
        label="Rotators",
    )

    corners = nw.new_node(
        nodegroup_node_group_013().name,
        input_kwargs={
            "Top": rotators.outputs["Top Rotator Cut"],
            "Right": rotators_1.outputs["Right Rotator Cut"],
            "Left": rotators_1.outputs["Left Rotator Cut"],
            "Bottom": rotators.outputs["Bottom Rotator Cut"],
        },
        label="Corners",
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                rotators.outputs["Top Rotator"],
                rotators.outputs["Bottom Rotator"],
                rotators_1.outputs["Right Rotator"],
                rotators_1.outputs["Left Rotator "],
            ]
        },
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [corners, join_geometry_1, reroute_10]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Face 1": join_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_022", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_022(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "V", 0),
            ("NodeSocketFloat", "Distanc Opp Site", 0.0000),
            ("NodeSocketInt", "Face Number", 0),
            ("NodeSocketInt", "Side Number", 0),
            ("NodeSocketFloat", "FlapFracCover", 0.0000),
            ("NodeSocketFloat", "Thickness", 0.0000),
            ("NodeSocketInt", "A", 0),
        ],
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["V"], 3: 4},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Face Number"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_and = nw.new_node(Nodes.BooleanMath, input_kwargs={0: equal, 1: equal_1})

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": op_and})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["V"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_1, 3: 3},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Face Number"], 3: 2},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Face Number"], 3: 4},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_or = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: equal_3, 1: equal_4},
        attrs={"operation": "OR"},
    )

    op_and_1 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: equal_2, 1: op_or})

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": op_and_1})

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    equal_5 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_14},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": equal_5})

    equal_6 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["V"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": equal_6})

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Face Number"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    equal_7 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_3, 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_and_2 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: reroute_12, 1: equal_7})

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": op_and_2})

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    equal_8 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["V"], 3: 4},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_or_1 = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: equal_6, 1: equal_8},
        attrs={"operation": "OR"},
    )

    equal_9 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_3, 3: 3},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_and_3 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: op_or_1, 1: equal_9})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.5000

    value_3 = nw.new_node(Nodes.Value)
    value_3.outputs[0].default_value = 0.0000

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": op_and_3, "False": value, "True": value_3},
        attrs={"input_type": "FLOAT"},
    )

    value_4 = nw.new_node(Nodes.Value)
    value_4.outputs[0].default_value = 1.0000

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_19, "False": switch_6, "True": value_4},
        attrs={"input_type": "FLOAT"},
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": value})

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_20, "False": switch_3, "True": reroute_8},
        attrs={"input_type": "FLOAT"},
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["FlapFracCover"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    switch_7 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_17, "False": switch, "True": reroute_7},
        attrs={"input_type": "FLOAT"},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Distanc Opp Site"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_23, 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: reroute_23},
        attrs={"operation": "DIVIDE"},
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide})

    switch_8 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_13, "False": switch_7, "True": reroute_16},
        attrs={"input_type": "FLOAT"},
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 0.0000

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": value_3, "False": value_3, "True": value_2},
        attrs={"input_type": "FLOAT"},
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": value_2})

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal_5, "False": switch_5, "True": reroute_10},
        attrs={"input_type": "FLOAT"},
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_2})

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.0000

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": op_and_2, "False": value_3, "True": value_1},
        attrs={"input_type": "FLOAT"},
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": value_1})

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal_5, "False": switch_4, "True": reroute_9},
        attrs={"input_type": "FLOAT"},
    )

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_1})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "FFC": switch_8,
            "DTO": reroute_15,
            "FIStre": reroute_21,
            "FISta": reroute_22,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_002", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_002(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Depth", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "Thickness", 0.0000),
            ("NodeSocketFloat", "Distance Opposite Site", 0.0000),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Depth"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Thickness"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    rotators = nw.new_node(
        nodegroup_node_group_009().name,
        input_kwargs={"Depth": reroute_6, "Height": reroute_9, "Thickness": reroute_8},
        label="Rotators",
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": reroute_3, "Z": reroute_5}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    transform_geometry = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube.outputs["Mesh"]}
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": transform_geometry}
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_1.outputs["Vector"]}
    )

    rotators_1 = nw.new_node(
        nodegroup_node_group_012().name,
        input_kwargs={
            "Parent Width": reroute_7,
            "Height": reroute_9,
            "Thickness": reroute_8,
            "Parent Center": separate_xyz.outputs["Y"],
        },
        label="Rotators",
    )

    corners = nw.new_node(
        nodegroup_node_group_013().name,
        input_kwargs={
            "Top": rotators.outputs["Top Rotator Cut"],
            "Right": rotators_1.outputs["Right Rotator Cut"],
            "Left": rotators_1.outputs["Left Rotator Cut"],
            "Bottom": rotators.outputs["Bottom Rotator Cut"],
        },
        label="Corners",
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                rotators.outputs["Top Rotator"],
                rotators.outputs["Bottom Rotator"],
                rotators_1.outputs["Right Rotator"],
                rotators_1.outputs["Left Rotator "],
            ]
        },
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [corners, join_geometry_1, reroute_10]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Face 1": join_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_004", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_004(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "V", 0),
            ("NodeSocketFloat", "Distanc Opp Site", 0.0000),
            ("NodeSocketInt", "Face Number", 0),
            ("NodeSocketInt", "Side Number", 0),
            ("NodeSocketFloat", "FlapFracCover", 0.0000),
            ("NodeSocketFloat", "Thickness", 0.0000),
        ],
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["V"], 3: 4},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Face Number"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_and = nw.new_node(Nodes.BooleanMath, input_kwargs={0: equal, 1: equal_1})

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": op_and})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["V"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_1, 3: 3},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Face Number"], 3: 2},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Face Number"], 3: 4},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_or = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: equal_3, 1: equal_4},
        attrs={"operation": "OR"},
    )

    op_and_1 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: equal_2, 1: op_or})

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": op_and_1})

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    equal_5 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_14},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": equal_5})

    equal_6 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["V"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": equal_6})

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Face Number"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    equal_7 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_3, 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_and_2 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: reroute_12, 1: equal_7})

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": op_and_2})

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    equal_8 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["V"], 3: 4},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_or_1 = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: equal_6, 1: equal_8},
        attrs={"operation": "OR"},
    )

    equal_9 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_3, 3: 3},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_and_3 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: op_or_1, 1: equal_9})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.5000

    value_3 = nw.new_node(Nodes.Value)
    value_3.outputs[0].default_value = 0.0000

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": op_and_3, "False": value, "True": value_3},
        attrs={"input_type": "FLOAT"},
    )

    value_4 = nw.new_node(Nodes.Value)
    value_4.outputs[0].default_value = 1.0000

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_19, "False": switch_6, "True": value_4},
        attrs={"input_type": "FLOAT"},
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": value})

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_20, "False": switch_3, "True": reroute_8},
        attrs={"input_type": "FLOAT"},
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["FlapFracCover"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    switch_7 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_17, "False": switch, "True": reroute_7},
        attrs={"input_type": "FLOAT"},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Distanc Opp Site"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_23, 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: reroute_23},
        attrs={"operation": "DIVIDE"},
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide})

    switch_8 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_13, "False": switch_7, "True": reroute_16},
        attrs={"input_type": "FLOAT"},
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 0.0000

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": value_3, "False": value_3, "True": value_2},
        attrs={"input_type": "FLOAT"},
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": value_2})

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal_5, "False": switch_5, "True": reroute_10},
        attrs={"input_type": "FLOAT"},
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_2})

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.0000

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": op_and_2, "False": value_3, "True": value_1},
        attrs={"input_type": "FLOAT"},
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": value_1})

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal_5, "False": switch_4, "True": reroute_9},
        attrs={"input_type": "FLOAT"},
    )

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_1})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "FFC": switch_8,
            "DTO": reroute_15,
            "FIStre": reroute_21,
            "FISta": reroute_22,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_011", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_011(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Thickness", 0.0000),
            ("NodeSocketFloat", "Flap Fraction Cover", 0.0000),
            ("NodeSocketFloat", "Distance to Opposite Side", 0.0000),
            ("NodeSocketFloat", "Flap Inward Strength", 0.0000),
            ("NodeSocketFloat", "Flap Inward Start", 0.0000),
        ],
    )

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Flap Inward Start"]}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Flap Fraction Cover"],
            1: group_input.outputs["Distance to Opposite Side"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_5, 1: multiply},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Thickness"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_1, "Y": reroute_3, "Z": reroute_7}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_1, 1: (-0.5000, -1.0000, -1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Translation": multiply_2.outputs["Vector"],
        },
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Distance to Opposite Side"]},
    )

    reroute_12 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Flap Fraction Cover"]},
    )

    reroute_15 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Thickness"]}
    )

    reroute_14 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Flap Inward Strength"]},
    )

    prism = nw.new_node(
        nodegroup_node_group_007().name,
        input_kwargs={
            "W": reroute_13,
            "T": reroute_12,
            "FFC": reroute_15,
            "Dist": reroute_17,
            "FIStrength": reroute_14,
            "FIStart": reroute_16,
        },
        label="Prism",
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": prism})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_geometry, reroute_6]}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Rotation": (0.0000, 0.0000, 1.5708)},
    )

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_1}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": transform_geometry_1}
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Min"]}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_3})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_11, "Translation": combine_xyz_2},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    pivot = nw.new_node(
        nodegroup_node_group_010().name,
        input_kwargs={"Base Width": reroute_1, "Extrusion": divide},
        label="Pivot",
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide_1})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": pivot,
            "Translation": combine_xyz_3,
            "Rotation": (0.0000, 4.7124, 1.5708),
        },
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry_2})

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_geometry_3, reroute_8]}
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_9, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": divide_2})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_1, "Translation": combine_xyz_4},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": transform_geometry_4},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_019", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_019(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "F1/3 Width", 0.0000),
            ("NodeSocketFloat", "Box Thickness", 0.0000),
            ("NodeSocketFloat", "F3 Height", 0.5000),
            ("NodeSocketInt", "Box Version", 0),
            ("NodeSocketFloat", "V5 Faction Length", 0.2500),
            ("NodeSocketVector", "BoundingBoxTopFlapV6", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "V3 Fraction of Width", 0.0000),
            ("NodeSocketFloat", "F2/4 Width", 0.0000),
            ("NodeSocketVector", "F1/3 Half Negative", (0.0000, 0.0000, 0.0000)),
        ],
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["F3 Height"],
            1: group_input.outputs["Box Thickness"],
        },
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide})

    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_14})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["F1/3 Width"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Box Thickness"], 1: 1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Box Thickness"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Box Version"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["F2/4 Width"]}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Box Thickness"], 1: 2.5000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_8, 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract_1})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 3.0000

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 1.0000

    reroute_6 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["V3 Fraction of Width"]},
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    flap_length_calculator = nw.new_node(
        nodegroup_node_group_004().name,
        input_kwargs={
            "V": reroute_5,
            "Distanc Opp Site": reroute_9,
            "Face Number": value,
            "Side Number": value_1,
            "FlapFracCover": reroute_7,
            "Thickness": reroute_3,
        },
        label="Flap Length Calculator",
    )

    flaps = nw.new_node(
        nodegroup_node_group_011().name,
        input_kwargs={
            "Width": subtract,
            "Thickness": reroute_16,
            "Flap Fraction Cover": flap_length_calculator.outputs["FFC"],
            "Distance to Opposite Side": flap_length_calculator.outputs["DTO"],
            "Flap Inward Strength": flap_length_calculator.outputs["FIStre"],
            "Flap Inward Start": flap_length_calculator.outputs["FISta"],
        },
        label="Flaps",
    )

    convex_hull = nw.new_node(Nodes.ConvexHull, input_kwargs={"Geometry": flaps})

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": convex_hull})

    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 3.0000

    value_3 = nw.new_node(Nodes.Value)
    value_3.outputs[0].default_value = 2.0000

    flap_length_calculator_1 = nw.new_node(
        nodegroup_node_group_004().name,
        input_kwargs={
            "V": reroute_5,
            "Distanc Opp Site": reroute_9,
            "Face Number": value_2,
            "Side Number": value_3,
            "FlapFracCover": reroute_7,
            "Thickness": reroute_3,
        },
        label="Flap Length Calculator",
    )

    flaps_1 = nw.new_node(
        nodegroup_node_group_011().name,
        input_kwargs={
            "Width": subtract,
            "Thickness": reroute_16,
            "Flap Fraction Cover": flap_length_calculator_1.outputs["FFC"],
            "Distance to Opposite Side": flap_length_calculator_1.outputs["DTO"],
            "Flap Inward Strength": flap_length_calculator_1.outputs["FIStre"],
            "Flap Inward Start": flap_length_calculator_1.outputs["FISta"],
        },
        label="Flaps",
    )

    convex_hull_1 = nw.new_node(Nodes.ConvexHull, input_kwargs={"Geometry": flaps_1})

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": convex_hull_1})

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_17, 3: 6},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["F3 Height"]}
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    rotators = nw.new_node(
        nodegroup_node_group_009().name,
        input_kwargs={
            "Depth": reroute_15,
            "Height": reroute_20,
            "Thickness": reroute_22,
        },
        label="Rotators",
    )

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract})

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": reroute_16, "Z": reroute_13}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_7})

    transform_geometry = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube.outputs["Mesh"]}
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": transform_geometry}
    )

    subtract_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_2.outputs["Vector"]}
    )

    rotators_1 = nw.new_node(
        nodegroup_node_group_012().name,
        input_kwargs={
            "Parent Width": reroute_19,
            "Height": reroute_20,
            "Thickness": reroute_22,
            "Parent Center": separate_xyz.outputs["Y"],
        },
        label="Rotators",
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                rotators.outputs["Bottom Rotator"],
                rotators_1.outputs["Left Rotator "],
                rotators.outputs["Top Rotator"],
                rotators_1.outputs["Right Rotator"],
            ]
        },
    )

    corners = nw.new_node(
        nodegroup_node_group_013().name,
        input_kwargs={
            "Top": rotators.outputs["Top Rotator Cut"],
            "Right": rotators_1.outputs["Right Rotator Cut"],
            "Left": rotators_1.outputs["Left Rotator Cut"],
            "Bottom": rotators.outputs["Bottom Rotator Cut"],
        },
        label="Corners",
    )

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_28})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [join_geometry_1, corners, reroute_29]},
    )

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry})

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_3, 1: 4.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["F3 Height"],
            1: group_input.outputs["V5 Faction Length"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_3, 1: 1.2000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_10 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_2, "Y": 1.0000, "Z": multiply_4}
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_10})

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": group_input.outputs["BoundingBoxTopFlapV6"]},
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: 1.0000},
        attrs={"operation": "DIVIDE"},
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_1, 1: 0.0000},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_11 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": subtract_3})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Translation": combine_xyz_11,
        },
    )

    reroute_25 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_3}
    )

    reroute_23 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cube_1.outputs["Mesh"]}
    )

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    multiply_5 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_11, 1: (-1.0000, -1.0000, -1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_24,
            "Translation": multiply_5.outputs["Vector"],
        },
    )

    reroute_30 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_4}
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": join_geometry, "Mesh 2": [reroute_25, reroute_30]},
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal,
            "False": reroute_31,
            "True": difference.outputs["Mesh"],
        },
    )

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["F1/3 Half Negative"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": switch_1, "Translation": reroute_11}
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Box Version"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Box Version"], 3: 4},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_or = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: equal_1, 1: equal_2},
        attrs={"operation": "OR"},
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Box Version"], 3: 5},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Box Version"], 3: 6},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_or_1 = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: equal_3, 1: equal_4},
        attrs={"operation": "OR"},
    )

    op_or_2 = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: op_or, 1: op_or_1},
        attrs={"operation": "OR"},
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": op_or_2})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Pos Top Lid Axis": combine_xyz_9,
            "Pos Bottom Lid Axis": combine_xyz_9,
            "Top Flap": reroute_27,
            "Bottom Flap": reroute_26,
            "Side": transform_geometry_1,
            "F3 Exists": reroute_18,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_020", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_020(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "F4 Height", 0.5000),
            ("NodeSocketFloat", "Box Thickness", 0.5000),
            ("NodeSocketInt", "Version", 0),
            ("NodeSocketFloat", "V4 Space Open for cap", 0.5000),
            ("NodeSocketFloat", "Height Flap V5", 0.5000),
            ("NodeSocketFloat", "V3 Fraction of Width Coverede", 0.0000),
            ("NodeSocketFloat", "F1/3 Width", 0.0000),
            ("NodeSocketFloat", "F2/4 Width", 0.0000),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["F2/4 Width"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["F4 Height"]}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Box Thickness"]}
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    reroute_32 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_32})

    reroute_38 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_37})

    rotators = nw.new_node(
        nodegroup_node_group_009().name,
        input_kwargs={
            "Depth": reroute_17,
            "Height": reroute_21,
            "Thickness": reroute_38,
        },
        label="Rotators",
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Box Thickness"], 1: 1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract})

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    reroute_30 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_30})

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": reroute_6, "Z": reroute_4}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_7})

    transform_geometry = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube.outputs["Mesh"]}
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": transform_geometry}
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_1.outputs["Vector"]}
    )

    rotators_1 = nw.new_node(
        nodegroup_node_group_012().name,
        input_kwargs={
            "Parent Width": reroute_31,
            "Height": reroute_21,
            "Thickness": reroute_38,
            "Parent Center": separate_xyz.outputs["Y"],
        },
        label="Rotators",
    )

    corners = nw.new_node(
        nodegroup_node_group_013().name,
        input_kwargs={
            "Top": rotators.outputs["Top Rotator Cut"],
            "Right": rotators_1.outputs["Right Rotator Cut"],
            "Left": rotators_1.outputs["Left Rotator Cut"],
            "Bottom": rotators.outputs["Bottom Rotator Cut"],
        },
        label="Corners",
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                rotators_1.outputs["Left Rotator "],
                rotators.outputs["Top Rotator"],
                rotators.outputs["Bottom Rotator"],
                rotators_1.outputs["Right Rotator"],
            ]
        },
    )

    reroute_34 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_34})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [corners, join_geometry_1, reroute_35]},
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["F4 Height"],
            1: group_input.outputs["Box Thickness"],
        },
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide})

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_27})

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Version"]}
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_24})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_33, 3: 5},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Version"], 3: 4},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Version"], 3: 5},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_or = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: equal_1, 1: equal_2},
        attrs={"operation": "OR"},
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": op_or})

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["V4 Space Open for cap"],
            1: group_input.outputs["Box Thickness"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_1})

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: reroute_11},
        attrs={"operation": "SUBTRACT"},
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_15, "False": reroute_19, "True": subtract_2},
        attrs={"input_type": "FLOAT"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["F1/3 Width"]}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Box Thickness"], 1: 1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_2, 1: multiply_2},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract_3})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 4.0000

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 1.0000

    reroute_9 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["V3 Fraction of Width Coverede"]},
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_25})

    flap_length_calculator = nw.new_node(
        nodegroup_node_group_004().name,
        input_kwargs={
            "V": reroute_24,
            "Distanc Opp Site": reroute_20,
            "Face Number": value,
            "Side Number": value_1,
            "FlapFracCover": reroute_26,
        },
        label="Flap Length Calculator",
    )

    flaps = nw.new_node(
        nodegroup_node_group_011().name,
        input_kwargs={
            "Width": switch_1,
            "Thickness": reroute_32,
            "Flap Fraction Cover": flap_length_calculator.outputs["FFC"],
            "Distance to Opposite Side": flap_length_calculator.outputs["DTO"],
            "Flap Inward Strength": flap_length_calculator.outputs["FIStre"],
            "Flap Inward Start": flap_length_calculator.outputs["FISta"],
        },
        label="Flaps",
    )

    reroute_39 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": flaps})

    convex_hull = nw.new_node(Nodes.ConvexHull, input_kwargs={"Geometry": flaps})

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height Flap V5"], 1: 1.2000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Box Thickness"], 1: 4.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_3, "Y": multiply_4, "Z": 10.0000}
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_9})

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_4})

    combine_xyz_10 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": reroute_12})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Translation": combine_xyz_10,
        },
    )

    reroute_29 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_3}
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": convex_hull, "Mesh 2": reroute_29},
        attrs={"solver": "EXACT"},
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal,
            "False": reroute_39,
            "True": difference.outputs["Mesh"],
        },
    )

    reroute_42 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_2})

    reroute_40 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_42})

    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 4.0000

    value_3 = nw.new_node(Nodes.Value)
    value_3.outputs[0].default_value = 2.0000

    flap_length_calculator_1 = nw.new_node(
        nodegroup_node_group_004().name,
        input_kwargs={
            "V": reroute_8,
            "Distanc Opp Site": subtract_3,
            "Face Number": value_2,
            "Side Number": value_3,
            "FlapFracCover": reroute_10,
        },
        label="Flap Length Calculator",
    )

    flaps_1 = nw.new_node(
        nodegroup_node_group_011().name,
        input_kwargs={
            "Width": reroute_19,
            "Thickness": reroute_23,
            "Flap Fraction Cover": flap_length_calculator_1.outputs["FFC"],
            "Distance to Opposite Side": flap_length_calculator_1.outputs["DTO"],
            "Flap Inward Strength": flap_length_calculator_1.outputs["FIStre"],
            "Flap Inward Start": flap_length_calculator_1.outputs["FISta"],
        },
        label="Flaps",
    )

    convex_hull_1 = nw.new_node(Nodes.ConvexHull, input_kwargs={"Geometry": flaps_1})

    reroute_41 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": convex_hull_1})

    reroute_36 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_41})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": join_geometry,
            "Value": reroute_28,
            "Top Flap": reroute_40,
            "Bottom Flap": reroute_36,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "F2 Height", 0.5000),
            ("NodeSocketFloat", "Box Thickness", 0.5000),
            ("NodeSocketInt", "Box Version", 0),
            ("NodeSocketFloat", "V4 Space Open", 6.0000),
            ("NodeSocketFloat", "Height Flaps V5", 0.5000),
            ("NodeSocketFloat", "V3 Fraction of Width", 0.0000),
            ("NodeSocketFloat", "F1/3 Width", 0.0000),
            ("NodeSocketFloat", "F2/4 Width", 0.0000),
            ("NodeSocketVector", "Vector", (0.0000, 0.0000, 0.0000)),
        ],
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["F2 Height"],
            1: group_input.outputs["Box Thickness"],
        },
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide})

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["F2/4 Width"]}
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Box Thickness"], 1: 1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_6, 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Box Thickness"]}
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Box Version"]}
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["F1/3 Width"]}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Box Thickness"], 1: 2.5000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_3, 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract_1})

    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 2.0000

    value_3 = nw.new_node(Nodes.Value)
    value_3.outputs[0].default_value = 2.0000

    reroute_1 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["V3 Fraction of Width"]},
    )

    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    flap_length_calculator = nw.new_node(
        nodegroup_node_group_004().name,
        input_kwargs={
            "V": reroute,
            "Distanc Opp Site": reroute_4,
            "Face Number": value_2,
            "Side Number": value_3,
            "FlapFracCover": reroute_2,
        },
        label="Flap Length Calculator",
    )

    flaps = nw.new_node(
        nodegroup_node_group_018().name,
        input_kwargs={
            "Width": subtract,
            "Thickness": reroute_12,
            "Flap Fraction Cover": flap_length_calculator.outputs["FFC"],
            "Distance to Opposite Side": flap_length_calculator.outputs["DTO"],
            "Flap Inward Strength": flap_length_calculator.outputs["FIStre"],
            "Flap Inward Start": flap_length_calculator.outputs["FISta"],
        },
        label="Flaps",
    )

    convex_hull_1 = nw.new_node(Nodes.ConvexHull, input_kwargs={"Geometry": flaps})

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": convex_hull_1})

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["F2 Height"]}
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_25})

    reroute_32 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_31})

    rotators = nw.new_node(
        nodegroup_node_group_009().name,
        input_kwargs={
            "Depth": reroute_22,
            "Height": reroute_24,
            "Thickness": reroute_32,
        },
        label="Rotators",
    )

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract})

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": reroute_12, "Z": reroute_10}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    transform_geometry = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube.outputs["Mesh"]}
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": transform_geometry}
    )

    subtract_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_2.outputs["Vector"]}
    )

    rotators_1 = nw.new_node(
        nodegroup_node_group_012().name,
        input_kwargs={
            "Parent Width": reroute_27,
            "Height": reroute_24,
            "Thickness": reroute_32,
            "Parent Center": separate_xyz.outputs["Y"],
        },
        label="Rotators",
    )

    corners = nw.new_node(
        nodegroup_node_group_013().name,
        input_kwargs={
            "Top": rotators.outputs["Top Rotator Cut"],
            "Right": rotators_1.outputs["Right Rotator Cut"],
            "Left": rotators_1.outputs["Left Rotator Cut"],
            "Bottom": rotators.outputs["Bottom Rotator Cut"],
        },
        label="Corners",
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                rotators.outputs["Top Rotator"],
                rotators.outputs["Bottom Rotator"],
                rotators_1.outputs["Right Rotator"],
                rotators_1.outputs["Left Rotator "],
            ]
        },
    )

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    reroute_30 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_29})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [corners, join_geometry_1, reroute_30]},
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Vector"]}
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Translation": reroute_8},
    )

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_17})

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_26, 3: 5},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Box Version"], 3: 4},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Box Version"], 3: 5},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_or = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: equal_1, 1: equal_2},
        attrs={"operation": "OR"},
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": op_or})

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Box Thickness"],
            1: group_input.outputs["V4 Space Open"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_2})

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: reroute_13},
        attrs={"operation": "SUBTRACT"},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_16, "False": reroute_23, "True": subtract_3},
        attrs={"input_type": "FLOAT"},
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 2.0000

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 1.0000

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    flap_length_calculator_1 = nw.new_node(
        nodegroup_node_group_004().name,
        input_kwargs={
            "V": reroute_18,
            "Distanc Opp Site": reroute_20,
            "Face Number": value,
            "Side Number": value_1,
            "FlapFracCover": reroute_19,
        },
        label="Flap Length Calculator",
    )

    flaps_1 = nw.new_node(
        nodegroup_node_group_017().name,
        input_kwargs={
            "Width": switch,
            "Thickness": reroute_25,
            "Flap Fraction Cover": flap_length_calculator_1.outputs["FFC"],
            "Distance to Opposite Side": flap_length_calculator_1.outputs["DTO"],
            "Flap Inward Strength": flap_length_calculator_1.outputs["FIStre"],
            "Flap Inward Start": flap_length_calculator_1.outputs["FISta"],
        },
        label="Flaps",
    )

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": flaps_1})

    convex_hull = nw.new_node(Nodes.ConvexHull, input_kwargs={"Geometry": flaps_1})

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height Flaps V5"], 1: 1.2000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Box Thickness"], 1: 4.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_3, "Y": multiply_4, "Z": 10.0000}
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_1})

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_4})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": reroute_14})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube_1.outputs["Mesh"], "Translation": combine_xyz_2},
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": transform_geometry_2}
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": convex_hull, "Mesh 2": transform_geometry_3},
        attrs={"solver": "EXACT"},
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal,
            "False": reroute_33,
            "True": difference.outputs["Mesh"],
        },
    )

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_1})

    reroute_34 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_35})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Value": reroute_15,
            "Bottom Lid": reroute_28,
            "Box Side": transform_geometry_1,
            "Top Lid": reroute_34,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_002",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_002(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Name": group_input.outputs["Label"],
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": store_named_attribute},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_004",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_004(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Name": group_input.outputs["Label"],
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": store_named_attribute},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_006", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_006(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "Version", 0),
            ("NodeSocketVector", "Vector", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "Thickness", 0.0000),
        ],
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Vector"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Version"], 3: 6},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 4.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_2, 1: group_input.outputs["Thickness"]},
        attrs={"operation": "SUBTRACT"},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": subtract_2, "True": subtract_3},
        attrs={"input_type": "FLOAT"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 6.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply_2},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply_2},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "F1 Height": subtract_1,
            "F3 Height": switch,
            "F1/3 Width": separate_xyz.outputs["Y"],
            "F2 Height": subtract_4,
            "F4 Height": subtract_5,
            "F2/4 Width": subtract_6,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_021", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_021(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "F1 Height", 0.5000),
            ("NodeSocketFloat", "Box Thickness", 0.5000),
            ("NodeSocketFloat", "F2/4 Width", 0.5000),
            ("NodeSocketInt", "Version", 0),
            ("NodeSocketFloat", "V5 Fraction Length Side Flap", 0.2500),
            ("NodeSocketFloat", "V345 Depth Flaps Fraction Height", 0.5000),
            ("NodeSocketFloat", "F1/3 Width", 0.0000),
            ("NodeSocketFloat", "V45 Space Open For Cap", 0.0000),
        ],
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["F1 Height"],
            1: group_input.outputs["Box Thickness"],
        },
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide})

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Version"]}
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    reroute_42 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    reroute_60 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_42})

    reroute_61 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_60})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_61, 3: 6},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    reroute_97 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": equal})

    reroute_107 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_97})

    reroute_108 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_107})

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_4, 3: 5},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_4, 3: 4},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_or = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: equal_1, 1: equal_2},
        attrs={"operation": "OR"},
    )

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": op_or})

    reroute_36 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_35})

    op_or_1 = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: reroute_108, 1: reroute_36},
        attrs={"operation": "OR"},
    )

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": equal_2})

    reroute_30 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_29})

    reroute_40 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_30})

    reroute_41 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_40})

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": equal_1})

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_27})

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["F1/3 Width"]}
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Box Thickness"], 1: 1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_10, 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract})

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Box Thickness"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_3, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: multiply_2},
        attrs={"operation": "SUBTRACT"},
    )

    switch_7 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_28, "False": reroute_31, "True": subtract_2},
        attrs={"input_type": "FLOAT"},
    )

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract_1})

    reroute_49 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_37})

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_41, "False": switch_7, "True": reroute_49},
        attrs={"input_type": "FLOAT"},
    )

    reroute_77 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_2})

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    reroute_34 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_33})

    reroute_58 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_34})

    reroute_59 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_58})

    reroute_73 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_59})

    reroute_74 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_73})

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["F2/4 Width"]}
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["F2/4 Width"],
            1: group_input.outputs["Box Thickness"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract_3})

    reroute_13 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["V45 Space Open For Cap"]},
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Box Thickness"], 1: 4.0000},
        attrs={"operation": "DIVIDE"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_14, 1: divide_1},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_15, 1: multiply_3},
        attrs={"operation": "SUBTRACT"},
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": op_or, "False": reroute_12, "True": subtract_4},
        attrs={"input_type": "FLOAT"},
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 1.0000

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 1.0000

    value_4 = nw.new_node(Nodes.Value)
    value_4.outputs[0].default_value = 0.0000

    top_flap = nw.new_node(
        nodegroup_node_group_004().name,
        input_kwargs={
            "V": reroute_42,
            "Distanc Opp Site": switch_6,
            "Face Number": value,
            "Side Number": value_1,
            "FlapFracCover": value_4,
        },
        label="Top Flap",
    )

    reroute_65 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": top_flap.outputs["FFC"]}
    )

    reroute_66 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_65})

    subtract_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: top_flap.outputs["DTO"],
            1: group_input.outputs["Box Thickness"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    reroute_67 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": top_flap.outputs["FIStre"]}
    )

    reroute_68 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_67})

    reroute_69 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": top_flap.outputs["FISta"]}
    )

    flaps = nw.new_node(
        nodegroup_node_group_011().name,
        input_kwargs={
            "Width": reroute_77,
            "Thickness": reroute_74,
            "Flap Fraction Cover": reroute_66,
            "Distance to Opposite Side": subtract_5,
            "Flap Inward Strength": reroute_68,
            "Flap Inward Start": reroute_69,
        },
        label="Flaps",
    )

    reroute_103 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": flaps})

    reroute_64 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": top_flap.outputs["DTO"]}
    )

    flap_top_v456 = nw.new_node(
        nodegroup_node_group_002().name,
        input_kwargs={
            "Depth": reroute_77,
            "Height": reroute_64,
            "Thickness": reroute_74,
            "Distance Opposite Site": reroute_64,
        },
        label="FlapTopV456",
    )

    reroute_104 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": flap_top_v456})

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: top_flap.outputs["DTO"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_59, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: divide_2, 1: divide_3})

    combine_xyz_12 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add_1})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_104, "Translation": combine_xyz_12},
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": op_or_1,
            "False": reroute_103,
            "True": transform_geometry_3,
        },
    )

    reroute_123 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_3})

    convex_hull = nw.new_node(Nodes.ConvexHull, input_kwargs={"Geometry": reroute_123})

    reroute_50 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_31})

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_50})

    reroute_79 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_51})

    reroute_80 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_79})

    reroute_90 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_74})

    reroute_91 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_90})

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Version"], 3: 4},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Version"], 3: 5},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_or_2 = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: equal_3, 1: equal_4},
        attrs={"operation": "OR"},
    )

    equal_5 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_4, 3: 6},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_or_3 = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: op_or_2, 1: equal_5},
        attrs={"operation": "OR"},
    )

    integer = nw.new_node(Nodes.Integer)
    integer.integer = 1

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": op_or_3, "False": reroute_21, "True": integer},
        attrs={"input_type": "INT"},
    )

    reroute_46 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch})

    reroute_62 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_46})

    reroute_63 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_62})

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": op_or_2})

    reroute_43 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_6})

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Box Thickness"], 1: 3.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: switch_6, 1: multiply_4})

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_17, "False": reroute_43, "True": add_2},
        attrs={"input_type": "FLOAT"},
    )

    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 1.0000

    value_3 = nw.new_node(Nodes.Value)
    value_3.outputs[0].default_value = 2.0000

    flap_length_calculator = nw.new_node(
        nodegroup_node_group_022().name,
        input_kwargs={
            "V": reroute_63,
            "Distanc Opp Site": switch_1,
            "Face Number": value_2,
            "Side Number": value_3,
            "Thickness": group_input.outputs["Box Thickness"],
        },
        label="Flap Length Calculator",
    )

    reroute_93 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": flap_length_calculator.outputs["FFC"]}
    )

    reroute_94 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_93})

    subtract_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: flap_length_calculator.outputs["DTO"],
            1: group_input.outputs["Box Thickness"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    reroute_95 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": flap_length_calculator.outputs["FIStre"]}
    )

    reroute_96 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": flap_length_calculator.outputs["FISta"]}
    )

    flaps_1 = nw.new_node(
        nodegroup_node_group_011().name,
        input_kwargs={
            "Width": reroute_80,
            "Thickness": reroute_91,
            "Flap Fraction Cover": reroute_94,
            "Distance to Opposite Side": subtract_6,
            "Flap Inward Strength": reroute_95,
            "Flap Inward Start": reroute_96,
        },
        label="Flaps",
    )

    convex_hull_1 = nw.new_node(Nodes.ConvexHull, input_kwargs={"Geometry": flaps_1})

    reroute_121 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": convex_hull_1})

    reroute_119 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": op_or_1})

    reroute_120 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_119})

    bounding_box_1 = nw.new_node(Nodes.BoundingBox, input_kwargs={"Geometry": switch_3})

    subtract_7 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_1.outputs["Max"],
            1: bounding_box_1.outputs["Min"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_7.outputs["Vector"]}
    )

    divide_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Y"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    reroute_112 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_91})

    reroute_113 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_112})

    divide_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_113, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: divide_4, 1: divide_5})

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_5, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_10 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": add_3, "Z": multiply_5}
    )

    reroute_105 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_12})

    reroute_106 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_105})

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_120,
            "False": combine_xyz_10,
            "True": reroute_106,
        },
        attrs={"input_type": "VECTOR"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["F1 Height"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_32 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    reroute_52 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_32})

    reroute_53 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_52})

    reroute_7 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["V345 Depth Flaps Fraction Height"]},
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_24})

    reroute_48 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_25})

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_53, 1: reroute_48},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_85 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_6})

    reroute_5 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["V5 Fraction Length Side Flap"]},
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    reroute_47 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: top_flap.outputs["DTO"], 1: reroute_47},
        attrs={"operation": "MULTIPLY"},
    )

    right_top_v5 = nw.new_node(
        nodegroup_node_group_001().name,
        input_kwargs={
            "Depth": reroute_85,
            "Height": multiply_7,
            "Thickness": reroute_74,
        },
        label="RightTopV5",
    )

    reroute_102 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": right_top_v5})

    multiply_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_85, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_14 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_8})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_102, "Translation": combine_xyz_14},
    )

    reroute_117 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_4}
    )

    reroute_110 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_102})

    reroute_111 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_110})

    multiply_9 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_14, 1: (-1.0000, -1.0000, 1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_111,
            "Translation": multiply_9.outputs["Vector"],
        },
    )

    reroute_124 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_5}
    )

    multiply_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: reroute_6},
        attrs={"operation": "MULTIPLY"},
    )

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["F2/4 Width"],
            1: group_input.outputs["F1 Height"],
        },
        attrs={"operation": "MINIMUM"},
    )

    multiply_11 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum, 1: reroute_8},
        attrs={"operation": "MULTIPLY"},
    )

    right_top_v5_1 = nw.new_node(
        nodegroup_node_group_001().name,
        input_kwargs={
            "Depth": multiply_10,
            "Height": multiply_11,
            "Thickness": reroute_19,
        },
        label="RightTopV5",
    )

    reroute_38 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": right_top_v5_1})

    reroute_55 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_38})

    divide_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_11, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    divide_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_19, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: divide_6, 1: divide_7})

    combine_xyz_16 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": add_4})

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_55,
            "Translation": combine_xyz_16,
            "Rotation": (1.5708, 1.5708, 0.0000),
        },
    )

    reroute_71 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_6}
    )

    reroute_72 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_71})

    reroute_100 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_72})

    reroute_101 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_100})

    reroute_75 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_2})

    reroute_76 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_75})

    reroute_78 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_2})

    multiply_12 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_59, 1: 8.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_78, 1: multiply_12},
        attrs={"operation": "SUBTRACT"},
    )

    switch_12 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_97, "False": reroute_76, "True": subtract_8},
        attrs={"input_type": "FLOAT"},
    )

    divide_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: switch_12, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    reroute_114 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide_8})

    combine_xyz_15 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": reroute_114})

    reroute_128 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_15})

    right_top_v5_2 = nw.new_node(
        nodegroup_node_group_001().name,
        input_kwargs={
            "Depth": multiply_10,
            "Height": multiply_11,
            "Thickness": reroute_19,
        },
        label="RightTopV5",
    )

    reroute_39 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": right_top_v5_2})

    reroute_56 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_39})

    reroute_57 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_56})

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_57,
            "Translation": combine_xyz_16,
            "Rotation": (1.5708, 1.5708, 0.0000),
        },
    )

    reroute_86 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_7}
    )

    reroute_126 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_86})

    reroute_127 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_126})

    multiply_13 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_15, 1: (-1.0000, 1.0000, 1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_115 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_108})

    reroute_116 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_115})

    reroute_92 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_76})

    reroute_44 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_28})

    reroute_45 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_44})

    reroute_88 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_45})

    reroute_89 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_88})

    reroute_131 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_120})

    reroute_132 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_131})

    reroute_83 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_6})

    reroute_84 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_83})

    reroute_70 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_53})

    switch_10 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_97, "False": reroute_84, "True": reroute_70},
        attrs={"input_type": "FLOAT"},
    )

    reroute_98 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_84})

    top_lid_flap = nw.new_node(
        nodegroup_node_group_003().name,
        input_kwargs={
            "Depth": switch_12,
            "Height": switch_10,
            "Thickness": reroute_91,
            "Distance Opposite Site": reroute_98,
        },
        label="TopLidFlap",
    )

    reroute_118 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": top_lid_flap})

    multiply_14 = nw.new_node(
        Nodes.Math, input_kwargs={0: switch_10}, attrs={"operation": "MULTIPLY"}
    )

    reroute_87 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide_3})

    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_14, 1: reroute_87})

    combine_xyz_11 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": add_5})

    reroute_122 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_11})

    reroute_129 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_122})

    reroute_130 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_129})

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_81 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_26})

    reroute_82 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_81})

    reroute_99 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_70})

    rotators = nw.new_node(
        nodegroup_node_group_009().name,
        input_kwargs={
            "Depth": reroute_82,
            "Height": reroute_99,
            "Thickness": reroute_91,
        },
        label="Rotators",
    )

    reroute_109 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_80})

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": reroute_19, "Z": reroute_18}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_7})

    transform_geometry = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube.outputs["Mesh"]}
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": transform_geometry}
    )

    subtract_9 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_9.outputs["Vector"]}
    )

    rotators_1 = nw.new_node(
        nodegroup_node_group_012().name,
        input_kwargs={
            "Parent Width": reroute_109,
            "Height": reroute_99,
            "Thickness": reroute_91,
            "Parent Center": separate_xyz.outputs["Y"],
        },
        label="Rotators",
    )

    corners = nw.new_node(
        nodegroup_node_group_013().name,
        input_kwargs={
            "Top": rotators.outputs["Top Rotator Cut"],
            "Right": rotators_1.outputs["Right Rotator Cut"],
            "Left": rotators_1.outputs["Left Rotator Cut"],
            "Bottom": rotators.outputs["Bottom Rotator Cut"],
        },
        label="Corners",
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                rotators.outputs["Top Rotator"],
                rotators.outputs["Bottom Rotator"],
                rotators_1.outputs["Right Rotator"],
                rotators_1.outputs["Left Rotator "],
            ]
        },
    )

    reroute_54 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [corners, join_geometry_1, reroute_54]},
    )

    reroute_125 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "V1": reroute_16,
            "CH1": convex_hull,
            "CH2": reroute_121,
            "Vec1": switch_4,
            "Geom1": reroute_117,
            "Geom2": reroute_124,
            "Geom3": reroute_101,
            "Vec2": reroute_128,
            "Geom4": reroute_127,
            "Vec3": multiply_13.outputs["Vector"],
            "Bool1": reroute_116,
            "V2": reroute_92,
            "Bool2": reroute_89,
            "Bool3": reroute_132,
            "F1": reroute_118,
            "Vec4": reroute_130,
            "Geom5": reroute_125,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_003",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_003(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Name": group_input.outputs["Label"],
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": store_named_attribute},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_001",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_001(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Name": group_input.outputs["Label"],
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": store_named_attribute},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketVector", "Dimensions", (1.0000, 1.0000, 2.0000)),
            ("NodeSocketFloat", "Box Thickness", 0.0000),
            ("NodeSocketInt", "Version", 0),
            ("NodeSocketFloat", "V3 Fraction of Width Covered", 0.0000),
            ("NodeSocketFloat", "V45 Space Open For Cap", 0.0000),
            ("NodeSocketBool", "Bottom Closed", False),
            ("NodeSocketFloat", "V5 Fraction Length Side Flap", 0.0000),
            ("NodeSocketFloat", "V345 Depth Flaps Fraction Height", 0.0000),
            ("NodeSocketMaterial", "BoxMaterial", None),
        ],
    )

    calculate_face_dimensions = nw.new_node(
        nodegroup_node_group_006().name,
        input_kwargs={
            "Version": group_input.outputs["Version"],
            "Vector": group_input.outputs["Dimensions"],
            "Thickness": group_input.outputs["Box Thickness"],
        },
        label="Calculate Face Dimensions",
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Box Thickness"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Version"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_6 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["V5 Fraction Length Side Flap"]},
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    reroute_8 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["V345 Depth Flaps Fraction Height"]},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["V45 Space Open For Cap"]},
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    face_1_components = nw.new_node(
        nodegroup_node_group_021().name,
        input_kwargs={
            "F1 Height": calculate_face_dimensions.outputs["F1 Height"],
            "Box Thickness": reroute_1,
            "F2/4 Width": calculate_face_dimensions.outputs["F2/4 Width"],
            "Version": reroute_3,
            "V5 Fraction Length Side Flap": reroute_7,
            "V345 Depth Flaps Fraction Height": reroute_8,
            "F1/3 Width": calculate_face_dimensions.outputs["F1/3 Width"],
            "V45 Space Open For Cap": reroute_5,
        },
        label="Face 1 Components",
    )

    reroute_52 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_1_components.outputs["Geom5"]}
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_52, "Rotation": (0.0000, 0.0000, 3.1416)},
    )

    add_jointed_geometry_metadata_003 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_003().name,
        input_kwargs={
            "Geometry": transform_geometry_3,
            "Label": "F1_complete_body_no_lids",
        },
    )

    reroute_46 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_1_components.outputs["Bool2"]}
    )

    reroute_47 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_46})

    reroute_48 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_1_components.outputs["Bool3"]}
    )

    reroute_49 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_48})

    reroute_53 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_1_components.outputs["CH1"]}
    )

    reroute_54 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_53})

    reroute_85 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_54})

    add_jointed_geometry_metadata_003_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_003().name,
        input_kwargs={"Geometry": reroute_54, "Label": "F1_top_lid_no_top_flap"},
    )

    reroute_43 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_1_components.outputs["Bool1"]}
    )

    reroute_44 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_43})

    reroute_31 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_1_components.outputs["F1"]}
    )

    reroute_32 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_31})

    reroute_66 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_32})

    reroute_67 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_66})

    add_jointed_geometry_metadata_003_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_003().name,
        input_kwargs={"Geometry": reroute_32, "Label": "F1_top_flap_top_right_flap"},
    )

    transform_geometry_19 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": face_1_components.outputs["Geom3"],
            "Rotation": (3.1416, 0.0000, 0.0000),
        },
    )

    add_jointed_geometry_metadata_003_3 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_003().name,
        input_kwargs={
            "Geometry": transform_geometry_19,
            "Label": "F1_top_flap_central_flap",
        },
    )

    reroute_39 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_1_components.outputs["Vec2"]}
    )

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "F1_top_flap_top_right_flap_axis",
            "Parent": add_jointed_geometry_metadata_003_2,
            "Child": add_jointed_geometry_metadata_003_3,
            "Position": reroute_39,
            "Max": 3.1416,
        },
    )

    add_jointed_geometry_metadata_003_4 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_003().name,
        input_kwargs={
            "Geometry": hinge_joint.outputs["Geometry"],
            "Label": "F1_top_flap_central_right_flap",
        },
    )

    reroute_40 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_1_components.outputs["Geom4"]}
    )

    transform_geometry_18 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_40, "Rotation": (3.1416, 0.0000, 0.0000)},
    )

    add_jointed_geometry_metadata_003_5 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_003().name,
        input_kwargs={
            "Geometry": transform_geometry_18,
            "Label": "F1_top_flap_top_left_flap",
        },
    )

    reroute_41 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_1_components.outputs["Vec3"]}
    )

    reroute_42 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_41})

    hinge_joint_1 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "F1_top_flap_top_left_flap_axis",
            "Parent": add_jointed_geometry_metadata_003_4,
            "Child": add_jointed_geometry_metadata_003_5,
            "Position": reroute_42,
            "Min": -3.1416,
        },
    )

    switch_11 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_44,
            "False": reroute_67,
            "True": hinge_joint_1.outputs["Geometry"],
        },
    )

    reroute_50 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_1_components.outputs["Vec4"]}
    )

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_50})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": switch_11,
            "Translation": reroute_51,
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    add_jointed_geometry_metadata_003_6 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_003().name,
        input_kwargs={
            "Geometry": transform_geometry_1,
            "Label": "F1_complete_top_flap",
        },
    )

    reroute_35 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_1_components.outputs["Vec1"]}
    )

    reroute_36 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_35})

    hinge_joint_2 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "F1_top_lid_top_flap_axis",
            "Parent": add_jointed_geometry_metadata_003_1,
            "Child": add_jointed_geometry_metadata_003_6,
            "Position": reroute_36,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Max": 3.1416,
        },
    )

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": hinge_joint_2.outputs["Geometry"],
            "Rotation": (-1.5708, 0.0000, 0.0000),
        },
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_49,
            "False": reroute_85,
            "True": transform_geometry_6,
        },
    )

    reroute_91 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_1})

    add_jointed_geometry_metadata_003_7 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_003().name,
        input_kwargs={"Geometry": switch_1, "Label": "F1_top_lid_no_side_flaps"},
    )

    reroute_37 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_1_components.outputs["Geom1"]}
    )

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_37, "Rotation": (-1.5708, 0.0000, 0.0000)},
    )

    transform_geometry_17 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_7,
            "Rotation": (0.0000, -1.5708, 0.0000),
        },
    )

    add_jointed_geometry_metadata_003_8 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_003().name,
        input_kwargs={"Geometry": transform_geometry_17, "Label": "F1_top_left_flap"},
    )

    reroute_45 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_1_components.outputs["V2"]}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_45, 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_13 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide})

    reroute_88 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_13})

    hinge_joint_3 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "F1_top_left_lid_axis",
            "Parent": add_jointed_geometry_metadata_003_7,
            "Child": add_jointed_geometry_metadata_003_8,
            "Position": reroute_88,
            "Axis": (0.0000, 1.0000, 0.0000),
            "Max": 3.1416,
        },
    )

    add_jointed_geometry_metadata_003_9 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_003().name,
        input_kwargs={
            "Geometry": hinge_joint_3.outputs["Geometry"],
            "Label": "F1_top_lid_left_flap",
        },
    )

    reroute_38 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_1_components.outputs["Geom2"]}
    )

    transform_geometry_8 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_38, "Rotation": (-1.5708, 0.0000, 0.0000)},
    )

    transform_geometry_11 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_8,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    add_jointed_geometry_metadata_003_10 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_003().name,
        input_kwargs={"Geometry": transform_geometry_11, "Label": "F1_top_right_lid"},
    )

    reroute_90 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_88})

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_90, 1: (-1.0000, 1.0000, 1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_93 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply.outputs["Vector"]}
    )

    hinge_joint_4 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "F1_top_right_lid_axis",
            "Parent": add_jointed_geometry_metadata_003_9,
            "Child": add_jointed_geometry_metadata_003_10,
            "Position": reroute_93,
            "Axis": (0.0000, 1.0000, 0.0000),
            "Min": -3.1416,
        },
    )

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_47,
            "False": reroute_91,
            "True": hinge_joint_4.outputs["Geometry"],
        },
    )

    add_jointed_geometry_metadata_003_11 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_003().name,
        input_kwargs={"Geometry": switch_5, "Label": "F1_complete_top_lid"},
    )

    reroute_33 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_1_components.outputs["V1"]}
    )

    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_33})

    hinge_joint_5 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "F1_top_lid_axis",
            "Parent": add_jointed_geometry_metadata_003,
            "Child": add_jointed_geometry_metadata_003_11,
            "Position": combine_xyz_9,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Max": 3.1416,
        },
    )

    add_jointed_geometry_metadata_003_12 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_003().name,
        input_kwargs={
            "Geometry": hinge_joint_5.outputs["Geometry"],
            "Label": "F1_complete_body_with_top_lid",
        },
    )

    reroute_34 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_1_components.outputs["CH2"]}
    )

    add_jointed_geometry_metadata_003_13 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_003().name,
        input_kwargs={"Geometry": reroute_34, "Label": "F1_bottom_lid"},
    )

    reroute_94 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_9})

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_94, 1: (-1.0000, -1.0000, -1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    hinge_joint_6 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "F1_bottom_lid_axis",
            "Parent": add_jointed_geometry_metadata_003_12,
            "Child": add_jointed_geometry_metadata_003_13,
            "Position": multiply_1.outputs["Vector"],
            "Axis": (1.0000, 0.0000, 0.0000),
            "Min": -3.1416,
        },
    )

    flip_faces_1 = nw.new_node(
        Nodes.FlipFaces, input_kwargs={"Mesh": hinge_joint_6.outputs["Geometry"]}
    )

    add_jointed_geometry_metadata_001 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_001().name,
        input_kwargs={"Geometry": flip_faces_1, "Label": "comlete_F1"},
    )

    reroute_19 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": calculate_face_dimensions.outputs["F2 Height"]},
    )

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_62 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_26})

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    reroute_63 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_27})

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    reroute_64 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_28})

    reroute_65 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_64})

    bounding_box_4 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": face_1_components.outputs["Geom1"]}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_4.outputs["Max"],
            1: bounding_box_4.outputs["Min"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_4 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    reroute_11 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["V3 Fraction of Width Covered"]},
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    reroute_61 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    reroute_22 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": calculate_face_dimensions.outputs["F1/3 Width"]},
    )

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    reroute_57 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    reroute_24 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": calculate_face_dimensions.outputs["F2/4 Width"]},
    )

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_24})

    reroute_58 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_25})

    reroute_59 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_58})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": reroute_25})

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_2, 1: (-0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    face_2_components = nw.new_node(
        nodegroup_node_group().name,
        input_kwargs={
            "F2 Height": reroute_19,
            "Box Thickness": reroute_62,
            "Box Version": reroute_63,
            "V4 Space Open": reroute_65,
            "Height Flaps V5": separate_xyz_4.outputs["Z"],
            "V3 Fraction of Width": reroute_61,
            "F1/3 Width": reroute_57,
            "F2/4 Width": reroute_59,
            "Vector": multiply_2.outputs["Vector"],
        },
        label="Face 2 Components",
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": face_2_components.outputs["Box Side"],
            "Scale": (-1.0000, 1.0000, 1.0000),
        },
    )

    add_jointed_geometry_metadata_002 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_002().name,
        input_kwargs={
            "Geometry": transform_geometry_4,
            "Label": "F2_complete_body_no_lids",
        },
    )

    flip_faces_4 = nw.new_node(
        Nodes.FlipFaces, input_kwargs={"Mesh": face_2_components.outputs["Top Lid"]}
    )

    add_jointed_geometry_metadata_003_14 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_003().name,
        input_kwargs={"Geometry": flip_faces_4, "Label": "F2_top_lid"},
    )

    reroute_78 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_2_components.outputs["Value"]}
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_78})

    hinge_joint_7 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "F2_top_lid_axis",
            "Parent": add_jointed_geometry_metadata_002,
            "Child": add_jointed_geometry_metadata_003_14,
            "Position": combine_xyz_6,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Max": 3.1416,
        },
    )

    add_jointed_geometry_metadata_004 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_004().name,
        input_kwargs={
            "Geometry": hinge_joint_7.outputs["Geometry"],
            "Label": "F2_complete_body_with_top_lid",
        },
    )

    reroute_79 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_2_components.outputs["Bottom Lid"]}
    )

    flip_faces_3 = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": reroute_79})

    add_jointed_geometry_metadata_001_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_001().name,
        input_kwargs={"Geometry": flip_faces_3, "Label": "F2_bottom_lid"},
    )

    reroute_84 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_6})

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_84, 1: (-1.0000, -1.0000, -1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    hinge_joint_8 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "F2_bottom_lid_axis",
            "Parent": add_jointed_geometry_metadata_004,
            "Child": add_jointed_geometry_metadata_001_1,
            "Position": multiply_3.outputs["Vector"],
            "Axis": (1.0000, 0.0000, 0.0000),
            "Min": -3.1416,
        },
    )

    transform_geometry_10 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": hinge_joint_8.outputs["Geometry"],
            "Rotation": (0.0000, 0.0000, 1.5708),
        },
    )

    add_jointed_geometry_metadata_001_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_001().name,
        input_kwargs={"Geometry": transform_geometry_10, "Label": "complete_F2"},
    )

    reroute_92 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_001_2}
    )

    reroute_75 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_57})

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": reroute_75})

    divide_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: (2.0000, 2.0000, 2.0000)},
        attrs={"operation": "DIVIDE"},
    )

    hinge_joint_9 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "F12_folding_edge",
            "Parent": add_jointed_geometry_metadata_001,
            "Child": reroute_92,
            "Position": divide_1.outputs["Vector"],
            "Axis": (0.0000, 0.0000, -1.0000),
            "Value": -0.5000,
            "Max": 3.1416,
        },
    )

    add_jointed_geometry_metadata_001_3 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_001().name,
        input_kwargs={
            "Geometry": hinge_joint_9.outputs["Geometry"],
            "Label": "complete_F12",
        },
    )

    reroute_20 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": calculate_face_dimensions.outputs["F4 Height"]},
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    reroute_55 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    reroute_56 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_55})

    nodegroup_020 = nw.new_node(
        nodegroup_node_group_020().name,
        input_kwargs={
            "F4 Height": reroute_56,
            "Box Thickness": reroute_62,
            "Version": reroute_63,
            "V4 Space Open for cap": reroute_65,
            "Height Flap V5": separate_xyz_4.outputs["Z"],
            "V3 Fraction of Width Coverede": reroute_61,
            "F1/3 Width": reroute_57,
            "F2/4 Width": reroute_59,
        },
    )

    reroute_80 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": nodegroup_020.outputs["Geometry"]}
    )

    add_jointed_geometry_metadata_002_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_002().name,
        input_kwargs={"Geometry": reroute_80, "Label": "F4"},
    )

    flip_faces_7 = nw.new_node(
        Nodes.FlipFaces, input_kwargs={"Mesh": nodegroup_020.outputs["Top Flap"]}
    )

    transform_geometry_9 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": flip_faces_7, "Scale": (1.0000, -1.0000, 1.0000)},
    )

    add_jointed_geometry_metadata_003_15 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_003().name,
        input_kwargs={"Geometry": transform_geometry_9, "Label": "F4_lid1"},
    )

    reroute_81 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": nodegroup_020.outputs["Value"]}
    )

    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_81})

    hinge_joint_10 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "F4_top_lid",
            "Parent": add_jointed_geometry_metadata_002_1,
            "Child": add_jointed_geometry_metadata_003_15,
            "Position": combine_xyz_8,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Min": -3.1416,
        },
    )

    add_jointed_geometry_metadata_004_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_004().name,
        input_kwargs={
            "Geometry": hinge_joint_10.outputs["Geometry"],
            "Label": "F4_top_lid_attached",
        },
    )

    reroute_82 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": nodegroup_020.outputs["Bottom Flap"]}
    )

    flip_faces_6 = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": reroute_82})

    transform_geometry_12 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": flip_faces_6, "Scale": (1.0000, -1.0000, 1.0000)},
    )

    add_jointed_geometry_metadata_001_4 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_001().name,
        input_kwargs={"Geometry": transform_geometry_12, "Label": "F4_lid2"},
    )

    reroute_86 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_8})

    multiply_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_86, 1: (-1.0000, -1.0000, -1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    hinge_joint_11 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "F4_both_lids",
            "Parent": add_jointed_geometry_metadata_004_1,
            "Child": add_jointed_geometry_metadata_001_4,
            "Position": multiply_4.outputs["Vector"],
            "Axis": (1.0000, 0.0000, 0.0000),
            "Max": 3.1416,
        },
    )

    flip_faces_2 = nw.new_node(
        Nodes.FlipFaces, input_kwargs={"Mesh": hinge_joint_11.outputs["Geometry"]}
    )

    add_jointed_geometry_metadata_001_5 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_001().name,
        input_kwargs={"Geometry": flip_faces_2, "Label": "complete_F4"},
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": calculate_face_dimensions.outputs["F3 Height"]},
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_17})

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    reroute_30 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_29})

    bounding_box_2 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": face_1_components.outputs["F1"]}
    )

    reroute_60 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": bounding_box_2.outputs["Max"]}
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": calculate_face_dimensions.outputs["F1/3 Width"]},
    )

    multiply_5 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_3, 1: (-0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    face_3_components = nw.new_node(
        nodegroup_node_group_019().name,
        input_kwargs={
            "F1/3 Width": reroute_23,
            "Box Thickness": reroute_26,
            "F3 Height": reroute_18,
            "Box Version": reroute_27,
            "V5 Faction Length": reroute_30,
            "BoundingBoxTopFlapV6": reroute_60,
            "V3 Fraction of Width": reroute_12,
            "F2/4 Width": reroute_25,
            "F1/3 Half Negative": multiply_5.outputs["Vector"],
        },
        label="Face 3 Components",
    )

    reroute_73 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_3_components.outputs["F3 Exists"]}
    )

    reroute_74 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_73})

    transform_geometry_14 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": face_3_components.outputs["Side"],
            "Scale": (-1.0000, 1.0000, 1.0000),
        },
    )

    add_jointed_geometry_metadata_002_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_002().name,
        input_kwargs={
            "Geometry": transform_geometry_14,
            "Label": "F3_complete_body_no_lids",
        },
    )

    transform_geometry_15 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": face_3_components.outputs["Top Flap"],
            "Scale": (1.0000, -1.0000, 1.0000),
        },
    )

    add_jointed_geometry_metadata_003_16 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_003().name,
        input_kwargs={"Geometry": transform_geometry_15, "Label": "F1_top_lid"},
    )

    reroute_68 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": face_3_components.outputs["Pos Top Lid Axis"]},
    )

    reroute_69 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_68})

    hinge_joint_12 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "F3_top_lid_axis",
            "Parent": add_jointed_geometry_metadata_002_2,
            "Child": add_jointed_geometry_metadata_003_16,
            "Position": reroute_69,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Value": 0.7000,
            "Min": -3.1416,
        },
    )

    add_jointed_geometry_metadata_004_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_004().name,
        input_kwargs={
            "Geometry": hinge_joint_12.outputs["Geometry"],
            "Label": "F3_complete_body_with_top_lid",
        },
    )

    reroute_71 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_3_components.outputs["Bottom Flap"]}
    )

    transform_geometry_16 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_71, "Scale": (1.0000, -1.0000, 1.0000)},
    )

    add_jointed_geometry_metadata_001_6 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_001().name,
        input_kwargs={"Geometry": transform_geometry_16, "Label": "F3_bottom_lid"},
    )

    reroute_70 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": face_3_components.outputs["Pos Bottom Lid Axis"]},
    )

    multiply_6 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_70, 1: (-1.0000, -1.0000, -1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_83 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply_6.outputs["Vector"]}
    )

    hinge_joint_13 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "F3_bottom_lid_axis",
            "Parent": add_jointed_geometry_metadata_004_2,
            "Child": add_jointed_geometry_metadata_001_6,
            "Position": reroute_83,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Max": 3.1416,
        },
    )

    reroute_72 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": face_3_components.outputs["Side"]}
    )

    transform_geometry_21 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_72, "Scale": (-1.0000, 1.0000, 1.0000)},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_74,
            "False": hinge_joint_13.outputs["Geometry"],
            "True": transform_geometry_21,
        },
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": switch, "Rotation": (0.0000, 0.0000, 4.7141)},
    )

    add_jointed_geometry_metadata_001_7 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_001().name,
        input_kwargs={"Geometry": transform_geometry_5, "Label": "comlete_F3"},
    )

    reroute_89 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_001_7}
    )

    reroute_76 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_59})

    reroute_77 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_76})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": reroute_77})

    divide_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_1, 1: (2.0000, 2.0000, 2.0000)},
        attrs={"operation": "DIVIDE"},
    )

    hinge_joint_14 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "F34_folding_edge",
            "Parent": add_jointed_geometry_metadata_001_5,
            "Child": reroute_89,
            "Position": divide_2.outputs["Vector"],
            "Axis": (0.0000, 0.0000, -1.0000),
            "Min": -3.1416,
        },
    )

    reroute_87 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_77})

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_87, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide_3})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": hinge_joint_14.outputs["Geometry"],
            "Translation": combine_xyz_5,
        },
    )

    transform_geometry_20 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_2,
            "Rotation": (0.0000, 0.0000, 1.5708),
        },
    )

    add_jointed_geometry_metadata_001_8 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_001().name,
        input_kwargs={"Geometry": transform_geometry_20, "Label": "complete_F34"},
    )

    reroute_95 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_001_8}
    )

    reroute_96 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_75})

    divide_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_96, 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide_4})

    hinge_joint_15 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "F1234_folding_edge",
            "Parent": add_jointed_geometry_metadata_001_3,
            "Child": reroute_95,
            "Position": combine_xyz_4,
            "Axis": (0.0000, 0.0000, -1.0000),
            "Min": -3.1416,
        },
    )

    reroute_16 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": calculate_face_dimensions.outputs["F1 Height"]},
    )

    divide_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_16, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Box Thickness"]}
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: divide_5, 1: reroute_13})

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": hinge_joint_15.outputs["Geometry"],
            "Translation": combine_xyz_7,
        },
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BoxMaterial"]}
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry, "Material": reroute_10},
    )

    flip_faces = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": set_material})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": flip_faces},
        attrs={"is_active_output": True},
    )


class BoxFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    @gin.configurable(module="BoxFactory")
    def sample_joint_parameters(
        cls,
    ):
        return {
            "F1_bottom_lid_axis": {
                "stiffness": uniform(
                    F1_bottom_lid_axis_stiffness_min, F1_bottom_lid_axis_stiffness_max
                ),
                "damping": uniform(
                    F1_bottom_lid_axis_damping_min, F1_bottom_lid_axis_damping_max
                ),
            },
            "F1_top_flap_top_left_flap_axis": {
                "stiffness": uniform(
                    F1_top_flap_top_left_flap_axis_stiffness_min,
                    F1_top_flap_top_left_flap_axis_stiffness_max,
                ),
                "damping": uniform(
                    F1_top_flap_top_left_flap_axis_damping_min,
                    F1_top_flap_top_left_flap_axis_damping_max,
                ),
            },
            "F4_both_lids": {
                "stiffness": uniform(
                    F4_both_lids_stiffness_min, F4_both_lids_stiffness_max
                ),
                "damping": uniform(F4_both_lids_damping_min, F4_both_lids_damping_max),
            },
            "F1_top_right_lid_axis": {
                "stiffness": uniform(
                    F1_top_right_lid_axis_stiffness_min,
                    F1_top_right_lid_axis_stiffness_max,
                ),
                "damping": uniform(
                    F1_top_right_lid_axis_damping_min, F1_top_right_lid_axis_damping_max
                ),
            },
            "F12_folding_edge": {
                "stiffness": uniform(
                    F12_folding_edge_stiffness_min, F12_folding_edge_stiffness_max
                ),
                "damping": uniform(
                    F12_folding_edge_damping_min, F12_folding_edge_damping_max
                ),
            },
            "F3_top_lid_axis": {
                "stiffness": uniform(
                    F3_top_lid_axis_stiffness_min, F3_top_lid_axis_stiffness_max
                ),
                "damping": uniform(
                    F3_top_lid_axis_damping_min, F3_top_lid_axis_damping_max
                ),
            },
            "F1234_folding_edge": {
                "stiffness": uniform(
                    F1234_folding_edge_stiffness_min, F1234_folding_edge_stiffness_max
                ),
                "damping": uniform(
                    F1234_folding_edge_damping_min, F1234_folding_edge_damping_max
                ),
            },
            "F1_top_lid_top_flap_axis": {
                "stiffness": uniform(
                    F1_top_lid_top_flap_axis_stiffness_min,
                    F1_top_lid_top_flap_axis_stiffness_max,
                ),
                "damping": uniform(
                    F1_top_lid_top_flap_axis_damping_min,
                    F1_top_lid_top_flap_axis_damping_max,
                ),
            },
            "F1_top_left_lid_axis": {
                "stiffness": uniform(
                    F1_top_left_lid_axis_stiffness_min,
                    F1_top_left_lid_axis_stiffness_max,
                ),
                "damping": uniform(
                    F1_top_left_lid_axis_damping_min, F1_top_left_lid_axis_damping_max
                ),
            },
            "F2_bottom_lid_axis": {
                "stiffness": uniform(
                    F2_bottom_lid_axis_stiffness_min, F2_bottom_lid_axis_stiffness_max
                ),
                "damping": uniform(
                    F2_bottom_lid_axis_damping_min, F2_bottom_lid_axis_damping_max
                ),
            },
            "F1_top_lid_axis": {
                "stiffness": uniform(
                    F1_top_lid_axis_stiffness_min, F1_top_lid_axis_stiffness_max
                ),
                "damping": uniform(
                    F1_top_lid_axis_damping_min, F1_top_lid_axis_damping_max
                ),
            },
            "F4_top_lid": {
                "stiffness": uniform(
                    F4_top_lid_stiffness_min, F4_top_lid_stiffness_max
                ),
                "damping": uniform(F4_top_lid_damping_min, F4_top_lid_damping_max),
            },
            "F3_bottom_lid_axis": {
                "stiffness": uniform(
                    F3_bottom_lid_axis_stiffness_min, F3_bottom_lid_axis_stiffness_max
                ),
                "damping": uniform(
                    F3_bottom_lid_axis_damping_min, F3_bottom_lid_axis_damping_max
                ),
            },
            "F1_top_flap_top_right_flap_axis": {
                "stiffness": uniform(
                    F1_top_flap_top_right_flap_axis_stiffness_min,
                    F1_top_flap_top_right_flap_axis_stiffness_max,
                ),
                "damping": uniform(
                    F1_top_flap_top_right_flap_axis_damping_min,
                    F1_top_flap_top_right_flap_axis_damping_max,
                ),
            },
            "F34_folding_edge": {
                "stiffness": uniform(
                    F34_folding_edge_stiffness_min, F34_folding_edge_stiffness_max
                ),
                "damping": uniform(
                    F34_folding_edge_damping_min, F34_folding_edge_damping_max
                ),
            },
            "F2_top_lid_axis": {
                "stiffness": uniform(
                    F2_top_lid_axis_stiffness_min, F2_top_lid_axis_stiffness_max
                ),
                "damping": uniform(
                    F2_top_lid_axis_damping_min, F2_top_lid_axis_damping_max
                ),
            },
        }

    def sample_parameters(self):
        # add code here to randomly sample from parameters
        version = randint(0, 7)

        w = uniform(0.2, 1)
        d = uniform(0.2, 1)
        h = uniform(0.2, 1)

        thickness = uniform(0.01, 0.04) * min([w, d, h])
        open_space_slid = uniform(0.1, 0.13) * w / thickness
        return {
            "Dimensions": (w, d, h),
            "Box Thickness": thickness,
            "Version": version,
            "V3 Fraction of Width Covered": uniform(0.2, 0.45),
            "V45 Space Open For Cap": open_space_slid,
            "V5 Fraction Length Side Flap": uniform(0.2, 0.6),
            "V345 Depth Flaps Fraction Height": uniform(0.2, 0.6),
            "BoxMaterial": weighted_sample(material_assignments.woods)()(),
        }

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
