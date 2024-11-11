# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick, Stamatis Alexandropolous, Yiming Zuo


import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.core import surface, tagging
from infinigen.core import tags as t
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import clip_gaussian


@node_utils.to_nodegroup(
    "nodegroup_array_fill_line", singleton=False, type="GeometryNodeTree"
)
def nodegroup_array_fill_line(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Line Start", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "Line End", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "Instance Dimensions", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketInt", "Count", 10),
            ("NodeSocketGeometry", "Instance", None),
        ],
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Instance Dimensions"],
            1: (0.0000, -0.5000, 0.0000),
        },
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Line End"],
            1: multiply.outputs["Vector"],
        },
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Line Start"],
            1: multiply.outputs["Vector"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    mesh_line = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={
            "Count": group_input.outputs["Count"],
            "Start Location": add.outputs["Vector"],
            "Offset": subtract.outputs["Vector"],
        },
        attrs={"mode": "END_POINTS"},
    )

    instance_on_points_1 = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={"Points": mesh_line, "Instance": group_input.outputs["Instance"]},
    )

    realize_instances_1 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points_1}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": realize_instances_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_corner_cube", singleton=False, type="GeometryNodeTree"
)
def nodegroup_corner_cube(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Location", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "CenteringLoc", (0.5000, 0.5000, 0.0000)),
            ("NodeSocketVector", "Dimensions", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "SupportingEdgeFac", 0.0000),
            ("NodeSocketInt", "Vertices X", 4),
            ("NodeSocketInt", "Vertices Y", 4),
            ("NodeSocketInt", "Vertices Z", 4),
        ],
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": group_input.outputs["Dimensions"],
            "Vertices X": group_input.outputs["Vertices X"],
            "Vertices Y": group_input.outputs["Vertices Y"],
            "Vertices Z": group_input.outputs["Vertices Z"],
        },
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Vector": group_input.outputs["CenteringLoc"],
            9: (0.5000, 0.5000, 0.5000),
            10: (-0.5000, -0.5000, -0.5000),
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: map_range.outputs["Vector"],
            1: group_input.outputs["Dimensions"],
            2: group_input.outputs["Location"],
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Translation": multiply_add.outputs["Vector"],
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": transform_geometry,
            "Name": "UVMap",
            3: cube.outputs["UV Map"],
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": store_named_attribute},
        attrs={"is_active_output": True},
    )


ARM_TYPE_SQUARE = 0
ARM_TYPE_ROUND = 1
ARM_TYPE_ANGULAR = 2


@node_utils.to_nodegroup(
    "nodegroup_sofa_geometry", singleton=False, type="GeometryNodeTree"
)
def nodegroup_sofa_geometry(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketVector", "Dimensions", (0.0000, 0.9000, 2.5000)),
            ("NodeSocketVector", "Arm Dimensions", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "Back Dimensions", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "Seat Dimensions", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "Foot Dimensions", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "Baseboard Height", 0.1300),
            ("NodeSocketFloat", "Backrest Width", 0.1100),
            ("NodeSocketFloat", "Seat Margin", 0.9700),
            ("NodeSocketFloat", "Backrest Angle", -0.2000),
            ("NodeSocketFloat", "arm_width", 0.7000),
            ("NodeSocketInt", "Arm Type", 0),
            ("NodeSocketFloat", "Arm_height", 0.7318),
            ("NodeSocketFloat", "arms_angle", 0.8727),
            ("NodeSocketBool", "Footrest", False),
            ("NodeSocketInt", "Count", 4),
            ("NodeSocketFloat", "Scaling footrest", 1.5000),
            ("NodeSocketInt", "Reflection", 0),
            ("NodeSocketBool", "leg_type", False),
            ("NodeSocketFloat", "leg_dimensions", 0.5000),
            ("NodeSocketFloat", "leg_z", 1.0000),
            ("NodeSocketInt", "leg_faces", 20),
            ("NodeSocketBool", "Subdivide", True),
        ],
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Dimensions"],
            1: (0.0000, 0.5000, 0.0000),
        },
        attrs={"operation": "MULTIPLY"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Arm Dimensions"]}
    )

    arm_cube = nw.new_node(
        nodegroup_corner_cube().name,
        input_kwargs={
            "Location": multiply.outputs["Vector"],
            "CenteringLoc": (0.0000, 1.0000, 0.0000),
            "Dimensions": reroute,
            "Vertices Z": 10,
        },
        label="ArmCube",
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": arm_cube})

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": reroute})

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": separate_xyz.outputs["Z"],
            1: -0.1000,
            2: separate_xyz_1.outputs["Z"],
            3: -0.1000,
            4: 0.2000,
        },
    )

    float_curve = nw.new_node(
        Nodes.FloatCurve,
        input_kwargs={
            "Factor": group_input.outputs["arm_width"],
            "Value": map_range.outputs["Result"],
        },
    )
    node_utils.assign_curve(
        float_curve.mapping.curves[0],
        [
            (0.0092, 0.7688),
            (0.1011, 0.5937),
            (0.1494, 0.4062),
            (0.3954, 0.0781),
            (1.0000, 0.2187),
        ],
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": multiply.outputs["Vector"]}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: separate_xyz_2.outputs["Y"]},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: float_curve, 1: subtract},
        attrs={"operation": "MULTIPLY"},
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_14 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": position_1}
    )

    map_range_1 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": separate_xyz_14.outputs["X"],
            1: -1.0000,
            2: 0.6000,
            3: 2.1000,
            4: -1.1000,
        },
    )

    float_curve_1 = nw.new_node(
        Nodes.FloatCurve,
        input_kwargs={
            "Factor": group_input.outputs["Arm_height"],
            "Value": map_range_1.outputs["Result"],
        },
    )
    node_utils.assign_curve(
        float_curve_1.mapping.curves[0],
        [(0.1341, 0.2094), (0.7386, 1.0000), (0.9682, 0.0781), (1.0000, 0.0000)],
    )

    separate_xyz_15 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": (-2.9000, 3.3000, 0.0000)}
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_14.outputs["Z"], 1: separate_xyz_15.outputs["Z"]},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: float_curve_1, 1: subtract_1},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply_1, "Z": multiply_2}
    )

    vector_rotate = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={
            "Vector": combine_xyz,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Angle": group_input.outputs["arms_angle"],
        },
    )

    set_position = nw.new_node(
        Nodes.SetPosition, input_kwargs={"Geometry": reroute_1, "Offset": vector_rotate}
    )

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Dimensions"],
            1: (0.0000, 0.5000, 0.0000),
        },
        attrs={"operation": "MULTIPLY"},
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": group_input.outputs["Arm Dimensions"]},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["Z"], 1: separate_xyz_3.outputs["Y"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz_3.outputs["X"],
            "Y": separate_xyz_3.outputs["Y"],
            "Z": subtract_2,
        },
    )

    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_1})

    arm_cube_1 = nw.new_node(
        nodegroup_corner_cube().name,
        input_kwargs={
            "Location": multiply_3.outputs["Vector"],
            "CenteringLoc": (0.0000, 1.0000, 0.0000),
            "Dimensions": reroute_2,
        },
        label="ArmCube",
    )

    separate_xyz_4 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": reroute_2})

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_4.outputs["X"], 1: 1.0001},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_4})

    arm_cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Side Segments": 4,
            "Radius": separate_xyz_4.outputs["Y"],
            "Depth": reroute_3,
        },
        attrs={"fill_type": "TRIANGLE_FAN"},
    )

    arm_cylinder = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": arm_cylinder.outputs["Mesh"],
            "Name": "UVMap",
            3: arm_cylinder.outputs["UV Map"],
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_3, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    separate_xyz_5 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": multiply_3.outputs["Vector"]}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": divide,
            "Y": separate_xyz_5.outputs["Y"],
            "Z": separate_xyz_4.outputs["Z"],
        },
    )

    arm_cylinder = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": arm_cylinder,
            "Translation": combine_xyz_2,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    roundtop = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [arm_cube_1, arm_cylinder]}
    )

    square_or_round = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": nw.compare(
                "EQUAL", group_input.outputs["Arm Type"], ARM_TYPE_SQUARE
            ),
            "False": roundtop,
            "True": arm_cube_1,
        },
    )

    angular_or_squareround = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": nw.compare(
                "EQUAL", group_input.outputs["Arm Type"], ARM_TYPE_ANGULAR
            ),
            "False": square_or_round,
            "True": set_position,
        },
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": angular_or_squareround,
            "Scale": (1.0000, -1.0000, 1.0000),
        },
    )

    flip_faces = nw.new_node(
        Nodes.FlipFaces, input_kwargs={"Mesh": transform_geometry_1}
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [flip_faces, angular_or_squareround]},
    )

    separate_xyz_6 = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": group_input.outputs["Back Dimensions"]},
    )

    separate_xyz_7 = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": group_input.outputs["Arm Dimensions"]},
    )

    separate_xyz_8 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Dimensions"]}
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_7.outputs["Y"],
            1: -2.0000,
            2: separate_xyz_8.outputs["Y"],
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz_6.outputs["X"],
            "Y": multiply_add,
            "Z": separate_xyz_6.outputs["Z"],
        },
    )

    back_board = nw.new_node(
        nodegroup_corner_cube().name,
        input_kwargs={
            "CenteringLoc": (0.0000, 0.5000, -1.0000),
            "Dimensions": combine_xyz_3,
            "Vertices X": 2,
            "Vertices Y": 2,
            "Vertices Z": 2,
        },
        label="BackBoard",
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [join_geometry_2, back_board]}
    )

    multiply_5 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_3, 1: (1.0000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Arm Dimensions"],
            1: (0.0000, -2.0000, 0.0000),
            2: group_input.outputs["Dimensions"],
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    multiply_add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Back Dimensions"],
            1: (-1.0000, 0.0000, 0.0000),
            2: multiply_add_1.outputs["Vector"],
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    separate_xyz_9 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": multiply_add_2.outputs["Vector"]}
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz_9.outputs["X"],
            "Y": separate_xyz_9.outputs["Y"],
            "Z": group_input.outputs["Baseboard Height"],
        },
    )

    base_board = nw.new_node(
        nodegroup_corner_cube().name,
        input_kwargs={
            "Location": multiply_5.outputs["Vector"],
            "CenteringLoc": (0.0000, 0.5000, -1.0000),
            "Dimensions": combine_xyz_4,
            "Vertices X": 2,
            "Vertices Y": 2,
            "Vertices Z": 2,
        },
        label="BaseBoard",
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Count"]}
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_13, 3: 4},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz_9.outputs["Y"]}
    )

    separate_xyz_10 = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": group_input.outputs["Seat Dimensions"]},
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_5, 1: separate_xyz_10.outputs["Y"]},
        attrs={"operation": "DIVIDE"},
    )

    ceil = nw.new_node(
        Nodes.Math, input_kwargs={0: divide_1}, attrs={"operation": "CEIL"}
    )

    combine_xyz_14 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 1.0000, "Y": ceil, "Z": 1.0000}
    )

    divide_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_4, 1: combine_xyz_14},
        attrs={"operation": "DIVIDE"},
    )

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": divide_2.outputs["Vector"]}
    )

    base_board_1 = nw.new_node(
        nodegroup_corner_cube().name,
        input_kwargs={
            "Location": multiply_5.outputs["Vector"],
            "CenteringLoc": (0.0000, 0.5000, -1.0000),
            "Dimensions": reroute_12,
            "Vertices X": 2,
            "Vertices Y": 2,
            "Vertices Z": 2,
        },
        label="BaseBoard",
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_13, 3: 4},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    switch_8 = nw.new_node(
        Nodes.Switch,
        input_kwargs={0: equal_1, 1: divide_2.outputs["Vector"], 2: combine_xyz_4},
        attrs={"input_type": "VECTOR"},
    )

    separate_xyz_16 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": switch_8})

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_16.outputs["Y"], 1: 0.7000},
        attrs={"operation": "MULTIPLY"},
    )

    grid_1 = nw.new_node(
        Nodes.MeshGrid,
        input_kwargs={"Size Y": multiply_6, "Vertices X": 1, "Vertices Y": 2},
    )

    combine_xyz_18 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": 0.1000,
            "Y": separate_xyz_16.outputs["Y"],
            "Z": separate_xyz_16.outputs["Z"],
        },
    )

    subtract_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: switch_8, 1: combine_xyz_18},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_7 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Back Dimensions"],
            1: (1.0000, 0.0000, 0.0000),
        },
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: subtract_3.outputs["Vector"], 1: multiply_7.outputs["Vector"]},
    )

    transform_geometry_10 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": grid_1.outputs["Mesh"],
            "Translation": add.outputs["Vector"],
            "Scale": (1.0000, 1.0000, 0.9000),
        },
    )

    cone = nw.new_node(
        "GeometryNodeMeshCone",
        input_kwargs={
            "Vertices": group_input.outputs["leg_faces"],
            "Side Segments": 4,
            "Radius Top": 0.0100,
            "Radius Bottom": 0.0250,
            "Depth": 0.0700,
        },
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["leg_dimensions"]}
    )

    combine_xyz_17 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": reroute_9,
            "Y": reroute_9,
            "Z": group_input.outputs["leg_z"],
        },
    )

    transform_geometry_9 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cone.outputs["Mesh"],
            "Translation": (0.0000, 0.0000, 0.0100),
            "Rotation": (0.0000, 3.1416, 0.0000),
            "Scale": combine_xyz_17,
        },
    )

    foot_cube = nw.new_node(
        nodegroup_corner_cube().name,
        input_kwargs={
            "CenteringLoc": (0.5000, 0.5000, 0.9000),
            "Dimensions": group_input.outputs["Foot Dimensions"],
        },
        label="FootCube",
    )

    transform_geometry_12 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": foot_cube, "Scale": (0.5000, 0.8000, 0.8000)},
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            0: group_input.outputs["leg_type"],
            1: transform_geometry_9,
            2: transform_geometry_12,
        },
    )

    transform_geometry_8 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": switch_6}
    )

    instance_on_points_1 = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": transform_geometry_10,
            "Instance": transform_geometry_8,
            "Scale": (1.0000, 1.0000, 1.2000),
        },
    )

    realize_instances_1 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points_1}
    )

    join_geometry_10 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [base_board_1, realize_instances_1]},
    )

    subtract_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_14, 1: (1.0000, 1.0000, 1.0000)},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_8 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: subtract_4.outputs["Vector"], 1: (0.0000, 0.5000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_9 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: divide_2.outputs["Vector"], 1: multiply_8.outputs["Vector"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_16 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": 1.0000, "Y": group_input.outputs["Reflection"], "Z": 1.0000},
    )

    multiply_10 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_9.outputs["Vector"], 1: combine_xyz_16},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_12 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Scaling footrest"],
            "Y": 1.0000,
            "Z": 1.0000,
        },
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_10,
            "Translation": multiply_10.outputs["Vector"],
            "Scale": combine_xyz_12,
        },
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={0: group_input.outputs["Footrest"], 1: transform_geometry_5},
    )

    combine_xyz_19 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Scaling footrest"],
            "Y": 1.3000,
            "Z": 1.0000,
        },
    )

    transform_geometry_11 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": realize_instances_1, "Scale": combine_xyz_19},
    )

    base_board_2 = nw.new_node(
        nodegroup_corner_cube().name,
        input_kwargs={
            "Location": multiply_5.outputs["Vector"],
            "CenteringLoc": (0.0000, 0.5000, -1.0000),
            "Dimensions": combine_xyz_4,
            "Vertices X": 3,
            "Vertices Y": 3,
            "Vertices Z": 3,
        },
        label="BaseBoard",
    )

    combine_xyz_13 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Scaling footrest"],
            "Y": 1.0000,
            "Z": 1.0000,
        },
    )

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": base_board_2, "Scale": combine_xyz_13},
    )

    join_geometry_11 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_11, transform_geometry_6]},
    )

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={0: group_input.outputs["Footrest"], 2: join_geometry_11},
    )

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={0: equal, 1: switch_2, 2: switch_4},
    )

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [join_geometry_3, base_board, switch_5]},
    )

    grid = nw.new_node(Nodes.MeshGrid, input_kwargs={"Vertices X": 2, "Vertices Y": 2})

    multiply_11 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Dimensions"],
            1: (0.5000, 0.0000, 0.0000),
        },
        attrs={"operation": "MULTIPLY"},
    )

    multiply_12 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Dimensions"],
            1: (1.0000, 1.0000, 0.0000),
        },
        attrs={"operation": "MULTIPLY"},
    )

    multiply_13 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Foot Dimensions"],
            1: (2.5000, 2.5000, 0.0000),
        },
        attrs={"operation": "MULTIPLY"},
    )

    subtract_5 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: multiply_12.outputs["Vector"],
            1: multiply_13.outputs["Vector"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": grid.outputs["Mesh"],
            "Translation": multiply_11.outputs["Vector"],
            "Scale": subtract_5.outputs["Vector"],
        },
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={"Points": transform_geometry_2, "Instance": transform_geometry_8},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points}
    )

    join_geometry_5 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [join_geometry_4, realize_instances]},
    )

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Count"]}
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_10, 3: 4},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_4})

    multiply_14 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_4, 1: (0.0000, -0.5000, 1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_15 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_4, 1: (0.0000, 0.5000, 1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_10, 3: 4},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Reflection"]}
    )

    switch_7 = nw.new_node(
        Nodes.Switch,
        input_kwargs={0: equal_3, 1: reroute_11, 2: 1},
        attrs={"input_type": "INT"},
    )

    combine_xyz_15 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": 1.0000, "Y": switch_7, "Z": 1.1000},
    )

    multiply_16 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_15.outputs["Vector"], 1: combine_xyz_15},
        attrs={"operation": "MULTIPLY"},
    )

    divide_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_5, 1: ceil}, attrs={"operation": "DIVIDE"}
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz_10.outputs["X"],
            "Y": divide_3,
            "Z": separate_xyz_10.outputs["Z"],
        },
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_5})

    multiply_17 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_6, 1: combine_xyz_15},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_18 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_5, 1: (1.0000, 1.0300, 1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    seat_cushion = nw.new_node(
        nodegroup_corner_cube().name,
        input_kwargs={
            "CenteringLoc": (0.0000, 0.5000, 0.0000),
            "Dimensions": multiply_18.outputs["Vector"],
            "Vertices X": 2,
            "Vertices Y": 2,
            "Vertices Z": 2,
        },
        label="SeatCushion",
    )

    upwards_part = nw.new_node(
        Nodes.Compare,
        input_kwargs={"A": nw.new_node(Nodes.Index), "B": 2},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )
    seat_cushion = tagging.tag_nodegroup(
        nw, seat_cushion, t.Subpart.SupportSurface, selection=upwards_part
    )

    index = nw.new_node(Nodes.Index)

    equal_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: 1},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": seat_cushion,
            "Selection": equal_4,
            "Name": "TAG_support",
            "Value": True,
        },
        attrs={"data_type": "BOOLEAN", "domain": "FACE"},
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 1.0000

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_1,
            "Selection": value,
            "Name": "TAG_cushion",
            "Value": True,
        },
        attrs={"data_type": "BOOLEAN", "domain": "FACE"},
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Seat Margin"],
            "Y": group_input.outputs["Seat Margin"],
            "Z": 1.0000,
        },
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": store_named_attribute_2, "Scale": combine_xyz_6},
    )

    combine_xyz_11 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Scaling footrest"],
            "Y": 1.0000,
            "Z": 1.1000,
        },
    )

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_3, "Scale": combine_xyz_11},
    )

    nodegroup_array_fill_line_002 = nw.new_node(
        nodegroup_array_fill_line().name,
        input_kwargs={
            "Line Start": multiply_14.outputs["Vector"],
            "Line End": multiply_16.outputs["Vector"],
            "Instance Dimensions": multiply_17.outputs["Vector"],
            "Count": reroute_10,
            "Instance": transform_geometry_7,
        },
    )

    separate_xyz_17 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": multiply_16.outputs["Vector"]}
    )

    combine_xyz_21 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": separate_xyz_17.outputs["Z"]}
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": ceil})

    combine_xyz_20 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 1.0000, "Y": reroute_14, "Z": 1.0000}
    )

    transform_geometry_13 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_7, "Scale": combine_xyz_20},
    )

    nodegroup_array_fill_line_002_1 = nw.new_node(
        nodegroup_array_fill_line().name,
        input_kwargs={
            "Line End": combine_xyz_21,
            "Count": 1,
            "Instance": transform_geometry_13,
        },
    )

    switch_9 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            0: equal_2,
            1: nodegroup_array_fill_line_002,
            2: nodegroup_array_fill_line_002_1,
        },
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={0: group_input.outputs["Footrest"], 2: switch_9},
    )

    nodegroup_array_fill_line_002_2 = nw.new_node(
        nodegroup_array_fill_line().name,
        input_kwargs={
            "Line Start": multiply_14.outputs["Vector"],
            "Line End": multiply_15.outputs["Vector"],
            "Instance Dimensions": reroute_6,
            "Count": reroute_14,
            "Instance": transform_geometry_3,
        },
    )

    join_geometry_9 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [switch_3, nodegroup_array_fill_line_002_2]},
    )

    subdivide_mesh = nw.new_node(
        Nodes.SubdivideMesh, input_kwargs={"Mesh": join_geometry_9, "Level": 2}
    )

    separate_xyz_11 = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": group_input.outputs["Seat Dimensions"]},
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Backrest Width"],
            "Z": separate_xyz_11.outputs["Z"],
        },
    )

    add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_14.outputs["Vector"], 1: combine_xyz_7},
    )

    add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_15.outputs["Vector"], 1: combine_xyz_7},
    )

    separate_xyz_12 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Dimensions"]}
    )

    subtract_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_12.outputs["Z"], 1: separate_xyz_11.outputs["Z"]},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_6, 1: group_input.outputs["Baseboard Height"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": subtract_7,
            "Y": divide_3,
            "Z": group_input.outputs["Backrest Width"],
        },
    )

    seat_cushion_1 = nw.new_node(
        nodegroup_corner_cube().name,
        input_kwargs={
            "CenteringLoc": (0.1000, 0.5000, 1.0000),
            "Dimensions": combine_xyz_8,
            "Vertices X": 2,
            "Vertices Y": 2,
            "Vertices Z": 2,
        },
        label="SeatCushion",
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh, input_kwargs={"Mesh": seat_cushion_1, "Offset Scale": 0.0300}
    )

    scale_elements = nw.new_node(
        Nodes.ScaleElements,
        input_kwargs={
            "Geometry": extrude_mesh.outputs["Mesh"],
            "Selection": extrude_mesh.outputs["Top"],
            "Scale": 0.6000,
        },
    )

    subdivision_surface_1 = nw.new_node(
        Nodes.SubdivisionSurface, input_kwargs={"Mesh": scale_elements}
    )

    random_value = nw.new_node(Nodes.RandomValue, attrs={"data_type": "FLOAT_VECTOR"})

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": subdivision_surface_1,
            "Name": "UVMap",
            3: random_value.outputs["Value"],
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    multiply_19 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Backrest Width"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    separate_xyz_13 = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": group_input.outputs["Back Dimensions"]},
    )

    add_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: separate_xyz_13.outputs["X"], 1: 0.1000}
    )

    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_19, 1: add_3})

    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": add_4})

    add_5 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Backrest Angle"], 1: -1.5708}
    )

    combine_xyz_10 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": add_5})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_3,
            "Translation": combine_xyz_9,
            "Rotation": combine_xyz_10,
            "Scale": combine_xyz_6,
        },
    )

    nodegroup_array_fill_line_003 = nw.new_node(
        nodegroup_array_fill_line().name,
        input_kwargs={
            "Line Start": add_1.outputs["Vector"],
            "Line End": add_2.outputs["Vector"],
            "Instance Dimensions": reroute_6,
            "Count": ceil,
            "Instance": transform_geometry_4,
        },
    )

    join_geometry_6 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [subdivide_mesh, nodegroup_array_fill_line_003]},
    )

    join_geometry_7 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [join_geometry_5, realize_instances, join_geometry_6]
        },
    )

    subdivide_mesh_1 = nw.new_node(
        Nodes.SubdivideMesh, input_kwargs={"Mesh": join_geometry_5, "Level": 2}
    )

    join_geometry_8 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [subdivide_mesh_1, realize_instances, join_geometry_6]
        },
    )

    subdivision_surface_2 = nw.new_node(
        Nodes.SubdivisionSurface, input_kwargs={"Mesh": join_geometry_8, "Level": 1}
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={0: True, 1: join_geometry_7, 2: subdivision_surface_2},
    )
    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            0: group_input.outputs["Subdivide"],
            1: join_geometry_7,
            2: subdivision_surface_2,
        },
    )

    bounding_box = nw.new_node(
        nodegroup_corner_cube().name,
        input_kwargs={
            "CenteringLoc": (0.0000, 0.5000, -1.0000),
            "Dimensions": group_input.outputs["Dimensions"],
            "Vertices X": 2,
            "Vertices Y": 2,
            "Vertices Z": 2,
        },
        label="BoundingBox",
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": bounding_box})

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": switch_1, "BoundingBox": reroute_8},
        attrs={"is_active_output": True},
    )


def sofa_parameter_distribution(dimensions=None):
    if dimensions is None:
        dimensions = (
            uniform(0.95, 1.1),
            clip_gaussian(1.75, 0.75, 0.9, 3),
            uniform(0.69, 0.97),
        )

    return {
        "Dimensions": dimensions,
        "Arm Dimensions": (
            uniform(1, 1),
            uniform(0.06, 0.15),
            uniform(0.5, 0.75),
        ),
        "Back Dimensions": (uniform(0.15, 0.25), 0.0000, uniform(0.5, 0.75)),
        "Seat Dimensions": (dimensions[0], uniform(0.7, 1), uniform(0.15, 0.3)),
        "Foot Dimensions": (uniform(0.07, 0.25), 0.06, 0.06),
        "Baseboard Height": uniform(0.05, 0.09),
        "Backrest Width": uniform(0.1, 0.2),
        "Seat Margin": uniform(0.9700, 1),
        "Backrest Angle": uniform(-0.15, -0.5),
        "Arm Type": np.random.choice(
            [ARM_TYPE_SQUARE, ARM_TYPE_ROUND, ARM_TYPE_ANGULAR], p=[0.4, 0.2, 0.4]
        ),
        "arm_width": uniform(0.6, 0.9),
        "Arm_height": uniform(0.7, 1.0),
        "arms_angle": uniform(0.0, 1.08),
        "Footrest": True if uniform() > 0.5 and dimensions[1] > 2 else False,
        "Count": 1 if uniform() > 0.2 else 4,
        "Scaling footrest": uniform(1.3, 1.6),
        "Reflection": 1 if uniform() > 0.5 else -1,
        "leg_type": True if uniform() > 0.5 else False,
        "leg_dimensions": uniform(0.4, 0.9),
        "leg_z": uniform(1.1, 2.5),
        "leg_faces": uniform(4, 25),
    }


class SofaFactory(AssetFactory):
    def __init__(self, factory_seed):
        super().__init__(factory_seed)
        with FixedSeed(factory_seed):
            self.params = sofa_parameter_distribution()
            # from infinigen.assets.scatters.clothes import ClothesCover
            # self.clothes_scatter = ClothesCover(factory_fn=blanket.BlanketFactory, width=log_uniform(1, 1.5),
            #                                    size=uniform(.8, 1.2)) if uniform() < .3 else NoApply()
            materials = AssetList["SofaFactory"]()
            self.sofa_fabric = materials["sofa_fabric"].assign_material()

    def create_placeholder(self, **_):
        obj = butil.spawn_vert()
        butil.modify_mesh(
            obj,
            "NODES",
            node_group=nodegroup_sofa_geometry(),
            ng_inputs={
                **self.params,
            },
            apply=True,
        )
        tagging.tag_system.relabel_obj(obj)
        surface.add_material(obj, self.sofa_fabric)
        return obj

    def create_asset(self, i, placeholder, face_size, **_):
        hipoly = butil.copy(placeholder, keep_materials=True)

        butil.modify_mesh(hipoly, "SUBSURF", levels=1, apply=True)

        with butil.SelectObjects(hipoly):
            bpy.ops.object.shade_smooth()

        return hipoly


class ArmChairFactory(SofaFactory):
    def __init__(self, factory_seed):
        super().__init__(factory_seed)
        with FixedSeed(factory_seed):
            dimensions = (uniform(0.8, 1), uniform(0.9, 1.1), uniform(0.69, 0.97))
            self.params = sofa_parameter_distribution(dimensions=dimensions)
