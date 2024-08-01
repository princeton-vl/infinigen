# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


from infinigen.assets.objects.tables.table_top import nodegroup_capped_cylinder
from infinigen.assets.objects.tables.table_utils import (
    nodegroup_align_bottom_to_floor,
    nodegroup_arc_top,
    nodegroup_create_anchors,
    nodegroup_create_legs_and_strechers,
    nodegroup_n_gon_cylinder,
)
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


@node_utils.to_nodegroup(
    "nodegroup_chair_wheel", singleton=False, type="GeometryNodeTree"
)
def nodegroup_chair_wheel(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Arc Sweep Angle", 240.0000),
            ("NodeSocketFloat", "Wheel Width", 0.0000),
            ("NodeSocketFloat", "Wheel Rotation", 0.5000),
            ("NodeSocketFloat", "Pole Width", 0.0000),
            ("NodeSocketFloat", "Pole Aspect Ratio", 0.6000),
            ("NodeSocketFloat", "Pole Length", 3.0000),
        ],
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": group_input.outputs["Wheel Width"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Wheel Width"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply})

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_1, "End": combine_xyz_2}
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0200

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.5000

    cappedcylinder = nw.new_node(
        nodegroup_capped_cylinder().name,
        input_kwargs={
            "Thickness": value,
            "Radius": value_1,
            "Cap Relative Scale": 0.0100,
        },
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: value, 1: -1.0000}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_1})

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cappedcylinder,
            "Translation": combine_xyz,
            "Rotation": (-1.5708, 0.0000, 0.0000),
        },
    )

    position = nw.new_node(Nodes.InputPosition)

    align_euler_to_vector = nw.new_node(
        Nodes.AlignEulerToVector, input_kwargs={"Vector": position}, attrs={"axis": "Y"}
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": curve_line,
            "Instance": transform,
            "Rotation": align_euler_to_vector,
        },
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: value_1, 1: 0.0800})

    arctop = nw.new_node(
        nodegroup_arc_top().name,
        input_kwargs={
            "Diameter": add,
            "Sweep Angle": group_input.outputs["Arc Sweep Angle"],
        },
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Wheel Width"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": multiply_2, "Height": 0.0200},
    )

    fillet_curve = nw.new_node(
        "GeometryNodeFilletCurve",
        input_kwargs={
            "Curve": quadrilateral,
            "Count": 4,
            "Radius": 0.0300,
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": arctop,
            "Profile Curve": fillet_curve,
            "Fill Caps": True,
        },
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: value_1, 1: 0.1000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: value_1, 1: 0.4000},
        attrs={"operation": "MULTIPLY"},
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Side Segments": 8,
            "Fill Segments": 4,
            "Radius": multiply_3,
            "Depth": multiply_4,
        },
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: value_1, 1: 0.4400},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: value_1, 1: 0.4500},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_5, "Z": multiply_6}
    )

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_3,
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [instance_on_points, curve_to_mesh, transform_2]},
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_5, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_7})

    transform_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Translation": combine_xyz_4},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Pole Length"], 1: 0.1500},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Pole Width"], 1: -0.3535, 2: -0.3000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Z": multiply_add}
    )

    radians = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Wheel Rotation"]},
        attrs={"operation": "RADIANS"},
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": radians})

    transform_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_6,
            "Translation": combine_xyz_5,
            "Rotation": combine_xyz_6,
        },
    )

    curve_line_1 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (1.0000, 0.0000, -1.0000),
            "End": (1.0000, 0.0000, 1.0000),
        },
    )

    ngoncylinder = nw.new_node(
        nodegroup_n_gon_cylinder().name,
        input_kwargs={
            "Radius Curve": curve_line_1,
            "Height": group_input.outputs["Pole Length"],
            "N-gon": 4,
            "Profile Width": group_input.outputs["Pole Width"],
            "Aspect Ratio": group_input.outputs["Pole Aspect Ratio"],
            "Fillet Ratio": 0.1500,
            "Resolution": 32,
        },
    )

    transform_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": ngoncylinder.outputs["Mesh"],
            "Rotation": (0.0000, -1.5708, 0.0000),
        },
    )

    subdivision_surface_1 = nw.new_node(
        Nodes.SubdivisionSurface, input_kwargs={"Mesh": transform_3, "Level": 0}
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_4, subdivision_surface_1]},
    )

    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 0.1500

    transform_geometry = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": join_geometry_1, "Scale": value_2}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_wheeled_leg", singleton=False, type="GeometryNodeTree"
)
def nodegroup_wheeled_leg(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Joint Height", 0.0000),
            ("NodeSocketFloat", "Leg Diameter", 0.0000),
            ("NodeSocketFloat", "Top Height", 0.0000),
            ("NodeSocketFloat", "Arc Sweep Angle", 240.0000),
            ("NodeSocketFloat", "Wheel Width", 0.1300),
            ("NodeSocketFloat", "Wheel Rotation", 0.5000),
            ("NodeSocketFloat", "Pole Length", 1.8000),
            ("NodeSocketInt", "Leg Number", 5),
        ],
    )

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.0010

    createanchors = nw.new_node(
        nodegroup_create_anchors().name,
        input_kwargs={
            "Profile N-gon": group_input.outputs["Leg Number"],
            "Profile Width": value_1,
            "Profile Aspect Ratio": 1.0000,
        },
    )

    chair_wheel = nw.new_node(
        nodegroup_chair_wheel().name,
        input_kwargs={
            "Arc Sweep Angle": group_input.outputs["Arc Sweep Angle"],
            "Wheel Width": group_input.outputs["Wheel Width"],
            "Wheel Rotation": group_input.outputs["Wheel Rotation"],
            "Pole Width": 0.5000,
            "Pole Length": group_input.outputs["Pole Length"],
        },
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": chair_wheel, "Rotation": (0.0000, 1.5708, 0.0000)},
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: 2.0000, 1: value_1}, attrs={"operation": "DIVIDE"}
    )

    createlegsandstrechers = nw.new_node(
        nodegroup_create_legs_and_strechers().name,
        input_kwargs={
            "Anchors": createanchors,
            "Keep Legs": True,
            "Leg Instance": transform_geometry,
            "Table Height": 0.0250,
            "Leg Bottom Relative Scale": divide,
            "Strecher Index Increment": 1,
            "Strecher Relative Position": 1.0000,
            "Leg Bottom Offset": 0.0250,
            "Align Leg X rot": True,
        },
    )

    alignbottomtofloor = nw.new_node(
        nodegroup_align_bottom_to_floor().name,
        input_kwargs={"Geometry": createlegsandstrechers},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Leg Diameter"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Joint Height"],
            1: alignbottomtofloor.outputs["Offset"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 64, "Radius": multiply, "Depth": subtract},
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract}, attrs={"operation": "MULTIPLY"}
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: alignbottomtofloor.outputs["Offset"]},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_1,
        },
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: 0.0025},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Top Height"],
            1: group_input.outputs["Joint Height"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 64, "Radius": subtract_1, "Depth": subtract_2},
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={1: subtract_2}, attrs={"operation": "MULTIPLY"}
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Top Height"], 1: multiply_2},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": subtract_3})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": combine_xyz_2,
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                alignbottomtofloor.outputs["Geometry"],
                transform_geometry_2,
                transform_geometry_3,
            ]
        },
    )

    # multiply_3 = nw.new_node(Nodes.Math,
    #     input_kwargs={0: group_input.outputs["Top Height"], 1: -1.0000},
    #     attrs={'operation': 'MULTIPLY'})

    # combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Z': multiply_3})

    # transform_geometry_4 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': join_geometry, 'Translation': combine_xyz_3})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )
