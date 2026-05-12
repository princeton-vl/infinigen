# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: Version 2
# - Essence Toone: Version 1
# - Max Gonzalez Saez-Diez: Updates for sim

import gin
from numpy.random import randint, uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.joints import (
    nodegroup_add_jointed_geometry_metadata,
    nodegroup_distance_from_center,
    nodegroup_duplicate_joints_on_parent,
    nodegroup_hinge_joint,
    nodegroup_sliding_joint,
)
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import weighted_sample


@node_utils.to_nodegroup(
    "nodegroup_rounded_quad", singleton=False, type="GeometryNodeTree"
)
def nodegroup_rounded_quad(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": reroute, "Height": reroute_1},
    )

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute, 1: reroute_1},
        attrs={"operation": "MINIMUM"},
    )

    map_range = nw.new_node(Nodes.MapRange, input_kwargs={"Value": 0.5000, 4: 0.5000})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum, 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={"Curve": quadrilateral, "Count": 3, "Radius": multiply},
        attrs={"mode": "POLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Curve": fillet_curve},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_index_select", singleton=False, type="GeometryNodeTree"
)
def nodegroup_index_select(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "Index", 0)]
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: group_input.outputs["Index"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Result": equal},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_handle", singleton=False, type="GeometryNodeTree")
def nodegroup_handle(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "Handle Type", 0),
            ("NodeSocketFloat", "Handle Length", 0.0000),
            ("NodeSocketFloat", "Handle Radius", 0.0100),
            ("NodeSocketFloat", "Handle Protrude", 0.0300),
            ("NodeSocketFloat", "Handle Width", 0.0000),
            ("NodeSocketFloat", "Handle Height", 0.0000),
        ],
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Handle Type"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Handle Type"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    mesh_line = nw.new_node(Nodes.MeshLine, input_kwargs={"Count": 4})

    index_select = nw.new_node(nodegroup_index_select().name)

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide, 1: group_input.outputs["Handle Radius"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": subtract})

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": mesh_line,
            "Selection": index_select,
            "Position": combine_xyz_6,
        },
    )

    index_select_1 = nw.new_node(
        nodegroup_index_select().name, input_kwargs={"Index": 1}
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": group_input.outputs["Handle Protrude"], "Z": subtract},
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position,
            "Selection": index_select_1,
            "Position": combine_xyz_7,
        },
    )

    index_select_2 = nw.new_node(
        nodegroup_index_select().name, input_kwargs={"Index": 2}
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: divide_1, 1: group_input.outputs["Handle Radius"]}
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": group_input.outputs["Handle Protrude"], "Z": add},
    )

    set_position_2 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position_1,
            "Selection": index_select_2,
            "Position": combine_xyz_8,
        },
    )

    index_select_3 = nw.new_node(
        nodegroup_index_select().name, input_kwargs={"Index": 3}
    )

    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add})

    set_position_3 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position_2,
            "Selection": index_select_3,
            "Position": combine_xyz_9,
        },
    )

    mesh_to_curve = nw.new_node(
        Nodes.MeshToCurve, input_kwargs={"Mesh": set_position_3}
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": mesh_to_curve,
            "Count": 6,
            "Radius": group_input.outputs["Handle Radius"],
        },
        attrs={"mode": "POLY"},
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={"Resolution": 16, "Radius": group_input.outputs["Handle Radius"]},
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": fillet_curve,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Protrude"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_2})

    quadratic_b_zier = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Resolution": 10,
            "Start": combine_xyz_4,
            "Middle": combine_xyz_3,
            "End": combine_xyz_5,
        },
    )

    rounded_quad = nw.new_node(
        nodegroup_rounded_quad().name,
        input_kwargs={
            "Width": group_input.outputs["Handle Width"],
            "Height": group_input.outputs["Handle Height"],
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": quadratic_b_zier,
            "Profile Curve": rounded_quad,
            "Fill Caps": True,
        },
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_1,
            "False": curve_to_mesh_1,
            "True": curve_to_mesh,
        },
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Length"],
        },
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input.outputs["Handle Protrude"]}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cylinder.outputs["Mesh"], "Translation": combine_xyz},
    )

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Protrude"],
        },
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Protrude"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: 0.3500},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_2, "Z": multiply_3}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": combine_xyz_1,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    cylinder_2 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Protrude"],
        },
    )

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Protrude"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_3, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_3, "Z": multiply_4}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_2.outputs["Mesh"],
            "Translation": combine_xyz_2,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [transform_geometry_2, transform_geometry, transform_geometry_1]
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": switch_1, "True": join_geometry},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": switch},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_smooth_by_angle", singleton=False, type="GeometryNodeTree"
)
def nodegroup_smooth_by_angle(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Mesh", None),
            ("NodeSocketFloat", "Angle", 0.5236),
            ("NodeSocketBool", "Ignore Sharpness", False),
        ],
    )

    is_edge_smooth = nw.new_node("GeometryNodeInputEdgeSmooth")

    op_or = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: is_edge_smooth, 1: group_input.outputs["Ignore Sharpness"]},
        attrs={"operation": "OR"},
    )

    edge_angle = nw.new_node(Nodes.InputEdgeAngle)

    less_equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            0: edge_angle.outputs["Unsigned Angle"],
            1: group_input.outputs["Angle"],
        },
        attrs={"operation": "LESS_EQUAL"},
    )

    is_shade_smooth = nw.new_node("GeometryNodeInputShadeSmooth")

    op_or_1 = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: is_shade_smooth, 1: group_input.outputs["Ignore Sharpness"]},
        attrs={"operation": "OR"},
    )

    op_and = nw.new_node(Nodes.BooleanMath, input_kwargs={0: less_equal, 1: op_or_1})

    set_shade_smooth = nw.new_node(
        Nodes.SetShadeSmooth,
        input_kwargs={
            "Geometry": group_input.outputs["Mesh"],
            "Selection": op_or,
            "Shade Smooth": op_and,
        },
        attrs={"domain": "EDGE"},
    )

    set_shade_smooth_1 = nw.new_node(
        Nodes.SetShadeSmooth, input_kwargs={"Geometry": set_shade_smooth}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": set_shade_smooth_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_dials", singleton=False, type="GeometryNodeTree")
def nodegroup_dials(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Radius", 1.0000),
            ("NodeSocketFloat", "Depth", 2.0000),
            ("NodeSocketBool", "Is Knob", False),
            ("NodeSocketMaterial", "Knob Material", None),
        ],
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 16,
            "Radius": group_input.outputs["Radius"],
            "Depth": group_input.outputs["Depth"],
        },
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 0.9000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 0.0020, "Y": 0.0020, "Z": multiply}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_1})

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": group_input.outputs["Depth"], "Z": divide_1},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Translation": combine_xyz_2},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry, transform_geometry_1]},
    )

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Radius": group_input.outputs["Radius"], "Depth": 0.0050},
    )

    index = nw.new_node(Nodes.Index)

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: 31},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_or = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: equal, 1: equal_1},
        attrs={"operation": "OR"},
    )

    index_1 = nw.new_node(Nodes.Index)

    greater_equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index_1, 3: 15},
        attrs={"data_type": "INT", "operation": "GREATER_EQUAL"},
    )

    less_equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index_1, 3: 16},
        attrs={"data_type": "INT", "operation": "LESS_EQUAL"},
    )

    op_and = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: greater_equal, 1: less_equal}
    )

    op_or_1 = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: op_or, 1: op_and}, attrs={"operation": "OR"}
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Depth"]}
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": cylinder_1.outputs["Mesh"],
            "Selection": op_or_1,
            "Offset": combine_xyz_3,
        },
        attrs={"mode": "EDGES"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input.outputs["Radius"]}
    )

    normal = nw.new_node(Nodes.InputNormal)

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_4, 1: normal},
        attrs={"operation": "MULTIPLY"},
    )

    extrude_mesh_1 = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": extrude_mesh.outputs["Mesh"],
            "Selection": extrude_mesh.outputs["Side"],
            "Offset": multiply_1.outputs["Vector"],
        },
    )

    merge_by_distance = nw.new_node(
        Nodes.MergeByDistance, input_kwargs={"Geometry": extrude_mesh_1.outputs["Mesh"]}
    )

    separate_geometry = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={
            "Geometry": extrude_mesh.outputs["Mesh"],
            "Selection": extrude_mesh.outputs["Side"],
        },
    )

    flip_faces = nw.new_node(
        Nodes.FlipFaces, input_kwargs={"Mesh": separate_geometry.outputs["Selection"]}
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [merge_by_distance, flip_faces]}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Translation": (0.0025, 0.0000, 0.0000),
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Is Knob"],
            "False": join_geometry,
            "True": transform_geometry_2,
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": switch,
            "Material": group_input.outputs["Knob Material"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": set_material},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_door", singleton=False, type="GeometryNodeTree")
def nodegroup_door(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Radius", 0.5000),
            ("NodeSocketFloat", "Inner Door Box Height", -0.1000),
            ("NodeSocketFloat", "inner Door Box Width", -0.1000),
            ("NodeSocketFloat", "Window Height", 0.0000),
            ("NodeSocketFloat", "Window Width", 0.0000),
            ("NodeSocketFloat", "Window Radius", 0.8500),
            ("NodeSocketBool", "Has Handle", False),
            ("NodeSocketInt", "Handle Type", 0),
            ("NodeSocketFloat", "Handle Length", 0.0000),
            ("NodeSocketFloat", "Handle Radius", 0.0100),
            ("NodeSocketFloat", "Handle Protrude", 0.0300),
            ("NodeSocketFloat", "Handle Width", 0.0000),
            ("NodeSocketFloat", "Handle Height", 0.0000),
            ("NodeSocketFloat", "Handle Y Offset", 0.0000),
            ("NodeSocketFloat", "Handle Z Offset", 0.0000),
            ("NodeSocketBool", "Is Vertical Handle", False),
            ("NodeSocketMaterial", "Door Handle Material", None),
            ("NodeSocketMaterial", "Window Material", None),
            ("NodeSocketMaterial", "Door Box Material", None),
            ("NodeSocketMaterial", "Door Material", None),
        ],
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: (-0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: (0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    curve_line = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": multiply.outputs["Vector"],
            "End": multiply_1.outputs["Vector"],
        },
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Width": separate_xyz.outputs["Y"],
            "Height": separate_xyz.outputs["Z"],
        },
    )

    index = nw.new_node(Nodes.Index)

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    index_1 = nw.new_node(Nodes.Index)

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index_1, 3: 3},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_or = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: equal, 1: equal_1},
        attrs={"operation": "OR"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: op_or},
        attrs={"operation": "MULTIPLY"},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": quadrilateral,
            "Count": 6,
            "Radius": multiply_2,
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": fillet_curve,
            "Fill Caps": True,
        },
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "Y": group_input.outputs["Inner Door Box Height"],
            "Z": group_input.outputs["inner Door Box Width"],
        },
    )

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Size"],
            1: (1.0000, 0.0000, 0.0000),
            2: combine_xyz,
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    cube = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": multiply_add.outputs["Vector"]}
    )

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Scale": (2.0000, 1.0000, 1.0000),
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": curve_to_mesh, "Mesh 2": transform_geometry_7},
        attrs={"solver": "EXACT"},
    )

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": difference.outputs["Mesh"],
            "Material": group_input.outputs["Door Material"],
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Material": group_input.outputs["Door Box Material"],
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material_3, set_material]}
    )

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: (-0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: (0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    curve_line_1 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": multiply_3.outputs["Vector"],
            "End": multiply_4.outputs["Vector"],
        },
    )

    quadrilateral_1 = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Width": group_input.outputs["Window Width"],
            "Height": group_input.outputs["Window Height"],
        },
    )

    fillet_curve_1 = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": quadrilateral_1,
            "Count": 6,
            "Radius": group_input.outputs["Window Radius"],
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line_1,
            "Profile Curve": fillet_curve_1,
            "Fill Caps": True,
        },
    )

    transform_geometry_8 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": curve_to_mesh_1, "Scale": (2.0000, 1.0000, 1.0000)},
    )

    difference_1 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 1": join_geometry,
            "Mesh 2": transform_geometry_8,
            "Self Intersection": True,
        },
        attrs={"solver": "EXACT"},
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": curve_to_mesh_1,
            "Material": group_input.outputs["Window Material"],
        },
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [difference_1.outputs["Mesh"], set_material_1]},
    )

    multiply_5 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: (0.5000, 0.5000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_2,
            "Translation": multiply_5.outputs["Vector"],
        },
    )

    smooth_by_angle = nw.new_node(
        nodegroup_smooth_by_angle().name, input_kwargs={"Mesh": transform_geometry}
    )

    handle = nw.new_node(
        nodegroup_handle().name,
        input_kwargs={
            "Handle Type": group_input.outputs["Handle Type"],
            "Handle Length": group_input.outputs["Handle Length"],
            "Handle Radius": group_input.outputs["Handle Radius"],
            "Handle Protrude": group_input.outputs["Handle Protrude"],
            "Handle Width": group_input.outputs["Handle Width"],
            "Handle Height": group_input.outputs["Handle Height"],
        },
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": handle, "Rotation": (1.5708, 0.0000, 0.0000)},
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Handle Type"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_2,
            "False": group_input.outputs["Handle Radius"],
            "True": group_input.outputs["Handle Height"],
        },
        attrs={"input_type": "FLOAT"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": switch_4})

    multiply_6 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_3, 1: (0.0000, 0.0000, -1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Size"],
            1: (1.0000, 0.5000, 0.5000),
            2: multiply_6.outputs["Vector"],
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_4,
            "Translation": multiply_add_1.outputs["Vector"],
        },
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "Y": group_input.outputs["Handle Y Offset"],
            "Z": group_input.outputs["Handle Z Offset"],
        },
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_3, "Translation": combine_xyz_2},
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Handle Type"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_3,
            "False": group_input.outputs["Handle Radius"],
            "True": group_input.outputs["Handle Width"],
        },
        attrs={"input_type": "FLOAT"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": switch_2})

    multiply_7 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_1, 1: (0.0000, -1.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Size"],
            1: (1.0000, 1.0000, 0.0000),
            2: multiply_7.outputs["Vector"],
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": handle,
            "Translation": multiply_add_2.outputs["Vector"],
        },
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_1, "Translation": combine_xyz_2},
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Is Vertical Handle"],
            "False": transform_geometry_5,
            "True": transform_geometry_2,
        },
    )

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": group_input.outputs["Has Handle"], "True": switch_3},
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": switch_5,
            "Material": group_input.outputs["Door Handle Material"],
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [smooth_by_angle, set_material_2]}
    )

    multiply_8 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: (0.0000, 0.0000, 0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Translation": multiply_8.outputs["Vector"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": transform_geometry_6},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_plate", singleton=False, type="GeometryNodeTree")
def nodegroup_plate(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketFloat", "Radius", 1.0000)]
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide})

    curve_circle = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 17, "Radius": reroute}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": reroute})

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_1, 1: (-1.0000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: combine_xyz_1,
            1: (1.0000, 0.0000, 0.0000),
            2: (0.0000, -0.0100, 0.0000),
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    b_zier_segment = nw.new_node(
        Nodes.CurveBezierSegment,
        input_kwargs={
            "Resolution": 3,
            "Start": multiply.outputs["Vector"],
            "Start Handle": combine_xyz_1,
            "End Handle": combine_xyz_1,
            "End": multiply_add.outputs["Vector"],
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_circle.outputs["Curve"],
            "Profile Curve": b_zier_segment,
        },
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh, input_kwargs={"Mesh": curve_to_mesh, "Offset Scale": 0.0010}
    )

    smooth_by_angle = nw.new_node(
        nodegroup_smooth_by_angle().name,
        input_kwargs={"Mesh": extrude_mesh.outputs["Mesh"]},
    )

    flip_faces = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": curve_to_mesh})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [smooth_by_angle, flip_faces]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": join_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_microwave_base", singleton=False, type="GeometryNodeTree"
)
def nodegroup_microwave_base(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Radius", 0.0100),
            ("NodeSocketFloat", "Horizontal Base Thickness", 0.0000),
            ("NodeSocketFloat", "Vertical Base Thickness", 0.0000),
            ("NodeSocketMaterial", "Base Material", None),
        ],
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: (-0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: (0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    curve_line = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": multiply.outputs["Vector"],
            "End": multiply_1.outputs["Vector"],
        },
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Width": separate_xyz.outputs["Y"],
            "Height": separate_xyz.outputs["Z"],
        },
    )

    index = nw.new_node(Nodes.Index)

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    index_1 = nw.new_node(Nodes.Index)

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index_1, 3: 3},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_or = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: equal, 1: equal_1},
        attrs={"operation": "OR"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: op_or},
        attrs={"operation": "MULTIPLY"},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": quadrilateral,
            "Count": 2,
            "Radius": multiply_2,
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": fillet_curve,
            "Fill Caps": True,
        },
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Horizontal Base Thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Vertical Base Thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply_3, "Z": multiply_4}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: combine_xyz},
        attrs={"operation": "SUBTRACT"},
    )

    cube = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": subtract.outputs["Vector"]}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Translation": (0.0100, 0.0000, 0.0000),
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": curve_to_mesh, "Mesh 2": transform_geometry_1},
    )

    smooth_by_angle = nw.new_node(
        nodegroup_smooth_by_angle().name,
        input_kwargs={"Mesh": difference.outputs["Mesh"]},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": smooth_by_angle,
            "Material": group_input.outputs["Base Material"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": set_material},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_button", singleton=False, type="GeometryNodeTree")
def nodegroup_button(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketMaterial", "Button Material", None),
            ("NodeSocketFloat", "Radius", 0.0020),
            ("NodeSocketFloat", "zOffset", 0.0000),
        ],
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["zOffset"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    not_equal = nw.new_node(
        Nodes.Compare, input_kwargs={0: reroute_3}, attrs={"operation": "NOT_EQUAL"}
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["Z"]}
    )

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["Y"]}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": reroute_13, "Height": reroute_14},
    )

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_14, 1: reroute_13},
        attrs={"operation": "MINIMUM"},
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: minimum, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Radius"]}
    )

    minimum_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide, 1: reroute_1},
        attrs={"operation": "MINIMUM"},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={"Curve": quadrilateral, "Count": 5, "Radius": minimum_1},
        attrs={"mode": "POLY"},
    )

    fill_curve = nw.new_node(Nodes.FillCurve, input_kwargs={"Curve": fillet_curve})

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["X"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh, input_kwargs={"Mesh": fill_curve, "Offset Scale": reroute_9}
    )

    flip_faces = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": fill_curve})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [extrude_mesh.outputs["Mesh"], flip_faces]},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Rotation": (0.0000, 1.5708, 0.0000)},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Size"]}
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute, 1: (-1.0000, -0.5000, 0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry,
            "Translation": multiply.outputs["Vector"],
        },
    )

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["Y"]}
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_4, "Y": reroute_6, "Z": multiply_1}
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_1})

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_1, 1: (-0.5000, -0.5000, 1.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Translation": multiply_2.outputs["Vector"],
        },
    )

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_3}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"]},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["Z"]}
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_4, "Y": multiply_3, "Z": reroute_8}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    multiply_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: (-0.5000, -1.5000, 0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Translation": multiply_4.outputs["Vector"],
        },
    )

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_2}
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_1, reroute_10, reroute_11]},
    )

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_1}
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": not_equal,
            "False": join_geometry_1,
            "True": reroute_12,
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": switch,
            "Material": group_input.outputs["Button Material"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": set_material},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_side_panel", singleton=False, type="GeometryNodeTree"
)
def nodegroup_side_panel(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input_1 = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Radius", 0.5000),
            ("NodeSocketFloat", "Panel Width", 0.0000),
            ("NodeSocketFloat", "Panel Height", 0.0000),
            ("NodeSocketString", "Time", ""),
            ("NodeSocketGeometry", "Button Geom", None),
            ("NodeSocketFloat", "Button Y Offset", 0.0000),
            ("NodeSocketFloat", "Button Z Offset", 0.0000),
            ("NodeSocketBool", "Has Handle", False),
            ("NodeSocketFloat", "Side Box Inner Box Width", 0.0000),
            ("NodeSocketFloat", "Side Box Inner Box Height", 0.0000),
            ("NodeSocketFloat", "Panel Z Offset", 0.0000),
            ("NodeSocketMaterial", "Time Material", None),
            ("NodeSocketMaterial", "Time Panel Material", None),
            ("NodeSocketMaterial", "Side Panel Material", None),
            ("NodeSocketMaterial", "Side Panel Box Material", None),
        ],
    )

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["Has Handle"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["Size"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_1, 1: (-0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_1, 1: (0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    curve_line = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": multiply.outputs["Vector"],
            "End": multiply_1.outputs["Vector"],
        },
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input_1.outputs["Size"]}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Width": separate_xyz.outputs["Y"],
            "Height": separate_xyz.outputs["Z"],
        },
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["Radius"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    index = nw.new_node(Nodes.Index)

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    index_1 = nw.new_node(Nodes.Index)

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index_1, 3: 2},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_or = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: equal, 1: equal_1},
        attrs={"operation": "OR"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_3, 1: op_or},
        attrs={"operation": "MULTIPLY"},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": quadrilateral,
            "Count": 6,
            "Radius": multiply_2,
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": fillet_curve,
            "Fill Caps": True,
        },
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": 0.0010,
            "Y": group_input_1.outputs["Side Box Inner Box Width"],
            "Z": group_input_1.outputs["Side Box Inner Box Height"],
        },
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_4})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Translation": multiply_1.outputs["Vector"],
        },
    )

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_3}
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 1": curve_to_mesh,
            "Mesh 2": reroute_14,
            "Self Intersection": True,
        },
        attrs={"solver": "EXACT"},
    )

    set_material_4 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": difference.outputs["Mesh"],
            "Material": group_input_1.outputs["Side Panel Material"],
        },
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": -0.0005})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_3, "Translation": combine_xyz_5},
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_4,
            "Material": group_input_1.outputs["Side Panel Box Material"],
        },
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_1})

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material_4, reroute_15]}
    )

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_3})

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["Button Geom"]}
    )

    bounding_box = nw.new_node(Nodes.BoundingBox, input_kwargs={"Geometry": reroute_9})

    distance_from_center = nw.new_node(
        nodegroup_distance_from_center().name,
        input_kwargs={"Geometry": join_geometry_3},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "Y": group_input_1.outputs["Button Y Offset"],
            "Z": group_input_1.outputs["Button Z Offset"],
        },
    )

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: distance_from_center,
            1: (1.0000, 1.0000, -1.0000),
            2: combine_xyz_3,
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Translation": multiply_add.outputs["Vector"],
        },
    )

    difference_1 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 1": reroute_20,
            "Mesh 2": transform_geometry_2,
            "Self Intersection": True,
        },
        attrs={"solver": "EXACT"},
    )

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_11,
            "False": difference_1.outputs["Mesh"],
            "True": reroute_22,
        },
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": 0.0010,
            "Y": group_input_1.outputs["Panel Width"],
            "Z": group_input_1.outputs["Panel Height"],
        },
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cube.outputs["Mesh"]}
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_1.outputs["Panel Height"], 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide})

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["Panel Z Offset"]}
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_6})

    add = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: combine_xyz_1, 1: combine_xyz_6}
    )

    multiply_add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: reroute_13,
            1: (0.5000, 0.0000, 0.5000),
            2: add.outputs["Vector"],
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_12,
            "Translation": multiply_add_1.outputs["Vector"],
        },
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry,
            "Material": group_input_1.outputs["Time Panel Material"],
        },
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_2})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [switch, reroute_18]}
    )

    smooth_by_angle = nw.new_node(
        nodegroup_smooth_by_angle().name,
        input_kwargs={"Mesh": join_geometry, "Angle": 0.0000},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["Time"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input_1.outputs["Panel Height"],
            1: group_input_1.outputs["Panel Width"],
        },
        attrs={"operation": "MINIMUM"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum, 1: 0.4000},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["Panel Width"]}
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    string_to_curves = nw.new_node(
        "GeometryNodeStringToCurves",
        input_kwargs={
            "String": reroute_5,
            "Size": multiply_3,
            "Character Spacing": 2.1000,
            "Text Box Width": reroute_8,
        },
        attrs={
            "align_x": "CENTER",
            "overflow": "SCALE_TO_FIT",
            "pivot_mode": "MIDPOINT",
            "align_y": "MIDDLE",
        },
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve,
        input_kwargs={
            "Curve": string_to_curves.outputs["Curve Instances"],
            "Count": 24,
        },
    )

    fill_curve = nw.new_node(
        Nodes.FillCurve, input_kwargs={"Curve": resample_curve}, attrs={"mode": "NGONS"}
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh, input_kwargs={"Mesh": fill_curve, "Offset Scale": 0.0006}
    )

    flip_faces = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": fill_curve})

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [extrude_mesh.outputs["Mesh"], flip_faces]},
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_8, 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": divide_1})

    add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_2, 1: multiply_add_1.outputs["Vector"]},
    )

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_1.outputs["Vector"]}
    )

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_2,
            "Translation": reroute_17,
            "Rotation": (1.5708, 0.0000, 1.5708),
        },
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": transform_geometry_1}
    )

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": realize_instances,
            "Material": group_input_1.outputs["Time Material"],
        },
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_3})

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [smooth_by_angle, reroute_21]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": join_geometry_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input_2 = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Base Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Radius", 0.0100),
            ("NodeSocketFloat", "Horizontal Base Thickness", 0.0000),
            ("NodeSocketFloat", "Vertical Base Thickness", 0.0000),
            ("NodeSocketFloat", "Side Box Width", 0.0000),
            ("NodeSocketFloat", "Panel Width", 0.0000),
            ("NodeSocketFloat", "Panel Height", 0.0000),
            ("NodeSocketFloat", "Panel Z Offset", 0.0000),
            ("NodeSocketString", "Time", ""),
            ("NodeSocketFloat", "Side Box Inner Box Width", 0.0000),
            ("NodeSocketFloat", "Side Box Inner Box Height", 0.0000),
            ("NodeSocketFloat", "Plate Radius", 1.0000),
            ("NodeSocketFloat", "Door Thickness", 0.0000),
            ("NodeSocketFloat", "Door Box Width", 0.1000),
            ("NodeSocketFloat", "Door Box Height", 0.1000),
            ("NodeSocketFloat", "Window Width", 0.0000),
            ("NodeSocketFloat", "Window Height", 0.0000),
            ("NodeSocketFloat", "Window Radius", 0.8500),
            ("NodeSocketBool", "Has Handle", False),
            ("NodeSocketInt", "Handle Type", 0),
            ("NodeSocketFloat", "Handle Length", 0.0000),
            ("NodeSocketFloat", "Handle Radius", 0.0100),
            ("NodeSocketFloat", "Handle Protrude", 0.0300),
            ("NodeSocketFloat", "Handle Width", 0.0000),
            ("NodeSocketFloat", "Handle Height", 0.0000),
            ("NodeSocketFloat", "Handle Y Offset", 0.0000),
            ("NodeSocketFloat", "Handle Z Offset", 0.0000),
            ("NodeSocketBool", "Is Vertical Handle", False),
            ("NodeSocketVector", "Button Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Button Y Offset", 0.0000),
            ("NodeSocketFloat", "Button Z Offset", 0.0000),
            ("NodeSocketBool", "Is Knob", False),
            ("NodeSocketInt", "Num Dials", 2),
            ("NodeSocketFloat", "Dial Radius", 1.0000),
            ("NodeSocketFloat", "Dial Depth", 2.0000),
            ("NodeSocketFloat", "Dial Y Offset", 0.0000),
            ("NodeSocketFloat", "Dial Z Offset", 0.0000),
            ("NodeSocketBool", "Has Knobs", False),
            ("NodeSocketVector", "Knob Start Loc", (0.0000, 0.0000, -0.0500)),
            ("NodeSocketVector", "Knob End Loc", (0.0000, 0.0000, 0.0500)),
            ("NodeSocketMaterial", "Base Material", None),
            ("NodeSocketMaterial", "Plate Material", None),
            ("NodeSocketMaterial", "Time Material", None),
            ("NodeSocketMaterial", "Time Panel Material", None),
            ("NodeSocketMaterial", "Side Panel Material", None),
            ("NodeSocketMaterial", "Side Panel Box Material", None),
            ("NodeSocketMaterial", "Button Material", None),
            ("NodeSocketMaterial", "Door Handle Material", None),
            ("NodeSocketMaterial", "Window Material", None),
            ("NodeSocketMaterial", "Door Box Material", None),
            ("NodeSocketMaterial", "Door Material", None),
            ("NodeSocketMaterial", "Knob Material", None),
        ],
    )

    reroute_53 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Has Knobs"]}
    )

    reroute_54 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_53})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Base Size"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input_2.outputs["Door Thickness"],
            "Y": group_input_2.outputs["Side Box Width"],
        },
    )

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_1, 1: (1.0000, 0.0000, 1.0000), 2: combine_xyz},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_46 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Radius"]}
    )

    reroute_47 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_46})

    reroute_62 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_47})

    reroute_63 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_62})

    reroute_27 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Panel Width"]}
    )

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_27})

    reroute_29 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Panel Height"]}
    )

    reroute_30 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_29})

    reroute_33 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Time"]}
    )

    reroute_34 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_33})

    button = nw.new_node(
        nodegroup_button().name,
        input_kwargs={
            "Size": group_input_2.outputs["Button Size"],
            "Button Material": group_input_2.outputs["Button Material"],
            "Radius": group_input_2.outputs["Radius"],
            "zOffset": group_input_2.outputs["Button Z Offset"],
        },
    )

    reroute_58 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": button})

    reroute_39 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Button Y Offset"]}
    )

    reroute_40 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_39})

    reroute_41 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Button Z Offset"]}
    )

    reroute_42 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_41})

    reroute_48 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Has Handle"]}
    )

    reroute_49 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_48})

    reroute_64 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_49})

    reroute_65 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_64})

    reroute_35 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_2.outputs["Side Box Inner Box Width"]},
    )

    reroute_36 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_35})

    reroute_37 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_2.outputs["Side Box Inner Box Height"]},
    )

    reroute_38 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_37})

    reroute_31 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Panel Z Offset"]}
    )

    reroute_32 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_31})

    side_panel = nw.new_node(
        nodegroup_side_panel().name,
        input_kwargs={
            "Size": multiply_add.outputs["Vector"],
            "Radius": reroute_63,
            "Panel Width": reroute_28,
            "Panel Height": reroute_30,
            "Time": reroute_34,
            "Button Geom": reroute_58,
            "Button Y Offset": reroute_40,
            "Button Z Offset": reroute_42,
            "Has Handle": reroute_65,
            "Side Box Inner Box Width": reroute_36,
            "Side Box Inner Box Height": reroute_38,
            "Panel Z Offset": reroute_32,
            "Time Material": group_input_2.outputs["Time Material"],
            "Time Panel Material": group_input_2.outputs["Time Panel Material"],
            "Side Panel Material": group_input_2.outputs["Side Panel Material"],
            "Side Panel Box Material": group_input_2.outputs["Side Panel Box Material"],
        },
    )

    reroute_60 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_61 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_60})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_2.outputs["Door Thickness"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_2.outputs["Side Box Width"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide, "Y": divide_1}
    )

    multiply_add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_61, 1: (0.0000, 0.5000, 0.0000), 2: combine_xyz_1},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": side_panel,
            "Translation": multiply_add_1.outputs["Vector"],
        },
    )

    microwave_base = nw.new_node(
        nodegroup_microwave_base().name,
        input_kwargs={
            "Size": group_input_2.outputs["Base Size"],
            "Radius": group_input_2.outputs["Radius"],
            "Horizontal Base Thickness": group_input_2.outputs[
                "Horizontal Base Thickness"
            ],
            "Vertical Base Thickness": group_input_2.outputs["Vertical Base Thickness"],
            "Base Material": group_input_2.outputs["Base Material"],
        },
    )

    reroute_57 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": microwave_base})

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_geometry, reroute_57]}
    )

    reroute_69 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_1})

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_69, "Label": "microwave_base"},
    )

    reroute_43 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Plate Radius"]}
    )

    reroute_44 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_43})

    plate = nw.new_node(nodegroup_plate().name, input_kwargs={"Radius": reroute_44})

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": plate,
            "Material": group_input_2.outputs["Plate Material"],
        },
    )

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material, "Label": "plate"},
    )

    distance_from_center = nw.new_node(
        nodegroup_distance_from_center().name,
        input_kwargs={"Geometry": join_geometry_1},
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input_2.outputs["Base Size"],
            1: (0.0000, 0.5000, 0.0000),
        },
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"Z": group_input_2.outputs["Vertical Base Thickness"]},
    )

    add = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: multiply.outputs["Vector"], 1: combine_xyz_2}
    )

    reroute_59 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add.outputs["Vector"]}
    )

    multiply_add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: distance_from_center,
            1: (0.0000, -1.0000, -1.0000),
            2: reroute_59,
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "plate_joint",
            "Parent": add_jointed_geometry_metadata,
            "Child": add_jointed_geometry_metadata_1,
            "Position": multiply_add_2.outputs["Vector"],
            "Value": 4.2000,
        },
    )

    add_jointed_geometry_metadata_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": hinge_joint.outputs["Geometry"],
            "Label": "microwave_base_w_plate",
        },
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input_2.outputs["Door Thickness"]}
    )

    multiply_add_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_1, 1: (0.0000, 1.0000, 1.0000), 2: combine_xyz_3},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Door Box Width"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Door Box Height"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Window Height"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Window Width"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Window Radius"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Handle Type"]}
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Handle Length"]}
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Handle Radius"]}
    )

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    reroute_18 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Handle Protrude"]}
    )

    reroute_19 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Handle Width"]}
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    reroute_21 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Handle Height"]}
    )

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    reroute_23 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Handle Y Offset"]}
    )

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    reroute_25 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Handle Z Offset"]}
    )

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_25})

    reroute_50 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_2.outputs["Is Vertical Handle"]},
    )

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_50})

    door = nw.new_node(
        nodegroup_door().name,
        input_kwargs={
            "Size": multiply_add_3.outputs["Vector"],
            "Radius": reroute_63,
            "Inner Door Box Height": reroute_3,
            "inner Door Box Width": reroute_5,
            "Window Height": reroute_9,
            "Window Width": reroute_7,
            "Window Radius": reroute_11,
            "Has Handle": reroute_65,
            "Handle Type": reroute_13,
            "Handle Length": reroute_15,
            "Handle Radius": reroute_17,
            "Handle Protrude": reroute_18,
            "Handle Width": reroute_20,
            "Handle Height": reroute_22,
            "Handle Y Offset": reroute_24,
            "Handle Z Offset": reroute_26,
            "Is Vertical Handle": reroute_51,
            "Door Handle Material": group_input_2.outputs["Door Handle Material"],
            "Window Material": group_input_2.outputs["Window Material"],
            "Door Box Material": group_input_2.outputs["Door Box Material"],
            "Door Material": group_input_2.outputs["Door Material"],
        },
    )

    add_jointed_geometry_metadata_3 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": door, "Label": "microwave_door"},
    )

    reroute_68 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_3}
    )

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: distance_from_center, 1: (1.0000, -1.0000, -1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_45 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Door Thickness"]}
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_45, 1: 0.0006})

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": add_1})

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_1.outputs["Vector"], 1: combine_xyz_4},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_72 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": subtract.outputs["Vector"]}
    )

    reroute_66 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_51})

    reroute_70 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_66})

    reroute_71 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_70})

    vector = nw.new_node(Nodes.Vector)
    vector.vector = (0.0000, 1.0000, 0.0000)

    vector_1 = nw.new_node(Nodes.Vector)
    vector_1.vector = (0.0000, 0.0000, -1.0000)

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_71, "False": vector, "True": vector_1},
        attrs={"input_type": "VECTOR"},
    )

    hinge_joint_1 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "door_joint",
            "Parent": add_jointed_geometry_metadata_2,
            "Child": reroute_68,
            "Position": reroute_72,
            "Axis": switch,
            "Max": 1.8000,
        },
    )

    reroute_73 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_1.outputs["Geometry"]}
    )

    reroute_74 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_73})

    add_jointed_geometry_metadata_4 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_74, "Label": "microwave_base_w_door"},
    )

    reroute_67 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_58})

    add_jointed_geometry_metadata_5 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_67, "Label": "microwave_button"},
    )

    distance_from_center_1 = nw.new_node(
        nodegroup_distance_from_center().name,
        input_kwargs={"Geometry": hinge_joint_1.outputs["Geometry"]},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": -0.0006,
            "Y": group_input_2.outputs["Button Y Offset"],
            "Z": group_input_2.outputs["Button Z Offset"],
        },
    )

    reroute_55 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_5})

    multiply_add_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: distance_from_center_1,
            1: (1.0000, 1.0000, -1.0000),
            2: reroute_55,
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "button joint",
            "Parent": add_jointed_geometry_metadata_4,
            "Child": add_jointed_geometry_metadata_5,
            "Position": multiply_add_4.outputs["Vector"],
            "Axis": (-1.0000, 0.0000, 0.0000),
            "Value": -0.0000,
            "Max": 0.0200,
        },
    )

    reroute_75 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_74})

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input_2.outputs["Has Handle"],
            "False": sliding_joint.outputs["Geometry"],
            "True": reroute_75,
        },
    )

    reroute_76 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_1})

    reroute_77 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_76})

    reroute_78 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_77})

    add_jointed_geometry_metadata_6 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_77, "Label": "microwave_base_door_button"},
    )

    dials = nw.new_node(
        nodegroup_dials().name,
        input_kwargs={
            "Radius": group_input_2.outputs["Dial Radius"],
            "Depth": group_input_2.outputs["Dial Depth"],
            "Is Knob": group_input_2.outputs["Is Knob"],
            "Knob Material": group_input_2.outputs["Knob Material"],
        },
    )

    add_jointed_geometry_metadata_7 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": dials, "Label": "microwave_dials"},
    )

    distance_from_center_2 = nw.new_node(
        nodegroup_distance_from_center().name, input_kwargs={"Geometry": switch_1}
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": -0.0006,
            "Y": group_input_2.outputs["Dial Y Offset"],
            "Z": group_input_2.outputs["Dial Z Offset"],
        },
    )

    multiply_add_5 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: distance_from_center_2,
            1: (1.0000, 1.0000, 0.0000),
            2: combine_xyz_6,
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    hinge_joint_2 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "knob_joint",
            "Parent": add_jointed_geometry_metadata_6,
            "Child": add_jointed_geometry_metadata_7,
            "Position": multiply_add_5.outputs["Vector"],
            "Axis": (1.0000, 0.0000, 0.0000),
            "Value": 1.2000,
        },
    )

    mesh_line = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={
            "Count": group_input_2.outputs["Num Dials"],
            "Start Location": group_input_2.outputs["Knob Start Loc"],
            "Offset": group_input_2.outputs["Knob End Loc"],
        },
        attrs={"mode": "END_POINTS"},
    )

    reroute_56 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": mesh_line})

    duplicate_joints_on_parent = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Parent": hinge_joint_2.outputs["Parent"],
            "Child": hinge_joint_2.outputs["Child"],
            "Points": reroute_56,
        },
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_54,
            "False": reroute_78,
            "True": duplicate_joints_on_parent,
        },
    )

    reroute_52 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["Base Size"]}
    )

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": reroute_52})

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide_2})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": switch_2, "Translation": combine_xyz_7},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_1},
        attrs={"is_active_output": True},
    )


class MicrowaveFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    @gin.configurable(module="MicrowaveFactory")
    def sample_joint_parameters(
        cls,
        button_joint_stiffness_min: float = 2.0,
        button_joint_stiffness_max: float = 3.0,
        button_joint_damping_min: float = 0.0,
        button_joint_damping_max: float = 0.0,
        plate_joint_stiffness_min: float = 0.0,
        plate_joint_stiffness_max: float = 0.0,
        plate_joint_damping_min: float = 1.0,
        plate_joint_damping_max: float = 1.0,
        knob_joint_stiffness_min: float = 0.0,
        knob_joint_stiffness_max: float = 0.0,
        knob_joint_damping_min: float = 0.00001,
        knob_joint_damping_max: float = 0.00001,
        door_joint_stiffness_min: float = 120.0,
        door_joint_stiffness_max: float = 120.0,
        door_joint_damping_min: float = 5.0,
        door_joint_damping_max: float = 5.0,
    ):
        return {
            "button joint": {
                "stiffness": uniform(
                    button_joint_stiffness_min, button_joint_stiffness_max
                ),
                "damping": uniform(button_joint_damping_min, button_joint_damping_max),
            },
            "plate_joint": {
                "stiffness": uniform(
                    plate_joint_stiffness_min, plate_joint_stiffness_max
                ),
                "damping": uniform(plate_joint_damping_min, plate_joint_damping_max),
            },
            "knob_joint": {
                "stiffness": uniform(
                    knob_joint_stiffness_min, knob_joint_stiffness_max
                ),
                "damping": uniform(knob_joint_damping_min, knob_joint_damping_max),
            },
            "door_joint": {
                "stiffness": uniform(
                    door_joint_stiffness_min, door_joint_stiffness_max
                ),
                "damping": uniform(door_joint_damping_min, door_joint_damping_max),
                "friction": 10000,
            },
        }

    def sample_time(self):
        hour = randint(1, 13)
        minute = randint(0, 60)
        return f"{hour:02d}:{minute:02d}"

    def sample_parameters(self):
        import numpy as np

        from infinigen.assets.materials import (
            ceramic,
            metal,
        )
        from infinigen.assets.materials.plastic import plastic_rough
        from infinigen.core.util.color import hsv2rgba

        plate_material = weighted_sample(material_assignments.glasses)()()
        window_material = weighted_sample(
            material_assignments.appliance_front_glass
        )()()
        time_panel_material = weighted_sample(
            [
                (metal.BlackGlass, 10.0),
                (ceramic.Glass, 2.0),
            ]
        )()()

        handle_material = weighted_sample(material_assignments.appliance_handle)()()
        main_material = weighted_sample(material_assignments.kitchen_appliance_hard)()()
        knob_material = weighted_sample(material_assignments.plastics)()()

        hsv = None
        h = np.random.uniform(0, 1)
        s = np.random.uniform(0, 0.1)
        v = np.random.uniform(0, 0.1)
        hsv = (h, s, v)
        rgba = hsv2rgba(hsv)
        box_material = plastic_rough.PlasticRough().generate(base_color=rgba)

        door_material = main_material
        if uniform() < 0.2:
            door_material = weighted_sample(
                material_assignments.kitchen_appliance_hard
            )()()

        base_dimensions = (uniform(0.25, 0.35), uniform(0.30, 0.45), uniform(0.2, 0.3))
        radius = uniform(0, 0.03)

        horizontal_base_thickness = uniform(0.005, 0.04)
        vertical_base_thickness = uniform(0.005, 0.04)

        side_box_width = uniform(0.07, 0.12)
        panel_width = min(uniform(0.03, 0.1), side_box_width)
        panel_height = uniform(0.01, 0.03)
        panel_z_offset = -uniform(0.01, 0.02)

        time = self.sample_time()

        max_rad_plate = 0.35 * (
            min(base_dimensions[0], base_dimensions[1]) - 2 * horizontal_base_thickness
        )
        plate_radius = uniform(0.8, 1) * max_rad_plate

        door_thickness = uniform(0.02, 0.05)

        a = uniform()
        if a < 0.5:
            door_box_width_percentage = 1
        elif a < 0.8:
            door_box_width_percentage = uniform(0.6, 0.8)
        else:
            door_box_width_percentage = 0

        door_box_width = base_dimensions[1] * door_box_width_percentage
        door_box_height = uniform(0.6, 0.8) * base_dimensions[2]

        window_width = uniform(door_box_width * 0.6, door_box_width * 0.9)
        window_height = uniform(door_box_height * 0.6, door_box_height * 0.9)

        # is_vertical_handle = np.random.choice([True, False], p=[0.9, 0.1])
        is_vertical_handle = np.random.choice([True, False], p=[0.5, 0.5])

        handle_length = uniform(0.1, base_dimensions[2] * 0.8)

        has_handle = np.random.choice([True, False], p=[0.6, 0.4])
        handle_height = uniform(0.003, 0.03)
        handle_portrude = uniform(0.01, 0.05)
        handle_radius = uniform(0.005, 0.01)
        handle_type = randint(1, 2)

        if not is_vertical_handle:
            # Todo handle length
            dist_edge2glass_box = base_dimensions[2] / 2 - window_height / 2
            handle_width = uniform(dist_edge2glass_box * 0.1, dist_edge2glass_box * 0.9)
            handle_height = uniform(handle_portrude * 0.4, handle_portrude * 0.9)
            handle_height_vertical = uniform(
                0.05 * dist_edge2glass_box, 0.3 * dist_edge2glass_box
            )
            if handle_type == 0:
                handle_radius = handle_height_vertical
                handle_z_offset = uniform(
                    -dist_edge2glass_box + 2.5 * handle_height_vertical,
                    -handle_height_vertical * 0.5,
                )
            else:
                handle_width = handle_height_vertical
                handle_z_offset = uniform(
                    -dist_edge2glass_box + handle_height + handle_width,
                    -handle_width - handle_height,
                )

            handle_y_offset = -uniform(0, 0.05)
        else:
            handle_length = uniform(base_dimensions[2] * 0.3, base_dimensions[2] * 0.8)
            space_glass_right_side = base_dimensions[1] / 2 - window_width / 2
            handle_radius = min(
                uniform(0.2 * space_glass_right_side, 0.4 * space_glass_right_side),
                handle_length / 9,
            )
            handle_z_offset = uniform(
                0.8 * (-base_dimensions[2] / 2 + handle_length / 2),
                0.8 * (base_dimensions[2] / 2 - handle_length / 2),
            )

            if handle_type == 1:
                handle_width = min(
                    uniform(space_glass_right_side * 0.3, space_glass_right_side * 0.7),
                    0.05,
                )
                handle_y_offset = uniform(
                    -space_glass_right_side + handle_width / 2 + handle_width,
                    handle_width / 2,
                )
            else:
                handle_y_offset = uniform(
                    -space_glass_right_side + 2 * handle_radius, 0
                )
                handle_width = uniform(
                    space_glass_right_side * 0.1, space_glass_right_side * 0.9
                )

        num_knobs = randint(1, 4)
        dial_radius = uniform(0.008, 0.012)
        side_box_inner_box = uniform(0.5 * side_box_width, 0.9 * side_box_width)

        inbox_off_outwall = side_box_width / 2 - side_box_inner_box / 2
        dial_y_offset = uniform(
            -inbox_off_outwall - 1.1 * dial_radius,
            -inbox_off_outwall - side_box_inner_box + 1.1 * dial_radius,
        )

        size_box_inner_box_height = uniform(0.10, 0.15)

        knob_start = [0.0, 0.0, 0.0]
        knob_end = [0.0, 0.0, 0.0]
        knob_start[2] = uniform(
            3 * dial_radius, (size_box_inner_box_height / 2 - dial_radius) * 0.8
        )
        knob_end[2] = -knob_start[2]

        if num_knobs == 1:
            tmp = uniform()
            if tmp < 0.3:
                knob_start[2] = 0
            elif tmp < 0.6:
                knob_start[2] = -knob_start[2]
                knob_end[2] = knob_start[2]

        space_below_sidebox = base_dimensions[2] / 2 - size_box_inner_box_height / 2

        button_size = [
            0.03,
            uniform(side_box_width * 0.6, side_box_width * 0.8),
            uniform(space_below_sidebox * 0.3, space_below_sidebox * 0.5),
        ]

        button_offset_z = space_below_sidebox / 2
        button_offset_y = (side_box_width - button_size[1]) / 2

        params = {
            "Base Size": base_dimensions,
            "Radius": radius,
            "Horizontal Base Thickness": horizontal_base_thickness,
            "Vertical Base Thickness": vertical_base_thickness,
            "Side Box Width": side_box_width,
            "Panel Width": panel_width,
            "Panel Height": panel_height,
            "Panel Z Offset": panel_z_offset,
            "Time": time,
            "Side Box Inner Box Width": side_box_inner_box,
            "Side Box Inner Box Height": size_box_inner_box_height,
            "Plate Radius": plate_radius,
            "Door Thickness": door_thickness,
            "Door Box Width": door_box_width,
            "Door Box Height": door_box_height,
            "Window Width": window_width,
            "Window Height": window_height,
            "Window Radius": uniform(0, min(window_width, window_height) / 2),
            "Has Handle": has_handle,
            "Handle Type": handle_type,
            "Handle Length": handle_length,
            "Handle Radius": handle_radius,
            "Handle Protrude": handle_portrude,
            "Handle Width": handle_width,
            "Handle Height": handle_height,
            "Handle Y Offset": handle_y_offset,
            "Handle Z Offset": handle_z_offset,
            "Is Vertical Handle": is_vertical_handle,
            "Button Size": tuple(button_size),
            "Button Y Offset": -button_offset_y,
            "Button Z Offset": button_offset_z,
            "Is Knob": np.random.choice([True, False], p=[0.8, 0.2]),
            "Num Dials": num_knobs,
            "Dial Radius": dial_radius,
            "Dial Depth": uniform(0.005, 0.01),
            "Dial Y Offset": dial_y_offset,
            "Dial Z Offset": 0,
            "Has Knobs": np.random.choice([True, False], p=[0.8, 0.2]),
            "Knob Start Loc": tuple(knob_start),
            "Knob End Loc": tuple(knob_end),
            "Base Material": main_material,
            "Plate Material": plate_material,
            "Time Material": None,
            "Time Panel Material": time_panel_material,
            "Side Panel Material": main_material,
            "Side Panel Box Material": box_material,
            "Button Material": box_material,
            "Door Handle Material": handle_material,
            "Window Material": window_material,
            "Door Box Material": box_material,
            "Door Material": door_material,
            "Knob Material": knob_material,
        }

        """ For testing in Blender
        import bpy
        import importlib

        from infinigen.assets.sim_objects import microwave
        from infinigen.assets.composition import material_assignments
        from infinigen.assets.materials.dev import basic_bsdf

        importlib.reload(microwave)
        importlib.reload(material_assignments)
        importlib.reload(basic_bsdf)

        seeds = 10

        for seed in range(seeds):
            obj = microwave.MicrowaveFactory(seed).spawn_asset(0)
        """

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
