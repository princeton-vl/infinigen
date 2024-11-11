# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.materials.shelf_shaders import (
    shader_glass,
    shader_shelves_black_wood,
    shader_shelves_black_wood_sampler,
    shader_shelves_white,
    shader_shelves_white_sampler,
    shader_shelves_wood,
    shader_shelves_wood_sampler,
)
from infinigen.core import surface, tagging
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory


@node_utils.to_nodegroup(
    "nodegroup_node_group", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (0.0120, 0.00060, 0.0400)})

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 64, "Radius": 0.0100, "Depth": 0.00050},
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": (0.0050, 0.0000, 0.0000),
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": (0.0200, 0.0006, 0.0120)}
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube_1, "Translation": (0.0080, 0.0000, 0.0000)},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [cube, transform, transform_1]}
    )

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "attach_height", 0.1000),
            ("NodeSocketFloat", "door_width", 0.5000),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["door_width"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: 0.0181},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": subtract, "Z": group_input.outputs["attach_height"]},
    )

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_1, "Translation": combine_xyz},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_2},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_knob_handle", singleton=False, type="GeometryNodeTree"
)
def nodegroup_knob_handle(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Radius", 0.0100),
            ("NodeSocketFloat", "thickness_1", 0.5000),
            ("NodeSocketFloat", "thickness_2", 0.5000),
            ("NodeSocketFloat", "length", 0.5000),
            ("NodeSocketFloat", "knob_mid_height", 0.0000),
            ("NodeSocketFloat", "edge_width", 0.5000),
            ("NodeSocketFloat", "door_width", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["thickness_2"],
            1: group_input.outputs["thickness_1"],
        },
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: group_input.outputs["length"]}
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 64,
            "Radius": group_input.outputs["Radius"],
            "Depth": add_1,
        },
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["door_width"],
            1: group_input.outputs["edge_width"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: -0.005})

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": add_2,
            "Y": multiply_1,
            "Z": group_input.outputs["knob_mid_height"],
        },
    )

    transform_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_6,
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_6},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_mid_board", singleton=False, type="GeometryNodeTree"
)
def nodegroup_mid_board(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "height", 0.5000),
            ("NodeSocketFloat", "thickness", 0.5000),
            ("NodeSocketFloat", "width", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["width"], 1: -0.0001}
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["thickness"], 1: 0.0000}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["height"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_k = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1, 1: 0.5000}, attrs={"operation": "MULTIPLY"}
    )

    add_k = nw.new_node(Nodes.Math, input_kwargs={0: multiply_k, 1: 0.004})

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: -0.0001})

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": add_1, "Z": add_2}
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_3,
            "Vertices X": 5,
            "Vertices Y": 5,
            "Vertices Z": 5,
        },
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": add_k, "Z": multiply_1}
    )

    transform_4 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_4}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_4,
            "Material": surface.shaderfunc_to_material(kwargs["material"][0]),
        },
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": add_1, "Z": add_2}
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_7,
            "Vertices X": 5,
            "Vertices Y": 5,
            "Vertices Z": 5,
        },
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: 1.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": add_k, "Z": multiply_2}
    )

    transform_7 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube_1, "Translation": combine_xyz_8}
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_7,
            "Material": surface.shaderfunc_to_material(kwargs["material"][1]),
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, set_material_1]}
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": join_geometry_1}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": realize_instances, "mid_height": multiply},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_mid_board_001", singleton=False, type="GeometryNodeTree"
)
def nodegroup_mid_board_001(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "height", 0.5000),
            ("NodeSocketFloat", "thickness", 0.5000),
            ("NodeSocketFloat", "width", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["width"], 1: -0.0001}
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["thickness"], 1: 0.0000}
    )

    multiply_k = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1, 1: 0.5000}, attrs={"operation": "MULTIPLY"}
    )

    add_k = nw.new_node(Nodes.Math, input_kwargs={0: multiply_k, 1: 0.004})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["height"], 1: 1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: -0.0001})

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": add_1, "Z": add_2}
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_3,
            "Vertices X": 5,
            "Vertices Y": 5,
            "Vertices Z": 5,
        },
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": add_k, "Z": multiply_1}
    )

    transform_4 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_4}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_4,
            "Material": surface.shaderfunc_to_material(kwargs["material"][0]),
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": set_material}
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": join_geometry_1}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": realize_instances, "mid_height": multiply},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_double_rampled_edge", singleton=False, type="GeometryNodeTree"
)
def nodegroup_double_rampled_edge(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "height", 0.5000),
            ("NodeSocketFloat", "thickness_2", 0.5000),
            ("NodeSocketFloat", "width", 0.5000),
            ("NodeSocketFloat", "thickness_1", 0.5000),
            ("NodeSocketFloat", "ramp_angle", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["height"], 1: 0.0000}
    )

    combine_xyz_10 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add})

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz_10})

    curve_circle = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 3, "Radius": 0.0100}
    )

    endpoint_selection = nw.new_node(
        Nodes.EndpointSelection, input_kwargs={"End Size": 0}
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["width"], 1: 0.0000}
    )

    add_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["ramp_angle"], 1: 0.0000}
    )

    tangent = nw.new_node(
        Nodes.Math, input_kwargs={0: add_2}, attrs={"operation": "TANGENT"}
    )

    add_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["thickness_2"], 1: 0.0000}
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: tangent, 1: add_3}, attrs={"operation": "MULTIPLY"}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 2.0000, 1: multiply},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_1, 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract}, attrs={"operation": "MULTIPLY"}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_2, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    add_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["thickness_1"], 1: 0.0000}
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_3, "Y": add_4}
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": curve_circle.outputs["Curve"],
            "Selection": endpoint_selection,
            "Position": combine_xyz_7,
        },
    )

    endpoint_selection_1 = nw.new_node(
        Nodes.EndpointSelection, input_kwargs={"Start Size": 0}
    )

    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: add_4, 1: add_3})

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_3, "Y": add_5}
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position,
            "Selection": endpoint_selection_1,
            "Position": combine_xyz_8,
        },
    )

    index = nw.new_node(Nodes.Index)

    less_than = nw.new_node(
        Nodes.Math, input_kwargs={0: index, 1: 1.0100}, attrs={"operation": "LESS_THAN"}
    )

    greater_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: index, 1: 0.9900},
        attrs={"operation": "GREATER_THAN"},
    )

    op_and = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: less_than, 1: greater_than}
    )

    multiply_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1}, attrs={"operation": "MULTIPLY"}
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_4, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_5, "Y": add_4}
    )

    set_position_2 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position_1,
            "Selection": op_and,
            "Position": combine_xyz_9,
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": set_position_2,
            "Fill Caps": True,
        },
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_1, "Y": add_4, "Z": add}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    multiply_6 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_4}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_6})

    transform = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_2}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": add_3, "Z": add}
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_1})

    multiply_7 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_3}, attrs={"operation": "MULTIPLY"}
    )

    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: add_4, 1: multiply_7})

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": add_6})

    transform_1 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube_1, "Translation": combine_xyz_3}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform, transform_1]}
    )

    multiply_8 = nw.new_node(
        Nodes.Math, input_kwargs={0: add}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_11 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_8})

    transform_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Translation": combine_xyz_11},
    )

    combine_xyz_12 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add})

    curve_line_1 = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz_12})

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": set_position_2, "Scale": (-1.0000, 1.0000, 1.0000)},
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line_1,
            "Profile Curve": transform_2,
            "Fill Caps": True,
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [curve_to_mesh, transform_4, curve_to_mesh_1]},
    )

    merge_by_distance = nw.new_node(
        Nodes.MergeByDistance,
        input_kwargs={"Geometry": join_geometry_1, "Distance": 0.0001},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": merge_by_distance}
    )

    subdivide_mesh = nw.new_node(
        Nodes.SubdivideMesh, input_kwargs={"Mesh": realize_instances, "Level": 4}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": subdivide_mesh},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_ramped_edge", singleton=False, type="GeometryNodeTree"
)
def nodegroup_ramped_edge(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "height", 0.5000),
            ("NodeSocketFloat", "thickness_2", 0.5000),
            ("NodeSocketFloat", "width", 0.5000),
            ("NodeSocketFloat", "thickness_1", 0.5000),
            ("NodeSocketFloat", "ramp_angle", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["height"], 1: 0.0000}
    )

    combine_xyz_10 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add})

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz_10})

    curve_circle = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 3, "Radius": 0.0100}
    )

    endpoint_selection = nw.new_node(
        Nodes.EndpointSelection, input_kwargs={"End Size": 0}
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["width"], 1: 0.0000}
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1}, attrs={"operation": "MULTIPLY"}
    )

    add_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["ramp_angle"], 1: 0.0000}
    )

    tangent = nw.new_node(
        Nodes.Math, input_kwargs={0: add_2}, attrs={"operation": "TANGENT"}
    )

    add_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["thickness_2"], 1: 0.0000}
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: tangent, 1: add_3}, attrs={"operation": "MULTIPLY"}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_1, 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: subtract},
        attrs={"operation": "SUBTRACT"},
    )

    add_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["thickness_1"], 1: 0.0000}
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract_1, "Y": add_4}
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": curve_circle.outputs["Curve"],
            "Selection": endpoint_selection,
            "Position": combine_xyz_7,
        },
    )

    endpoint_selection_1 = nw.new_node(
        Nodes.EndpointSelection, input_kwargs={"Start Size": 0}
    )

    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: add_4, 1: add_3})

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract_1, "Y": add_5}
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position,
            "Selection": endpoint_selection_1,
            "Position": combine_xyz_8,
        },
    )

    index = nw.new_node(Nodes.Index)

    less_than = nw.new_node(
        Nodes.Math, input_kwargs={0: index, 1: 1.0100}, attrs={"operation": "LESS_THAN"}
    )

    greater_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: index, 1: 0.9900},
        attrs={"operation": "GREATER_THAN"},
    )

    op_and = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: less_than, 1: greater_than}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_2, "Y": add_4}
    )

    set_position_2 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position_1,
            "Selection": op_and,
            "Position": combine_xyz_9,
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": set_position_2,
            "Fill Caps": True,
        },
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_1, "Y": add_4, "Z": add}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    multiply_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_4}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_3})

    transform = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_2}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": add_3, "Z": add}
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_1})

    multiply_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_1}, attrs={"operation": "MULTIPLY"}
    )

    multiply_5 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_3}, attrs={"operation": "MULTIPLY"}
    )

    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: add_4, 1: multiply_5})

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_4, "Y": add_6}
    )

    transform_1 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube_1, "Translation": combine_xyz_3}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform, transform_1]}
    )

    multiply_6 = nw.new_node(
        Nodes.Math, input_kwargs={0: add}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_11 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_6})

    transform_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Translation": combine_xyz_11},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [curve_to_mesh, transform_4]}
    )

    merge_by_distance = nw.new_node(
        Nodes.MergeByDistance,
        input_kwargs={"Geometry": join_geometry_1, "Distance": 0.0001},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": merge_by_distance}
    )

    subdivide_mesh = nw.new_node(
        Nodes.SubdivideMesh, input_kwargs={"Mesh": realize_instances, "Level": 4}
    )

    multiply_7 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1, 1: -0.5000}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_7})

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": subdivide_mesh, "Translation": combine_xyz_4},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_2},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_panel_edge_frame", singleton=False, type="GeometryNodeTree"
)
def nodegroup_panel_edge_frame(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "vertical_edge", None),
            ("NodeSocketFloat", "door_width", 0.5000),
            ("NodeSocketFloat", "door_height", 0.0000),
            ("NodeSocketGeometry", "horizontal_edge", None),
        ],
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["door_width"], 2: 0.0010},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_add, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    transform_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": group_input.outputs["horizontal_edge"],
            "Translation": (0.0000, -0.0001, 0.0000),
            "Scale": (0.9999, 1.0000, 1.0000),
        },
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_add, 1: 1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: -0.0001})

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["door_height"], 1: 0.0001}
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": add, "Z": add_1})

    transform_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_7,
            "Translation": combine_xyz_2,
            "Rotation": (0.0000, -1.5708, 0.0000),
        },
    )

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: 0.0001})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": add_2})

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_7,
            "Translation": combine_xyz_1,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_add})

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": group_input.outputs["vertical_edge"],
            "Translation": combine_xyz,
        },
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Scale": (-1.0000, 1.0000, 1.0000)},
    )

    # transform_1 = nw.new_node(Nodes.FlipFaces, input_kwargs={'Mesh': transform_1})

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_3, transform_2, transform_1, transform]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Value": multiply, "Geometry": join_geometry_1},
        attrs={"is_active_output": True},
    )


def geometry_door_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    door_height = nw.new_node(Nodes.Value, label="door_height")
    door_height.outputs[0].default_value = kwargs["door_height"]

    door_edge_thickness_2 = nw.new_node(Nodes.Value, label="door_edge_thickness_2")
    door_edge_thickness_2.outputs[0].default_value = kwargs["edge_thickness_2"]

    door_edge_width = nw.new_node(Nodes.Value, label="door_edge_width")
    door_edge_width.outputs[0].default_value = kwargs["edge_width"]

    door_edge_thickness_1 = nw.new_node(Nodes.Value, label="door_edge_thickness_1")
    door_edge_thickness_1.outputs[0].default_value = kwargs["edge_thickness_1"]

    door_edge_ramp_angle = nw.new_node(Nodes.Value, label="door_edge_ramp_angle")
    door_edge_ramp_angle.outputs[0].default_value = kwargs["edge_ramp_angle"]

    ramped_edge = nw.new_node(
        nodegroup_ramped_edge().name,
        input_kwargs={
            "height": door_height,
            "thickness_2": door_edge_thickness_2,
            "width": door_edge_width,
            "thickness_1": door_edge_thickness_1,
            "ramp_angle": door_edge_ramp_angle,
        },
    )

    door_width = nw.new_node(Nodes.Value, label="door_width")
    door_width.outputs[0].default_value = kwargs["door_width"]

    ramped_edge_1 = nw.new_node(
        nodegroup_ramped_edge().name,
        input_kwargs={
            "height": door_width,
            "thickness_2": door_edge_thickness_2,
            "width": door_edge_width,
            "thickness_1": door_edge_thickness_1,
            "ramp_angle": door_edge_ramp_angle,
        },
    )

    panel_edge_frame = nw.new_node(
        nodegroup_panel_edge_frame().name,
        input_kwargs={
            "vertical_edge": ramped_edge,
            "door_width": door_width,
            "door_height": door_height,
            "horizontal_edge": ramped_edge_1,
        },
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: panel_edge_frame.outputs["Value"], 1: 0.0001}
    )

    mid_board_thickness = nw.new_node(Nodes.Value, label="mid_board_thickness")
    mid_board_thickness.outputs[0].default_value = kwargs["board_thickness"]

    if kwargs["has_mid_ramp"]:
        mid_board = nw.new_node(
            nodegroup_mid_board(material=kwargs["panel_material"]).name,
            input_kwargs={
                "height": door_height,
                "thickness": mid_board_thickness,
                "width": door_width,
            },
        )
    else:
        mid_board = nw.new_node(
            nodegroup_mid_board_001(material=kwargs["panel_material"]).name,
            input_kwargs={
                "height": door_height,
                "thickness": mid_board_thickness,
                "width": door_width,
            },
        )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": add, "Y": -0.0001, "Z": mid_board.outputs["mid_height"]},
    )

    frame = [panel_edge_frame.outputs["Geometry"]]
    if kwargs["has_mid_ramp"]:
        double_rampled_edge = nw.new_node(
            nodegroup_double_rampled_edge().name,
            input_kwargs={
                "height": door_width,
                "thickness_2": door_edge_thickness_2,
                "width": door_edge_width,
                "thickness_1": door_edge_thickness_1,
                "ramp_angle": door_edge_ramp_angle,
            },
        )

        transform_5 = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": double_rampled_edge,
                "Translation": combine_xyz_5,
                "Rotation": (0.0000, 1.5708, 0.0000),
            },
        )
        frame.append(transform_5)

    knob_raduis = nw.new_node(Nodes.Value, label="knob_raduis")
    knob_raduis.outputs[0].default_value = kwargs["knob_R"]

    know_length = nw.new_node(Nodes.Value, label="know_length")
    know_length.outputs[0].default_value = kwargs["knob_length"]

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: door_height}, attrs={"operation": "MULTIPLY"}
    )

    knob_handle = nw.new_node(
        nodegroup_knob_handle().name,
        input_kwargs={
            "Radius": knob_raduis,
            "thickness_1": door_edge_thickness_1,
            "thickness_2": door_edge_thickness_2,
            "length": know_length,
            "knob_mid_height": multiply,
            "edge_width": door_edge_width,
            "door_width": door_width,
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": frame + [knob_handle]}
    )

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Material": surface.shaderfunc_to_material(kwargs["frame_material"]),
        },
    )

    geos = [set_material_3, mid_board.outputs["Geometry"]]
    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={"Geometry": geos})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: door_width, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply})

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Translation": combine_xyz},
    )

    realize_instances_1 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": transform}
    )

    triangulate = nw.new_node(
        "GeometryNodeTriangulate", input_kwargs={"Mesh": realize_instances_1}
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": triangulate,
            "Scale": (-1.0 if kwargs["door_left_hinge"] else 1.0, 1.0000, 1.0000),
        },
    )

    if kwargs["door_left_hinge"]:
        transform_1 = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": transform_1})

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_1, "Rotation": (0.0000, 0.0000, -1.5708)},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_2},
        attrs={"is_active_output": True},
    )


class CabinetDoorBaseFactory(AssetFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(CabinetDoorBaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.params = {}

    def get_asset_params(self, i=0):
        params = self.params.copy()
        if params.get("door_height", None) is None:
            params["door_height"] = uniform(0.7, 2.2)
        if params.get("door_width", None) is None:
            params["door_width"] = uniform(0.3, 0.4)
        if params.get("edge_thickness_1", None) is None:
            params["edge_thickness_1"] = uniform(0.01, 0.018)
        if params.get("edge_width", None) is None:
            params["edge_width"] = uniform(0.03, 0.05)
        if params.get("edge_thickness_2", None) is None:
            params["edge_thickness_2"] = uniform(0.005, 0.008)
        if params.get("edge_ramp_angle", None) is None:
            params["edge_ramp_angle"] = uniform(0.6, 0.8)
        params["board_thickness"] = params["edge_thickness_1"] - 0.005
        if params.get("knob_R", None) is None:
            params["knob_R"] = uniform(0.003, 0.006)
        if params.get("knob_length", None) is None:
            params["knob_length"] = uniform(0.018, 0.035)
        if params.get("attach_height", None) is None:
            gap = uniform(0.05, 0.15)
            params["attach_height"] = [gap, params["door_height"] - gap]
        if params.get("has_mid_ramp", None) is None:
            params["has_mid_ramp"] = np.random.choice([True, False], p=[0.6, 0.4])
        if params.get("door_left_hinge", None) is None:
            params["door_left_hinge"] = False

        if params.get("frame_material", None) is None:
            params["frame_material"] = np.random.choice(
                ["white", "black_wood", "wood"], p=[0.5, 0.2, 0.3]
            )
        if params.get("panel_material", None) is None:
            if params["has_mid_ramp"]:
                lower_mat = np.random.choice(
                    [params["frame_material"], "glass"], p=[0.7, 0.3]
                )
                upper_mat = np.random.choice([lower_mat, "glass"], p=[0.6, 0.4])
                params["panel_material"] = [lower_mat, upper_mat]
            else:
                params["panel_material"] = [params["frame_material"]]

        params = self.get_material_func(params)
        return params

    def get_material_func(self, params, randomness=True):
        white_wood_params = shader_shelves_white_sampler()
        black_wood_params = shader_shelves_black_wood_sampler()
        normal_wood_params = shader_shelves_wood_sampler()
        if params["frame_material"] == "white":
            if randomness:
                params["frame_material"] = lambda x: shader_shelves_white(
                    x, **white_wood_params
                )
            else:
                params["frame_material"] = shader_shelves_white
        elif params["frame_material"] == "black_wood":
            if randomness:
                params["frame_material"] = lambda x: shader_shelves_black_wood(
                    x, **black_wood_params, z_axis_texture=True
                )
            else:
                params["frame_material"] = lambda x: shader_shelves_black_wood(
                    x, z_axis_texture=True
                )
        elif params["frame_material"] == "wood":
            if randomness:
                params["frame_material"] = lambda x: shader_shelves_wood(
                    x, **normal_wood_params, z_axis_texture=True
                )
            else:
                params["frame_material"] = lambda x: shader_shelves_wood(
                    x, z_axis_texture=True
                )

        materials = []
        if not isinstance(params["panel_material"], list):
            params["panel_material"] = [params["board_material"]]
        for mat in params["panel_material"]:
            if mat == "white":
                if randomness:

                    def mat(x):
                        return shader_shelves_white(x, **white_wood_params)
                else:
                    mat = shader_shelves_white
            elif mat == "black_wood":
                if randomness:

                    def mat(x):
                        return shader_shelves_black_wood(
                            x, **black_wood_params, z_axis_texture=True
                        )
                else:

                    def mat(x):
                        return shader_shelves_black_wood(x, z_axis_texture=True)
            elif mat == "wood":
                if randomness:

                    def mat(x):
                        return shader_shelves_wood(
                            x, **normal_wood_params, z_axis_texture=True
                        )
                else:

                    def mat(x):
                        return shader_shelves_wood(x, z_axis_texture=True)
            elif mat == "glass":
                if randomness:

                    def mat(x):
                        return shader_glass(x)
                else:
                    mat = shader_glass
            materials.append(mat)
        params["panel_material"] = materials
        return params

    def create_asset(self, i=0, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=1,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        obj = bpy.context.active_object

        obj_params = self.get_asset_params(i)
        surface.add_geomod(
            obj, geometry_door_nodes, apply=True, attributes=[], input_kwargs=obj_params
        )
        tagging.tag_system.relabel_obj(obj)

        if params.get("ret_params", False):
            return obj, obj_params

        return obj
