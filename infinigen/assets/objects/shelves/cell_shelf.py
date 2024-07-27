# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han

import bpy
import numpy as np
from numpy.random import normal, randint, uniform

from infinigen.assets.materials import metal
from infinigen.assets.materials.shelf_shaders import (
    shader_shelves_black_metallic,
    shader_shelves_black_metallic_sampler,
    shader_shelves_black_wood,
    shader_shelves_black_wood_sampler,
    shader_shelves_white,
    shader_shelves_white_metallic,
    shader_shelves_white_metallic_sampler,
    shader_shelves_white_sampler,
    shader_shelves_wood,
    shader_shelves_wood_sampler,
)
from infinigen.assets.objects.shelves.utils import nodegroup_tagged_cube
from infinigen.assets.utils.object import new_bbox
from infinigen.core import surface, tagging
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util.math import FixedSeed


@node_utils.to_nodegroup(
    "nodegroup_screw_head", singleton=False, type="GeometryNodeTree"
)
def nodegroup_screw_head(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder", input_kwargs={"Radius": 0.0050, "Depth": 0.0010}
    )

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Z", 0.5000),
            ("NodeSocketFloat", "leg", 0.5000),
            ("NodeSocketFloat", "X", 0.5000),
            ("NodeSocketFloat", "external", 0.5000),
            ("NodeSocketFloat", "depth", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["external"], 1: 0.0000}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["X"], 1: add},
        attrs={"operation": "SUBTRACT"},
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract}, attrs={"operation": "MULTIPLY"}
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add}, attrs={"operation": "MULTIPLY"}
    )

    add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Z"], 1: group_input.outputs["leg"]},
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 2.0000}, attrs={"operation": "MULTIPLY"}
    )

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: add_1, 1: multiply_2})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply, "Y": multiply_1, "Z": add_2}
    )

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cylinder.outputs["Mesh"], "Translation": combine_xyz},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["depth"], 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply, "Y": subtract_1, "Z": add_2}
    )

    transform_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_1,
        },
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_3, "Y": subtract_1, "Z": add_2}
    )

    transform_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_2,
        },
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_3, "Y": multiply_1, "Z": add_2}
    )

    transform_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_3,
        },
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_2, transform_3, transform_4, transform_5]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_3},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_base_frame", singleton=False, type="GeometryNodeTree"
)
def nodegroup_base_frame(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "leg_height", 0.5000),
            ("NodeSocketFloat", "leg_size", 0.5000),
            ("NodeSocketFloat", "depth", 0.5000),
            ("NodeSocketFloat", "bottom_x", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["leg_size"], 1: 0.0000}
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["leg_height"], 1: 0.0000}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": add, "Z": add_1}
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz,
            "Vertices X": 5,
            "Vertices Y": 5,
            "Vertices Z": 5,
        },
    )

    add_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["bottom_x"], 1: 0.0000}
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: add_2}, attrs={"operation": "MULTIPLY"}
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add}, attrs={"operation": "MULTIPLY"}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": multiply_1, "Z": multiply_2}
    )

    transform_2 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_1}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": multiply_3, "Y": multiply_1, "Z": multiply_2},
    )

    transform_3 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_2}
    )

    add_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["depth"], 1: 0.0000}
    )

    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: add_3, 1: 0.0000})

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_4, 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": subtract_1, "Z": multiply_2}
    )

    transform_4 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_3}
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": multiply_3, "Y": subtract_1, "Z": multiply_2},
    )

    transform_5 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_4}
    )

    multiply_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 2.0000}, attrs={"operation": "MULTIPLY"}
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_2, 1: multiply_4},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract_2, "Y": add, "Z": add}
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_5,
            "Vertices X": 5,
            "Vertices Y": 5,
            "Vertices Z": 5,
        },
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_1, 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply_1, "Z": subtract_3}
    )

    transform_6 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube_1, "Translation": combine_xyz_6}
    )

    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: add_3, 1: 0.0000})

    subtract_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_5, 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": subtract_4, "Z": subtract_3}
    )

    transform_7 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube_1, "Translation": combine_xyz_7}
    )

    subtract_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_3, 1: multiply_4},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": subtract_5, "Z": add}
    )

    cube_2 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_8,
            "Vertices X": 5,
            "Vertices Y": 5,
            "Vertices Z": 5,
        },
    )

    subtract_6 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_2, 1: add}, attrs={"operation": "SUBTRACT"}
    )

    multiply_5 = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract_6}, attrs={"operation": "MULTIPLY"}
    )

    multiply_6 = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract_5}, attrs={"operation": "MULTIPLY"}
    )

    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_6, 1: add})

    subtract_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_1, 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_5, "Y": add_6, "Z": subtract_7}
    )

    transform_8 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube_2, "Translation": combine_xyz_9}
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_5, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_10 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_7, "Y": add_6, "Z": subtract_7}
    )

    transform_9 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube_2, "Translation": combine_xyz_10},
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                transform_2,
                transform_3,
                transform_4,
                transform_5,
                transform_6,
                transform_7,
                transform_8,
                transform_9,
            ]
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_3},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_back_board", singleton=False, type="GeometryNodeTree"
)
def nodegroup_back_board(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "X", 0.0000),
            ("NodeSocketFloat", "Z", 0.5000),
            ("NodeSocketFloat", "leg", 0.5000),
            ("NodeSocketFloat", "external", 0.5000),
        ],
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Z"], 1: 0.0000})

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": group_input.outputs["X"], "Y": 0.01, "Z": add},
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_4,
            "Vertices X": 5,
            "Vertices Y": 5,
            "Vertices Z": 5,
        },
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: add}, attrs={"operation": "MULTIPLY"}
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply, 1: group_input.outputs["leg"]}
    )

    add_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1, 1: group_input.outputs["external"]}
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add_2})

    transform_6 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_5}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_6},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_attach_gadget", singleton=False, type="GeometryNodeTree"
)
def nodegroup_attach_gadget(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "z", 0.5000),
            ("NodeSocketFloat", "base_leg", 0.5000),
            ("NodeSocketFloat", "x", 0.5000),
            ("NodeSocketFloat", "thickness", 0.5000),
            ("NodeSocketFloat", "size", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["size"], 1: 0.0000}
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": 0.0010, "Z": add}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_4})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["x"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: group_input.outputs["thickness"]},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add}, attrs={"operation": "MULTIPLY"}
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["base_leg"], 1: group_input.outputs["z"]},
    )

    add_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1, 1: group_input.outputs["thickness"]}
    )

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: add_2, 1: -0.02})

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_3, 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_2, "Z": subtract_2}
    )

    transform_6 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_5}
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract_1, "Z": subtract_2}
    )

    transform_7 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_6}
    )

    join_geometry_5 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_6, transform_7]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_5},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_h_division_placement", singleton=False, type="GeometryNodeTree"
)
def nodegroup_h_division_placement(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "depth", 0.5000),
            ("NodeSocketFloat", "cell_size", 0.5000),
            ("NodeSocketFloat", "leg_height", 0.5000),
            ("NodeSocketFloat", "division_board_thickness", 0.5000),
            ("NodeSocketFloat", "external_board_thickness", 0.5000),
            ("NodeSocketFloat", "index", 0.5000),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["index"], 1: 0.0000}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add, 1: group_input.outputs["cell_size"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: -1.0000})

    add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["external_board_thickness"], 1: 0.0000},
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1, 1: add_2}, attrs={"operation": "MULTIPLY"}
    )

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: multiply_2})

    add_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["division_board_thickness"],
            1: group_input.outputs["leg_height"],
        },
    )

    multiply_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_2}, attrs={"operation": "MULTIPLY"}
    )

    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: add_4, 1: multiply_3})

    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: add_3, 1: add_5})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply, "Z": add_6}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Vector": combine_xyz},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_h_division_board", singleton=False, type="GeometryNodeTree"
)
def nodegroup_h_division_board(nw: NodeWrangler, tag_support=False):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "cell_size", 0.5000),
            ("NodeSocketFloat", "horizontal_cell_num", 0.5000),
            ("NodeSocketFloat", "division_board_thickness", 0.5000),
            ("NodeSocketFloat", "depth", 0.0000),
        ],
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["horizontal_cell_num"], 1: 0.0000},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add, 1: group_input.outputs["cell_size"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: -1.0000})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_1, 1: group_input.outputs["division_board_thickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: multiply_1})

    add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["division_board_thickness"], 1: 0.0000},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": add_2, "Y": group_input.outputs["depth"], "Z": add_3},
    )
    if tag_support:
        cube = nw.new_node(
            nodegroup_tagged_cube().name, input_kwargs={"Size": combine_xyz}
        )
    else:
        cube = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": combine_xyz,
                "Vertices X": 5,
                "Vertices Y": 5,
                "Vertices Z": 5,
            },
        )

    group_output = nw.new_node(
        Nodes.GroupOutput, input_kwargs={"Mesh": cube}, attrs={"is_active_output": True}
    )


@node_utils.to_nodegroup(
    "nodegroup_v_division_board_placement", singleton=False, type="GeometryNodeTree"
)
def nodegroup_v_division_board_placement(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "depth", 0.5000),
            ("NodeSocketFloat", "base_leg", 0.5000),
            ("NodeSocketFloat", "external_thickness", 0.5000),
            ("NodeSocketFloat", "side_z", 0.5000),
            ("NodeSocketFloat", "index", 0.5000),
            ("NodeSocketFloat", "h_cell_num", 0.5000),
            ("NodeSocketFloat", "division_thickness", 0.5000),
            ("NodeSocketFloat", "cell_size", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["h_cell_num"], 1: 0.0000}
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: -1.0000})

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={1: add_1}, attrs={"operation": "MULTIPLY"}
    )

    add_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["index"], 1: 0.0000}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: add_2},
        attrs={"operation": "SUBTRACT"},
    )

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: subtract})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_3, 1: group_input.outputs["division_thickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: add}, attrs={"operation": "MULTIPLY"}
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_2, 1: add_2},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["cell_size"], 1: subtract_1},
        attrs={"operation": "MULTIPLY"},
    )

    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: multiply_3})

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["base_leg"],
            1: group_input.outputs["external_thickness"],
        },
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["side_z"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: add_5, 1: multiply_5})

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_4, "Y": multiply_4, "Z": add_6}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Vector": combine_xyz_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_v_division_board", singleton=False, type="GeometryNodeTree"
)
def nodegroup_v_division_board(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "division_board_thickness", 0.0000),
            ("NodeSocketFloat", "depth", 0.0000),
            ("NodeSocketFloat", "cell_size", 0.5000),
            ("NodeSocketFloat", "vertical_cell_num", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["vertical_cell_num"], 1: 0.0000},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["cell_size"], 1: add},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 1.0000}, attrs={"operation": "SUBTRACT"}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: group_input.outputs["division_board_thickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: multiply_1})

    add_200 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["depth"], 1: -0.001}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["division_board_thickness"],
            "Y": add_200,
            "Z": add_1,
        },
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz,
            "Vertices X": 5,
            "Vertices Y": 5,
            "Vertices Z": 5,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": cube, "Value": add_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_top_bottom_board", singleton=False, type="GeometryNodeTree"
)
def nodegroup_top_bottom_board(nw: NodeWrangler, tag_support=False):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "base_leg_height", 0.5000),
            ("NodeSocketFloat", "horizontal_cell_num", 0.5000),
            ("NodeSocketFloat", "vertical_cell_num", 0.5000),
            ("NodeSocketFloat", "cell_size", 0.5000),
            ("NodeSocketFloat", "depth", 0.5000),
            ("NodeSocketFloat", "division_board_thickness", 0.5000),
            ("NodeSocketFloat", "external_board_thickness", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["external_board_thickness"], 1: 0.0000},
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 2.0000}, attrs={"operation": "MULTIPLY"}
    )

    add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["division_board_thickness"], 1: 0.0000},
    )

    add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["horizontal_cell_num"], 1: 0.0000},
    )

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: add_2, 1: -1.0000})

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1, 1: add_3}, attrs={"operation": "MULTIPLY"}
    )

    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: multiply_1})

    add_5 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["cell_size"], 1: 0.0000}
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_5, 1: add_2}, attrs={"operation": "MULTIPLY"}
    )

    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: add_4, 1: multiply_2})

    add_7 = nw.new_node(Nodes.Math, input_kwargs={0: add_6, 1: 0.0020})

    add_8 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["depth"], 1: 0.0000}
    )

    add_9 = nw.new_node(Nodes.Math, input_kwargs={0: add_8, 1: 0.0000})

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_7, "Y": add_9, "Z": add}
    )

    if tag_support:
        cube_1 = nw.new_node(
            nodegroup_tagged_cube().name, input_kwargs={"Size": combine_xyz_3}
        )
    else:
        cube_1 = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": combine_xyz_3,
                "Vertices X": 5,
                "Vertices Y": 5,
                "Vertices Z": 5,
            },
        )

    multiply_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_8}, attrs={"operation": "MULTIPLY"}
    )

    multiply_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: add}, attrs={"operation": "MULTIPLY"}
    )

    add_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_4, 1: group_input.outputs["base_leg_height"]},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply_3, "Z": add_10}
    )

    transform_2 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube_1, "Translation": combine_xyz}
    )

    add_11 = nw.new_node(Nodes.Math, input_kwargs={0: add_10, 1: add})

    add_12 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["vertical_cell_num"], 1: 0.0000},
    )

    multiply_5 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_12, 1: add_5}, attrs={"operation": "MULTIPLY"}
    )

    add_13 = nw.new_node(Nodes.Math, input_kwargs={0: add_11, 1: multiply_5})

    add_14 = nw.new_node(Nodes.Math, input_kwargs={0: add_12, 1: -1.0000})

    multiply_6 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1, 1: add_14}, attrs={"operation": "MULTIPLY"}
    )

    add_15 = nw.new_node(Nodes.Math, input_kwargs={0: add_13, 1: multiply_6})

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply_3, "Z": add_15}
    )

    transform = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube_1, "Translation": combine_xyz_1}
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_2, transform]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_1, "x": add_7},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_side_board", singleton=False, type="GeometryNodeTree"
)
def nodegroup_side_board(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "base_leg_height", 0.5000),
            ("NodeSocketFloat", "horizontal_cell_num", 0.5000),
            ("NodeSocketFloat", "vertical_cell_num", 0.5000),
            ("NodeSocketFloat", "cell_size", 0.5000),
            ("NodeSocketFloat", "depth", 0.5000),
            ("NodeSocketFloat", "division_thickness", 0.5000),
            ("NodeSocketFloat", "external_thickness", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["external_thickness"], 1: 0.0000},
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["depth"], 1: 0.0000}
    )

    add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["vertical_cell_num"], 1: 0.0000},
    )

    subtract = nw.new_node(
        Nodes.Math, input_kwargs={0: add_2, 1: 1.0000}, attrs={"operation": "SUBTRACT"}
    )

    add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["division_thickness"], 1: 0.0000},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: add_3},
        attrs={"operation": "MULTIPLY"},
    )

    add_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["cell_size"], 1: 0.0000}
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_2, 1: add_4}, attrs={"operation": "MULTIPLY"}
    )

    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: multiply_1})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": add_1, "Z": add_5}
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz,
            "Vertices X": 5,
            "Vertices Y": 5,
            "Vertices Z": 5,
        },
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_4, 1: group_input.outputs["horizontal_cell_num"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["horizontal_cell_num"], 1: 1.0000},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_3, 1: subtract_1},
        attrs={"operation": "MULTIPLY"},
    )

    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: multiply_3})

    add_7 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_2, 1: add_6})

    multiply_4 = nw.new_node(
        Nodes.Math, input_kwargs={1: add_7}, attrs={"operation": "MULTIPLY"}
    )

    multiply_5 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1}, attrs={"operation": "MULTIPLY"}
    )

    multiply_6 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_5}, attrs={"operation": "MULTIPLY"}
    )

    add_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_6, 1: group_input.outputs["base_leg_height"]},
    )

    add_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["external_thickness"], 1: add_8},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_4, "Y": multiply_5, "Z": add_9}
    )

    transform = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_1}
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_4, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_7, "Y": multiply_5, "Z": add_9}
    )

    transform_1 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_2}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform, transform_1]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


def geometry_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    base_leg_height = nw.new_node(Nodes.Value, label="base_leg_height")
    base_leg_height.outputs[0].default_value = kwargs["base_leg_height"]

    horizontal_cell_num = nw.new_node(Nodes.Integer, label="horizontal_cell_num")
    horizontal_cell_num.integer = kwargs["horizontal_cell_num"]

    vertical_cell_num = nw.new_node(Nodes.Integer, label="vertical_cell_num")
    vertical_cell_num.integer = kwargs["vertical_cell_num"]

    cell_size = nw.new_node(Nodes.Value, label="cell_size")
    cell_size.outputs[0].default_value = kwargs["cell_size"]

    depth = nw.new_node(Nodes.Value, label="depth")
    depth.outputs[0].default_value = kwargs["depth"]

    division_board_thickness = nw.new_node(
        Nodes.Value, label="division_board_thickness"
    )
    division_board_thickness.outputs[0].default_value = kwargs[
        "division_board_thickness"
    ]

    external_board_thickness = nw.new_node(
        Nodes.Value, label="external_board_thickness"
    )
    external_board_thickness.outputs[0].default_value = kwargs[
        "external_board_thickness"
    ]

    sideboard = nw.new_node(
        nodegroup_side_board().name,
        input_kwargs={
            "base_leg_height": base_leg_height,
            "horizontal_cell_num": horizontal_cell_num,
            "vertical_cell_num": vertical_cell_num,
            "cell_size": cell_size,
            "depth": depth,
            "division_thickness": division_board_thickness,
            "external_thickness": external_board_thickness,
        },
    )

    topbottomboard = nw.new_node(
        nodegroup_top_bottom_board(tag_support=kwargs.get("tag_support", False)).name,
        input_kwargs={
            "base_leg_height": base_leg_height,
            "horizontal_cell_num": horizontal_cell_num,
            "vertical_cell_num": vertical_cell_num,
            "cell_size": cell_size,
            "depth": depth,
            "division_board_thickness": division_board_thickness,
            "external_board_thickness": external_board_thickness,
        },
    )

    vdivisionboard = nw.new_node(
        nodegroup_v_division_board().name,
        input_kwargs={
            "division_board_thickness": division_board_thickness,
            "depth": depth,
            "cell_size": cell_size,
            "vertical_cell_num": vertical_cell_num,
        },
    )

    all_components = [sideboard, topbottomboard.outputs["Geometry"]]

    v_division_boards = []
    for i in range(1, kwargs["horizontal_cell_num"]):
        v_division_index = nw.new_node(Nodes.Integer, label="VDivisionIndex")
        v_division_index.integer = i

        vdivisionboardplacement = nw.new_node(
            nodegroup_v_division_board_placement().name,
            input_kwargs={
                "depth": depth,
                "base_leg": base_leg_height,
                "external_thickness": external_board_thickness,
                "side_z": vdivisionboard.outputs["Value"],
                "index": v_division_index,
                "h_cell_num": horizontal_cell_num,
                "division_thickness": division_board_thickness,
                "cell_size": cell_size,
            },
        )

        transform_1 = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": vdivisionboard.outputs["Mesh"],
                "Translation": vdivisionboardplacement,
            },
        )
        v_division_boards.append(transform_1)

    if len(v_division_boards) > 0:
        join_geometry_1 = nw.new_node(
            Nodes.JoinGeometry, input_kwargs={"Geometry": v_division_boards}
        )
        all_components.append(join_geometry_1)

    hdivisionboard = nw.new_node(
        nodegroup_h_division_board(tag_support=kwargs.get("tag_support", False)).name,
        input_kwargs={
            "cell_size": cell_size,
            "horizontal_cell_num": horizontal_cell_num,
            "division_board_thickness": division_board_thickness,
            "depth": depth,
        },
    )

    h_division_boards = []
    for j in range(1, kwargs["vertical_cell_num"]):
        h_division_index = nw.new_node(Nodes.Integer, label="HDivisionIndex")
        h_division_index.integer = j

        hdivisionplacement = nw.new_node(
            nodegroup_h_division_placement().name,
            input_kwargs={
                "depth": depth,
                "cell_size": cell_size,
                "leg_height": base_leg_height,
                "division_board_thickness": external_board_thickness,
                "external_board_thickness": division_board_thickness,
                "index": h_division_index,
            },
        )

        transform = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": hdivisionboard,
                "Translation": hdivisionplacement,
            },
        )
        h_division_boards.append(transform)

    if len(h_division_boards) > 0:
        join_geometry = nw.new_node(
            Nodes.JoinGeometry, input_kwargs={"Geometry": h_division_boards}
        )
        all_components.append(join_geometry)

    if kwargs["has_backboard"]:
        backboard = nw.new_node(
            nodegroup_back_board().name,
            input_kwargs={
                "X": topbottomboard.outputs["x"],
                "Z": vdivisionboard.outputs["Value"],
                "leg": base_leg_height,
                "external": external_board_thickness,
            },
        )
        all_components.append(backboard)
    else:
        attach_square_size = nw.new_node(Nodes.Value, label="attach_square_size")
        attach_square_size.outputs[0].default_value = kwargs["attachment_size"]

        attachgadget = nw.new_node(
            nodegroup_attach_gadget().name,
            input_kwargs={
                "z": vdivisionboard.outputs["Value"],
                "base_leg": base_leg_height,
                "x": topbottomboard.outputs["x"],
                "thickness": external_board_thickness,
                "size": attach_square_size,
            },
        )
        all_components.append(attachgadget)

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": all_components}
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": join_geometry_4}
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": realize_instances,
            "Material": surface.shaderfunc_to_material(kwargs["wood_material"]),
        },
    )

    base_leg_size = nw.new_node(Nodes.Value, label="base_leg_size")
    base_leg_size.outputs[0].default_value = kwargs["base_leg_size"]

    merge_components = [set_material_1]
    if kwargs["has_base_frame"]:
        baseframe = nw.new_node(
            nodegroup_base_frame().name,
            input_kwargs={
                "leg_height": base_leg_height,
                "leg_size": base_leg_size,
                "depth": depth,
                "bottom_x": topbottomboard.outputs["x"],
            },
        )

        realize_instances_1 = nw.new_node(
            Nodes.RealizeInstances, input_kwargs={"Geometry": baseframe}
        )

        set_material = nw.new_node(
            Nodes.SetMaterial,
            input_kwargs={
                "Geometry": realize_instances_1,
                "Material": surface.shaderfunc_to_material(kwargs["base_material"]),
            },
        )
        merge_components.append(set_material)

    screwhead = nw.new_node(
        nodegroup_screw_head().name,
        input_kwargs={
            "Z": vdivisionboard.outputs["Value"],
            "leg": base_leg_height,
            "X": topbottomboard.outputs["x"],
            "external": external_board_thickness,
            "depth": depth,
        },
    )

    realize_instances_2 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": screwhead}
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": realize_instances_2,
            "Material": surface.shaderfunc_to_material(metal.get_shader()),
        },
    )
    merge_components.append(set_material_2)

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": merge_components}
    )

    triangulate = nw.new_node(
        "GeometryNodeTriangulate", input_kwargs={"Mesh": join_geometry_2}
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": triangulate, "Rotation": (0.0000, 0.0000, -1.5708)},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform},
        attrs={"is_active_output": True},
    )


class CellShelfBaseFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(CellShelfBaseFactory, self).__init__(factory_seed, coarse=coarse)
        with FixedSeed(factory_seed):
            self.params = self.sample_params()
            self.params = self.get_asset_params(self.params)

    def get_asset_params(self, params):
        if params is None:
            params = {}

        if params.get("depth", None) is None:
            params["depth"] = np.clip(normal(0.39, 0.05), 0.29, 0.49)
        if params.get("cell_size", None) is None:
            params["cell_size"] = np.clip(normal(0.335, 0.03), 0.26, 0.40)
        if params.get("vertical_cell_num", None) is None:
            params["vertical_cell_num"] = randint(1, 7)
        if params.get("horizontal_cell_num", None) is None:
            params["horizontal_cell_num"] = randint(1, 7)
        if params.get("division_board_thickness", None) is None:
            params["division_board_thickness"] = np.clip(
                normal(0.015, 0.005), 0.008, 0.022
            )
        if params.get("external_board_thickness", None) is None:
            params["external_board_thickness"] = np.clip(
                normal(0.04, 0.005), 0.028, 0.052
            )
        if params.get("has_backboard", None) is None:
            params["has_backboard"] = False
        if params.get("has_base_frame", None) is None:
            params["has_base_frame"] = np.random.choice([True, False], p=[0.4, 0.6])
        if params["has_base_frame"]:
            if params.get("base_leg_height", None) is None:
                params["base_leg_height"] = np.clip(normal(0.174, 0.03), 0.1, 0.25)
            if params.get("base_leg_size", None) is None:
                params["base_leg_size"] = np.clip(normal(0.035, 0.007), 0.02, 0.05)
            if params.get("base_material", None) is None:
                params["base_material"] = np.random.choice(
                    ["black", "white"], p=[0.4, 0.6]
                )
        else:
            params["base_leg_height"] = 0.0
            params["base_leg_size"] = 0.0
            params["base_material"] = "white"
        if params.get("attachment_size", None) is None:
            params["attachment_size"] = np.clip(normal(0.05, 0.02), 0.02, 0.1)
        if params.get("wood_material", None) is None:
            params["wood_material"] = np.random.choice(
                ["black_wood", "white", "wood"], p=[0.3, 0.2, 0.5]
            )
        params["tag_support"] = True
        params = self.get_material_func(params, randomness=True)
        return params

    def get_material_func(self, params, randomness=True):
        if params["wood_material"] == "white":
            if randomness:
                params["wood_material"] = lambda x: shader_shelves_white(
                    x, **shader_shelves_white_sampler()
                )
            else:
                params["wood_material"] = shader_shelves_white
        elif params["wood_material"] == "black_wood":
            if randomness:
                params["wood_material"] = lambda x: shader_shelves_black_wood(
                    x, **shader_shelves_black_wood_sampler()
                )
            else:
                params["wood_material"] = shader_shelves_black_wood
        elif params["wood_material"] == "wood":
            if randomness:
                params["wood_material"] = lambda x: shader_shelves_wood(
                    x, **shader_shelves_wood_sampler()
                )
            else:
                params["wood_material"] = shader_shelves_wood
        else:
            raise NotImplementedError

        if params["base_material"] == "white":
            if randomness:
                params["base_material"] = lambda x: shader_shelves_white_metallic(
                    x, **shader_shelves_white_metallic_sampler()
                )
            else:
                params["base_material"] = shader_shelves_white_metallic
        elif params["base_material"] == "black":
            if randomness:
                params["base_material"] = lambda x: shader_shelves_black_metallic(
                    x, **shader_shelves_black_metallic_sampler()
                )
            else:
                params["base_material"] = shader_shelves_black_metallic
        else:
            raise NotImplementedError

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

        obj_params = self.params
        surface.add_geomod(
            obj, geometry_nodes, attributes=[], input_kwargs=obj_params, apply=True
        )
        tagging.tag_system.relabel_obj(obj)

        return obj


class CellShelfFactory(CellShelfBaseFactory):
    def sample_params(self):
        params = dict()
        params["Dimensions"] = (
            uniform(0.3, 0.45),
            uniform(2 * 0.35, 6 * 0.35),
            uniform(1 * 0.35, 6 * 0.35),
        )
        h_cell_num = int(params["Dimensions"][1] / 0.35)
        params["cell_size"] = params["Dimensions"][1] / h_cell_num
        params["horizontal_cell_num"] = h_cell_num
        params["vertical_cell_num"] = max(
            int(params["Dimensions"][2] / params["cell_size"]), 1
        )
        params["depth"] = params["Dimensions"][0]
        params["has_base_frame"] = False
        params["Dimensions"] = list(params["Dimensions"])
        params["Dimensions"][2] = params["vertical_cell_num"] * params["cell_size"]
        return params

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        x, y, z = (
            self.params["Dimensions"][0],
            self.params["Dimensions"][1],
            self.params["Dimensions"][2],
        )
        return new_bbox(
            0,
            x,
            -y / 2 * 1.1,
            y / 2 * 1.1,
            0,
            z
            + (self.params["vertical_cell_num"] - 1)
            * self.params["division_board_thickness"]
            + 2 * self.params["external_board_thickness"],
        )


class TVStandFactory(CellShelfFactory):
    def sample_params(
        self,
    ):  # TODO HACK copied code just following the pattern to get this working
        params = dict()
        params["Dimensions"] = (
            uniform(0.3, 0.45),
            uniform(2 * 0.35, 6 * 0.35),
            uniform(0.3, 0.5),
        )
        h_cell_num = int(params["Dimensions"][1] / 0.35)
        params["cell_size"] = params["Dimensions"][1] / h_cell_num
        params["horizontal_cell_num"] = h_cell_num
        params["vertical_cell_num"] = max(
            int(params["Dimensions"][2] / params["cell_size"]), 1
        )
        params["depth"] = params["Dimensions"][0]
        params["has_base_frame"] = False
        params["Dimensions"] = list(params["Dimensions"])
        params["Dimensions"][2] = params["vertical_cell_num"] * params["cell_size"]
        return params
