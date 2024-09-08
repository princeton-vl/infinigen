# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han

import bpy
import numpy as np
from numpy.random import normal, uniform

from infinigen.assets.materials.shelf_shaders import get_shelf_material
from infinigen.assets.objects.shelves.utils import nodegroup_tagged_cube
from infinigen.core import surface, tagging
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory


@node_utils.to_nodegroup(
    "nodegroup_attach_gadget", singleton=False, type="GeometryNodeTree"
)
def nodegroup_attach_gadget(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "division_thickness", 0.5000),
            ("NodeSocketFloat", "height", 0.5000),
            ("NodeSocketFloat", "attach_thickness", 0.5000),
            ("NodeSocketFloat", "attach_width", 0.5000),
            ("NodeSocketFloat", "attach_back_len", 0.5000),
            ("NodeSocketFloat", "attach_top_len", 0.5000),
            ("NodeSocketFloat", "depth", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["attach_width"], 1: 0.0000}
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["attach_top_len"], 1: 0.0000}
    )

    add_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["attach_thickness"], 1: 0.0000}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": add_1, "Z": add_2}
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

    add_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["depth"], 1: 0.0000}
    )

    subtract = nw.new_node(
        Nodes.Math, input_kwargs={0: add_3, 1: add_1}, attrs={"operation": "SUBTRACT"}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["height"],
            1: group_input.outputs["division_thickness"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply, "Z": subtract_1}
    )

    transform = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_2}
    )

    add_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["attach_back_len"], 1: 0.0000}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": add_2, "Z": add_4}
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_1,
            "Vertices X": 5,
            "Vertices Y": 5,
            "Vertices Z": 5,
        },
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_3, 1: -0.5000}, attrs={"operation": "MULTIPLY"}
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_4}, attrs={"operation": "MULTIPLY"}
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: multiply_2},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply_1, "Z": subtract_2}
    )

    transform_1 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube_1, "Translation": combine_xyz_3}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"attach1": transform, "attach2": transform_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_screw_head", singleton=False, type="GeometryNodeTree"
)
def nodegroup_screw_head(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Depth", 0.0050),
            ("NodeSocketFloat", "Radius", 1.0000),
            ("NodeSocketFloat", "bottom_gap", 0.5000),
            ("NodeSocketFloat", "division_thickness", 0.5000),
            ("NodeSocketFloat", "width", 0.5000),
            ("NodeSocketFloat", "height", 0.5000),
            ("NodeSocketFloat", "depth", 0.5000),
            ("NodeSocketFloat", "screw_gap", 0.5000),
        ],
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["Radius"],
            "Depth": group_input.outputs["Depth"],
        },
        attrs={"fill_type": "TRIANGLE_FAN"},
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["width"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["screw_gap"], 1: 0.0000}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: add},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["division_thickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["height"], 1: multiply_2},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply, "Y": subtract, "Z": subtract_1}
    )

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Translation": combine_xyz_1},
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_2, 1: group_input.outputs["bottom_gap"]}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply, "Y": subtract, "Z": add_1}
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Translation": combine_xyz},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply, "Y": multiply_3, "Z": subtract_1}
    )

    transform_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Translation": combine_xyz_2},
    )

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_1, 1: add_1})

    multiply_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_2}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply, "Z": multiply_4}
    )

    transform_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Translation": combine_xyz_3},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply, "Y": multiply_3, "Z": add_1}
    )

    transform_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Translation": combine_xyz_4},
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                transform_2,
                transform_1,
                transform_3,
                transform_5,
                transform_6,
            ]
        },
    )

    transform_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_2, "Scale": (-1.0000, 1.0000, 1.0000)},
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_4, join_geometry_2]}
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
            ("NodeSocketFloat", "width", 0.0000),
            ("NodeSocketFloat", "thickness", 0.5000),
            ("NodeSocketFloat", "height", 0.5000),
            ("NodeSocketFloat", "depth", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["thickness"], 1: 0.0000}
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["height"], 1: 0.0000}
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": group_input.outputs["width"], "Y": add, "Z": add_1},
    )

    cube_2 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_4,
            "Vertices X": 10,
            "Vertices Y": 10,
            "Vertices Z": 10,
        },
    )

    add_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["depth"], 1: 0.0000}
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: -0.5000}, attrs={"operation": "MULTIPLY"}
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_2, 1: -0.5000, 2: multiply},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply_add, "Z": multiply_1}
    )

    transform_5 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube_2, "Translation": combine_xyz_5}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_5},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_division_board", singleton=False, type="GeometryNodeTree"
)
def nodegroup_division_board(nw: NodeWrangler, tag_support=False):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "board_thickness", 0.0000),
            ("NodeSocketFloat", "depth", 0.5000),
            ("NodeSocketFloat", "width", 0.5000),
            ("NodeSocketFloat", "side_thickness", 0.5000),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["side_thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["width"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["depth"], 1: 0.0000}
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": subtract,
            "Y": add,
            "Z": group_input.outputs["board_thickness"],
        },
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
                "Vertices X": 10,
                "Vertices Y": 10,
                "Vertices Z": 10,
            },
        )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": cube_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_division_boards", singleton=False, type="GeometryNodeTree"
)
def nodegroup_division_boards(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "thickness", 0.5000),
            ("NodeSocketFloat", "height", 0.5000),
            ("NodeSocketFloat", "gap", 0.5000),
            ("NodeSocketGeometry", "Geometry", None),
        ],
    )

    realize_instances_1 = nw.new_node(
        Nodes.RealizeInstances,
        input_kwargs={"Geometry": group_input.outputs["Geometry"]},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["thickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["gap"], 1: multiply}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add})

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": realize_instances_1, "Translation": combine_xyz_1},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["height"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: subtract, 1: add})

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_1})

    transform_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": realize_instances_1, "Translation": combine_xyz_2},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": subtract})

    transform_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": realize_instances_1, "Translation": combine_xyz},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "board1": transform_2,
            "board2": transform_3,
            "board3": transform_4,
        },
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
            ("NodeSocketFloat", "board_thickness", 0.5000),
            ("NodeSocketFloat", "depth", 0.5000),
            ("NodeSocketFloat", "height", 0.5000),
            ("NodeSocketFloat", "width", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["board_thickness"], 1: 0.0000}
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["depth"], 1: 0.0000}
    )

    add_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["height"], 1: 0.0000}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": add_1, "Z": add_2}
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz,
            "Vertices X": 10,
            "Vertices Y": 10,
            "Vertices Z": 10,
        },
    )

    add_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["width"], 1: 0.0000}
    )

    subtract = nw.new_node(
        Nodes.Math, input_kwargs={0: add_3, 1: add}, attrs={"operation": "SUBTRACT"}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_2, 1: 0.5000}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply, "Z": multiply_1}
    )

    transform = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_1}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: 0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_2, "Z": multiply_1}
    )

    transform_1 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_2}
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform, transform_1]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_1},
        attrs={"is_active_output": True},
    )


def geometry_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    side_board_thickness = nw.new_node(Nodes.Value, label="side_board_thickness")
    side_board_thickness.outputs[0].default_value = kwargs["side_board_thickness"]

    shelf_depth = nw.new_node(Nodes.Value, label="shelf_depth")
    shelf_depth.outputs[0].default_value = kwargs["depth"]

    shelf_height = nw.new_node(Nodes.Value, label="shelf_height")
    shelf_height.outputs[0].default_value = kwargs["height"]

    shelf_width = nw.new_node(Nodes.Value, label="shelf_width")
    shelf_width.outputs[0].default_value = kwargs["width"]

    side_board = nw.new_node(
        nodegroup_side_board().name,
        input_kwargs={
            "board_thickness": side_board_thickness,
            "depth": shelf_depth,
            "height": shelf_height,
            "width": shelf_width,
        },
    )

    division_board_thickness = nw.new_node(
        Nodes.Value, label="division_board_thickness"
    )
    division_board_thickness.outputs[0].default_value = kwargs[
        "division_board_thickness"
    ]

    bottom_gap = nw.new_node(Nodes.Value, label="bottom_gap")
    bottom_gap.outputs[0].default_value = kwargs["bottom_gap"]

    division_board = nw.new_node(
        nodegroup_division_board(tag_support=kwargs["tag_support"]).name,
        input_kwargs={
            "board_thickness": division_board_thickness,
            "depth": shelf_depth,
            "width": shelf_width,
            "side_thickness": side_board_thickness,
        },
    )

    division_boards = nw.new_node(
        nodegroup_division_boards().name,
        input_kwargs={
            "thickness": division_board_thickness,
            "height": shelf_height,
            "gap": bottom_gap,
            "Geometry": division_board,
        },
    )

    backboard_thickness = nw.new_node(Nodes.Value, label="backboard_thickness")
    backboard_thickness.outputs[0].default_value = kwargs["backboard_thickness"]

    back_board = nw.new_node(
        nodegroup_back_board().name,
        input_kwargs={
            "width": shelf_width,
            "thickness": backboard_thickness,
            "height": shelf_height,
            "depth": shelf_depth,
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                side_board,
                division_boards.outputs["board1"],
                division_boards.outputs["board2"],
                back_board,
                division_boards.outputs["board3"],
            ]
        },
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": join_geometry}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": realize_instances,
            "Material": kwargs["frame_material"],
        },
    )

    screw_depth_head = nw.new_node(Nodes.Value, label="screw_depth_head")
    screw_depth_head.outputs[0].default_value = kwargs["screw_head_depth"]

    screw_head_radius = nw.new_node(Nodes.Value, label="screw_head_radius")
    screw_head_radius.outputs[0].default_value = kwargs["screw_head_radius"]

    screw_head_gap = nw.new_node(Nodes.Value, label="screw_head_gap")
    screw_head_gap.outputs[0].default_value = kwargs["screw_head_dist"]

    screw_head = nw.new_node(
        nodegroup_screw_head().name,
        input_kwargs={
            "Depth": screw_depth_head,
            "Radius": screw_head_radius,
            "bottom_gap": bottom_gap,
            "division_thickness": division_board_thickness,
            "width": shelf_width,
            "height": shelf_height,
            "depth": shelf_depth,
            "screw_gap": screw_head_gap,
        },
    )

    realize_instances_1 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": screw_head}
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": realize_instances_1,
            "Material": kwargs["metal_material"],
        },
    )

    attach_thickness = nw.new_node(Nodes.Value, label="attach_thickness")
    attach_thickness.outputs[0].default_value = kwargs["attach_thickness"]

    attach_width = nw.new_node(Nodes.Value, label="attach_width")
    attach_width.outputs[0].default_value = kwargs["attach_width"]

    attach_back_length = nw.new_node(Nodes.Value, label="attach_back_length")
    attach_back_length.outputs[0].default_value = kwargs["attach_back_length"]

    attach_top_length = nw.new_node(Nodes.Value, label="attach_top_length")
    attach_top_length.outputs[0].default_value = kwargs["attach_top_length"]

    attach_gadget = nw.new_node(
        nodegroup_attach_gadget().name,
        input_kwargs={
            "division_thickness": division_board_thickness,
            "height": shelf_height,
            "attach_thickness": attach_thickness,
            "attach_width": attach_width,
            "attach_back_len": attach_back_length,
            "attach_top_len": attach_top_length,
            "depth": shelf_depth,
        },
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                attach_gadget.outputs["attach1"],
                attach_gadget.outputs["attach2"],
            ]
        },
    )

    realize_instances_2 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": join_geometry_2}
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": realize_instances_2,
            "Material": kwargs["metal_material"],
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [set_material, set_material_1, set_material_2]},
    )

    realize_instances_3 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": join_geometry_1}
    )

    triangulate = nw.new_node(
        "GeometryNodeTriangulate", input_kwargs={"Mesh": realize_instances_3}
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


class SimpleBookcaseBaseFactory(AssetFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(SimpleBookcaseBaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.params = params

    def sample_params(self):
        return self.params.copy()

    def get_asset_params(self, i=0):
        params = self.sample_params()
        if params.get("depth", None) is None:
            params["depth"] = np.clip(normal(0.3, 0.05), 0.15, 0.45)
        if params.get("width", None) is None:
            params["width"] = np.clip(normal(0.5, 0.1), 0.25, 0.75)
        if params.get("height", None) is None:
            params["height"] = np.clip(normal(0.8, 0.1), 0.5, 1.0)
        params["side_board_thickness"] = uniform(0.005, 0.03)
        params["division_board_thickness"] = np.clip(normal(0.015, 0.005), 0.005, 0.025)
        params["bottom_gap"] = np.clip(normal(0.14, 0.05), 0.0, 0.2)
        params["backboard_thickness"] = uniform(0.01, 0.02)
        params["screw_head_depth"] = uniform(0.002, 0.008)
        params["screw_head_radius"] = uniform(0.003, 0.008)
        params["screw_head_dist"] = uniform(0.03, 0.1)
        params["attach_thickness"] = uniform(0.002, 0.005)
        params["attach_width"] = uniform(0.01, 0.04)
        params["attach_top_length"] = uniform(0.03, 0.1)
        params["attach_back_length"] = uniform(0.02, 0.05)
        params["frame_material"] = get_shelf_material("white")
        params["metal_material"] = get_shelf_material("metal")
        params["tag_support"] = True
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
            obj, geometry_nodes, apply=True, attributes=[], input_kwargs=obj_params
        )
        tagging.tag_system.relabel_obj(obj)

        return obj


class SimpleBookcaseFactory(SimpleBookcaseBaseFactory):
    def sample_params(self):
        params = dict()
        params["Dimensions"] = (
            uniform(0.25, 0.4),
            uniform(0.5, 0.7),
            uniform(0.7, 0.9),
        )
        params["depth"] = params["Dimensions"][0] - 0.015
        params["width"] = params["Dimensions"][1]
        params["height"] = params["Dimensions"][2]
        return params
