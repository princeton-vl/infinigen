# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han

import bpy
import numpy as np
from numpy.random import normal, randint, uniform

from infinigen.assets.materials.shelf_shaders import (
    shader_shelves_black_wood,
    shader_shelves_black_wood_sampler,
    shader_shelves_white,
    shader_shelves_white_sampler,
    shader_shelves_wood,
    shader_shelves_wood_sampler,
)
from infinigen.assets.objects.shelves.utils import nodegroup_tagged_cube
from infinigen.core import surface, tagging
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory


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
            ("NodeSocketFloat", "division_thickness", 0.5000),
            ("NodeSocketFloat", "width", 0.5000),
            ("NodeSocketFloat", "depth", 0.5000),
            ("NodeSocketFloat", "screw_width_gap", 0.5000),
            ("NodeSocketFloat", "screw_depth_gap", 0.0000),
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
        Nodes.Transform, input_kwargs={"Geometry": cylinder.outputs["Mesh"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["width"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: group_input.outputs["screw_width_gap"]},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["screw_width_gap"], 1: 0.0000}
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: add},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["division_thickness"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": multiply_2, "Z": multiply_3}
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Translation": combine_xyz},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": subtract_1, "Z": multiply_3}
    )

    transform_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Translation": combine_xyz_4},
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_1, transform_6]}
    )

    transform_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_2, "Scale": (-1.0000, 1.0000, 1.0000)},
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_4, join_geometry_2]}
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": join_geometry_3}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": realize_instances},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_attachment", singleton=False, type="GeometryNodeTree"
)
def nodegroup_attachment(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "attach_thickness", 0.0000),
            ("NodeSocketFloat", "attach_length", 0.0000),
            ("NodeSocketFloat", "attach_z_translation", 0.0000),
            ("NodeSocketFloat", "depth", 0.5000),
            ("NodeSocketFloat", "width", 0.5000),
            ("NodeSocketFloat", "attach_gap", 0.5000),
            ("NodeSocketFloat", "attach_width", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["attach_width"], 1: 0.0000}
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["attach_length"], 1: 0.0000}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": add,
            "Y": add_1,
            "Z": group_input.outputs["attach_thickness"],
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

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["width"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: group_input.outputs["attach_gap"]},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract, 1: add}, attrs={"operation": "SUBTRACT"}
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1}, attrs={"operation": "MULTIPLY"}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["depth"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: multiply_2})

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": subtract_1,
            "Y": add_2,
            "Z": group_input.outputs["attach_z_translation"],
        },
    )

    transform = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_1}
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Scale": (-1.0000, 1.0000, 1.0000)},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_1, transform]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_division_board", singleton=False, type="GeometryNodeTree"
)
def nodegroup_division_board(nw: NodeWrangler, material, tag_support=False):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "thickness", 0.0000),
            ("NodeSocketFloat", "width", 0.0000),
            ("NodeSocketFloat", "depth", 0.0000),
            ("NodeSocketFloat", "z_translation", 0.0000),
            ("NodeSocketFloat", "x_translation", 0.0000),
            ("NodeSocketFloat", "screw_depth", 0.0000),
            ("NodeSocketFloat", "screw_radius", 0.0000),
            ("NodeSocketFloat", "screw_width_gap", 0.0000),
            ("NodeSocketFloat", "screw_depth_gap", 0.0000),
        ],
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["width"],
            "Y": group_input.outputs["depth"],
            "Z": group_input.outputs["thickness"],
        },
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
                "Vertices X": 10,
                "Vertices Y": 10,
                "Vertices Z": 10,
            },
        )

    screw_head = nw.new_node(
        nodegroup_screw_head().name,
        input_kwargs={
            "Depth": group_input.outputs["screw_depth"],
            "Radius": group_input.outputs["screw_radius"],
            "division_thickness": group_input.outputs["thickness"],
            "width": group_input.outputs["width"],
            "depth": group_input.outputs["depth"],
            "screw_width_gap": group_input.outputs["screw_width_gap"],
            "screw_depth_gap": group_input.outputs["screw_depth_gap"],
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [cube, screw_head]}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["x_translation"],
            "Z": group_input.outputs["z_translation"],
        },
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Translation": combine_xyz_1},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_bottom_board", singleton=False, type="GeometryNodeTree"
)
def nodegroup_bottom_board(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "thickness", 0.0000),
            ("NodeSocketFloat", "depth", 0.5000),
            ("NodeSocketFloat", "y_gap", 0.5000),
            ("NodeSocketFloat", "x_translation", 0.0000),
            ("NodeSocketFloat", "height", 0.5000),
            ("NodeSocketFloat", "width", 0.0000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["height"], 1: 0.0000}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["width"],
            "Y": group_input.outputs["thickness"],
            "Z": add,
        },
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

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: group_input.outputs["y_gap"]},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["x_translation"],
            "Y": subtract,
            "Z": multiply_1,
        },
    )

    transform = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_1}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform},
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
            ("NodeSocketFloat", "x_translation", 0.0000),
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

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: add_2, 1: 0.5000}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": group_input.outputs["x_translation"], "Z": multiply},
    )

    transform = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_1}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform},
        attrs={"is_active_output": True},
    )


def geometry_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    side_board_thickness = nw.new_node(Nodes.Value, label="side_board_thickness")
    side_board_thickness.outputs[0].default_value = kwargs["side_board_thickness"]

    shelf_depth = nw.new_node(Nodes.Value, label="shelf_depth")
    shelf_depth.outputs[0].default_value = kwargs["shelf_depth"]

    add = nw.new_node(Nodes.Math, input_kwargs={0: shelf_depth, 1: 0.0040})

    shelf_height = nw.new_node(Nodes.Value, label="shelf_height")
    shelf_height.outputs[0].default_value = kwargs["shelf_height"]

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: shelf_height, 1: 0.0020})
    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: shelf_height, 1: -0.0010})
    side_boards = []

    for x in kwargs["side_board_x_translation"]:
        side_board_x_translation = nw.new_node(
            Nodes.Value, label="side_board_x_translation"
        )
        side_board_x_translation.outputs[0].default_value = x

        side_board = nw.new_node(
            nodegroup_side_board().name,
            input_kwargs={
                "board_thickness": side_board_thickness,
                "depth": add,
                "height": add_1,
                "x_translation": side_board_x_translation,
            },
        )
        side_boards.append(side_board)

    shelf_width = nw.new_node(Nodes.Value, label="shelf_width")
    shelf_width.outputs[0].default_value = kwargs["shelf_width"]

    backboard_thickness = nw.new_node(Nodes.Value, label="backboard_thickness")
    backboard_thickness.outputs[0].default_value = kwargs["backboard_thickness"]

    add_side = nw.new_node(
        Nodes.Math, input_kwargs={0: shelf_width, 1: kwargs["side_board_thickness"] * 2}
    )
    back_board = nw.new_node(
        nodegroup_back_board().name,
        input_kwargs={
            "width": add_side,
            "thickness": backboard_thickness,
            "height": add_2,
            "depth": shelf_depth,
        },
    )

    bottom_board_y_gap = nw.new_node(Nodes.Value, label="bottom_board_y_gap")
    bottom_board_y_gap.outputs[0].default_value = kwargs["bottom_board_y_gap"]

    bottom_board_height = nw.new_node(Nodes.Value, label="bottom_board_height")
    bottom_board_height.outputs[0].default_value = kwargs["bottom_board_height"]

    bottom_boards = []
    for i in range(len(kwargs["shelf_cell_width"])):
        bottom_gap_x_translation = nw.new_node(
            Nodes.Value, label="bottom_gap_x_translation"
        )
        bottom_gap_x_translation.outputs[0].default_value = kwargs[
            "bottom_gap_x_translation"
        ][i]

        shelf_cell_width = nw.new_node(Nodes.Value, label="shelf_cell_width")
        shelf_cell_width.outputs[0].default_value = kwargs["shelf_cell_width"][i]

        bottomboard = nw.new_node(
            nodegroup_bottom_board().name,
            input_kwargs={
                "thickness": side_board_thickness,
                "depth": shelf_depth,
                "y_gap": bottom_board_y_gap,
                "x_translation": bottom_gap_x_translation,
                "height": bottom_board_height,
                "width": shelf_cell_width,
            },
        )

        bottom_boards.append(bottomboard)

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [back_board] + side_boards + bottom_boards},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": join_geometry}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": realize_instances,
            "Material": surface.shaderfunc_to_material(kwargs["frame_material"]),
        },
    )

    division_board_thickness = nw.new_node(
        Nodes.Value, label="division_board_thickness"
    )
    division_board_thickness.outputs[0].default_value = kwargs[
        "division_board_thickness"
    ]

    division_boards = []
    for i in range(len(kwargs["shelf_cell_width"])):
        for j in range(len(kwargs["division_board_z_translation"])):
            division_board_z_translation = nw.new_node(
                Nodes.Value, label="division_board_z_translation"
            )
            division_board_z_translation.outputs[0].default_value = kwargs[
                "division_board_z_translation"
            ][j]

            division_board_x_translation = nw.new_node(
                Nodes.Value, label="division_board_x_translation"
            )
            division_board_x_translation.outputs[0].default_value = kwargs[
                "division_board_x_translation"
            ][i]

            shelf_cell_width = nw.new_node(Nodes.Value, label="shelf_cell_width")
            shelf_cell_width.outputs[0].default_value = kwargs["shelf_cell_width"][i]

            screw_depth_head = nw.new_node(Nodes.Value, label="screw_depth_head")
            screw_depth_head.outputs[0].default_value = kwargs["screw_depth_head"]

            screw_head_radius = nw.new_node(Nodes.Value, label="screw_head_radius")
            screw_head_radius.outputs[0].default_value = kwargs["screw_head_radius"]

            screw_width_gap = nw.new_node(Nodes.Value, label="screw_width_gap")
            screw_width_gap.outputs[0].default_value = kwargs["screw_width_gap"]

            screw_depth_gap = nw.new_node(Nodes.Value, label="screw_depth_gap")
            screw_depth_gap.outputs[0].default_value = kwargs["screw_depth_gap"]

            division_board = nw.new_node(
                nodegroup_division_board(
                    material=kwargs["board_material"],
                    tag_support=kwargs.get("tag_support", False),
                ).name,
                input_kwargs={
                    "thickness": division_board_thickness,
                    "width": shelf_cell_width,
                    "depth": shelf_depth,
                    "z_translation": division_board_z_translation,
                    "x_translation": division_board_x_translation,
                    "screw_depth": screw_depth_head,
                    "screw_radius": screw_head_radius,
                    "screw_width_gap": screw_width_gap,
                    "screw_depth_gap": screw_depth_gap,
                },
            )
            division_boards.append(division_board)

    attach_thickness = nw.new_node(Nodes.Value, label="attach_thickness")
    attach_thickness.outputs[0].default_value = kwargs["attach_thickness"]

    attach_length = nw.new_node(Nodes.Value, label="attach_length")
    attach_length.outputs[0].default_value = kwargs["attach_length"]

    attach_z_translation = nw.new_node(Nodes.Value, label="attach_z_translation")
    attach_z_translation.outputs[0].default_value = kwargs["attach_z_translation"]

    attach_gap = nw.new_node(Nodes.Value, label="attach_gap")
    attach_gap.outputs[0].default_value = kwargs["attach_gap"]

    attach_width = nw.new_node(Nodes.Value, label="attach_width")
    attach_width.outputs[0].default_value = kwargs["attach_width"]

    join_geometry_k = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": division_boards}
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": join_geometry_k,
            "Material": surface.shaderfunc_to_material(kwargs["board_material"]),
        },
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, set_material_1]}
    )

    realize_instances_3 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": join_geometry_3}
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


class LargeShelfBaseFactory(AssetFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(LargeShelfBaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.params = {}

    def sample_params(self):
        return self.params.copy()

    def get_asset_params(self, i=0):
        params = self.sample_params()
        if params.get("shelf_depth", None) is None:
            params["shelf_depth"] = np.clip(normal(0.26, 0.03), 0.18, 0.36)
        if params.get("side_board_thickness", None) is None:
            params["side_board_thickness"] = np.clip(normal(0.02, 0.002), 0.015, 0.025)
        if params.get("back_board_thickness", None) is None:
            params["backboard_thickness"] = 0.01
        if params.get("bottom_board_y_gap", None) is None:
            params["bottom_board_y_gap"] = uniform(0.01, 0.05)
        if params.get("bottom_board_height", None) is None:
            params["bottom_board_height"] = np.clip(
                normal(0.083, 0.01), 0.05, 0.11
            ) * np.random.choice([1.0, 0.0], p=[0.8, 0.2])
        if params.get("division_board_thickness", None) is None:
            params["division_board_thickness"] = np.clip(
                normal(0.02, 0.002), 0.015, 0.025
            )
        if params.get("screw_depth_head", None) is None:
            params["screw_depth_head"] = uniform(0.001, 0.004)
        if params.get("screw_head_radius", None) is None:
            params["screw_head_radius"] = uniform(0.001, 0.004)
        if params.get("screw_width_gap", None) is None:
            params["screw_width_gap"] = uniform(0.0, 0.02)
        if params.get("screw_depth_gap", None) is None:
            params["screw_depth_gap"] = uniform(0.025, 0.06)
        if params.get("attach_length", None) is None:
            params["attach_length"] = uniform(0.05, 0.1)
        if params.get("attach_width", None) is None:
            params["attach_width"] = uniform(0.01, 0.025)
        if params.get("attach_thickness", None) is None:
            params["attach_thickness"] = uniform(0.002, 0.005)
        if params.get("attach_gap", None) is None:
            params["attach_gap"] = uniform(0.0, 0.05)
        if params.get("shelf_cell_width", None) is None:
            num_h_cells = randint(1, 4)
            shelf_cell_width = []
            for i in range(num_h_cells):
                shelf_cell_width.append(
                    np.random.choice([0.76, 0.36], p=[0.5, 0.5])
                    * np.clip(normal(1.0, 0.1), 0.75, 1.25)
                )
            params["shelf_cell_width"] = shelf_cell_width
        if params.get("shelf_cell_height", None) is None:
            num_v_cells = randint(3, 8)
            shelf_cell_height = []
            for i in range(num_v_cells):
                shelf_cell_height.append(0.3 * np.clip(normal(1.0, 0.1), 0.75, 1.25))
            params["shelf_cell_height"] = shelf_cell_height

        params = self.update_translation_params(params)
        if params.get("frame_material", None) is None:
            params["frame_material"] = np.random.choice(
                ["white", "black_wood", "wood"], p=[0.4, 0.3, 0.3]
            )
        if params.get("board_material", None) is None:
            params["board_material"] = params["frame_material"]

        params = self.get_material_func(params)
        params["tag_support"] = True
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

        if params["board_material"] == "white":
            if randomness:
                params["board_material"] = lambda x: shader_shelves_white(
                    x, **white_wood_params
                )
            else:
                params["board_material"] = shader_shelves_white
        elif params["board_material"] == "black_wood":
            if randomness:
                params["board_material"] = lambda x: shader_shelves_black_wood(
                    x, **black_wood_params
                )
            else:
                params["board_material"] = shader_shelves_black_wood
        elif params["board_material"] == "wood":
            if randomness:
                params["board_material"] = lambda x: shader_shelves_wood(
                    x, **normal_wood_params
                )
            else:
                params["board_material"] = shader_shelves_wood

        return params

    def update_translation_params(self, params):
        cell_widths = params["shelf_cell_width"]
        cell_heights = params["shelf_cell_height"]
        side_thickness = params["side_board_thickness"]
        div_thickness = params["division_board_thickness"]

        # get shelf_width and shelf_height
        width = (len(cell_widths) - 1) * side_thickness * 2 + (
            len(cell_widths) - 1
        ) * 0.001
        height = (len(cell_heights) + 1) * div_thickness + params["bottom_board_height"]
        for w in cell_widths:
            width += w
        for h in cell_heights:
            height += h

        params["shelf_width"] = width
        params["shelf_height"] = height
        params["attach_z_translation"] = height - div_thickness

        # get side_board_x_translation
        dist = -(width + side_thickness) / 2.0
        side_board_x_translation = [dist]

        for w in cell_widths:
            dist += side_thickness + w
            side_board_x_translation.append(dist)
            dist += side_thickness + 0.001
            side_board_x_translation.append(dist)
        side_board_x_translation = side_board_x_translation[:-1]

        # get division_board_z_translation
        dist = params["bottom_board_height"] + div_thickness / 2.0
        division_board_z_translation = [dist]
        for h in cell_heights:
            dist += h + div_thickness
            division_board_z_translation.append(dist)

        # get division_board_x_translation
        division_board_x_translation = []
        for i in range(len(cell_widths)):
            division_board_x_translation.append(
                (side_board_x_translation[2 * i] + side_board_x_translation[2 * i + 1])
                / 2.0
            )

        params["side_board_x_translation"] = side_board_x_translation
        params["division_board_x_translation"] = division_board_x_translation
        params["division_board_z_translation"] = division_board_z_translation
        params["bottom_gap_x_translation"] = division_board_x_translation

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
            obj, geometry_nodes, attributes=[], apply=True, input_kwargs=obj_params
        )

        if params.get("ret_params", False):
            return obj, obj_params

        tagging.tag_system.relabel_obj(obj)

        return obj


class LargeShelfFactory(LargeShelfBaseFactory):
    def sample_params(self):
        params = dict()
        params["Dimensions"] = (
            uniform(0.25, 0.35),
            uniform(0.3, 2.0),
            uniform(0.9, 2.0),
        )

        params["bottom_board_height"] = 0.083
        params["shelf_depth"] = params["Dimensions"][0] - 0.01
        num_h = int((params["Dimensions"][2] - 0.083) / 0.3)
        params["shelf_cell_height"] = [
            (params["Dimensions"][2] - 0.083) / num_h for _ in range(num_h)
        ]
        num_v = max(int(params["Dimensions"][1] / 0.5), 1)
        params["shelf_cell_width"] = [
            params["Dimensions"][1] / num_v for _ in range(num_v)
        ]
        return params
