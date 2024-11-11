# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.materials import metal
from infinigen.assets.materials.shelf_shaders import (
    shader_shelves_black_wood,
    shader_shelves_black_wood_sampler,
    shader_shelves_white,
    shader_shelves_white_sampler,
    shader_shelves_wood,
    shader_shelves_wood_sampler,
)
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory


@node_utils.to_nodegroup(
    "nodegroup_board_rail", singleton=False, type="GeometryNodeTree"
)
def nodegroup_board_rail(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 64, "Radius": 0.0040, "Depth": 0.0050},
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Name": "uv_map",
            3: cylinder_1.outputs["UV Map"],
        },
        attrs={"data_type": "FLOAT_VECTOR", "domain": "CORNER"},
    )

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "width", 0.0000),
            ("NodeSocketFloat", "thickness", 0.5000),
            ("NodeSocketFloat", "depth", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["depth"], 1: 0.0000}
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: -0.5000}, attrs={"operation": "MULTIPLY"}
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: 0.0200})

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": add_1})

    transform_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_2,
            "Translation": combine_xyz_3,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    subtract = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 0.0300}, attrs={"operation": "SUBTRACT"}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": 0.0020, "Y": subtract, "Z": group_input.outputs["width"]},
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Name": "uv_map",
            3: cube.outputs["UV Map"],
        },
        attrs={"data_type": "FLOAT_VECTOR", "domain": "CORNER"},
    )

    transform = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": store_named_attribute}
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 64, "Radius": 0.0030, "Depth": subtract},
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Name": "uv_map",
            3: cylinder.outputs["UV Map"],
        },
        attrs={"data_type": "FLOAT_VECTOR", "domain": "CORNER"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["width"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_1})

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_1,
            "Translation": combine_xyz_1,
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_1, "Scale": (1.0000, 1.0000, -1.0000)},
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_2, transform_1]}
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_5, transform, join_geometry_2]},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["thickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_2, 1: 0.0030})

    multiply_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: -0.5000}, attrs={"operation": "MULTIPLY"}
    )

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: 0.0200})

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_2, "Y": multiply_3, "Z": add_3}
    )

    transform_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_1, "Translation": combine_xyz_2},
    )

    transform_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_3, "Scale": (-1.0000, 1.0000, 1.0000)},
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_4, transform_3]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_3},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_kallax_drawer_frame", singleton=False, type="GeometryNodeTree"
)
def nodegroup_kallax_drawer_frame(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "depth", 0.5000),
            ("NodeSocketFloat", "height", 0.5000),
            ("NodeSocketFloat", "thickness", 0.5000),
            ("NodeSocketFloat", "width", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["thickness"], 1: 0.0000}
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
            "Vertices X": 4,
            "Vertices Y": 4,
            "Vertices Z": 4,
        },
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Name": "uv_map",
            3: cube.outputs["UV Map"],
        },
        attrs={"data_type": "FLOAT_VECTOR", "domain": "CORNER"},
    )

    add_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["width"], 1: 0.0000}
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: add_3}, attrs={"operation": "MULTIPLY"}
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1, 1: -0.5000}, attrs={"operation": "MULTIPLY"}
    )

    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: -0.0001})

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_2, 2: 0.0100},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply, "Y": add_4, "Z": multiply_add}
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_1,
            "Translation": combine_xyz_1,
        },
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Scale": (-1.0000, 1.0000, 1.0000)},
    )

    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: -0.0001})

    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: add_3, 1: add_5})

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_6, "Y": add_1, "Z": add}
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_2,
            "Vertices X": 4,
            "Vertices Y": 4,
            "Vertices Z": 4,
        },
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_1.outputs["UV Map"],
        },
        attrs={"data_type": "FLOAT_VECTOR", "domain": "CORNER"},
    )

    multiply_add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_1, 1: -0.5000, 2: -0.0001},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply_add_1, "Z": 0.0100}
    )

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_2,
            "Translation": combine_xyz_3,
        },
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_3, "Y": add, "Z": add_2}
    )

    cube_2 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_4,
            "Vertices X": 4,
            "Vertices Y": 4,
            "Vertices Z": 4,
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_2.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_2.outputs["UV Map"],
        },
        attrs={"data_type": "FLOAT_VECTOR", "domain": "CORNER"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: add}, attrs={"operation": "MULTIPLY"}
    )

    multiply_add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_1, 1: -1.0000, 2: multiply_2},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    multiply_add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_2, 2: 0.0100},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply_add_2, "Z": multiply_add_3}
    )

    transform_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": store_named_attribute, "Translation": combine_xyz_5},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_1, transform, transform_2, transform_3]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_door_knob", singleton=False, type="GeometryNodeTree"
)
def nodegroup_door_knob(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Radius", 0.0040),
            ("NodeSocketFloat", "length", 0.5000),
            ("NodeSocketFloat", "z", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["length"], 1: 0.0000}
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 64,
            "Radius": group_input.outputs["Radius"],
            "Depth": add,
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Name": "uv_map",
            3: cylinder.outputs["UV Map"],
        },
        attrs={"data_type": "FLOAT_VECTOR", "domain": "CORNER"},
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: add}, attrs={"operation": "MULTIPLY"}
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: 0.0001})

    add_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["z"], 1: 0.0000}
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_2}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": add_1, "Z": multiply_1}
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute,
            "Translation": combine_xyz_2,
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_drawer_door_board", singleton=False, type="GeometryNodeTree"
)
def nodegroup_drawer_door_board(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "thickness", 0.5000),
            ("NodeSocketFloat", "width", 0.5000),
            ("NodeSocketFloat", "height", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["width"], 1: 0.0000}
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["thickness"], 1: 0.0000}
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
            "Vertices X": 5,
            "Vertices Y": 5,
            "Vertices Z": 5,
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Name": "uv_map",
            3: cube.outputs["UV Map"],
        },
        attrs={"data_type": "FLOAT_VECTOR", "domain": "CORNER"},
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1, 1: -0.5000}, attrs={"operation": "MULTIPLY"}
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_2}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply, "Z": multiply_1}
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": store_named_attribute, "Translation": combine_xyz_1},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform},
        attrs={"is_active_output": True},
    )


def geometry_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    door_thickness = nw.new_node(Nodes.Value, label="door_thickness")
    door_thickness.outputs[0].default_value = kwargs["drawer_board_thickness"]

    drawer_board_width = nw.new_node(Nodes.Value, label="drawer_board_width")
    drawer_board_width.outputs[0].default_value = kwargs["drawer_board_width"]

    drawer_board_height = nw.new_node(Nodes.Value, label="drawer_board_height")
    drawer_board_height.outputs[0].default_value = kwargs["drawer_board_height"]

    drawer_door_board = nw.new_node(
        nodegroup_drawer_door_board().name,
        input_kwargs={
            "thickness": door_thickness,
            "width": drawer_board_width,
            "height": drawer_board_height,
        },
    )

    knob_radius = nw.new_node(Nodes.Value, label="knob_radius")
    knob_radius.outputs[0].default_value = kwargs["knob_radius"]

    knob_length = nw.new_node(Nodes.Value, label="knob_length")
    knob_length.outputs[0].default_value = kwargs["knob_length"]

    door_knob = nw.new_node(
        nodegroup_door_knob().name,
        input_kwargs={
            "Radius": knob_radius,
            "length": knob_length,
            "z": drawer_board_height,
        },
    )

    drawer_depth = nw.new_node(Nodes.Value, label="drawer_depth")
    drawer_depth.outputs[0].default_value = (
        kwargs["drawer_depth"] - kwargs["drawer_board_thickness"]
    )

    drawer_side_height = nw.new_node(Nodes.Value, label="drawer_side_height")
    drawer_side_height.outputs[0].default_value = kwargs["drawer_side_height"]

    drawer_width = nw.new_node(Nodes.Value, label="drawer_width")
    drawer_width.outputs[0].default_value = kwargs["drawer_width"]

    kallax_drawer_frame = nw.new_node(
        nodegroup_kallax_drawer_frame().name,
        input_kwargs={
            "depth": drawer_depth,
            "height": drawer_side_height,
            "thickness": door_thickness,
            "width": drawer_width,
        },
    )

    side_tilt_width = nw.new_node(Nodes.Value, label="side_tilt_width")
    side_tilt_width.outputs[0].default_value = kwargs["side_tilt_width"]

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [door_knob, drawer_door_board, kallax_drawer_frame]},
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": join_geometry,
            "Material": surface.shaderfunc_to_material(kwargs["frame_material"]),
        },
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": set_material_2}
    )

    triangulate = nw.new_node(
        "GeometryNodeTriangulate", input_kwargs={"Mesh": realize_instances}
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": triangulate, "Rotation": (0.0000, 0.0000, -1.5708)},
    )

    group_output_1 = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform},
        attrs={"is_active_output": True},
    )


class CabinetDrawerBaseFactory(AssetFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(CabinetDrawerBaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.params = {}

    def get_asset_params(self, i=0):
        params = self.params.copy()
        if params.get("drawer_board_thickness", None) is None:
            params["drawer_board_thickness"] = uniform(0.005, 0.01)
        if params.get("drawer_board_width", None) is None:
            params["drawer_board_width"] = uniform(0.3, 0.7)
        if params.get("drawer_board_height", None) is None:
            params["drawer_board_height"] = uniform(0.25, 0.4)
        if params.get("drawer_depth", None) is None:
            params["drawer_depth"] = uniform(0.3, 0.4)
        if params.get("drawer_side_height", None) is None:
            params["drawer_side_height"] = uniform(0.05, 0.2)
        if params.get("drawer_width", None) is None:
            params["drawer_width"] = params["drawer_board_width"] - uniform(
                0.015, 0.025
            )
        if params.get("side_tilt_width", None) is None:
            params["side_tilt_width"] = uniform(0.02, 0.03)
        if params.get("knob_radius", None) is None:
            params["knob_radius"] = uniform(0.003, 0.006)
        if params.get("knob_length", None) is None:
            params["knob_length"] = uniform(0.018, 0.035)

        if params.get("frame_material", None) is None:
            params["frame_material"] = np.random.choice(
                ["white", "black_wood", "wood"], p=[0.5, 0.2, 0.3]
            )
        if params.get("knob_material", None) is None:
            params["knob_material"] = np.random.choice(
                [params["frame_material"], "metal"], p=[0.5, 0.5]
            )

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

        if params["knob_material"] == "metal":
            params["knob_material"] = metal.get_shader()
        else:
            params["knob_material"] = params["frame_material"]

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

        if params.get("ret_params", False):
            return obj, obj_params

        return obj
