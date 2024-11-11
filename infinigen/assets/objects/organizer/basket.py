# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.materials.plastics.plastic_rough import shader_rough_plastic
from infinigen.core import surface, tagging
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory


@node_utils.to_nodegroup("nodegroup_holes", singleton=False, type="GeometryNodeTree")
def nodegroup_holes(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Value1", 0.5000),
            ("NodeSocketFloat", "Value2", 0.5000),
            ("NodeSocketFloat", "Value3", 0.5000),
            ("NodeSocketFloat", "Value4", 0.5000),
            ("NodeSocketFloat", "Value5", 0.5000),
            ("NodeSocketFloat", "Value6", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Value3"], 1: 0.0000}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Value1"], 1: add},
        attrs={"operation": "SUBTRACT"},
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Value6"], 1: 0.0000}
    )

    subtract_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1, 1: add}, attrs={"operation": "SUBTRACT"}
    )

    add_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Value4"], 1: 0.0000}
    )

    add_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_2, 1: group_input.outputs["Value2"]}
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract, 1: add_3}, attrs={"operation": "DIVIDE"}
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: add_3},
        attrs={"operation": "DIVIDE"},
    )

    grid = nw.new_node(
        Nodes.MeshGrid,
        input_kwargs={
            "Size X": subtract,
            "Size Y": subtract_1,
            "Vertices X": divide,
            "Vertices Y": divide_1,
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": grid.outputs["Mesh"],
            "Name": "uv_map",
            3: grid.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    add_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Value5"], 1: 0.0000}
    )

    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: add_4, 1: 0.1})

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_5, "Y": add_2, "Z": add_2}
    )

    cube_2 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_3})

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_2.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_2.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={"Points": transform_1, "Instance": store_named_attribute_1},
    )

    subtract_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_4, 1: add}, attrs={"operation": "SUBTRACT"}
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_2, 1: add_3},
        attrs={"operation": "DIVIDE"},
    )

    grid_1 = nw.new_node(
        Nodes.MeshGrid,
        input_kwargs={
            "Size X": subtract_2,
            "Size Y": subtract,
            "Vertices X": divide_2,
            "Vertices Y": divide,
        },
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": grid_1.outputs["Mesh"],
            "Name": "uv_map",
            3: grid_1.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_2,
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: add_1, 1: 0.1})

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_2, "Y": add_6, "Z": add_2}
    )

    cube_3 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_4})

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_3.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_3.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    instance_on_points_1 = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={"Points": transform_2, "Instance": store_named_attribute_3},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Instances1": instance_on_points,
            "Instances2": instance_on_points_1,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_handle_hole", singleton=False, type="GeometryNodeTree"
)
def nodegroup_handle_hole(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "X", 0.0000),
            ("NodeSocketFloat", "Z", 0.0000),
            ("NodeSocketFloat", "Value", 0.5000),
            ("NodeSocketFloat", "Value2", 0.5000),
            ("NodeSocketInt", "Level", 0),
        ],
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["X"],
            "Y": 1.0000,
            "Z": group_input.outputs["Z"],
        },
    )

    cube_2 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_3})

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_2.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_2.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    subdivide_mesh_2 = nw.new_node(
        Nodes.SubdivideMesh, input_kwargs={"Mesh": store_named_attribute}
    )

    subdivision_surface_2 = nw.new_node(
        Nodes.SubdivisionSurface,
        input_kwargs={"Mesh": subdivide_mesh_2, "Level": group_input.outputs["Level"]},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: group_input.outputs["Value2"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": subtract})

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": subdivision_surface_2, "Translation": combine_xyz_4},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_1},
        attrs={"is_active_output": True},
    )


def geometry_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    depth = nw.new_node(Nodes.Value, label="depth")
    depth.outputs[0].default_value = kwargs["depth"]

    width = nw.new_node(Nodes.Value, label="width")
    width.outputs[0].default_value = kwargs["width"]

    height = nw.new_node(Nodes.Value, label="height")
    height.outputs[0].default_value = kwargs["height"]

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": depth, "Y": width, "Z": height}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Name": "uv_map",
            3: cube.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    subdivide_mesh = nw.new_node(
        Nodes.SubdivideMesh, input_kwargs={"Mesh": store_named_attribute, "Level": 2}
    )

    sub_level = nw.new_node(Nodes.Integer, label="sub_level")
    sub_level.integer = kwargs["frame_sub_level"]

    subdivision_surface = nw.new_node(
        Nodes.SubdivisionSurface,
        input_kwargs={"Mesh": subdivide_mesh, "Level": sub_level},
    )

    differences = []

    if kwargs["has_handle"]:
        hole_depth = nw.new_node(Nodes.Value, label="hole_depth")
        hole_depth.outputs[0].default_value = kwargs["handle_depth"]

        hole_height = nw.new_node(Nodes.Value, label="hole_height")
        hole_height.outputs[0].default_value = kwargs["handle_height"]

        hole_dist = nw.new_node(Nodes.Value, label="hole_dist")
        hole_dist.outputs[0].default_value = kwargs["handle_dist_to_top"]

        handle_level = nw.new_node(Nodes.Integer, label="handle_level")
        handle_level.integer = kwargs["handle_sub_level"]
        handle_hole = nw.new_node(
            nodegroup_handle_hole().name,
            input_kwargs={
                "X": hole_depth,
                "Z": hole_height,
                "Value": height,
                "Value2": hole_dist,
                "Level": handle_level,
            },
        )
        differences.append(handle_hole)

    thickness = nw.new_node(Nodes.Value, label="thickness")
    thickness.outputs[0].default_value = kwargs["thickness"]

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: depth, 1: thickness},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: width, 1: thickness},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": subtract_1, "Z": height}
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_1})

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_1.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    subdivide_mesh_1 = nw.new_node(
        Nodes.SubdivideMesh, input_kwargs={"Mesh": store_named_attribute_1, "Level": 2}
    )

    subdivision_surface_1 = nw.new_node(
        Nodes.SubdivisionSurface,
        input_kwargs={"Mesh": subdivide_mesh_1, "Level": sub_level},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: thickness, 1: 0.2500},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": subdivision_surface_1, "Translation": combine_xyz_2},
    )

    if kwargs["has_holes"]:
        gap_size = nw.new_node(Nodes.Value, label="gap_size")
        gap_size.outputs[0].default_value = kwargs["hole_gap_size"]

        hole_edge_gap = nw.new_node(Nodes.Value, label="hole_edge_gap")
        hole_edge_gap.outputs[0].default_value = kwargs["hole_edge_gap"]

        hole_size = nw.new_node(Nodes.Value, label="hole_size")
        hole_size.outputs[0].default_value = kwargs["hole_size"]
        holes = nw.new_node(
            nodegroup_holes().name,
            input_kwargs={
                "Value1": height,
                "Value2": gap_size,
                "Value3": hole_edge_gap,
                "Value4": hole_size,
                "Value5": depth,
                "Value6": width,
            },
        )
        differences.extend([holes.outputs["Instances1"], holes.outputs["Instances2"]])

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 1": subdivision_surface,
            "Mesh 2": [transform] + differences,
        },
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": difference.outputs["Mesh"]}
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: height}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_1})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": realize_instances, "Translation": combine_xyz_3},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry,
            "Material": surface.shaderfunc_to_material(shader_rough_plastic),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_material},
        attrs={"is_active_output": True},
    )


class BasketBaseFactory(AssetFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(BasketBaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.params = params

    def sample_params(self):
        return self.params.copy()

    def get_asset_params(self, i=0):
        params = self.sample_params()
        if params.get("depth", None) is None:
            params["depth"] = uniform(0.15, 0.4)
        if params.get("width", None) is None:
            params["width"] = uniform(0.2, 0.6)
        if params.get("height", None) is None:
            params["height"] = uniform(0.06, 0.24)
        if params.get("frame_sub_level", None) is None:
            params["frame_sub_level"] = np.random.choice([0, 3], p=[0.5, 0.5])
        if params.get("thickness", None) is None:
            params["thickness"] = uniform(0.001, 0.005)

        if params.get("has_handle", None) is None:
            params["has_handle"] = np.random.choice([True, False], p=[0.8, 0.2])
        if params.get("handle_sub_level", None) is None:
            params["handle_sub_level"] = np.random.choice([0, 1, 2], p=[0.2, 0.4, 0.4])
        if params.get("handle_depth", None) is None:
            params["handle_depth"] = params["depth"] * uniform(0.2, 0.4)
        if params.get("handle_height", None) is None:
            params["handle_height"] = params["height"] * uniform(0.1, 0.25)
        if params.get("handle_dist_to_top", None) is None:
            params["handle_dist_to_top"] = params["handle_height"] * 0.5 + params[
                "height"
            ] * uniform(0.08, 0.15)

        if params.get("has_holes", None) is None:
            if params["height"] < 0.12:
                params["has_holes"] = False
            else:
                params["has_holes"] = np.random.choice([True, False], p=[0.5, 0.5])
        if params.get("hole_size", None) is None:
            params["hole_size"] = uniform(0.005, 0.01)
        if params.get("hole_gap_size", None) is None:
            params["hole_gap_size"] = params["hole_size"] * uniform(0.8, 1.1)
        if params.get("hole_edge_gap", None) is None:
            params["hole_edge_gap"] = uniform(0.04, 0.06)

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
        tagging.tag_system.relabel_obj(obj)

        return obj
