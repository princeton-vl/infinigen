# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han

import bpy
import numpy as np
from numpy.random import normal, uniform

from infinigen.assets.materials.shelf_shaders import (
    shader_shelves_black_metallic,
    shader_shelves_black_metallic_sampler,
    shader_shelves_black_wood,
    shader_shelves_black_wood_sampler,
    shader_shelves_white,
    shader_shelves_white_metallic,
    shader_shelves_white_metallic_sampler,
    shader_shelves_white_sampler,
)
from infinigen.assets.objects.shelves.utils import nodegroup_tagged_cube
from infinigen.core import surface, tagging
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory


@node_utils.to_nodegroup(
    "nodegroup_table_legs", singleton=False, type="GeometryNodeTree"
)
def nodegroup_table_legs(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "thickness", 0.5000),
            ("NodeSocketFloat", "height", 0.5000),
            ("NodeSocketFloat", "radius", 0.0200),
            ("NodeSocketFloat", "width", 0.5000),
            ("NodeSocketFloat", "depth", 0.5000),
            ("NodeSocketFloat", "dist", 0.5000),
        ],
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["height"],
            1: group_input.outputs["thickness"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["radius"],
            "Depth": subtract,
            "Vertices": 128,
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["width"]},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["dist"], 1: 0.0000}
    )

    subtract_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply, 1: add}, attrs={"operation": "SUBTRACT"}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={1: group_input.outputs["depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: add},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": subtract_1, "Y": subtract_2, "Z": multiply_2},
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_2,
        },
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": multiply_3, "Y": subtract_2, "Z": multiply_2},
    )

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_3,
        },
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_2, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": subtract_1, "Y": multiply_4, "Z": multiply_2},
    )

    transform_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_4,
        },
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": multiply_3, "Y": multiply_4, "Z": multiply_2},
    )

    transform_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_5,
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform, transform_2, transform_3, transform_4]},
    )

    realize_instances_1 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": join_geometry_1}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": realize_instances_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_table_top", singleton=False, type="GeometryNodeTree"
)
def nodegroup_table_top(nw: NodeWrangler, tag_support=True):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "depth", 0.0000),
            ("NodeSocketFloat", "width", 0.0000),
            ("NodeSocketFloat", "height", 0.5000),
            ("NodeSocketFloat", "thickness", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["thickness"], 1: 0.0000}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["width"],
            "Y": group_input.outputs["depth"],
            "Z": add,
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

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: add}, attrs={"operation": "MULTIPLY"}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["height"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": subtract})

    transform_1 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz_1}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_1},
        attrs={"is_active_output": True},
    )


def geometry_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    table_depth = nw.new_node(Nodes.Value, label="table_depth")
    table_depth.outputs[0].default_value = kwargs["depth"]

    table_width = nw.new_node(Nodes.Value, label="table_width")
    table_width.outputs[0].default_value = kwargs["width"]

    table_height = nw.new_node(Nodes.Value, label="table_height")
    table_height.outputs[0].default_value = kwargs["height"]

    top_thickness = nw.new_node(Nodes.Value, label="top_thickness")
    top_thickness.outputs[0].default_value = kwargs["thickness"]

    table_top = nw.new_node(
        nodegroup_table_top(tag_support=True).name,
        input_kwargs={
            "depth": table_depth,
            "width": table_width,
            "height": table_height,
            "thickness": top_thickness,
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": table_top,
            "Material": surface.shaderfunc_to_material(kwargs["top_material"]),
        },
    )

    leg_radius = nw.new_node(Nodes.Value, label="leg_radius")
    leg_radius.outputs[0].default_value = kwargs["leg_radius"]

    leg_center_to_edge = nw.new_node(Nodes.Value, label="leg_center_to_edge")
    leg_center_to_edge.outputs[0].default_value = kwargs["leg_dist"]

    table_legs = nw.new_node(
        nodegroup_table_legs().name,
        input_kwargs={
            "thickness": top_thickness,
            "height": table_height,
            "radius": leg_radius,
            "width": table_width,
            "depth": table_depth,
            "dist": leg_center_to_edge,
        },
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": table_legs,
            "Material": surface.shaderfunc_to_material(kwargs["leg_material"]),
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, set_material_1]}
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": join_geometry}
    )

    triangulate = nw.new_node(
        "GeometryNodeTriangulate", input_kwargs={"Mesh": realize_instances}
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": triangulate, "Rotation": (0.0000, 0.0000, 1.5708)},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform},
        attrs={"is_active_output": True},
    )


class SimpleDeskBaseFactory(AssetFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(SimpleDeskBaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.params = params

    def sample_params(self):
        return self.params.copy()

    def get_asset_params(self, i=0):
        params = self.sample_params()
        if params.get("depth", None) is None:
            params["depth"] = np.clip(normal(0.6, 0.05), 0.45, 0.7)
        if params.get("width", None) is None:
            params["width"] = np.clip(normal(1.0, 0.1), 0.7, 1.3)
        if params.get("height", None) is None:
            params["height"] = np.clip(normal(0.73, 0.05), 0.6, 0.83)
        if params.get("top_material", None) is None:
            params["top_material"] = np.random.choice(["white", "black_wood"])
        if params.get("leg_material", None) is None:
            params["leg_material"] = np.random.choice(["white", "black"])
        if params.get("leg_radius", None) is None:
            params["leg_radius"] = uniform(0.01, 0.025)
        if params.get("leg_dist", None) is None:
            params["leg_dist"] = uniform(0.035, 0.07)
        if params.get("thickness", None) is None:
            params["thickness"] = uniform(0.01, 0.03)

        params = self.get_material_func(params)
        return params

    def get_material_func(self, params, randomness=True):
        if params["top_material"] == "white":
            if randomness:
                params["top_material"] = lambda x: shader_shelves_white(
                    x, **shader_shelves_white_sampler()
                )
            else:
                params["top_material"] = shader_shelves_white
        elif params["top_material"] == "black_wood":
            if randomness:
                params["top_material"] = lambda x: shader_shelves_black_wood(
                    x, **shader_shelves_black_wood_sampler()
                )
            else:
                params["top_material"] = shader_shelves_black_wood
        else:
            raise NotImplementedError

        if params["leg_material"] == "white":
            if randomness:
                params["leg_material"] = lambda x: shader_shelves_white_metallic(
                    x, **shader_shelves_white_metallic_sampler()
                )
            else:
                params["leg_material"] = shader_shelves_white_metallic
        elif params["leg_material"] == "black":
            if randomness:
                params["leg_material"] = lambda x: shader_shelves_black_metallic(
                    x, **shader_shelves_black_metallic_sampler()
                )
            else:
                params["leg_material"] = shader_shelves_black_metallic
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

        obj_params = self.get_asset_params(i)
        surface.add_geomod(
            obj, geometry_nodes, attributes=[], apply=True, input_kwargs=obj_params
        )
        tagging.tag_system.relabel_obj(obj)

        return obj


class SimpleDeskFactory(SimpleDeskBaseFactory):
    def sample_params(self):
        params = dict()
        params["Dimensions"] = (uniform(0.5, 0.75), uniform(0.8, 2), uniform(0.6, 0.8))
        params["depth"] = params["Dimensions"][0]
        params["width"] = params["Dimensions"][1]
        params["height"] = params["Dimensions"][2]
        return params


class SidetableDeskFactory(SimpleDeskBaseFactory):
    def sample_params(self):
        params = dict()
        w = 0.55 * normal(1, 0.1)
        params["Dimensions"] = (w, w, w * normal(1, 0.05))
        params["depth"] = params["Dimensions"][0]
        params["width"] = params["Dimensions"][1]
        params["height"] = params["Dimensions"][2]
        return params
