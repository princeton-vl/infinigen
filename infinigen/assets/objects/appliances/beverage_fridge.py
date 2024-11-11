# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hongyu Wen


import numpy as np
from numpy.random import normal as N
from numpy.random import randint as RI
from numpy.random import uniform as U

from infinigen.assets.material_assignments import AssetList
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.bevelling import (
    add_bevel,
    complete_bevel,
    complete_no_bevel,
    get_bevel_edges,
)
from infinigen.core.util.blender import delete
from infinigen.core.util.math import FixedSeed


class BeverageFridgeFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, dimensions=[1.0, 1.0, 1.0]):
        super(BeverageFridgeFactory, self).__init__(factory_seed, coarse=coarse)

        self.dimensions = dimensions
        with FixedSeed(factory_seed):
            self.params = self.sample_parameters(dimensions)
            self.material_params, self.scratch, self.edge_wear = (
                self.get_material_params()
            )
        self.params.update(self.material_params)

    def get_material_params(self):
        material_assignments = AssetList["BeverageFridgeFactory"]()
        params = {
            "Surface": material_assignments["surface"].assign_material(),
            "Front": material_assignments["front"].assign_material(),
            "Handle": material_assignments["handle"].assign_material(),
            "Back": material_assignments["back"].assign_material(),
        }
        wrapped_params = {
            k: surface.shaderfunc_to_material(v) for k, v in params.items()
        }

        scratch_prob, edge_wear_prob = material_assignments["wear_tear_prob"]
        scratch, edge_wear = material_assignments["wear_tear"]

        is_scratch = np.random.uniform() < scratch_prob
        is_edge_wear = np.random.uniform() < edge_wear_prob
        if not is_scratch:
            scratch = None

        if not is_edge_wear:
            edge_wear = None

        return wrapped_params, scratch, edge_wear

    @staticmethod
    def sample_parameters(dimensions):
        depth = 1 + N(0, 0.1)
        width = 1 + N(0, 0.1)
        height = 1 + N(0, 0.1)
        # depth, width, height = dimensions
        door_thickness = U(0.05, 0.1) * depth
        door_rotation = 0  # Set to 0 for now

        rack_radius = U(0.01, 0.02) * depth
        rack_h_amount = RI(2, 4)
        rack_d_amount = RI(4, 6)
        brand_name = "BrandName"

        params = {
            "Depth": depth,
            "Width": width,
            "Height": height,
            "DoorThickness": door_thickness,
            "DoorRotation": door_rotation,
            "RackRadius": rack_radius,
            "RackHAmount": rack_h_amount,
            "RackDAmount": rack_d_amount,
            "BrandName": brand_name,
        }
        return params

    def create_asset(self, **params):
        obj = butil.spawn_cube()
        butil.modify_mesh(
            obj,
            "NODES",
            node_group=nodegroup_beverage_fridge_geometry(preprocess=True),
            ng_inputs=self.params,
            apply=True,
        )
        bevel_edges = get_bevel_edges(obj)
        delete(obj)
        obj = butil.spawn_cube()
        butil.modify_mesh(
            obj,
            "NODES",
            node_group=nodegroup_beverage_fridge_geometry(),
            ng_inputs=self.params,
            apply=True,
        )
        obj = add_bevel(obj, bevel_edges, offset=0.01)

        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)


@node_utils.to_nodegroup(
    "nodegroup_oven_rack", singleton=False, type="GeometryNodeTree"
)
def nodegroup_oven_rack(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 2.0000),
            ("NodeSocketFloat", "Height", 2.0000),
            ("NodeSocketFloat", "Radius", 0.0200),
            ("NodeSocketInt", "Amount", 5),
        ],
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Width": group_input.outputs["Width"],
            "Height": group_input.outputs["Height"],
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_1})

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_3, "End": combine_xyz_4}
    )

    geometry_to_instance = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": curve_line}
    )

    duplicate_elements = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={
            "Geometry": geometry_to_instance,
            "Amount": group_input.outputs["Amount"],
        },
        attrs={"domain": "INSTANCE"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"]},
        attrs={"operation": "MULTIPLY"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_2, 1: group_input.outputs["Amount"]},
        attrs={"operation": "DIVIDE"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: divide},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_3})

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": duplicate_elements.outputs["Geometry"],
            "Offset": combine_xyz,
        },
    )

    duplicate_elements_1 = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={
            "Geometry": geometry_to_instance,
            "Amount": group_input.outputs["Amount"],
        },
        attrs={"domain": "INSTANCE"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_4, 1: group_input.outputs["Amount"]},
        attrs={"operation": "DIVIDE"},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements_1.outputs["Duplicate Index"], 1: divide_1},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_5})

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": duplicate_elements_1.outputs["Geometry"],
            "Offset": combine_xyz_1,
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [quadrilateral, set_position, set_position_1]},
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Radius": group_input.outputs["Radius"]}
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": join_geometry,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": curve_to_mesh},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_text", singleton=False, type="GeometryNodeTree")
def nodegroup_text(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Translation", (1.5000, 0.0000, 0.0000)),
            ("NodeSocketString", "String", "BrandName"),
            ("NodeSocketFloat", "Size", 0.0500),
            ("NodeSocketFloat", "Offset Scale", 0.0020),
        ],
    )

    string_to_curves = nw.new_node(
        "GeometryNodeStringToCurves",
        input_kwargs={
            "String": group_input.outputs["String"],
            "Size": group_input.outputs["Size"],
        },
        attrs={"align_y": "BOTTOM_BASELINE", "align_x": "CENTER"},
    )

    fill_curve = nw.new_node(
        Nodes.FillCurve,
        input_kwargs={"Curve": string_to_curves.outputs["Curve Instances"]},
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": fill_curve,
            "Offset Scale": group_input.outputs["Offset Scale"],
        },
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": extrude_mesh.outputs["Mesh"],
            "Translation": group_input.outputs["Translation"],
            "Rotation": (1.5708, 0.0000, 1.5708),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_handle", singleton=False, type="GeometryNodeTree")
def nodegroup_handle(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "width", 0.0000),
            ("NodeSocketFloat", "length", 0.0000),
            ("NodeSocketFloat", "thickness", 0.0200),
        ],
    )

    cube = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": group_input.outputs["width"]}
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Name": "uv_map",
            3: cube.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": group_input.outputs["width"]}
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_1.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": group_input.outputs["length"]}
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": store_named_attribute_1, "Translation": combine_xyz},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute, transform]},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["width"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_1, "Translation": combine_xyz_3},
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["length"],
            1: group_input.outputs["width"],
        },
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["width"],
            "Y": add,
            "Z": group_input.outputs["thickness"],
        },
    )

    cube_2 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_1})

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_2.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_2.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["length"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["thickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["width"], 1: multiply_2}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply_1, "Z": add_1}
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_2,
            "Translation": combine_xyz_2,
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_2, transform_1]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_center", singleton=False, type="GeometryNodeTree")
def nodegroup_center(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketVector", "Vector", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "MarginX", 0.5000),
            ("NodeSocketFloat", "MarginY", 0.0000),
            ("NodeSocketFloat", "MarginZ", 0.0000),
        ],
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": group_input.outputs["Geometry"]}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Vector"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    greater_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: group_input.outputs["MarginX"]},
        attrs={"operation": "GREATER_THAN", "use_clamp": True},
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: group_input.outputs["Vector"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_1.outputs["Vector"]}
    )

    greater_than_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_1.outputs["X"],
            1: group_input.outputs["MarginX"],
        },
        attrs={"operation": "GREATER_THAN", "use_clamp": True},
    )

    op_and = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: greater_than, 1: greater_than_1}
    )

    greater_than_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: group_input.outputs["MarginY"]},
        attrs={"operation": "GREATER_THAN"},
    )

    greater_than_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_1.outputs["Y"],
            1: group_input.outputs["MarginY"],
        },
        attrs={"operation": "GREATER_THAN", "use_clamp": True},
    )

    op_and_1 = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: greater_than_2, 1: greater_than_3}
    )

    op_and_2 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: op_and, 1: op_and_1})

    greater_than_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: group_input.outputs["MarginZ"]},
        attrs={"operation": "GREATER_THAN", "use_clamp": True},
    )

    greater_than_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_1.outputs["Z"],
            1: group_input.outputs["MarginZ"],
        },
        attrs={"operation": "GREATER_THAN", "use_clamp": True},
    )

    op_and_3 = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: greater_than_4, 1: greater_than_5}
    )

    op_and_4 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: op_and_2, 1: op_and_3})

    op_not = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: op_and_4}, attrs={"operation": "NOT"}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"In": op_and_4, "Out": op_not},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_cube", singleton=False, type="GeometryNodeTree")
def nodegroup_cube(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (0.1000, 10.0000, 4.0000)),
            ("NodeSocketVector", "Pos", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketInt", "Resolution", 2),
        ],
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": group_input.outputs["Size"],
            "Vertices X": group_input.outputs["Resolution"],
            "Vertices Y": group_input.outputs["Resolution"],
            "Vertices Z": group_input.outputs["Resolution"],
        },
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Name": "uv_map",
            3: cube.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": store_named_attribute_1, "Name": "uv_map"},
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Size"],
            1: (0.5000, 0.5000, 0.5000),
            2: group_input.outputs["Pos"],
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute,
            "Translation": multiply_add.outputs["Vector"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_hollow_cube", singleton=False, type="GeometryNodeTree"
)
def nodegroup_hollow_cube(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (0.1000, 10.0000, 4.0000)),
            ("NodeSocketVector", "Pos", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketInt", "Resolution", 2),
            ("NodeSocketFloat", "Thickness", 0.0000),
            ("NodeSocketBool", "Switch1", False),
            ("NodeSocketBool", "Switch2", False),
            ("NodeSocketBool", "Switch3", False),
            ("NodeSocketBool", "Switch4", False),
            ("NodeSocketBool", "Switch5", False),
            ("NodeSocketBool", "Switch6", False),
        ],
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Thickness"],
            "Y": subtract,
            "Z": subtract_1,
        },
    )

    cube_2 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_4,
            "Vertices X": 2,
            "Vertices Y": 2,
            "Vertices Z": 2,
        },
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_2.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_2.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Pos"]}
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_1, 1: separate_xyz_1.outputs["X"]}
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], "Scale": 0.5000},
        attrs={"operation": "SCALE"},
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": scale.outputs["Vector"]}
    )

    add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: separate_xyz_1.outputs["Y"]},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: separate_xyz_1.outputs["Z"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": add_1, "Z": subtract_2}
    )

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_1,
            "Translation": combine_xyz_5,
        },
    )

    switch_2 = nw.new_node(
        Nodes.Switch, input_kwargs={0: group_input.outputs["Switch3"], 1: transform_2}
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz.outputs["X"],
            "Y": subtract_3,
            "Z": group_input.outputs["Thickness"],
        },
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_2,
            "Vertices X": 2,
            "Vertices Y": 2,
            "Vertices Z": 2,
        },
    )

    store_named_attribute_4 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_1.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: separate_xyz_1.outputs["X"]},
    )

    add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: separate_xyz_1.outputs["Y"]},
    )

    subtract_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_2, "Y": add_3, "Z": subtract_4}
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_4,
            "Translation": combine_xyz_3,
        },
    )

    switch_1 = nw.new_node(
        Nodes.Switch, input_kwargs={0: group_input.outputs["Switch2"], 1: transform_1}
    )

    subtract_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz.outputs["X"],
            "Y": subtract_5,
            "Z": group_input.outputs["Thickness"],
        },
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz,
            "Vertices X": 2,
            "Vertices Y": 2,
            "Vertices Z": 2,
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Name": "uv_map",
            3: cube.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    add_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: separate_xyz_1.outputs["X"]},
    )

    add_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: separate_xyz_1.outputs["Y"]},
    )

    add_6 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_1, 1: separate_xyz_1.outputs["Z"]}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_4, "Y": add_5, "Z": add_6}
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": store_named_attribute, "Translation": combine_xyz_1},
    )

    switch = nw.new_node(
        Nodes.Switch, input_kwargs={0: group_input.outputs["Switch1"], 1: transform}
    )

    subtract_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Thickness"],
            "Y": subtract_6,
            "Z": subtract_7,
        },
    )

    cube_3 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_6,
            "Vertices X": 2,
            "Vertices Y": 2,
            "Vertices Z": 2,
        },
    )

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_3.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_3.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    subtract_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    add_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: separate_xyz_1.outputs["Y"]},
    )

    subtract_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: separate_xyz_1.outputs["Z"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract_8, "Y": add_7, "Z": subtract_9}
    )

    transform_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_5,
            "Translation": combine_xyz_7,
        },
    )

    switch_3 = nw.new_node(
        Nodes.Switch, input_kwargs={0: group_input.outputs["Switch4"], 1: transform_3}
    )

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz.outputs["X"],
            "Y": group_input.outputs["Thickness"],
            "Z": separate_xyz.outputs["Z"],
        },
    )

    cube_4 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_9,
            "Vertices X": 2,
            "Vertices Y": 2,
            "Vertices Z": 2,
        },
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_4.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_4.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    add_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: separate_xyz_2.outputs["X"]},
    )

    add_9 = nw.new_node(
        Nodes.Math, input_kwargs={0: separate_xyz_1.outputs["Y"], 1: multiply_1}
    )

    add_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Z"], 1: separate_xyz_2.outputs["Z"]},
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_8, "Y": add_9, "Z": add_10}
    )

    transform_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_2,
            "Translation": combine_xyz_8,
        },
    )

    switch_4 = nw.new_node(
        Nodes.Switch, input_kwargs={0: group_input.outputs["Switch5"], 1: transform_4}
    )

    combine_xyz_10 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz.outputs["X"],
            "Y": group_input.outputs["Thickness"],
            "Z": separate_xyz.outputs["Z"],
        },
    )

    cube_5 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_10,
            "Vertices X": 2,
            "Vertices Y": 2,
            "Vertices Z": 2,
        },
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_5.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_5.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    add_11 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: separate_xyz_1.outputs["X"]},
    )

    subtract_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    add_12 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: separate_xyz_1.outputs["Z"]},
    )

    combine_xyz_11 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_11, "Y": subtract_10, "Z": add_12}
    )

    transform_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_3,
            "Translation": combine_xyz_11,
        },
    )

    switch_5 = nw.new_node(
        Nodes.Switch, input_kwargs={0: group_input.outputs["Switch6"], 1: transform_5}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                switch_2,
                switch_1,
                switch,
                switch_3,
                switch_4,
                switch_5,
            ]
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_beverage_fridge_geometry", singleton=False, type="GeometryNodeTree"
)
def nodegroup_beverage_fridge_geometry(nw: NodeWrangler, preprocess: bool = False):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Depth", 1.0000),
            ("NodeSocketFloat", "Width", 1.0000),
            ("NodeSocketFloat", "Height", 1.0000),
            ("NodeSocketFloat", "DoorThickness", 0.0700),
            ("NodeSocketFloat", "DoorRotation", 0.0000),
            ("NodeSocketFloat", "RackRadius", 0.0100),
            ("NodeSocketInt", "RackDAmount", 5),
            ("NodeSocketInt", "RackHAmount", 2),
            ("NodeSocketString", "BrandName", "BrandName"),
            ("NodeSocketMaterial", "Surface", None),
            ("NodeSocketMaterial", "Front", None),
            ("NodeSocketMaterial", "Handle", None),
            ("NodeSocketMaterial", "Back", None),
        ],
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Depth"],
            "Y": group_input.outputs["Width"],
            "Z": group_input.outputs["Height"],
        },
    )

    hollowcube = nw.new_node(
        nodegroup_hollow_cube().name,
        input_kwargs={
            "Size": combine_xyz,
            "Thickness": group_input.outputs["DoorThickness"],
            "Switch2": True,
            "Switch4": True,
        },
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": hollowcube,
            "Material": group_input.outputs["Surface"],
        },
    )

    subdivide_mesh = nw.new_node(
        Nodes.SubdivideMesh, input_kwargs={"Mesh": set_material_1, "Level": 0}
    )

    # set_shade_smooth_2 = nw.new_node(Nodes.SetShadeSmooth, input_kwargs={'Geometry': subdivide_mesh})

    body = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": subdivide_mesh}, label="Body"
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["DoorThickness"],
            "Y": group_input.outputs["Width"],
            "Z": group_input.outputs["Height"],
        },
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input.outputs["Depth"]}
    )

    cube = nw.new_node(
        nodegroup_cube().name,
        input_kwargs={"Size": combine_xyz_1, "Pos": combine_xyz_2},
    )

    position = nw.new_node(Nodes.InputPosition)

    center = nw.new_node(
        nodegroup_center().name,
        input_kwargs={
            "Geometry": cube,
            "Vector": position,
            "MarginX": -1.0000,
            "MarginY": 0.1000,
            "MarginZ": 0.1500,
        },
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": cube,
            "Selection": center.outputs["In"],
            "Material": group_input.outputs["Front"],
        },
    )

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": set_material_2,
            "Selection": center.outputs["Out"],
            "Material": group_input.outputs["Surface"],
        },
    )

    # set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth, input_kwargs={'Geometry': set_material_3})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: 0.0500},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: 0.8000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply}, attrs={"operation": "MULTIPLY"}
    )

    handle = nw.new_node(
        nodegroup_handle().name,
        input_kwargs={"width": multiply, "length": multiply_1, "thickness": multiply_2},
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Depth"],
            1: group_input.outputs["DoorThickness"],
        },
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: 0.1000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: 0.9000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_13 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": multiply_3, "Z": multiply_4}
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": handle,
            "Translation": combine_xyz_13,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    set_material_8 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_1,
            "Material": group_input.outputs["Handle"],
        },
    )

    geometry_to_instance_4 = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": set_material_8}
    )

    rotate_instances_2 = nw.new_node(
        Nodes.RotateInstances,
        input_kwargs={
            "Instances": geometry_to_instance_4,
            "Rotation": (-1.5708, 0.0000, 0.0000),
            "Pivot Point": combine_xyz_13,
        },
    )

    add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Depth"],
            1: group_input.outputs["DoorThickness"],
        },
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_12 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_1, "Y": multiply_5, "Z": 0.0300}
    )

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: 0.0500},
        attrs={"operation": "MULTIPLY"},
    )

    text = nw.new_node(
        nodegroup_text().name,
        input_kwargs={
            "Translation": combine_xyz_12,
            "String": group_input.outputs["BrandName"],
            "Size": multiply_6,
            "Offset Scale": 0.0020,
        },
    )

    text = complete_no_bevel(nw, text, preprocess)

    set_material_9 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": text, "Material": group_input.outputs["Handle"]},
    )

    rotate_instances_2 = complete_bevel(nw, rotate_instances_2, preprocess)

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [set_material_3, rotate_instances_2, set_material_9]},
    )

    geometry_to_instance = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": join_geometry_3}
    )

    z = nw.scalar_multiply(
        group_input.outputs["DoorRotation"], 1 if not preprocess else 0
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": z})

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Depth"],
            "Y": group_input.outputs["Width"],
        },
    )

    rotate_instances = nw.new_node(
        Nodes.RotateInstances,
        input_kwargs={
            "Instances": geometry_to_instance,
            "Rotation": combine_xyz_3,
            "Pivot Point": combine_xyz_4,
        },
    )

    door = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": nw.new_node(Nodes.RealizeInstances, [rotate_instances])},
        label="door",
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["DoorThickness"], 1: 2.1000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 1: multiply_7},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["DoorThickness"], 1: 2.1000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: multiply_8},
        attrs={"operation": "SUBTRACT"},
    )

    ovenrack = nw.new_node(
        nodegroup_oven_rack().name,
        input_kwargs={
            "Width": subtract,
            "Height": subtract_1,
            "Radius": group_input.outputs["RackRadius"],
            "Amount": group_input.outputs["RackDAmount"],
        },
    )

    geometry_to_instance_1 = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": ovenrack}
    )

    duplicate_elements = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={
            "Geometry": geometry_to_instance_1,
            "Amount": group_input.outputs["RackHAmount"],
        },
        attrs={"domain": "INSTANCE"},
    )

    multiply_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: 1.0000},
    )

    multiply_11 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["DoorThickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: multiply_11},
        attrs={"operation": "SUBTRACT"},
    )

    add_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["RackHAmount"], 1: 1.0000}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_2, 1: add_3},
        attrs={"operation": "DIVIDE"},
    )

    multiply_12 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_2, 1: divide}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": multiply_9, "Y": multiply_10, "Z": multiply_12},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": duplicate_elements.outputs["Geometry"],
            "Offset": combine_xyz_5,
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": set_position,
            "Material": group_input.outputs["Handle"],
        },
    )

    racks = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": nw.new_node(Nodes.RealizeInstances, [set_material])},
        label="racks",
    )

    add_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Depth"],
            1: group_input.outputs["DoorThickness"],
        },
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add_4})

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["DoorThickness"]}
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": reroute_10, "Y": reroute_11, "Z": reroute_8},
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_9})

    cube_1 = nw.new_node(
        nodegroup_cube().name,
        input_kwargs={"Size": combine_xyz_6, "Pos": combine_xyz_7},
    )

    set_material_5 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": cube_1, "Material": group_input.outputs["Back"]},
    )

    # set_shade_smooth_1 = nw.new_node(Nodes.SetShadeSmooth, input_kwargs={'Geometry': set_material_5})

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": set_material_5}
    )

    heater = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": join_geometry_2}, label="heater"
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [body, door, racks, heater]}
    )

    geometry = nw.new_node(Nodes.RealizeInstances, [join_geometry])

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geometry})
