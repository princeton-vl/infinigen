# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Stamatis Alexandropoulos: primary author
# - Abhishek Joshi: updates for sim integration
# Acknowledgment: This file draws inspiration
# from https://www.youtube.com/watch?v=o50FE2W1m8Y
# by Open Class

import functools

import numpy as np
from numpy.random import randint as RI
from numpy.random import uniform as U

from infinigen.assets.materials import (
    metal,
    plastic,
)
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import weighted_sample


@node_utils.to_nodegroup(
    "nodegroup_dish_rack", singleton=False, type="GeometryNodeTree"
)
def nodegroup_dish_rack(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    quadrilateral = nw.new_node("GeometryNodeCurvePrimitiveQuadrilateral")

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": quadrilateral, "Name": "joint11"},
        attrs={"data_type": "INT"},
    )

    curve_line = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (0.0000, -1.0000, 0.0000),
            "End": (0.0000, 1.0000, 0.0000),
        },
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": curve_line, "Name": "joint12"},
        attrs={"data_type": "INT"},
    )

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Depth", 2.0000),
            ("NodeSocketFloat", "Width", 2.0000),
            ("NodeSocketFloat", "Radius", 0.0200),
            ("NodeSocketInt", "Amount", 5),
            ("NodeSocketFloat", "Height", 0.5000),
            ("NodeSocketFloat", "Value", 0.5000),
        ],
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"Y": -1.0000, "Z": group_input.outputs["Height"]},
    )

    curve_line_1 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={"Start": (0.0000, -1.0000, 0.0000), "End": combine_xyz},
    )

    geometry_to_instance = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": curve_line_1}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Amount"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    duplicate_elements = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={"Geometry": geometry_to_instance, "Amount": multiply},
        attrs={"domain": "INSTANCE"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: group_input.outputs["Amount"]},
        attrs={"operation": "DIVIDE"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: divide},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_1})

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": duplicate_elements.outputs["Geometry"],
            "Offset": combine_xyz_1,
        },
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_position, "Name": "joint12", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_1, store_named_attribute_2]},
    )

    geometry_to_instance_1 = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": join_geometry}
    )

    duplicate_elements_1 = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={"Geometry": geometry_to_instance_1, "Amount": multiply},
        attrs={"domain": "INSTANCE"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: duplicate_elements_1.outputs["Duplicate Index"],
            1: group_input.outputs["Amount"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: divide},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_2})

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": duplicate_elements_1.outputs["Geometry"],
            "Offset": combine_xyz_2,
        },
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": set_position_1, "Rotation": (0.0000, 0.0000, 1.5708)},
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry, "Name": "joint11", "Value": 1},
        attrs={"data_type": "INT"},
    )

    duplicate_elements_2 = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={"Geometry": geometry_to_instance_1, "Amount": multiply},
        attrs={"domain": "INSTANCE"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: duplicate_elements_2.outputs["Duplicate Index"],
            1: group_input.outputs["Amount"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: divide},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_3})

    set_position_2 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": duplicate_elements_2.outputs["Geometry"],
            "Offset": combine_xyz_3,
        },
    )

    store_named_attribute_4 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_position_2, "Name": "joint11", "Value": 2},
        attrs={"data_type": "INT"},
    )

    quadrilateral_1 = nw.new_node("GeometryNodeCurvePrimitiveQuadrilateral")

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: 0.8000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_4})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": quadrilateral_1, "Translation": combine_xyz_4},
    )

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_1, "Name": "joint11", "Value": 3},
        attrs={"data_type": "INT"},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                store_named_attribute,
                store_named_attribute_3,
                store_named_attribute_4,
                store_named_attribute_5,
            ]
        },
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Radius": group_input.outputs["Radius"]}
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": join_geometry_1,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 1: 0.4800},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: group_input.outputs["Value"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_5, "Y": multiply_6, "Z": 0.5000}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_to_mesh,
            "Rotation": (0.0000, 0.0000, 1.5708),
            "Scale": combine_xyz_5,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": transform_geometry_2},
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
            ("NodeSocketFloat", "X", 1.0000),
            ("NodeSocketFloat", "Y", 1.0000),
            ("NodeSocketBool", "Switch", False),
            ("NodeSocketFloat", "Radius", 0.0400),
            ("NodeSocketFloat", "handle_position", 0.0000),
            ("NodeSocketBool", "curved_handle", True),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["width"]}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": reroute})

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Name": "uv_map",
            "Value": cube.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute_4 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": store_named_attribute, "Name": "joint8"},
        attrs={"data_type": "INT"},
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": group_input.outputs["width"]}
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Name": "uv_map",
            "Value": cube_1.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["length"]}
    )

    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": reroute_2})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": store_named_attribute_1, "Translation": combine_xyz},
    )

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry, "Name": "joint8", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_4, store_named_attribute_5]},
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_5}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry,
            "Translation": combine_xyz_1,
            "Rotation": (0.0070, 0.0000, 0.0000),
        },
    )

    store_named_attribute_6 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_1, "Name": "switch3"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_7 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_1, "Name": "switch4"},
        attrs={"data_type": "INT"},
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_1,
            "Scale": (1.0000, 1.0000, 0.4010),
        },
    )

    store_named_attribute_8 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_5, "Name": "switch4", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["curved_handle"],
            "False": store_named_attribute_7,
            "True": store_named_attribute_8,
        },
    )

    store_named_attribute_9 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_2, "Name": "switch3", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Switch"],
            "False": store_named_attribute_6,
            "True": store_named_attribute_9,
        },
    )

    store_named_attribute_10 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_3, "Name": "joint7"},
        attrs={"data_type": "INT"},
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["length"],
            1: group_input.outputs["width"],
        },
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 77,
            "Radius": group_input.outputs["Radius"],
            "Depth": add,
        },
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": transform_geometry_3,
            "Name": "uv_map",
            "Value": cylinder.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute_11 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": store_named_attribute_2, "Name": "switch5"},
        attrs={"data_type": "INT"},
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["thickness"]}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute, "Y": add, "Z": reroute_4}
    )

    cube_2 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_2})

    store_named_attribute_12 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": cube_2.outputs["Mesh"], "Name": "switch6"},
        attrs={"data_type": "INT"},
    )

    subdivide_mesh = nw.new_node(
        Nodes.SubdivideMesh, input_kwargs={"Mesh": cube_2.outputs["Mesh"], "Level": 6}
    )

    position = nw.new_node(Nodes.InputPosition)

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: position, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    vector_rotate = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={
            "Vector": position,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Angle": multiply_1,
        },
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": subdivide_mesh, "Position": vector_rotate},
    )

    store_named_attribute_13 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_position_1, "Name": "switch6", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["curved_handle"],
            "False": store_named_attribute_12,
            "True": store_named_attribute_13,
        },
    )

    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={"Geometry": switch_1})

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": set_position,
            "Name": "uv_map",
            "Value": cube_2.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute_14 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_3,
            "Name": "switch5",
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Switch"],
            "False": store_named_attribute_11,
            "True": store_named_attribute_14,
        },
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_2}, attrs={"operation": "MULTIPLY"}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["thickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: reroute, 1: multiply_3})

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply_2, "Z": add_1}
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_3})

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["X"],
            "Y": group_input.outputs["Y"],
            "Z": 1.0000,
        },
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": switch,
            "Translation": reroute_6,
            "Scale": combine_xyz_4,
        },
    )

    store_named_attribute_15 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_2, "Name": "joint7", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_10, store_named_attribute_15]},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input.outputs["handle_position"]}
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_1, "Translation": combine_xyz_5},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_4, "Handle_center": add_1},
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
        attrs={"use_clamp": True, "operation": "GREATER_THAN"},
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
        attrs={"use_clamp": True, "operation": "GREATER_THAN"},
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
        attrs={"use_clamp": True, "operation": "GREATER_THAN"},
    )

    op_and_1 = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: greater_than_2, 1: greater_than_3}
    )

    op_and_2 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: op_and, 1: op_and_1})

    greater_than_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: group_input.outputs["MarginZ"]},
        attrs={"use_clamp": True, "operation": "GREATER_THAN"},
    )

    greater_than_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_1.outputs["Z"],
            1: group_input.outputs["MarginZ"],
        },
        attrs={"use_clamp": True, "operation": "GREATER_THAN"},
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


@node_utils.to_nodegroup("nodegroup_buttons", singleton=False, type="GeometryNodeTree")
def nodegroup_buttons(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    string = nw.new_node("FunctionNodeInputString", attrs={"string": "joint5"})

    string_1 = nw.new_node("FunctionNodeInputString", attrs={"string": "duplicate1"})

    string_2 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint6"})

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "depth", 0.5000),
            ("NodeSocketFloat", "door_thickness", 0.5000),
            ("NodeSocketFloat", "height", 0.5000),
            ("NodeSocketFloat", "hinge_button", 1.3100),
            ("NodeSocketFloat", "relative_position", 0.3000),
            ("NodeSocketFloat", "width", 0.5000),
            ("NodeSocketGeometry", "Parent", None),
            ("NodeSocketBool", "button_type", False),
            ("NodeSocketInt", "buttons_amount", 6),
            ("NodeSocketFloat", "rotate_button", -11.0000),
            ("NodeSocketMaterial", "Material", None),
            ("NodeSocketFloat", "X", 0.0010),
            ("NodeSocketFloat", "Width", 0.5000),
            ("NodeSocketBool", "Switch", False),
            ("NodeSocketMaterial", "New_Material", None),
        ],
    )

    reroute_15 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Parent"]}
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_15})

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_25, "Name": "joint6"},
        attrs={"data_type": "INT"},
    )

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["button_type"]}
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": 0.0300, "Height": 0.0500},
    )

    fill_curve = nw.new_node(Nodes.FillCurve, input_kwargs={"Curve": quadrilateral})

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh, input_kwargs={"Mesh": fill_curve, "Offset Scale": 0.0200}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": extrude_mesh.outputs["Mesh"]}
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_2, "Name": "switch1"},
        attrs={"data_type": "INT"},
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 233, "Radius": 0.0200, "Depth": 0.0500},
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Name": "switch1",
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_6,
            "False": store_named_attribute_2,
            "True": store_named_attribute_3,
        },
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["door_thickness"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["relative_position"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["height"], 1: 0.4500},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_11 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide, "Y": reroute_4, "Z": multiply}
    )

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_11})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": switch,
            "Translation": reroute_23,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    reroute_19 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["New_Material"]}
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry, "Material": reroute_22},
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_1, "Label": "button"},
    )

    store_named_attribute_4 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata,
            "Name": "joint6",
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "button_joint",
            "Parent": store_named_attribute_1,
            "Child": store_named_attribute_4,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Min": -0.0200,
        },
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Switch"]}
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_17})

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["width"], 1: 4.0000},
        attrs={"operation": "DIVIDE"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["buttons_amount"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    grid = nw.new_node(
        Nodes.MeshGrid,
        input_kwargs={
            "Size X": 0.0000,
            "Size Y": divide_1,
            "Vertices X": 1,
            "Vertices Y": reroute_1,
        },
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": grid.outputs["Mesh"],
            "Translation": (0.0000, -0.2000, 0.0000),
        },
    )

    transform_geometry_6 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": grid.outputs["Mesh"]}
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_18,
            "False": transform_geometry_1,
            "True": transform_geometry_6,
        },
    )

    grid_1 = nw.new_node(
        Nodes.MeshGrid,
        input_kwargs={
            "Size X": 0.0000,
            "Size Y": divide_1,
            "Vertices X": 1,
            "Vertices Y": reroute_1,
        },
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": grid_1.outputs["Mesh"],
            "Translation": (0.0000, 0.6000, 0.0000),
        },
    )

    reroute_26 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_3}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [switch_3, reroute_26]}
    )

    reroute_30 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry})

    duplicate_joints_on_parent_002 = nw.new_node(
        nodegroup_duplicate_joints_on_parent_002().name,
        input_kwargs={
            "Duplicate ID (do not set)": string_1,
            "Parent": sliding_joint.outputs["Parent"],
            "Child": sliding_joint.outputs["Child"],
            "Points": reroute_30,
        },
    )

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": duplicate_joints_on_parent_002, "Name": "joint5"},
        attrs={"data_type": "INT"},
    )

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder", input_kwargs={"Radius": 0.0500, "Depth": 0.0600}
    )

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    add = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: combine_xyz_11, 1: (0.0200, 0.2000, 0.0000)}
    )

    reroute_24 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add.outputs["Vector"]}
    )

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_14, 1: -0.4200},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_1})

    add_1 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: add.outputs["Vector"], 1: combine_xyz_1}
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_27,
            "False": reroute_24,
            "True": add_1.outputs["Vector"],
        },
        attrs={"input_type": "VECTOR"},
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": switch_2,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_4, "Material": reroute_29},
    )

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_2, "Label": "knob"},
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_1,
            "Name": "joint5",
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "knob_joint",
            "Parent": store_named_attribute_5,
            "Child": store_named_attribute,
            "Axis": (1.0000, 0.0000, 0.0000),
        },
    )

    store_named_attribute_6 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": hinge_joint.outputs["Geometry"], "Name": "joint4"},
        attrs={"data_type": "INT"},
    )

    bounding_box = nw.new_node(Nodes.BoundingBox, input_kwargs={"Geometry": reroute_16})

    position = nw.new_node(Nodes.InputPosition)

    attribute_statistic = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": position,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: attribute_statistic.outputs["Max"],
            1: attribute_statistic.outputs["Min"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    cube = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": subtract.outputs["Vector"]}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["height"], 1: 0.9200},
        attrs={"operation": "MULTIPLY"},
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_2, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide_2})

    add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: attribute_statistic.outputs["Mean"], 1: combine_xyz_2},
    )

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Translation": add_2.outputs["Vector"],
            "Scale": (1.0000, 1.0000, 0.0800),
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_7,
            "Material": group_input.outputs["Material"],
        },
    )

    add_jointed_geometry_metadata_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material, "Label": "rotation_body"},
    )

    store_named_attribute_7 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_2,
            "Name": "joint4",
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_6, store_named_attribute_7]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_1},
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
        ],
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["X"]}
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

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

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Thickness"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_6, "Y": subtract, "Z": reroute_12}
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Resolution"]}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    cube_1 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_2,
            "Vertices X": reroute_14,
            "Vertices Y": reroute_14,
            "Vertices Z": reroute_14,
        },
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Name": "uv_map",
            "Value": cube_1.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], "Scale": 0.5000},
        attrs={"operation": "SCALE"},
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": scale.outputs["Vector"]}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Pos"]}
    )

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": reroute_2})

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: separate_xyz_1.outputs["X"]},
    )

    add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: separate_xyz_1.outputs["Y"]},
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["Z"]}
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_1}, attrs={"operation": "MULTIPLY"}
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_10, 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": add_1, "Z": subtract_1}
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_3})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": store_named_attribute_1, "Translation": reroute_15},
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_1, "Name": "joint10"},
        attrs={"data_type": "INT"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": reroute_12, "Y": subtract_2, "Z": subtract_3},
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz,
            "Vertices X": reroute_14,
            "Vertices Y": reroute_14,
            "Vertices Z": reroute_14,
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Name": "uv_map",
            "Value": cube.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    add_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_1, 1: separate_xyz_1.outputs["X"]}
    )

    add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: separate_xyz_1.outputs["Y"]},
    )

    subtract_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: separate_xyz_1.outputs["Z"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_2, "Y": add_3, "Z": subtract_4}
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_1})

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": store_named_attribute, "Translation": reroute_19},
    )

    store_named_attribute_6 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry, "Name": "joint10", "Value": 1},
        attrs={"data_type": "INT"},
    )

    subtract_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": reroute_6, "Y": subtract_5, "Z": reroute_12},
    )

    cube_2 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_4,
            "Vertices X": reroute_14,
            "Vertices Y": reroute_14,
            "Vertices Z": reroute_14,
        },
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_2.outputs["Mesh"],
            "Name": "uv_map",
            "Value": cube_2.outputs["UV Map"],
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

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_4, "Y": add_5, "Z": add_6}
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_5})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": store_named_attribute_2, "Translation": reroute_20},
    )

    store_named_attribute_7 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_2, "Name": "joint10", "Value": 2},
        attrs={"data_type": "INT"},
    )

    combine_xyz_10 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz.outputs["X"],
            "Y": reroute_1,
            "Z": separate_xyz.outputs["Z"],
        },
    )

    cube_5 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_10,
            "Vertices X": reroute_4,
            "Vertices Y": reroute_4,
            "Vertices Z": reroute_4,
        },
    )

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_5.outputs["Mesh"],
            "Name": "uv_map",
            "Value": cube_5.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    add_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: separate_xyz_1.outputs["X"]},
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["Y"]}
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    subtract_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_8, 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    add_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: separate_xyz_1.outputs["Z"]},
    )

    combine_xyz_11 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_7, "Y": subtract_6, "Z": add_8}
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_5,
            "Translation": combine_xyz_11,
        },
    )

    reroute_21 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_5}
    )

    store_named_attribute_8 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_21, "Name": "joint10", "Value": 3},
        attrs={"data_type": "INT"},
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz.outputs["X"],
            "Y": reroute_1,
            "Z": separate_xyz.outputs["Z"],
        },
    )

    cube_4 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_8,
            "Vertices X": reroute_4,
            "Vertices Y": reroute_4,
            "Vertices Z": reroute_4,
        },
    )

    store_named_attribute_4 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_4.outputs["Mesh"],
            "Name": "uv_map",
            "Value": cube_4.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    add_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: separate_xyz_2.outputs["X"]},
    )

    add_10 = nw.new_node(
        Nodes.Math, input_kwargs={0: separate_xyz_1.outputs["Y"], 1: multiply_1}
    )

    add_11 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Z"], 1: separate_xyz_2.outputs["Z"]},
    )

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_9, "Y": add_10, "Z": add_11}
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_4,
            "Translation": combine_xyz_9,
        },
    )

    reroute_22 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_4}
    )

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    store_named_attribute_9 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_23, "Name": "joint10", "Value": 4},
        attrs={"data_type": "INT"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                store_named_attribute_3,
                store_named_attribute_6,
                store_named_attribute_7,
                store_named_attribute_8,
                store_named_attribute_9,
            ]
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata", singleton=False, type="GeometryNodeTree"
)
def nodegroup_add_jointed_geometry_metadata(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Name": group_input.outputs["Label"],
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": store_named_attribute},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_hinge_joint", singleton=False, type="GeometryNodeTree"
)
def nodegroup_hinge_joint(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketString", "Joint ID (do not set)", ""),
            ("NodeSocketString", "Joint Label", ""),
            ("NodeSocketString", "Parent Label", ""),
            ("NodeSocketGeometry", "Parent", None),
            ("NodeSocketString", "Child Label", ""),
            ("NodeSocketGeometry", "Child", None),
            ("NodeSocketVector", "Position", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "Axis", (0.0000, 0.0000, 1.0000)),
            ("NodeSocketFloat", "Value", 0.0000),
            ("NodeSocketFloat", "Min", 0.0000),
            ("NodeSocketFloat", "Max", 0.0000),
            ("NodeSocketBool", "Show Center of Parent", False),
            ("NodeSocketBool", "Show Center of Child", False),
            ("NodeSocketBool", "Show Joint", False),
        ],
    )

    named_attribute_4 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    integer = nw.new_node(Nodes.Integer)
    integer.integer = 0

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": named_attribute_4.outputs["Exists"],
            "False": integer,
            "True": named_attribute_4.outputs["Attribute"],
        },
        attrs={"input_type": "INT"},
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Parent"],
            "Name": "part_id",
            "Value": switch_2,
        },
        attrs={"data_type": "INT"},
    )

    named_attribute_1 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    attribute_statistic = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": store_named_attribute_1,
            "Attribute": named_attribute_1.outputs["Attribute"],
        },
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_1.outputs["Attribute"],
            3: attribute_statistic.outputs["Min"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_2 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": store_named_attribute_1, "Selection": equal},
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                separate_geometry_2.outputs["Selection"],
                separate_geometry_2.outputs["Inverted"],
            ]
        },
    )

    named_attribute_3 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    integer_1 = nw.new_node(Nodes.Integer)
    integer_1.integer = 1

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: named_attribute_3.outputs["Attribute"], 1: 1.0000}
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": named_attribute_3.outputs["Exists"],
            "False": integer_1,
            "True": add,
        },
        attrs={"input_type": "INT"},
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Child"],
            "Name": "part_id",
            "Value": switch_3,
        },
        attrs={"data_type": "INT"},
    )

    named_attribute_2 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    attribute_statistic_1 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": store_named_attribute,
            "Attribute": named_attribute_2.outputs["Attribute"],
        },
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_2.outputs["Attribute"],
            3: attribute_statistic_1.outputs["Min"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_3 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": store_named_attribute, "Selection": equal_1},
    )

    named_attribute_11 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "is_jointed"},
        attrs={"data_type": "BOOLEAN"},
    )

    attribute_statistic_7 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": separate_geometry_3.outputs["Selection"],
            "Attribute": named_attribute_11.outputs["Attribute"],
        },
    )

    greater_than = nw.new_node(
        Nodes.Compare,
        input_kwargs={1: 1.0000, 2: attribute_statistic_7.outputs["Sum"]},
        attrs={"data_type": "INT"},
    )

    combine_matrix = nw.new_node("FunctionNodeCombineMatrix")

    named_attribute_10 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": greater_than,
            "False": combine_matrix,
            "True": named_attribute_10.outputs["Attribute"],
        },
        attrs={"input_type": "MATRIX"},
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute,
            "Name": "body_transform",
            "Value": switch_1,
        },
        attrs={"data_type": "FLOAT4X4"},
    )

    named_attribute_7 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "is_jointed"},
        attrs={"data_type": "BOOLEAN"},
    )

    attribute_statistic_4 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": separate_geometry_3.outputs["Selection"],
            "Attribute": named_attribute_7.outputs["Attribute"],
        },
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: attribute_statistic_4.outputs["Sum"]},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    position_4 = nw.new_node(Nodes.InputPosition)

    position_1 = nw.new_node(Nodes.InputPosition)

    bounding_box = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry_2.outputs["Selection"]},
    )

    position = nw.new_node(Nodes.InputPosition)

    attribute_statistic_2 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": position,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: position_1, 1: attribute_statistic_2.outputs["Mean"]},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_2,
            "False": position_4,
            "True": add_1.outputs["Vector"],
        },
        attrs={"input_type": "VECTOR"},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": store_named_attribute_2, "Position": switch},
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_position, "Name": "is_jointed", "Value": True},
        attrs={"data_type": "BOOLEAN"},
    )

    position_3 = nw.new_node(Nodes.InputPosition)

    named_attribute_12 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    attribute_statistic_6 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": set_position,
            "Attribute": named_attribute_12.outputs["Attribute"],
        },
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_12.outputs["Attribute"],
            3: attribute_statistic_6.outputs["Min"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_4 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": set_position, "Selection": equal_3},
    )

    bounding_box_1 = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry_4.outputs["Selection"]},
    )

    position_2 = nw.new_node(Nodes.InputPosition)

    attribute_statistic_5 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box_1.outputs["Bounding Box"],
            "Attribute": position_2,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    named_attribute_5 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    transpose_matrix = nw.new_node(
        "FunctionNodeTransposeMatrix",
        input_kwargs={"Matrix": named_attribute_5.outputs["Attribute"]},
    )

    transform_point = nw.new_node(
        "FunctionNodeTransformPoint",
        input_kwargs={
            "Vector": group_input.outputs["Position"],
            "Transform": transpose_matrix,
        },
    )

    add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: attribute_statistic_5.outputs["Mean"], 1: transform_point},
    )

    transform_direction = nw.new_node(
        "FunctionNodeTransformDirection",
        input_kwargs={
            "Direction": group_input.outputs["Axis"],
            "Transform": transpose_matrix,
        },
    )

    equal_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: group_input.outputs["Min"], "Epsilon": 0.0000},
        attrs={"operation": "EQUAL"},
    )

    equal_5 = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: group_input.outputs["Max"], "Epsilon": 0.0000},
        attrs={"operation": "EQUAL"},
    )

    op_and = nw.new_node(Nodes.BooleanMath, input_kwargs={0: equal_4, 1: equal_5})

    clamp = nw.new_node(
        Nodes.Clamp,
        input_kwargs={
            "Value": group_input.outputs["Value"],
            "Min": group_input.outputs["Min"],
            "Max": group_input.outputs["Max"],
        },
    )

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": op_and,
            "False": clamp,
            "True": group_input.outputs["Value"],
        },
        attrs={"input_type": "FLOAT"},
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_5})

    vector_rotate = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={
            "Vector": position_3,
            "Center": add_2.outputs["Vector"],
            "Axis": transform_direction,
            "Angle": reroute,
        },
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": store_named_attribute_3, "Position": vector_rotate},
    )

    named_attribute_6 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    separate_matrix = nw.new_node(
        "FunctionNodeSeparateMatrix",
        input_kwargs={"Matrix": named_attribute_6.outputs["Attribute"]},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_matrix.outputs["Column 1 Row 1"],
            "Y": separate_matrix.outputs["Column 2 Row 1"],
            "Z": separate_matrix.outputs["Column 3 Row 1"],
        },
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_direction})

    vector_rotate_1 = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={"Vector": combine_xyz, "Axis": reroute_1, "Angle": reroute},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": vector_rotate_1}
    )

    named_attribute_8 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    separate_matrix_1 = nw.new_node(
        "FunctionNodeSeparateMatrix",
        input_kwargs={"Matrix": named_attribute_8.outputs["Attribute"]},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_matrix_1.outputs["Column 1 Row 2"],
            "Y": separate_matrix_1.outputs["Column 2 Row 2"],
            "Z": separate_matrix_1.outputs["Column 3 Row 2"],
        },
    )

    vector_rotate_2 = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={"Vector": combine_xyz_1, "Axis": reroute_1, "Angle": reroute},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": vector_rotate_2}
    )

    named_attribute_9 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    separate_matrix_2 = nw.new_node(
        "FunctionNodeSeparateMatrix",
        input_kwargs={"Matrix": named_attribute_9.outputs["Attribute"]},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_matrix_2.outputs["Column 1 Row 3"],
            "Y": separate_matrix_2.outputs["Column 2 Row 3"],
            "Z": separate_matrix_2.outputs["Column 3 Row 3"],
        },
    )

    vector_rotate_3 = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={"Vector": combine_xyz_2, "Axis": reroute_1, "Angle": reroute},
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": vector_rotate_3}
    )

    combine_matrix_1 = nw.new_node(
        "FunctionNodeCombineMatrix",
        input_kwargs={
            "Column 1 Row 1": separate_xyz.outputs["X"],
            "Column 1 Row 2": separate_xyz_1.outputs["X"],
            "Column 1 Row 3": separate_xyz_2.outputs["X"],
            "Column 2 Row 1": separate_xyz.outputs["Y"],
            "Column 2 Row 2": separate_xyz_1.outputs["Y"],
            "Column 2 Row 3": separate_xyz_2.outputs["Y"],
            "Column 3 Row 1": separate_xyz.outputs["Z"],
            "Column 3 Row 2": separate_xyz_1.outputs["Z"],
            "Column 3 Row 3": separate_xyz_2.outputs["Z"],
        },
    )

    store_named_attribute_4 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": set_position_1,
            "Name": "body_transform",
            "Value": combine_matrix_1,
        },
        attrs={"data_type": "FLOAT4X4"},
    )

    string_1 = nw.new_node("FunctionNodeInputString", attrs={"string": "pos"})

    reroute_4 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Joint ID (do not set)"]},
    )

    join_strings_1 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_1, reroute_4]},
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Position"]}
    )

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_4,
            "Name": join_strings_1,
            "Value": reroute_3,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    string_2 = nw.new_node("FunctionNodeInputString", attrs={"string": "axis"})

    join_strings_2 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_2, reroute_4]},
    )

    store_named_attribute_6 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_5,
            "Name": join_strings_2,
            "Value": group_input.outputs["Axis"],
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    string_3 = nw.new_node("FunctionNodeInputString", attrs={"string": "min"})

    join_strings_3 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_3, reroute_4]},
    )

    store_named_attribute_8 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_6,
            "Name": join_strings_3,
            "Value": group_input.outputs["Min"],
        },
    )

    string_4 = nw.new_node("FunctionNodeInputString", attrs={"string": "max"})

    join_strings_4 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_4, reroute_4]},
    )

    store_named_attribute_7 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_8,
            "Name": join_strings_4,
            "Value": group_input.outputs["Max"],
        },
    )

    uv_sphere = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Segments": 10, "Rings": 10, "Radius": 0.0500}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere.outputs["Mesh"],
            "Translation": attribute_statistic_2.outputs["Mean"],
        },
    )

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Center of Parent"],
            "True": transform_geometry,
        },
    )

    store_named_attribute_13 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_4, "Name": "part_id", "Value": 999999999},
        attrs={"data_type": "INT"},
    )

    uv_sphere_1 = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Segments": 10, "Rings": 10, "Radius": 0.0500}
    )

    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_position_1})

    named_attribute_13 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    attribute_statistic_10 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": reroute_2,
            "Attribute": named_attribute_13.outputs["Attribute"],
        },
    )

    equal_6 = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_13.outputs["Attribute"],
            3: attribute_statistic_10.outputs["Min"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_5 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": reroute_2, "Selection": equal_6},
    )

    bounding_box_3 = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry_5.outputs["Selection"]},
    )

    position_7 = nw.new_node(Nodes.InputPosition)

    attribute_statistic_9 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box_3.outputs["Bounding Box"],
            "Attribute": position_7,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere_1.outputs["Mesh"],
            "Translation": attribute_statistic_9.outputs["Mean"],
        },
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Center of Child"],
            "True": transform_geometry_1,
        },
    )

    store_named_attribute_14 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_6, "Name": "part_id", "Value": 999999999},
        attrs={"data_type": "INT"},
    )

    cone = nw.new_node(
        "GeometryNodeMeshCone", input_kwargs={"Radius Bottom": 0.0500, "Depth": 0.2000}
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cone.outputs["Mesh"],
            "Translation": (0.0000, 0.0000, -0.0500),
        },
    )

    bounding_box_4 = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry_4.outputs["Selection"]},
    )

    position_8 = nw.new_node(Nodes.InputPosition)

    attribute_statistic_11 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box_4.outputs["Bounding Box"],
            "Attribute": position_8,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    add_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Position"],
            1: attribute_statistic_11.outputs["Mean"],
        },
    )

    attribute_statistic_12 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": separate_geometry_5.outputs["Selection"],
            "Attribute": transform_direction,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    normalize = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: attribute_statistic_12.outputs["Mean"]},
        attrs={"operation": "NORMALIZE"},
    )

    align_rotation_to_vector_1 = nw.new_node(
        "FunctionNodeAlignRotationToVector",
        input_kwargs={"Vector": normalize.outputs["Vector"]},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_3,
            "Translation": add_3.outputs["Vector"],
            "Rotation": align_rotation_to_vector_1,
        },
    )

    switch_7 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Joint"],
            "True": transform_geometry_2,
        },
    )

    store_named_attribute_15 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_7, "Name": "part_id", "Value": 999999999},
        attrs={"data_type": "INT"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                join_geometry_2,
                store_named_attribute_7,
                store_named_attribute_13,
                store_named_attribute_14,
                store_named_attribute_15,
            ]
        },
    )

    store_named_attribute_9 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_4,
            "Name": join_strings_1,
            "Value": reroute_3,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute_10 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_9,
            "Name": join_strings_2,
            "Value": group_input.outputs["Axis"],
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute_12 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_10,
            "Name": join_strings_3,
            "Value": group_input.outputs["Min"],
        },
    )

    store_named_attribute_11 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_12,
            "Name": join_strings_4,
            "Value": group_input.outputs["Max"],
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                store_named_attribute_11,
                store_named_attribute_13,
                store_named_attribute_14,
                store_named_attribute_15,
            ]
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": join_geometry,
            "Parent": join_geometry_2,
            "Child": join_geometry_1,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_sliding_joint", singleton=False, type="GeometryNodeTree"
)
def nodegroup_sliding_joint(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketString", "Joint ID (do not set)", ""),
            ("NodeSocketString", "Joint Label", ""),
            ("NodeSocketString", "Parent Label", ""),
            ("NodeSocketGeometry", "Parent", None),
            ("NodeSocketString", "Child Label", ""),
            ("NodeSocketGeometry", "Child", None),
            ("NodeSocketVector", "Position", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "Axis", (0.0000, 0.0000, 1.0000)),
            ("NodeSocketFloat", "Value", 0.0000),
            ("NodeSocketFloat", "Min", 0.0000),
            ("NodeSocketFloat", "Max", 0.0000),
            ("NodeSocketBool", "Show Center of Parent", False),
            ("NodeSocketBool", "Show Center of Child", False),
            ("NodeSocketBool", "Show Joint", False),
        ],
    )

    named_attribute_4 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    integer = nw.new_node(Nodes.Integer)
    integer.integer = 0

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": named_attribute_4.outputs["Exists"],
            "False": integer,
            "True": named_attribute_4.outputs["Attribute"],
        },
        attrs={"input_type": "INT"},
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Parent"],
            "Name": "part_id",
            "Value": switch_2,
        },
        attrs={"data_type": "INT"},
    )

    named_attribute_1 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    attribute_statistic = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": store_named_attribute_1,
            "Attribute": named_attribute_1.outputs["Attribute"],
        },
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_1.outputs["Attribute"],
            3: attribute_statistic.outputs["Min"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_2 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": store_named_attribute_1, "Selection": equal},
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                separate_geometry_2.outputs["Selection"],
                separate_geometry_2.outputs["Inverted"],
            ]
        },
    )

    cone = nw.new_node(
        "GeometryNodeMeshCone", input_kwargs={"Radius Bottom": 0.0500, "Depth": 0.2000}
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cone.outputs["Mesh"],
            "Translation": (0.0000, 0.0000, -0.0500),
        },
    )

    named_attribute_3 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    integer_1 = nw.new_node(Nodes.Integer)
    integer_1.integer = 1

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: named_attribute_3.outputs["Attribute"], 1: 1.0000}
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": named_attribute_3.outputs["Exists"],
            "False": integer_1,
            "True": add,
        },
        attrs={"input_type": "INT"},
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Child"],
            "Name": "part_id",
            "Value": switch_3,
        },
        attrs={"data_type": "INT"},
    )

    named_attribute_2 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    attribute_statistic_1 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": store_named_attribute,
            "Attribute": named_attribute_2.outputs["Attribute"],
        },
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_2.outputs["Attribute"],
            3: attribute_statistic_1.outputs["Min"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_3 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": store_named_attribute, "Selection": equal_1},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_geometry_3.outputs["Selection"]}
    )

    bounding_box_3 = nw.new_node(Nodes.BoundingBox, input_kwargs={"Geometry": reroute})

    position_7 = nw.new_node(Nodes.InputPosition)

    attribute_statistic_9 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box_3.outputs["Bounding Box"],
            "Attribute": position_7,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Position"],
            1: attribute_statistic_9.outputs["Mean"],
        },
    )

    named_attribute_5 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    transpose_matrix = nw.new_node(
        "FunctionNodeTransposeMatrix",
        input_kwargs={"Matrix": named_attribute_5.outputs["Attribute"]},
    )

    transform_direction = nw.new_node(
        "FunctionNodeTransformDirection",
        input_kwargs={
            "Direction": group_input.outputs["Axis"],
            "Transform": transpose_matrix,
        },
    )

    attribute_statistic_5 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={"Geometry": reroute, "Attribute": transform_direction},
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    normalize = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: attribute_statistic_5.outputs["Mean"]},
        attrs={"operation": "NORMALIZE"},
    )

    align_rotation_to_vector_1 = nw.new_node(
        "FunctionNodeAlignRotationToVector",
        input_kwargs={"Vector": normalize.outputs["Vector"]},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_3,
            "Translation": add_1.outputs["Vector"],
            "Rotation": align_rotation_to_vector_1,
        },
    )

    switch_7 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Joint"],
            "True": transform_geometry_2,
        },
    )

    store_named_attribute_15 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_7, "Name": "part_id", "Value": 999999999},
        attrs={"data_type": "INT"},
    )

    uv_sphere_1 = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Segments": 10, "Rings": 10, "Radius": 0.0500}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere_1.outputs["Mesh"],
            "Translation": attribute_statistic_9.outputs["Mean"],
        },
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Center of Child"],
            "True": transform_geometry_1,
        },
    )

    store_named_attribute_14 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_6, "Name": "part_id", "Value": 999999999},
        attrs={"data_type": "INT"},
    )

    uv_sphere = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Segments": 10, "Rings": 10, "Radius": 0.0500}
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry_2.outputs["Selection"]},
    )

    position = nw.new_node(Nodes.InputPosition)

    attribute_statistic_2 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": position,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere.outputs["Mesh"],
            "Translation": attribute_statistic_2.outputs["Mean"],
        },
    )

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Center of Parent"],
            "True": transform_geometry,
        },
    )

    store_named_attribute_13 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_4, "Name": "part_id", "Value": 999999999},
        attrs={"data_type": "INT"},
    )

    named_attribute_11 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "is_jointed"},
        attrs={"data_type": "BOOLEAN"},
    )

    attribute_statistic_7 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": separate_geometry_3.outputs["Selection"],
            "Attribute": named_attribute_11.outputs["Attribute"],
        },
    )

    greater_than = nw.new_node(
        Nodes.Compare,
        input_kwargs={1: 1.0000, 2: attribute_statistic_7.outputs["Sum"]},
        attrs={"data_type": "INT"},
    )

    combine_matrix = nw.new_node("FunctionNodeCombineMatrix")

    named_attribute_10 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": greater_than,
            "False": combine_matrix,
            "True": named_attribute_10.outputs["Attribute"],
        },
        attrs={"input_type": "MATRIX"},
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute,
            "Name": "body_transform",
            "Value": switch_1,
        },
        attrs={"data_type": "FLOAT4X4"},
    )

    named_attribute_7 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "is_jointed"},
        attrs={"data_type": "BOOLEAN"},
    )

    attribute_statistic_4 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": separate_geometry_3.outputs["Selection"],
            "Attribute": named_attribute_7.outputs["Attribute"],
        },
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: attribute_statistic_4.outputs["Sum"]},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    position_4 = nw.new_node(Nodes.InputPosition)

    position_1 = nw.new_node(Nodes.InputPosition)

    add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: position_1, 1: attribute_statistic_2.outputs["Mean"]},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_2,
            "False": position_4,
            "True": add_2.outputs["Vector"],
        },
        attrs={"input_type": "VECTOR"},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": store_named_attribute_2, "Position": switch},
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_position, "Name": "is_jointed", "Value": True},
        attrs={"data_type": "BOOLEAN"},
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: group_input.outputs["Min"], "Epsilon": 0.0000},
        attrs={"operation": "EQUAL"},
    )

    equal_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: group_input.outputs["Max"], "Epsilon": 0.0000},
        attrs={"operation": "EQUAL"},
    )

    op_and = nw.new_node(Nodes.BooleanMath, input_kwargs={0: equal_3, 1: equal_4})

    clamp = nw.new_node(
        Nodes.Clamp,
        input_kwargs={
            "Value": group_input.outputs["Value"],
            "Min": group_input.outputs["Min"],
            "Max": group_input.outputs["Max"],
        },
    )

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": op_and,
            "False": clamp,
            "True": group_input.outputs["Value"],
        },
        attrs={"input_type": "FLOAT"},
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: transform_direction, "Scale": switch_5},
        attrs={"operation": "SCALE"},
    )

    position_5 = nw.new_node(Nodes.InputPosition)

    add_3 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: scale.outputs["Vector"], 1: position_5}
    )

    set_position_2 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": store_named_attribute_3,
            "Position": add_3.outputs["Vector"],
        },
    )

    string = nw.new_node("FunctionNodeInputString", attrs={"string": "pos"})

    reroute_2 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Joint ID (do not set)"]},
    )

    join_strings = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string, reroute_2]},
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Position"]}
    )

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": set_position_2,
            "Name": join_strings,
            "Value": reroute_1,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    string_1 = nw.new_node("FunctionNodeInputString", attrs={"string": "axis"})

    join_strings_1 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_1, reroute_2]},
    )

    store_named_attribute_6 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_5,
            "Name": join_strings_1,
            "Value": group_input.outputs["Axis"],
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    string_2 = nw.new_node("FunctionNodeInputString", attrs={"string": "min"})

    join_strings_2 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_2, reroute_2]},
    )

    store_named_attribute_8 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_6,
            "Name": join_strings_2,
            "Value": group_input.outputs["Min"],
        },
    )

    string_3 = nw.new_node("FunctionNodeInputString", attrs={"string": "max"})

    join_strings_3 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_3, reroute_2]},
    )

    store_named_attribute_7 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_8,
            "Name": join_strings_3,
            "Value": group_input.outputs["Max"],
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                join_geometry_2,
                store_named_attribute_15,
                store_named_attribute_14,
                store_named_attribute_13,
                store_named_attribute_7,
            ]
        },
    )

    store_named_attribute_9 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": set_position_2,
            "Name": join_strings,
            "Value": reroute_1,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute_10 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_9,
            "Name": join_strings_1,
            "Value": group_input.outputs["Axis"],
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute_12 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_10,
            "Name": join_strings_2,
            "Value": group_input.outputs["Min"],
        },
    )

    store_named_attribute_11 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_12,
            "Name": join_strings_3,
            "Value": group_input.outputs["Max"],
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                store_named_attribute_15,
                store_named_attribute_14,
                store_named_attribute_13,
                store_named_attribute_11,
            ]
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": join_geometry,
            "Parent": join_geometry_2,
            "Child": join_geometry_1,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_duplicate_joints_on_parent_002", singleton=False, type="GeometryNodeTree"
)
def nodegroup_duplicate_joints_on_parent_002(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketString", "Duplicate ID (do not set)", ""),
            ("NodeSocketGeometry", "Parent", None),
            ("NodeSocketGeometry", "Child", None),
            ("NodeSocketGeometry", "Points", None),
        ],
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": group_input.outputs["Points"],
            "Instance": group_input.outputs["Child"],
        },
    )

    index = nw.new_node(Nodes.Index)

    add = nw.new_node(Nodes.Math, input_kwargs={0: index, 1: 1.0000})

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": instance_on_points,
            "Name": group_input.outputs["Duplicate ID (do not set)"],
            "Value": add,
        },
        attrs={"domain": "INSTANCE", "data_type": "INT"},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": store_named_attribute}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [group_input.outputs["Parent"], realize_instances]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Depth", 1.9600),
            ("NodeSocketFloat", "Width", 1.4000),
            ("NodeSocketFloat", "Height", 2.1400),
            ("NodeSocketFloat", "DoorThickness", 0.0300),
            ("NodeSocketFloat", "RackRadius", 0.0100),
            ("NodeSocketInt", "RackAmount", 3),
            ("NodeSocketMaterial", "Surface", None),
            ("NodeSocketMaterial", "Front", None),
            ("NodeSocketMaterial", "Button", None),
            ("NodeSocketMaterial", "WhiteMetal", None),
            ("NodeSocketMaterial", "Buttons", None),
            ("NodeSocketFloat", "hinge_rack", -1.4000),
            ("NodeSocketFloat", "hinge_door", 0.0000),
            ("NodeSocketFloat", "hinge_button", 1.3100),
            ("NodeSocketInt", "buttons_amount", 6),
            ("NodeSocketFloat", "RackHeight", 0.1000),
            ("NodeSocketFloat", "rotate_button", -11.0000),
            ("NodeSocketInt", "ElementAmountRack", 4),
            ("NodeSocketInt", "button_type", 0),
            ("NodeSocketFloat", "racks_depth", 0.5000),
            ("NodeSocketFloat", "handle_variations_x", 1.0000),
            ("NodeSocketFloat", "handle_variations_y", 1.0000),
            ("NodeSocketInt", "handle_type", 0),
            ("NodeSocketFloat", "Radius", 0.0400),
            ("NodeSocketInt", "has_handle", 0),
            ("NodeSocketFloat", "handle_position", 0.0000),
            ("NodeSocketInt", "button_position", 0),
            ("NodeSocketInt", "curved_handle", 0),
        ],
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["RackAmount"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    reroute_62 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    reroute_69 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_62})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_69},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    string = nw.new_node("FunctionNodeInputString", attrs={"string": "duplicate0"})

    string_1 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint1"})

    string_2 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint2"})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Depth"],
            "Y": group_input.outputs["Width"],
            "Z": group_input.outputs["Height"],
        },
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["DoorThickness"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    nodegroup_hollow_cube_no_gc = nw.new_node(
        nodegroup_hollow_cube().name,
        input_kwargs={"Size": combine_xyz, "Thickness": reroute_7},
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": nodegroup_hollow_cube_no_gc,
            "Translation": (0.0000, 0.0010, 0.0010),
            "Scale": (1.0000, 0.9990, 0.9990),
        },
    )

    reroute_47 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["WhiteMetal"]}
    )

    reroute_48 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_47})

    set_material_6 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_4, "Material": reroute_48},
    )

    store_named_attribute_6 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material_6, "Name": "joint9"},
        attrs={"data_type": "INT"},
    )

    reroute_67 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": nodegroup_hollow_cube_no_gc}
    )

    reroute_68 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_67})

    store_named_attribute_7 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_68, "Name": "joint9", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_6, store_named_attribute_7]},
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": join_geometry_2, "Label": "dishwasher_body"},
    )

    reroute_49 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Surface"]}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata,
            "Material": reroute_49,
        },
    )

    subdivide_mesh = nw.new_node(
        Nodes.SubdivideMesh, input_kwargs={"Mesh": set_material, "Level": 0}
    )

    body = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": subdivide_mesh}, label="Body"
    )

    reroute_91 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": body})

    store_named_attribute_8 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_91, "Name": "joint2"},
        attrs={"data_type": "INT"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Depth"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_57 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_75 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_57})

    reroute_61 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    reroute_71 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_61})

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_60 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    reroute_78 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_60})

    reroute_79 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_78})

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["hinge_button"]}
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_58 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    reroute_59 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_58})

    reroute_76 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_59})

    reroute_77 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_76})

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["DoorThickness"],
            "Y": group_input.outputs["Width"],
            "Z": group_input.outputs["Height"],
        },
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_1})

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Name": "uv_map",
            "Value": cube.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": store_named_attribute, "Name": "uv_map"},
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input.outputs["Depth"]}
    )

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_1, 1: (0.5000, 0.5000, 0.5000), 2: combine_xyz_2},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_55 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply_add.outputs["Vector"]}
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": store_named_attribute_1, "Translation": reroute_55},
    )

    reroute_74 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_3}
    )

    position = nw.new_node(Nodes.InputPosition)

    nodegroup_center_no_gc = nw.new_node(
        nodegroup_center().name,
        input_kwargs={
            "Geometry": transform_geometry_3,
            "Vector": position,
            "MarginX": -1.0000,
            "MarginY": 0.1000,
            "MarginZ": 0.1500,
        },
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": reroute_74,
            "Selection": nodegroup_center_no_gc.outputs["In"],
            "Material": reroute_49,
        },
    )

    reroute_84 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_49})

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": set_material_1, "Material": reroute_84},
    )

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_2, "Label": "rotation_door"},
    )

    reroute_18 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["button_type"]}
    )

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={3: reroute_19},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["buttons_amount"]}
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["rotate_button"]}
    )

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Button"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_37 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["has_handle"]}
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: 1, 3: reroute_37},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    reroute_73 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": equal_2})

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.0010

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0100

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_73, "False": value_1, "True": value},
        attrs={"input_type": "FLOAT"},
    )

    reroute_38 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["button_position"]}
    )

    reroute_39 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_38})

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_39, 3: 1},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    reroute_45 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Buttons"]}
    )

    reroute_46 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_45})

    reroute_80 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_46})

    reroute_82 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_80})

    reroute_83 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_82})

    buttons = nw.new_node(
        nodegroup_buttons().name,
        input_kwargs={
            "depth": reroute_75,
            "door_thickness": reroute_71,
            "height": reroute_79,
            "hinge_button": reroute_13,
            "relative_position": -0.2000,
            "width": reroute_77,
            "Parent": add_jointed_geometry_metadata_1,
            "button_type": equal_1,
            "buttons_amount": reroute_15,
            "rotate_button": reroute_17,
            "Material": reroute_11,
            "X": switch_1,
            "Width": reroute_77,
            "Switch": equal_3,
            "New_Material": reroute_83,
        },
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": buttons, "Name": "joint3"},
        attrs={"data_type": "INT"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: 0.0500},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_50 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply})

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_50})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_3, 1: 0.8000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply}, attrs={"operation": "MULTIPLY"}
    )

    reroute_28 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["handle_variations_x"]},
    )

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_28})

    reroute_30 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["handle_variations_y"]},
    )

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_30})

    reroute_36 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["handle_type"]}
    )

    equal_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_36, 3: 1},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    reroute_32 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Radius"]}
    )

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_32})

    reroute_34 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["handle_position"]}
    )

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_34})

    reroute_40 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["curved_handle"]}
    )

    equal_5 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_40},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    nodegroup_handle_no_gc = nw.new_node(
        nodegroup_handle().name,
        input_kwargs={
            "width": reroute_51,
            "length": multiply_1,
            "thickness": multiply_2,
            "X": reroute_29,
            "Y": reroute_31,
            "Switch": equal_4,
            "Radius": reroute_33,
            "handle_position": reroute_35,
            "curved_handle": equal_5,
        },
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

    reroute_52 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_4})

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": multiply_3, "Z": reroute_52}
    )

    reroute_63 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_3})

    reroute_64 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_63})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": nodegroup_handle_no_gc.outputs["Geometry"],
            "Translation": reroute_64,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    reroute_70 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_48})

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry, "Material": reroute_70},
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material_3, "Name": "switch2"},
        attrs={"data_type": "INT"},
    )

    switch = nw.new_node(
        Nodes.Switch, input_kwargs={"Switch": equal_2, "False": store_named_attribute_3}
    )

    realize_instances_1 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": switch, "Depth": 1}
    )

    vector_1 = nw.new_node(Nodes.Vector)
    vector_1.vector = (0.0000, 0.0000, 0.0000)

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": realize_instances_1, "Offset": vector_1},
    )

    add_jointed_geometry_metadata_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_position_1, "Label": "handle"},
    )

    reroute_89 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_2}
    )

    store_named_attribute_4 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_89, "Name": "joint3", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_2, store_named_attribute_4]},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_57, 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_59, 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_60, 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_12 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide, "Y": divide_1, "Z": divide_2}
    )

    reroute_85 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_12})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Translation": reroute_85},
    )

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_1, "Name": "joint2", "Value": 1},
        attrs={"data_type": "INT"},
    )

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: nodegroup_handle_no_gc.outputs["Handle_center"], 1: 1.6700},
        attrs={"operation": "DIVIDE"},
    )

    divide_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_61, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: divide_3, 1: divide_4})

    subtract = nw.new_node(
        Nodes.Math, input_kwargs={0: 0.0000, 1: add_1}, attrs={"operation": "SUBTRACT"}
    )

    reroute_86 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract})

    reroute_81 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide_2})

    combine_xyz_10 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_86, "Z": reroute_81}
    )

    reroute_88 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_10})

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "door_joint",
            "Parent": store_named_attribute_8,
            "Child": store_named_attribute_5,
            "Position": reroute_88,
            "Axis": (0.0000, 1.0000, 0.0000),
            "Max": 1.5000,
        },
    )

    store_named_attribute_9 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": hinge_joint.outputs["Geometry"], "Name": "joint1"},
        attrs={"data_type": "INT"},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["DoorThickness"], 1: 2.1000},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_53 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_5})

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_3, 1: reroute_53},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_66 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract_1})

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["DoorThickness"], 1: 2.1000},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_54 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_6})

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: reroute_54},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_65 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract_2})

    reroute_22 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["RackRadius"]}
    )

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    reroute_26 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ElementAmountRack"]}
    )

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_26})

    reroute_24 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["RackHeight"]}
    )

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_24})

    reroute_20 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["racks_depth"]}
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    nodegroup_dish_rack_no_gc = nw.new_node(
        nodegroup_dish_rack().name,
        input_kwargs={
            "Depth": reroute_66,
            "Width": reroute_65,
            "Radius": reroute_23,
            "Amount": reroute_27,
            "Height": reroute_25,
            "Value": reroute_21,
        },
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_7, "Y": multiply_8, "Z": 0.5000}
    )

    reroute_56 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_7})

    set_position_2 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": nodegroup_dish_rack_no_gc, "Offset": reroute_56},
    )

    set_material_5 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": set_position_2, "Material": reroute_70},
    )

    realize_instances_3 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": set_material_5}
    )

    reroute_87 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": realize_instances_3})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_87, "Translation": combine_xyz_12},
    )

    add_jointed_geometry_metadata_3 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_2, "Label": "shelf"},
    )

    reroute_90 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_3}
    )

    store_named_attribute_10 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_90, "Name": "joint1", "Value": 1},
        attrs={"data_type": "INT"},
    )

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "rack_joint",
            "Parent": store_named_attribute_9,
            "Child": store_named_attribute_10,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Max": 1.0000,
        },
    )

    divide_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: 2.6000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_11 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide_5})

    mesh_line = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={"Count": reroute_62, "Offset": combine_xyz_11},
        attrs={"mode": "END_POINTS"},
    )

    reroute_72 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": mesh_line})

    duplicate_joints_on_parent_002 = nw.new_node(
        nodegroup_duplicate_joints_on_parent_002().name,
        input_kwargs={
            "Duplicate ID (do not set)": string,
            "Parent": sliding_joint.outputs["Parent"],
            "Child": sliding_joint.outputs["Child"],
            "Points": reroute_72,
        },
    )

    store_named_attribute_11 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": duplicate_joints_on_parent_002, "Name": "switch0"},
        attrs={"data_type": "INT"},
    )

    reroute_92 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint.outputs["Geometry"]}
    )

    reroute_93 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_92})

    store_named_attribute_12 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_93, "Name": "switch0", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal,
            "False": store_named_attribute_11,
            "True": store_named_attribute_12,
        },
    )

    store_named_attribute_13 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_2, "Name": "joint0"},
        attrs={"data_type": "INT"},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": store_named_attribute_13}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_1},
        attrs={"is_active_output": True},
    )


def sample_white_interior():
    """Generate a white or near-white color for the lamp shade interior"""
    # Very high value (brightness), low saturation
    h = np.random.uniform(0, 1)  # Hue can be any value since saturation is low
    s = np.random.uniform(0, 0.1)  # Very low saturation to keep it close to white
    v = np.random.uniform(0.9, 1.0)  # High value for brightness

    # Convert to RGB
    rgb = hsv2rgba(h, s, v)
    return rgb


def sample_light_exterior():
    """Generate a light color for the lamp shade exterior"""
    # Light pastel colors - high value, moderate-low saturation
    h = np.random.uniform(0, 1)  # Any hue is possible
    s = np.random.uniform(0.1, 0.3)  # Low-moderate saturation for pastel effect
    v = np.random.uniform(0.8, 0.95)  # High value but not too bright

    # Convert to RGB
    rgb = hsv2rgba(h, s, v)
    return rgb


def get_all_metal_shaders(color):
    metal_shaders_list = [
        metal.brushed_metal.shader_brushed_metal,
        metal.galvanized_metal.shader_galvanized_metal,
        metal.grained_and_polished_metal.shader_grained_metal,
        metal.hammered_metal.shader_hammered_metal,
    ]
    new_shaders = [
        functools.partial(shader, base_color=color) for shader in metal_shaders_list
    ]
    for idx, ns in enumerate(new_shaders):
        # fix taken from: https://github.com/elastic/apm-agent-python/issues/293
        ns.__name__ = metal_shaders_list[idx].__name__

    return new_shaders


def sample_gold():
    """Generate a gold color variation"""
    # Gold colors are generally in yellow-orange hue range
    # 36/360 to 56/360 converted to 0-1 scale
    h = np.random.uniform(0.1, 0.155)  # Gold hue range
    s = np.random.uniform(0.65, 0.9)  # Moderate to high saturation
    v = np.random.uniform(0.75, 1.0)  # Bright

    # Convert to RGB
    rgb = hsv2rgba(h, s, v)
    return rgb


def sample_silver():
    """Generate a silver color variation"""
    # Silver colors are desaturated with high brightness
    h = np.random.uniform(0, 1)  # Hue doesn't matter much due to low saturation
    s = np.random.uniform(0, 0.1)  # Very low saturation
    v = np.random.uniform(0.75, 0.9)  # High but not maximum brightness

    # Convert to RGB
    rgb = hsv2rgba(h, s, v)
    return rgb


class DishwasherFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    def sample_joint_parameters(self):
        return {
            "door_joint": {
                "stiffness": np.random.uniform(1000, 10000),
                "damping": np.random.uniform(500, 2000),
                "frictionloss": np.random.uniform(1000, 1500),
            },
            "rack_joint": {
                "stiffness": 0,
                "damping": np.random.uniform(60, 80),
            },
            "button_joint": {
                "stiffness": np.random.uniform(1, 5),
                "damping": np.random.uniform(0, 10),
            },
            "knob_joint": {"stiffness": 0, "damping": np.random.uniform(0, 10)},
        }

    def sample_parameters(self):
        with FixedSeed(self.factory_seed):
            # add code here to randomly sample from parameters
            depth = U(1.0, 1.5)
            width = U(1.2, 1.6)
            height = U(1.7, 2.1)
            door_thickness = U(0.05, 0.07) * depth
            rack_radius = U(0.015, 0.02)
            rack_h_amount = np.random.choice([1, 2, 3])
            buttons_amount = RI(0, 6)
            button_type = RI(0, 2)  # 0 for round, 1 for square
            ElementAmountRack = RI(3, 6)  # Number of elements on the rack
            RackHeight = U(0.18, 0.3)  # Height of the rack elements
            racks_depth = U(0.3, 0.5)  # Depth of the racks
            handle_variations_x = U(1, 1.9)
            handle_variations_y = U(1, 1.15)
            handle_type = RI(0, 2)  # 0 for round, 1 for square
            Radius = U(0.03, 0.04)
            handle_position = U(0.1, 0.2)  # Position of the handle
            button_position = RI(0, 1)
            curved_handle = RI(0, 1)

            dishwasher_materials = (
                (metal.MetalBasic, 2.0),
                (metal.BrushedMetal, 2.0),
                (metal.GalvanizedMetal, 2.0),
                (metal.BrushedBlackMetal, 2.0),
                (plastic.Plastic, 1.0),
                (plastic.PlasticRough, 1.0),
            )

            body_shader = weighted_sample(dishwasher_materials)()
            front_shader = weighted_sample(dishwasher_materials)()
            button_shader = weighted_sample(dishwasher_materials)()
            knob_shader = weighted_sample(dishwasher_materials)()
            knob_shader_2 = weighted_sample(dishwasher_materials)()

            body_mat = body_shader.generate()
            front_mat = front_shader.generate()
            button_mat = button_shader.generate()
            knob_mat = knob_shader.generate()
            knob_mat_2 = knob_shader_2.generate()

            params = {
                "Depth": depth,
                "Width": width,
                "Height": height,
                "DoorThickness": door_thickness,
                "RackRadius": rack_radius,
                "RackAmount": rack_h_amount,
                "button_type": button_type,
                "buttons_amount": buttons_amount,
                "ElementAmountRack": ElementAmountRack,
                "RackHeight": RackHeight,
                "racks_depth": racks_depth,
                "handle_variations_x": handle_variations_x,
                "handle_variations_y": handle_variations_y,
                "handle_type": handle_type,
                "Radius": Radius,
                "handle_position": handle_position,
                "button_position": button_position,
                "curved_handle": curved_handle,
                "Surface": body_mat,
                "Front": front_mat,
                "Button": knob_mat,
                "WhiteMetal": button_mat,
                "Buttons": knob_mat_2,
            }
            return params

    def create_asset(self, **kwargs):
        obj = butil.spawn_vert()
        butil.modify_mesh(
            obj,
            "NODES",
            apply=False,
            node_group=geometry_nodes(),
            ng_inputs=self.sample_parameters(),
        )

        return obj
