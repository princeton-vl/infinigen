# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Stamatis Alexandropoulos: primary author
# - Abhishek Joshi: Updates for sim
# - Max Gonzalez Saez-Diez: Updates for sim
# - Hongyu Wen: developed original dishwasher

import gin
import numpy as np
from numpy.random import randint as RI
from numpy.random import uniform as U

from infinigen.assets.materials import (
    metal,
    plastic,
)
from infinigen.assets.utils.joints import (
    nodegroup_add_jointed_geometry_metadata,
    nodegroup_duplicate_joints_on_parent,
    nodegroup_hinge_joint,
    nodegroup_sliding_joint,
)
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import weighted_sample


@node_utils.to_nodegroup("nodegroup_center", singleton=False, type="GeometryNodeTree")
def nodegroup_center(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

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


@node_utils.to_nodegroup("nodegroup_handle", singleton=False, type="GeometryNodeTree")
def nodegroup_handle(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

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

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute, transform_geometry]},
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
            "Scale": (0.4000, 1.0000, 1.8000),
        },
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_1,
            "Scale": (1.0000, 1.0000, 0.5000),
        },
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["curved_handle"],
            "False": transform_geometry_1,
            "True": transform_geometry_5,
        },
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Switch"],
            "False": transform_geometry_1,
            "True": switch_2,
        },
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
            "Vertices": 3,
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

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["thickness"]}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute, "Y": add, "Z": reroute_4}
    )

    cube_2 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_2})

    subdivide_mesh = nw.new_node(
        Nodes.SubdivideMesh, input_kwargs={"Mesh": cube_2.outputs["Mesh"], "Level": 2}
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

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["curved_handle"],
            "False": cube_2.outputs["Mesh"],
            "True": set_position_1,
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

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Switch"],
            "False": store_named_attribute_2,
            "True": store_named_attribute_3,
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

    add_2 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: reroute_6, 1: (0.0000, 0.0000, 0.0200)}
    )

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
            "Translation": add_2.outputs["Vector"],
            "Scale": combine_xyz_4,
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [switch_3, transform_geometry_2]}
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


@node_utils.to_nodegroup(
    "nodegroup_dish_rack", singleton=False, type="GeometryNodeTree"
)
def nodegroup_dish_rack(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    quadrilateral = nw.new_node("GeometryNodeCurvePrimitiveQuadrilateral")

    curve_line = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (0.0000, -1.0000, 0.0000),
            "End": (0.0000, 1.0000, 0.0000),
        },
    )

    geometry_to_instance_2 = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": curve_line}
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

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Amount"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    duplicate_elements_1 = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={"Geometry": geometry_to_instance_2, "Amount": multiply},
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

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: group_input.outputs["Amount"]},
        attrs={"operation": "DIVIDE"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: divide},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

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

    quadrilateral_1 = nw.new_node("GeometryNodeCurvePrimitiveQuadrilateral")

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: 0.8000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_2})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": quadrilateral_1, "Translation": combine_xyz_4},
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

    duplicate_elements = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={"Geometry": geometry_to_instance, "Amount": multiply},
        attrs={"domain": "INSTANCE"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: divide},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_3})

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": duplicate_elements.outputs["Geometry"],
            "Offset": combine_xyz_1,
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [curve_line, set_position]}
    )

    geometry_to_instance_1 = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": join_geometry}
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

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: divide},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_4})

    set_position_2 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": duplicate_elements_2.outputs["Geometry"],
            "Offset": combine_xyz_3,
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                quadrilateral,
                transform_geometry,
                transform_geometry_1,
                set_position_2,
            ]
        },
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={"Resolution": 3, "Radius": group_input.outputs["Radius"]},
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


@node_utils.to_nodegroup(
    "nodegroup_nodegrou_hollow_cube", singleton=False, type="GeometryNodeTree"
)
def nodegroup_nodegrou_hollow_cube(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

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

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz.outputs["X"],
            "Y": subtract,
            "Z": group_input.outputs["Thickness"],
        },
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_2,
            "Vertices X": group_input.outputs["Resolution"],
            "Vertices Y": group_input.outputs["Resolution"],
            "Vertices Z": group_input.outputs["Resolution"],
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

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Pos"]}
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: separate_xyz_1.outputs["X"]},
    )

    add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: separate_xyz_1.outputs["Y"]},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": add_1, "Z": subtract_1}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_1,
            "Translation": combine_xyz_3,
        },
    )

    combine_xyz_8 = nw.new_node(
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
            "Size": combine_xyz_8,
            "Vertices X": group_input.outputs["Resolution"],
            "Vertices Y": group_input.outputs["Resolution"],
            "Vertices Z": group_input.outputs["Resolution"],
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

    add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: separate_xyz_2.outputs["X"]},
    )

    add_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: separate_xyz_1.outputs["Y"], 1: multiply_1}
    )

    add_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Z"], 1: separate_xyz_2.outputs["Z"]},
    )

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_2, "Y": add_3, "Z": add_4}
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_4,
            "Translation": combine_xyz_9,
        },
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
        input_kwargs={
            "X": group_input.outputs["Thickness"],
            "Y": subtract_2,
            "Z": subtract_3,
        },
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz,
            "Vertices X": group_input.outputs["Resolution"],
            "Vertices Y": group_input.outputs["Resolution"],
            "Vertices Z": group_input.outputs["Resolution"],
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

    add_5 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_1, 1: separate_xyz_1.outputs["X"]}
    )

    add_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: separate_xyz_1.outputs["Y"]},
    )

    subtract_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: separate_xyz_1.outputs["Z"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_5, "Y": add_6, "Z": subtract_4}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": store_named_attribute, "Translation": combine_xyz_1},
    )

    subtract_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz.outputs["X"],
            "Y": subtract_5,
            "Z": group_input.outputs["Thickness"],
        },
    )

    cube_2 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_4,
            "Vertices X": group_input.outputs["Resolution"],
            "Vertices Y": group_input.outputs["Resolution"],
            "Vertices Z": group_input.outputs["Resolution"],
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

    add_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: separate_xyz_1.outputs["X"]},
    )

    add_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: separate_xyz_1.outputs["Y"]},
    )

    add_9 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_1, 1: separate_xyz_1.outputs["Z"]}
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_7, "Y": add_8, "Z": add_9}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_2,
            "Translation": combine_xyz_5,
            "Scale": (1.0000, 1.0000, 0.9800),
        },
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
            "Vertices X": group_input.outputs["Resolution"],
            "Vertices Y": group_input.outputs["Resolution"],
            "Vertices Z": group_input.outputs["Resolution"],
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

    add_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: separate_xyz_1.outputs["X"]},
    )

    subtract_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    add_11 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: separate_xyz_1.outputs["Z"]},
    )

    combine_xyz_11 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_10, "Y": subtract_6, "Z": add_11}
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_5,
            "Translation": combine_xyz_11,
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                transform_geometry_1,
                transform_geometry_4,
                transform_geometry,
                transform_geometry_2,
                transform_geometry_5,
            ]
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry, "base": transform_geometry_2},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

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

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["RackAmount"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Depth"],
            "Y": group_input.outputs["Width"],
            "Z": group_input.outputs["Height"],
        },
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["DoorThickness"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    nodegroup_hollow_cube = nw.new_node(
        nodegroup_nodegrou_hollow_cube().name,
        input_kwargs={"Size": combine_xyz, "Thickness": reroute_3},
        label="nodegroup_hollow_cube",
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": nodegroup_hollow_cube.outputs["Geometry"],
            "Translation": (0.0000, 0.0010, 0.0010),
            "Scale": (1.0000, 0.9990, 0.9990),
        },
    )

    set_material_6 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_4,
            "Material": group_input.outputs["WhiteMetal"],
        },
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [set_material_6, nodegroup_hollow_cube.outputs["Geometry"]]
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": join_geometry_2, "Label": "dishwasher_body"},
    )

    reroute_34 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Surface"]}
    )

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_34})

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata,
            "Material": reroute_35,
        },
    )

    subdivide_mesh = nw.new_node(
        Nodes.SubdivideMesh, input_kwargs={"Mesh": set_material, "Level": 0}
    )

    body = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": subdivide_mesh}, label="Body"
    )

    reroute_61 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": body})

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_61, "Label": "base"},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_46 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["DoorThickness"], 1: 2.1000},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_40 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply})

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_46, 1: reroute_40},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_55 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Depth"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["DoorThickness"], 1: 2.1000},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_39 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_1})

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: reroute_39},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract_1})

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["RackRadius"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ElementAmountRack"]}
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["RackHeight"]}
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    nodegroup_dish_rack_no_gc = nw.new_node(
        nodegroup_dish_rack().name,
        input_kwargs={
            "Depth": reroute_55,
            "Width": reroute_51,
            "Radius": reroute_11,
            "Amount": reroute_15,
            "Height": reroute_13,
            "Value": group_input.outputs["racks_depth"],
        },
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_2, "Y": multiply_3, "Z": 0.5000}
    )

    reroute_42 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_7})

    set_position_2 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": nodegroup_dish_rack_no_gc, "Offset": reroute_42},
    )

    reroute_20 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["WhiteMetal"]}
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    set_material_5 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": set_position_2, "Material": reroute_21},
    )

    realize_instances_3 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": set_material_5}
    )

    reroute_44 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_44, 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    reroute_52 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_46})

    reroute_53 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_52})

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_53, 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    reroute_47 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_47, 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_12 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide, "Y": divide_1, "Z": divide_2}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": realize_instances_3, "Translation": combine_xyz_12},
    )

    reroute_70 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_2}
    )

    add_jointed_geometry_metadata_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_70, "Label": "racks"},
    )

    reroute_24 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["hinge_rack"]}
    )

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_24})

    sliding_joint_001 = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "rack_joint",
            "Parent": add_jointed_geometry_metadata_1,
            "Child": add_jointed_geometry_metadata_2,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Value": reroute_25,
            "Max": 1.0000,
        },
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["RackAmount"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    reroute_48 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    reroute_49 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_48})

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: 2.6000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_11 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide_3})

    mesh_line = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={"Count": reroute_49, "Offset": combine_xyz_11},
        attrs={"mode": "END_POINTS"},
    )

    reroute_56 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": mesh_line})

    duplicate_joints_on_parent = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Duplicate ID (do not set)": "duplicate_racks",
            "Parent": sliding_joint_001.outputs["Parent"],
            "Child": sliding_joint_001.outputs["Child"],
            "Points": reroute_56,
        },
    )

    set_material_10 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": nodegroup_hollow_cube.outputs["base"],
            "Material": group_input.outputs["Surface"],
        },
    )

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": set_material_10, "Scale": (0.9990, 1.0000, 0.0000)},
    )

    add_jointed_geometry_metadata_3 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_6, "Label": "base"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: 1, 3: group_input.outputs["has_handle"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: 0.0500},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_36 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_4})

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_36})

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_46, 1: 0.8000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_6 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_4}, attrs={"operation": "MULTIPLY"}
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["handle_type"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["curved_handle"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    nodegroup_handle_no_gc = nw.new_node(
        nodegroup_handle().name,
        input_kwargs={
            "width": reroute_37,
            "length": multiply_5,
            "thickness": multiply_6,
            "X": group_input.outputs["handle_variations_x"],
            "Y": group_input.outputs["handle_variations_y"],
            "Switch": equal_2,
            "Radius": group_input.outputs["Radius"],
            "handle_position": group_input.outputs["handle_position"],
            "curved_handle": equal_3,
        },
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Depth"],
            1: group_input.outputs["DoorThickness"],
        },
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: 0.1000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: 0.9000},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_38 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_8})

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": multiply_7, "Z": reroute_38}
    )

    subtract_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_3, 1: (0.0100, 0.0000, 0.0000)},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_50 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": subtract_2.outputs["Vector"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": nodegroup_handle_no_gc.outputs["Geometry"],
            "Translation": reroute_50,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry, "Material": reroute_21},
    )

    switch = nw.new_node(
        Nodes.Switch, input_kwargs={"Switch": equal_1, "False": set_material_3}
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

    reroute_68 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_position_1})

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

    reroute_41 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply_add.outputs["Vector"]}
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": store_named_attribute_1, "Translation": reroute_41},
    )

    reroute_57 = nw.new_node(
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

    reroute_54 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_35})

    reroute_62 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_54})

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": reroute_57,
            "Selection": nodegroup_center_no_gc.outputs["In"],
            "Material": reroute_62,
        },
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": set_material_1, "Material": reroute_62},
    )

    add_jointed_geometry_metadata_4 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_2, "Label": "door"},
    )

    equal_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={3: group_input.outputs["button_type"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": 0.0300, "Height": 0.0500},
    )

    fill_curve = nw.new_node(Nodes.FillCurve, input_kwargs={"Curve": quadrilateral})

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh, input_kwargs={"Mesh": fill_curve, "Offset Scale": 0.0200}
    )

    transform_geometry_9 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": extrude_mesh.outputs["Mesh"]}
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 6, "Radius": 0.0200, "Depth": 0.0500},
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_4,
            "False": transform_geometry_9,
            "True": cylinder.outputs["Mesh"],
        },
    )

    reroute_45 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    divide_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_45, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    reroute_60 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_47})

    multiply_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_60, 1: 0.4500},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_13 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_4, "Y": -0.2000, "Z": multiply_9}
    )

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": switch_3,
            "Translation": combine_xyz_13,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    set_material_8 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_7,
            "Material": group_input.outputs["Buttons"],
        },
    )

    add_jointed_geometry_metadata_5 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_8, "Label": "buttons"},
    )

    reroute_28 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["hinge_button"]}
    )

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_28})

    sliding_joint_001_1 = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "button_joint",
            "Parent": add_jointed_geometry_metadata_4,
            "Child": add_jointed_geometry_metadata_5,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Value": reroute_29,
            "Min": -0.0200,
        },
    )

    reroute_64 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_53})

    reroute_65 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_64})

    divide_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_65, 1: 3.8000},
        attrs={"operation": "DIVIDE"},
    )

    reroute_30 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["buttons_amount"]}
    )

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_30})

    grid_1 = nw.new_node(
        Nodes.MeshGrid,
        input_kwargs={
            "Size X": 0.0000,
            "Size Y": divide_5,
            "Vertices X": 1,
            "Vertices Y": reroute_31,
        },
    )

    transform_geometry_10 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": grid_1.outputs["Mesh"],
            "Translation": (0.0000, 0.6000, 0.0000),
        },
    )

    equal_5 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["button_position"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    grid = nw.new_node(
        Nodes.MeshGrid,
        input_kwargs={
            "Size X": 0.0000,
            "Size Y": divide_5,
            "Vertices X": 1,
            "Vertices Y": reroute_31,
        },
    )

    transform_geometry_8 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": grid.outputs["Mesh"],
            "Translation": (0.0000, -0.2000, 0.0000),
        },
    )

    transform_geometry_13 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": grid.outputs["Mesh"]}
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_5,
            "False": transform_geometry_8,
            "True": transform_geometry_13,
        },
    )

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_geometry_10, switch_6]}
    )

    duplicate_joints_on_parent_1 = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Parent": sliding_joint_001_1.outputs["Parent"],
            "Child": sliding_joint_001_1.outputs["Child"],
            "Points": join_geometry_4,
        },
    )

    add_jointed_geometry_metadata_6 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": duplicate_joints_on_parent_1,
            "Label": "_with_buttons",
        },
    )

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 7, "Radius": 0.0500, "Depth": 0.0600},
    )

    transform_geometry_11 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    set_material_9 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_11,
            "Material": group_input.outputs["Buttons"],
        },
    )

    add_jointed_geometry_metadata_7 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_9, "Label": "button"},
    )

    add_1 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: combine_xyz_13, 1: (0.0200, 0.2000, 0.0000)}
    )

    multiply_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: -0.4200},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_10})

    add_2 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: add_1.outputs["Vector"], 1: combine_xyz_9}
    )

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_5,
            "False": add_1.outputs["Vector"],
            "True": add_2.outputs["Vector"],
        },
        attrs={"input_type": "VECTOR"},
    )

    reroute_32 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["rotate_button"]}
    )

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_32})

    hinge_joint_001 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "knob_joint",
            "Parent": add_jointed_geometry_metadata_6,
            "Child": add_jointed_geometry_metadata_7,
            "Position": switch_5,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Value": reroute_33,
        },
    )

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.0010

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0100

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal_1, "False": value_1, "True": value},
        attrs={"input_type": "FLOAT"},
    )

    multiply_11 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_60, 1: 0.9200},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": switch_1, "Y": 0.0010, "Z": multiply_11}
    )

    transform_geometry_12 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": set_material_2,
            "Translation": combine_xyz_8,
            "Scale": (1.0000, 0.9990, 0.0798),
        },
    )

    set_material_7 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_12,
            "Material": group_input.outputs["Button"],
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 1": set_material_7,
            "Mesh 2": nodegroup_hollow_cube.outputs["Geometry"],
        },
    )

    join_geometry_5 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                hinge_joint_001.outputs["Geometry"],
                difference.outputs["Mesh"],
            ]
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [reroute_68, join_geometry_5]}
    )

    reroute_69 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_12})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Translation": reroute_69},
    )

    divide_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    divide_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_6, "Z": divide_7}
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_1, "Translation": combine_xyz_5},
    )

    add_jointed_geometry_metadata_8 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_5, "Label": "door"},
    )

    divide_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide_8})

    reroute_26 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["hinge_door"]}
    )

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_26})

    hinge_joint_001_1 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "door_joint",
            "Parent": add_jointed_geometry_metadata_3,
            "Child": add_jointed_geometry_metadata_8,
            "Position": combine_xyz_6,
            "Axis": (0.0000, 1.0000, 0.0000),
            "Value": reroute_27,
            "Max": 1.5000,
        },
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                duplicate_joints_on_parent,
                hinge_joint_001_1.outputs["Geometry"],
            ]
        },
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal,
            "False": join_geometry_3,
            "True": hinge_joint_001_1.outputs["Geometry"],
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": switch_2}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_1},
        attrs={"is_active_output": True},
    )


class DishwasherFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    @gin.configurable(module="DishwasherFactory")
    def sample_joint_parameters(
        cls,
        door_joint_stiffness_min: float = 500.0,
        door_joint_stiffness_max: float = 1000.0,
        door_joint_damping_min: float = 500.0,
        door_joint_damping_max: float = 2000.0,
        door_joint_friction_min: float = 1000.0,
        door_joint_friction_max: float = 1500.0,
        rack_joint_stiffness_min: float = 0.0,
        rack_joint_stiffness_max: float = 0.0,
        rack_joint_damping_min: float = 60.0,
        rack_joint_damping_max: float = 80.0,
        button_joint_stiffness_min: float = 5.0,
        button_joint_stiffness_max: float = 10.0,
        button_joint_damping_min: float = 1.0,
        button_joint_damping_max: float = 3.0,
        knob_joint_stiffness_min: float = 0.0,
        knob_joint_stiffness_max: float = 0.0,
        knob_joint_damping_min: float = 0.0,
        knob_joint_damping_max: float = 5.0,
    ):
        return {
            "door_joint": {
                "stiffness": np.random.uniform(
                    door_joint_stiffness_min, door_joint_stiffness_max
                ),
                "damping": np.random.uniform(
                    door_joint_damping_min, door_joint_damping_max
                ),
                "friction": np.random.uniform(
                    door_joint_friction_min, door_joint_friction_max
                ),
            },
            "rack_joint": {
                "stiffness": np.random.uniform(
                    rack_joint_stiffness_min, rack_joint_stiffness_max
                ),
                "damping": np.random.uniform(
                    rack_joint_damping_min, rack_joint_damping_max
                ),
            },
            "button_joint": {
                "stiffness": np.random.uniform(
                    button_joint_stiffness_min, button_joint_stiffness_max
                ),
                "damping": np.random.uniform(
                    button_joint_damping_min, button_joint_damping_max
                ),
            },
            "knob_joint": {
                "stiffness": np.random.uniform(
                    knob_joint_stiffness_min, knob_joint_stiffness_max
                ),
                "damping": np.random.uniform(
                    knob_joint_damping_min, knob_joint_damping_max
                ),
            },
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
            ElementAmountRack = RI(2, 4)  # Number of elements on the rack
            RackHeight = U(0.18, 0.3)  # Height of the rack elements
            racks_depth = U(0.3, 0.5)  # Depth of the racks
            handle_variations_x = U(1, 1.9)
            handle_variations_y = U(1.02, 1.15)
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
