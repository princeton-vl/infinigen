# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Hongyu Wen: primary author
# Acknowledgment: This file draws inspiration
# from https://www.youtube.com/watch?v=o50FE2W1m8Y
# by Open Class

import functools

import numpy as np

from infinigen.assets.materials import metal
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.paths import blueprint_path_completion


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
    "nodegroup_duplicate_joints_on_parent", singleton=False, type="GeometryNodeTree"
)
def nodegroup_duplicate_joints_on_parent(nw: NodeWrangler):
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
        attrs={"data_type": "INT", "domain": "INSTANCE"},
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


@node_utils.to_nodegroup(
    "nodegroup_round_quad", singleton=False, type="GeometryNodeTree"
)
def nodegroup_round_quad(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "Roundness", 1.0000),
        ],
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": reroute_1, "Height": reroute},
    )

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: reroute},
        attrs={"operation": "MINIMUM"},
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": group_input.outputs["Roundness"], 4: 0.5000},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum, 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={"Curve": quadrilateral, "Count": 10, "Radius": multiply},
        attrs={"mode": "POLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Curve": fillet_curve},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_rack", singleton=False, type="GeometryNodeTree")
def nodegroup_rack(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketInt", "FBNum", 0),
            ("NodeSocketInt", "LRNum", 0),
            ("NodeSocketMaterial", "Material", None),
        ],
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz, "End": combine_xyz_1}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"]},
        attrs={"operation": "MULTIPLY"},
    )

    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={"Radius": multiply_2})

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Size"]}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["Z"]}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": separate_xyz.outputs["X"], "Y": reroute, "Z": reroute},
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": 1.0000, "Y": group_input.outputs["LRNum"], "Z": 1.0000},
    )

    multipleobjects = nw.new_node(
        nodegroup_multiple_objects().name,
        input_kwargs={
            "Object": curve_to_mesh,
            "SpaceSize": reroute_1,
            "ObjectSize": combine_xyz_2,
            "ObjectNum": combine_xyz_8,
        },
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": multipleobjects}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_3})

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_4})

    curve_line_1 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_6, "End": combine_xyz_7}
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line_1,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_to_mesh_1,
            "Rotation": (0.0000, 0.0000, 1.5708),
        },
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": reroute, "Y": separate_xyz.outputs["Y"], "Z": reroute},
    )

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": group_input.outputs["FBNum"], "Y": 1.0000, "Z": 1.0000},
    )

    multipleobjects_1 = nw.new_node(
        nodegroup_multiple_objects().name,
        input_kwargs={
            "Object": transform_geometry,
            "SpaceSize": reroute_1,
            "ObjectSize": combine_xyz_3,
            "ObjectNum": combine_xyz_9,
        },
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": multipleobjects_1}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_2, transform_geometry_1]},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": join_geometry,
            "Material": group_input.outputs["Material"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_material},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_multi_drawer", singleton=False, type="GeometryNodeTree"
)
def nodegroup_multi_drawer(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    string = nw.new_node("FunctionNodeInputString", attrs={"string": "duplicate0"})

    string_1 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint6"})

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (0.8000, 1.0000, 0.4000)),
            ("NodeSocketFloat", "WallThickness", 0.0200),
            ("NodeSocketFloat", "HandleMargin", 0.1000),
            ("NodeSocketVector", "HandleTopSize", (0.4000, 0.1000, 0.1000)),
            ("NodeSocketFloat", "HandleTopThickness", 0.0100),
            ("NodeSocketFloat", "HandleTopRoundness", 1.0000),
            ("NodeSocketVector", "HandleSupportSize", (0.2000, 0.0800, 0.0200)),
            ("NodeSocketFloat", "HandleSupportMargin", 0.1000),
            ("NodeSocketFloat", "SlidingJointValue", 1.0000),
            ("NodeSocketFloat", "BodyRoundness", 0.0000),
            ("NodeSocketFloat", "SlideRoundness", 0.0000),
            ("NodeSocketFloat", "InnerRoundness", 0.0000),
            ("NodeSocketMaterial", "InnerMaterial", None),
            ("NodeSocketMaterial", "OuterMaterial", None),
            ("NodeSocketMaterial", "HandleMaterial", None),
            ("NodeSocketInt", "DrawerNum", 0),
        ],
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["SlideRoundness"]}
    )

    openroundcube = nw.new_node(
        nodegroup_open_round_cube().name,
        input_kwargs={
            "Size": group_input.outputs["Size"],
            "Thickness": group_input.outputs["WallThickness"],
            "OuterRoundness": group_input.outputs["BodyRoundness"],
            "InnerRoundness": reroute_1,
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": openroundcube,
            "Material": group_input.outputs["OuterMaterial"],
        },
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material, "Name": "joint6"},
        attrs={"data_type": "INT"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["WallThickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["DrawerNum"]}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: reroute_2},
        attrs={"operation": "DIVIDE"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz.outputs["X"],
            1: group_input.outputs["WallThickness"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": divide, "Z": subtract_2}
    )

    openroundcube_1 = nw.new_node(
        nodegroup_open_round_cube().name,
        input_kwargs={
            "Size": combine_xyz,
            "Thickness": group_input.outputs["WallThickness"],
            "OuterRoundness": reroute_1,
            "InnerRoundness": group_input.outputs["InnerRoundness"],
        },
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["WallThickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": openroundcube_1,
            "Translation": combine_xyz_1,
            "Rotation": (0.0000, -1.5708, 0.0000),
        },
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry,
            "Material": group_input.outputs["InnerMaterial"],
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material_2, "Name": "joint7"},
        attrs={"data_type": "INT"},
    )

    handle = nw.new_node(
        nodegroup_handle().name,
        input_kwargs={
            "TopSize": group_input.outputs["HandleTopSize"],
            "TopThickness": group_input.outputs["HandleTopThickness"],
            "TopRoundness": group_input.outputs["HandleTopRoundness"],
            "SupportSize": group_input.outputs["HandleSupportSize"],
            "SupportMargin": group_input.outputs["HandleSupportMargin"],
            "Material": group_input.outputs["HandleMaterial"],
        },
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"]},
        attrs={"operation": "MULTIPLY"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": group_input.outputs["HandleSupportSize"]},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_2, 1: separate_xyz_1.outputs["Z"]}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_3, 1: group_input.outputs["HandleMargin"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["HandleTopSize"]}
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_3, 1: multiply_4})

    subtract_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_1, 1: group_input.outputs["WallThickness"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Z": subtract_4}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": handle,
            "Translation": combine_xyz_2,
            "Rotation": (1.5708, 0.0000, 1.5708),
        },
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_1, "Name": "joint7", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute, store_named_attribute_1]},
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_1, "Name": "joint6", "Value": 1},
        attrs={"data_type": "INT"},
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract_2})

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "internal_drawer",
            "Parent": store_named_attribute_3,
            "Child": store_named_attribute_2,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Max": reroute,
        },
    )

    subtract_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: divide},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": subtract_5})

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_3,
            "Vertices X": 1,
            "Vertices Y": reroute_2,
            "Vertices Z": 1,
        },
    )

    duplicate_joints_on_parent = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Duplicate ID (do not set)": string,
            "Parent": sliding_joint.outputs["Parent"],
            "Child": sliding_joint.outputs["Child"],
            "Points": cube.outputs["Mesh"],
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": duplicate_joints_on_parent, "Label": "drawer"},
    )

    group_output_1 = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": add_jointed_geometry_metadata},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_multi_drawer_top", singleton=False, type="GeometryNodeTree"
)
def nodegroup_multi_drawer_top(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    string = nw.new_node("FunctionNodeInputString", attrs={"string": "duplicate0"})

    string_1 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint6"})

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (0.8000, 1.0000, 0.4000)),
            ("NodeSocketFloat", "WallThickness", 0.0200),
            ("NodeSocketFloat", "HandleMargin", 0.1000),
            ("NodeSocketVector", "HandleTopSize", (0.4000, 0.1000, 0.1000)),
            ("NodeSocketFloat", "HandleTopThickness", 0.0100),
            ("NodeSocketFloat", "HandleTopRoundness", 1.0000),
            ("NodeSocketVector", "HandleSupportSize", (0.2000, 0.0800, 0.0200)),
            ("NodeSocketFloat", "HandleSupportMargin", 0.1000),
            ("NodeSocketFloat", "SlidingJointValue", 1.0000),
            ("NodeSocketFloat", "BodyRoundness", 0.0000),
            ("NodeSocketFloat", "SlideRoundness", 0.0000),
            ("NodeSocketFloat", "InnerRoundness", 0.0000),
            ("NodeSocketMaterial", "InnerMaterial", None),
            ("NodeSocketMaterial", "OuterMaterial", None),
            ("NodeSocketMaterial", "HandleMaterial", None),
            ("NodeSocketInt", "DrawerNum", 0),
            ("NodeSocketInt", "Value", 0),
        ],
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["SlideRoundness"]}
    )

    openroundcube = nw.new_node(
        nodegroup_open_round_cube().name,
        input_kwargs={
            "Size": group_input.outputs["Size"],
            "Thickness": group_input.outputs["WallThickness"],
            "OuterRoundness": group_input.outputs["BodyRoundness"],
            "InnerRoundness": reroute_1,
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": openroundcube,
            "Material": group_input.outputs["OuterMaterial"],
        },
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material, "Name": "joint6"},
        attrs={"data_type": "INT"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["WallThickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["DrawerNum"]}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: reroute_2},
        attrs={"operation": "DIVIDE"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz.outputs["X"],
            1: group_input.outputs["WallThickness"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": divide, "Z": subtract_2}
    )

    openroundcube_1 = nw.new_node(
        nodegroup_open_round_cube().name,
        input_kwargs={
            "Size": combine_xyz,
            "Thickness": group_input.outputs["WallThickness"],
            "OuterRoundness": reroute_1,
            "InnerRoundness": group_input.outputs["InnerRoundness"],
        },
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["WallThickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": openroundcube_1,
            "Translation": combine_xyz_1,
            "Rotation": (0.0000, -1.5708, 0.0000),
        },
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry,
            "Material": group_input.outputs["InnerMaterial"],
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material_2, "Name": "joint7"},
        attrs={"data_type": "INT"},
    )

    handle = nw.new_node(
        nodegroup_handle().name,
        input_kwargs={
            "TopSize": group_input.outputs["HandleTopSize"],
            "TopThickness": group_input.outputs["HandleTopThickness"],
            "TopRoundness": group_input.outputs["HandleTopRoundness"],
            "SupportSize": group_input.outputs["HandleSupportSize"],
            "SupportMargin": group_input.outputs["HandleSupportMargin"],
            "Material": group_input.outputs["HandleMaterial"],
        },
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"]},
        attrs={"operation": "MULTIPLY"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": group_input.outputs["HandleSupportSize"]},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_2, 1: separate_xyz_1.outputs["Z"]}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_3, 1: group_input.outputs["HandleMargin"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["HandleTopSize"]}
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_3, 1: multiply_4})

    subtract_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_1, 1: group_input.outputs["WallThickness"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Z": subtract_4}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": handle,
            "Translation": combine_xyz_2,
            "Rotation": (1.5708, 0.0000, 1.5708),
        },
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_1, "Name": "joint7", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute, store_named_attribute_1]},
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_1, "Name": "joint6", "Value": 1},
        attrs={"data_type": "INT"},
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract_2})

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "freezer_drawer",
            "Parent": store_named_attribute_3,
            "Child": store_named_attribute_2,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Max": reroute,
        },
    )

    subtract_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: divide},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": subtract_5})

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_3,
            "Vertices X": 1,
            "Vertices Y": reroute_2,
            "Vertices Z": 1,
        },
    )

    duplicate_joints_on_parent = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Duplicate ID (do not set)": string,
            "Parent": sliding_joint.outputs["Parent"],
            "Child": sliding_joint.outputs["Child"],
            "Points": cube.outputs["Mesh"],
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": duplicate_joints_on_parent, "Label": "drawer"},
    )

    store_named_attribute_100 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata,
            "Name": "jointr",
            "Value": group_input.outputs["Value"],
        },
        attrs={"data_type": "INT"},
    )

    group_output_1 = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": store_named_attribute_100},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_center_of_geometry", singleton=False, type="GeometryNodeTree"
)
def nodegroup_center_of_geometry(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": group_input.outputs["Geometry"]}
    )

    add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Min"], 1: bounding_box.outputs["Max"]},
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add.outputs["Vector"], 1: (0.5000, 0.5000, 0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Center": multiply.outputs["Vector"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_handle", singleton=False, type="GeometryNodeTree")
def nodegroup_handle(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "TopSize", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "TopThickness", 0.1000),
            ("NodeSocketFloat", "TopRoundness", 0.0000),
            ("NodeSocketVector", "SupportSize", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "SupportMargin", -0.5000),
            ("NodeSocketMaterial", "Material", None),
        ],
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["TopSize"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply})

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": separate_xyz.outputs["Z"]}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    quadratic_b_zier = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Start": combine_xyz,
            "Middle": combine_xyz_2,
            "End": combine_xyz_1,
        },
    )

    roundquad = nw.new_node(
        nodegroup_round_quad().name,
        input_kwargs={
            "Width": separate_xyz.outputs["Y"],
            "Height": group_input.outputs["TopThickness"],
            "Roundness": group_input.outputs["TopRoundness"],
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": quadratic_b_zier,
            "Profile Curve": roundquad,
            "Fill Caps": True,
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": curve_to_mesh, "Name": "joint4"},
        attrs={"data_type": "INT"},
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["SupportSize"]}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["Z"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_2})

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": separate_xyz.outputs["Z"]}
    )

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_5, "End": combine_xyz_6}
    )

    roundquad_1 = nw.new_node(
        nodegroup_round_quad().name,
        input_kwargs={
            "Width": separate_xyz_3.outputs["X"],
            "Height": separate_xyz_3.outputs["Y"],
            "Roundness": 0.2000,
        },
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": roundquad_1,
            "Fill Caps": True,
        },
    )

    set_shade_smooth = nw.new_node(
        Nodes.SetShadeSmooth,
        input_kwargs={"Geometry": curve_to_mesh_1, "Shade Smooth": False},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["X"]}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={1: group_input.outputs["SupportMargin"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["X"]},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_4, 1: multiply_5})

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_3, 1: add})

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": add_1})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": set_shade_smooth, "Translation": combine_xyz_3},
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry, "Name": "joint5"},
        attrs={"data_type": "INT"},
    )

    multiply_6 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute}, attrs={"operation": "MULTIPLY"}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_6, 1: add},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": subtract})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": set_shade_smooth, "Translation": combine_xyz_4},
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_1, "Name": "joint5", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_1, store_named_attribute_2]},
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 1": join_geometry_1,
            "Mesh 2": curve_to_mesh,
            "Self Intersection": True,
            "Hole Tolerant": True,
        },
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    separate_xyz_4 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["TopSize"]}
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_4.outputs["X"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: multiply_7},
        attrs={"operation": "SUBTRACT"},
    )

    sample_curve = nw.new_node(
        Nodes.SampleCurve,
        input_kwargs={"Curves": quadratic_b_zier, "Length": subtract_1},
        attrs={"mode": "LENGTH"},
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": sample_curve.outputs["Position"]}
    )

    less_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: separate_xyz_1.outputs["Z"]},
        attrs={"operation": "LESS_THAN"},
    )

    delete_geometry = nw.new_node(
        Nodes.DeleteGeometry,
        input_kwargs={"Geometry": difference.outputs["Mesh"], "Selection": less_than},
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": delete_geometry, "Name": "joint4", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute, store_named_attribute_3]},
    )

    set_shade_smooth_1 = nw.new_node(
        Nodes.SetShadeSmooth,
        input_kwargs={"Geometry": join_geometry, "Shade Smooth": False},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": set_shade_smooth_1,
            "Material": group_input.outputs["Material"],
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material, "Label": "handle"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": add_jointed_geometry_metadata},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_size_of_object", singleton=False, type="GeometryNodeTree"
)
def nodegroup_size_of_object(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": group_input.outputs["Geometry"]}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Vector": subtract.outputs["Vector"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_multiple_objects", singleton=False, type="GeometryNodeTree"
)
def nodegroup_multiple_objects(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Object", None),
            ("NodeSocketVector", "SpaceSize", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "ObjectSize", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "ObjectNum", (0.0000, 0.0000, 0.0000)),
        ],
    )

    geometry_to_instance = nw.new_node(
        "GeometryNodeGeometryToInstance",
        input_kwargs={"Geometry": group_input.outputs["Object"]},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["ObjectNum"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Y"], 1: separate_xyz_1.outputs["Z"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: separate_xyz_1.outputs["X"]},
        attrs={"operation": "MULTIPLY"},
    )

    duplicate_elements = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={"Geometry": geometry_to_instance, "Amount": multiply_1},
        attrs={"domain": "INSTANCE"},
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["SpaceSize"],
            1: group_input.outputs["ObjectSize"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["ObjectNum"], 1: (1.0000, 1.0000, 1.0000)},
        attrs={"operation": "SUBTRACT"},
    )

    divide = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"], 1: subtract_1.outputs["Vector"]},
        attrs={"operation": "DIVIDE"},
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": divide.outputs["Vector"]}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Y"], 1: separate_xyz_1.outputs["Z"]},
        attrs={"operation": "MULTIPLY"},
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: multiply_2},
        attrs={"operation": "DIVIDE"},
    )

    floor = nw.new_node(
        Nodes.Math, input_kwargs={0: divide_1}, attrs={"operation": "FLOOR"}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: floor},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: floor, 1: multiply_2},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: multiply_4},
        attrs={"operation": "SUBTRACT"},
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_2, 1: separate_xyz_1.outputs["Z"]},
        attrs={"operation": "DIVIDE"},
    )

    floor_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: divide_2}, attrs={"operation": "FLOOR"}
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: floor_1},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: floor_1, 1: separate_xyz_1.outputs["Z"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_2, 1: multiply_6},
        attrs={"operation": "SUBTRACT"},
    )

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_3, 1: 1.0000},
        attrs={"operation": "DIVIDE"},
    )

    floor_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: divide_3}, attrs={"operation": "FLOOR"}
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: floor_2},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": multiply_3, "Y": multiply_5, "Z": multiply_7},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": duplicate_elements.outputs["Geometry"],
            "Offset": combine_xyz_1,
        },
    )

    multiply_8 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["ObjectSize"],
            1: (0.5000, 0.5000, 0.5000),
        },
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": set_position,
            "Translation": multiply_8.outputs["Vector"],
        },
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": transform_geometry}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": realize_instances},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_two_material_cube", singleton=False, type="GeometryNodeTree"
)
def nodegroup_two_material_cube(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "LMargin", 0.0000),
            ("NodeSocketFloat", "RMargin", 0.0000),
            ("NodeSocketFloat", "UMargin", 0.0000),
            ("NodeSocketFloat", "BMargin", 0.0000),
            ("NodeSocketMaterial", "InnerMaterial", None),
            ("NodeSocketMaterial", "OuterMaterial", None),
        ],
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": group_input.outputs["Size"],
            "Vertices X": 100,
            "Vertices Y": 100,
            "Vertices Z": 100,
        },
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply, 1: group_input.outputs["LMargin"]}
    )

    greater_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Y"], 1: add},
        attrs={"operation": "GREATER_THAN"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: group_input.outputs["RMargin"]},
        attrs={"operation": "SUBTRACT"},
    )

    less_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Y"], 1: subtract},
        attrs={"operation": "LESS_THAN"},
    )

    op_and = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: greater_than, 1: less_than}
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_1})

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_2, 1: group_input.outputs["BMargin"]}
    )

    greater_than_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: add_1},
        attrs={"operation": "GREATER_THAN"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_3, 1: group_input.outputs["UMargin"]},
        attrs={"operation": "SUBTRACT"},
    )

    less_than_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: subtract_1},
        attrs={"operation": "LESS_THAN"},
    )

    op_and_1 = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: greater_than_1, 1: less_than_1}
    )

    op_and_2 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: op_and, 1: op_and_1})

    separate_geometry = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Selection": op_and_2},
        attrs={"domain": "FACE"},
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": separate_geometry.outputs["Inverted"],
            "Material": group_input.outputs["OuterMaterial"],
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material_1, "Name": "joint3"},
        attrs={"data_type": "INT"},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": separate_geometry.outputs["Selection"],
            "Material": group_input.outputs["InnerMaterial"],
        },
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material, "Name": "joint3", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute, store_named_attribute_1]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": join_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_open_round_cube", singleton=False, type="GeometryNodeTree"
)
def nodegroup_open_round_cube(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "Thickness", 0.0000),
            ("NodeSocketFloat", "OuterRoundness", 0.1000),
            ("NodeSocketFloat", "InnerRoundness", 0.1000),
        ],
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_1})

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_2, "End": combine_xyz_3}
    )

    roundquad = nw.new_node(
        nodegroup_round_quad().name,
        input_kwargs={
            "Width": separate_xyz.outputs["X"],
            "Height": separate_xyz.outputs["Y"],
            "Roundness": group_input.outputs["OuterRoundness"],
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": roundquad,
            "Fill Caps": True,
        },
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Thickness"],
            "Y": multiply_2,
            "Z": multiply_2,
        },
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: combine_xyz},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_3})

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_4})

    curve_line_1 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_4, "End": combine_xyz_5}
    )

    roundquad_1 = nw.new_node(
        nodegroup_round_quad().name,
        input_kwargs={
            "Width": separate_xyz_1.outputs["Y"],
            "Height": separate_xyz_1.outputs["Z"],
            "Roundness": group_input.outputs["InnerRoundness"],
        },
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line_1,
            "Profile Curve": roundquad_1,
            "Fill Caps": True,
        },
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Thickness"], 1: 0.0000}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": add})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_to_mesh_1,
            "Translation": combine_xyz_1,
            "Rotation": (1.5708, 0.0000, 1.5708),
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": curve_to_mesh, "Mesh 2": transform_geometry},
        attrs={"solver": "EXACT"},
    )

    set_shade_smooth = nw.new_node(
        Nodes.SetShadeSmooth,
        input_kwargs={"Geometry": difference.outputs["Mesh"], "Shade Smooth": False},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_shade_smooth},
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


@node_utils.to_nodegroup("nodegroup_shelf", singleton=False, type="GeometryNodeTree")
def nodegroup_shelf(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketInt", "LayerNum", 0),
            ("NodeSocketFloat", "ShelfThickness", 0.0000),
            ("NodeSocketFloat", "BoardMargin", -0.5000),
            ("NodeSocketInt", "NetFBINum", 0),
            ("NodeSocketInt", "NetLRNum", 0),
            ("NodeSocketBool", "NettedShelf", True),
            ("NodeSocketMaterial", "InnerMaterial", None),
            ("NodeSocketMaterial", "OuterMaterial", None),
        ],
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["ShelfThickness"],
            "Y": separate_xyz.outputs["X"],
            "Z": separate_xyz.outputs["Y"],
        },
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BoardMargin"]}
    )

    twomaterialcube = nw.new_node(
        nodegroup_two_material_cube().name,
        input_kwargs={
            "Size": combine_xyz,
            "LMargin": reroute,
            "RMargin": reroute,
            "UMargin": reroute,
            "BMargin": reroute,
            "InnerMaterial": group_input.outputs["InnerMaterial"],
            "OuterMaterial": group_input.outputs["OuterMaterial"],
        },
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": twomaterialcube,
            "Rotation": (0.0000, 1.5708, 1.5708),
        },
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["LayerNum"]}
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: reroute_1, 1: 2.0000})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Z"], 1: add},
        attrs={"operation": "DIVIDE"},
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: divide, 1: 2.0000}, attrs={"operation": "MULTIPLY"}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Z"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz_1.outputs["X"],
            "Y": separate_xyz_1.outputs["Y"],
            "Z": subtract,
        },
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz.outputs["X"],
            "Y": separate_xyz.outputs["Y"],
            "Z": group_input.outputs["ShelfThickness"],
        },
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 1.0000, "Y": 1.0000, "Z": reroute_1}
    )

    multipleobjects = nw.new_node(
        nodegroup_multiple_objects().name,
        input_kwargs={
            "Object": transform_geometry_1,
            "SpaceSize": combine_xyz_3,
            "ObjectSize": combine_xyz_1,
            "ObjectNum": combine_xyz_4,
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": multipleobjects, "Name": "switch0"},
        attrs={"data_type": "INT"},
    )

    rack = nw.new_node(
        nodegroup_rack().name,
        input_kwargs={
            "Size": combine_xyz_1,
            "FBNum": group_input.outputs["NetFBINum"],
            "LRNum": group_input.outputs["NetLRNum"],
            "Material": group_input.outputs["OuterMaterial"],
        },
    )

    multipleobjects_1 = nw.new_node(
        nodegroup_multiple_objects().name,
        input_kwargs={
            "Object": rack,
            "SpaceSize": combine_xyz_3,
            "ObjectSize": combine_xyz_1,
            "ObjectNum": combine_xyz_4,
        },
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": multipleobjects_1, "Name": "switch0", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["NettedShelf"],
            "False": store_named_attribute,
            "True": store_named_attribute_1,
        },
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide})

    transform_geometry = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": switch, "Translation": combine_xyz_2}
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry, "Label": "shelf"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": add_jointed_geometry_metadata},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_move_to_origin", singleton=False, type="GeometryNodeTree"
)
def nodegroup_move_to_origin(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": group_input.outputs["Geometry"]}
    )

    add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Min"], 1: bounding_box.outputs["Max"]},
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add.outputs["Vector"], 1: (-0.5000, -0.5000, -0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Translation": multiply.outputs["Vector"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_singlefridge", singleton=False, type="GeometryNodeTree"
)
def nodegroup_singlefridge(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input_2 = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "WallThickness", 0.1000),
            ("NodeSocketFloat", "BodyOuterRoundness", 0.0000),
            ("NodeSocketFloat", "BodyInnerRoundness", 0.0000),
            ("NodeSocketBool", "DoorOnRight", False),
            ("NodeSocketBool", "TwoDoors", False),
            ("NodeSocketFloat", "DoorHandleMargin", 0.5000),
            ("NodeSocketVector", "DoorShelfSize", (0.3000, 1.0000, 0.2000)),
            ("NodeSocketFloat", "DoorShelfThickness", 0.0200),
            ("NodeSocketFloat", "DoorShelfOuterRoundness", 0.1000),
            ("NodeSocketFloat", "DoorShelfInnerRoundness", 0.1000),
            ("NodeSocketInt", "DoorShelfNum", 0),
            ("NodeSocketFloat", "DoorShelfMargin", 0.5000),
            ("NodeSocketVector", "DoorHandleTopSize", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "DoorHandleTopThickness", 0.0100),
            ("NodeSocketFloat", "DoorHandleTopRoundness", 1.0000),
            ("NodeSocketVector", "DoorHandleSupportSize", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "DoorHandleSupportMargin", 0.0500),
            ("NodeSocketFloat", "DoorLMargin", 0.1000),
            ("NodeSocketFloat", "DoorRMargin", 0.2000),
            ("NodeSocketFloat", "DoorUMargin", 0.1000),
            ("NodeSocketFloat", "DoorBMargin", 0.1000),
            ("NodeSocketFloat", "DoorHingeJointValue", 2.0000),
            ("NodeSocketFloat", "ShelfMargin", 0.0000),
            ("NodeSocketInt", "ShelfLayerNum", 3),
            ("NodeSocketFloat", "ShelfThickness", 0.0100),
            ("NodeSocketFloat", "ShelfBoardMargin", 0.0500),
            ("NodeSocketInt", "ShelfNetFBNum", 10),
            ("NodeSocketInt", "ShelfNetLRNum", 10),
            ("NodeSocketBool", "ShelfNettedShelf", True),
            ("NodeSocketBool", "DrawerOnBottom", True),
            ("NodeSocketFloat", "DrawerHeight", 0.3000),
            ("NodeSocketFloat", "DrawerWallThickness", 0.0200),
            ("NodeSocketFloat", "DrawerHandleMargin", 0.1000),
            ("NodeSocketVector", "DrawerHandleTopSize", (0.4000, 0.1000, 0.1000)),
            ("NodeSocketFloat", "DrawerHandleTopThickness", 0.0100),
            ("NodeSocketFloat", "DrawerHandleTopRoundness", 1.0000),
            ("NodeSocketVector", "DrawerHandleSupportSize", (0.0500, 0.0800, 0.0100)),
            ("NodeSocketFloat", "DrawerHandleSupportMargin", 0.1000),
            ("NodeSocketFloat", "DrawerBodyRoundness", 0.0000),
            ("NodeSocketFloat", "DrawerSlideRoundness", 0.0000),
            ("NodeSocketFloat", "DrawernnerRoundness", 0.0000),
            ("NodeSocketFloat", "DrawerSlidingJointValue", 0.2000),
            ("NodeSocketInt", "DrawerNum", 0),
            ("NodeSocketMaterial", "BodyMaterial", None),
            ("NodeSocketMaterial", "HandleMaterial", None),
            ("NodeSocketMaterial", "DoorShelfMaterial", None),
            ("NodeSocketMaterial", "DoorGlassMaterial", None),
            ("NodeSocketMaterial", "ShelfMaterial", None),
            ("NodeSocketMaterial", "ShelfGlassMaterial", None),
            ("NodeSocketMaterial", "DrawerMaterial", None),
            ("NodeSocketMaterial", "DrawerHandleMaterial", None),
            ("NodeSocketInt", "Value", 0),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_2.outputs["WallThickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input_2.outputs["ShelfMargin"], 1: multiply}
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input_2.outputs["DrawerHeight"], 1: multiply}
    )

    combine_xyz_11 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": multiply, "Z": add_1}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input_2.outputs["Size"], 1: combine_xyz_11},
        attrs={"operation": "SUBTRACT"},
    )

    shelf = nw.new_node(
        nodegroup_shelf().name,
        input_kwargs={
            "Size": subtract.outputs["Vector"],
            "LayerNum": group_input_2.outputs["ShelfLayerNum"],
            "ShelfThickness": group_input_2.outputs["ShelfThickness"],
            "BoardMargin": group_input_2.outputs["ShelfBoardMargin"],
            "NetFBINum": group_input_2.outputs["ShelfNetFBNum"],
            "NetLRNum": group_input_2.outputs["ShelfNetLRNum"],
            "NettedShelf": group_input_2.outputs["ShelfNettedShelf"],
            "InnerMaterial": group_input_2.outputs["ShelfGlassMaterial"],
            "OuterMaterial": group_input_2.outputs["ShelfMaterial"],
        },
    )

    movetoorigin = nw.new_node(
        nodegroup_move_to_origin().name, input_kwargs={"Geometry": shelf}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_2.outputs["ShelfMargin"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: group_input_2.outputs["WallThickness"]},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_2.outputs["DrawerHeight"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_2, "Z": multiply_2}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": movetoorigin, "Translation": combine_xyz_7},
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_2, "Name": "joint0"},
        attrs={"data_type": "INT"},
    )

    string = nw.new_node("FunctionNodeInputString", attrs={"string": "joint1"})

    size = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_2.outputs["Size"]},
        label="Size",
    )

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": size})

    thickness = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_2.outputs["WallThickness"]},
        label="Thickness",
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: thickness},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": subtract_1,
            "Y": separate_xyz.outputs["Y"],
            "Z": separate_xyz.outputs["Z"],
        },
    )

    openroundcube = nw.new_node(
        nodegroup_open_round_cube().name,
        input_kwargs={
            "Size": combine_xyz,
            "Thickness": thickness,
            "OuterRoundness": group_input_2.outputs["BodyOuterRoundness"],
            "InnerRoundness": group_input_2.outputs["BodyInnerRoundness"],
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": openroundcube,
            "Material": group_input_2.outputs["BodyMaterial"],
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material, "Label": "refrigerator_body"},
    )

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": add_jointed_geometry_metadata, "Name": "joint1"},
        attrs={"data_type": "INT"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input_2.outputs["Size"]}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input_2.outputs["WallThickness"],
            "Y": separate_xyz_1.outputs["Y"],
            "Z": separate_xyz_1.outputs["Z"],
        },
    )

    twomaterialcube = nw.new_node(
        nodegroup_two_material_cube().name,
        input_kwargs={
            "Size": combine_xyz_1,
            "LMargin": group_input_2.outputs["DoorLMargin"],
            "RMargin": group_input_2.outputs["DoorRMargin"],
            "UMargin": group_input_2.outputs["DoorUMargin"],
            "BMargin": group_input_2.outputs["DoorBMargin"],
            "InnerMaterial": group_input_2.outputs["DoorGlassMaterial"],
            "OuterMaterial": group_input_2.outputs["BodyMaterial"],
        },
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": twomaterialcube, "Name": "joint2"},
        attrs={"data_type": "INT"},
    )

    openroundcube_1 = nw.new_node(
        nodegroup_open_round_cube().name,
        input_kwargs={
            "Size": group_input_2.outputs["DoorShelfSize"],
            "Thickness": group_input_2.outputs["DoorShelfThickness"],
            "OuterRoundness": group_input_2.outputs["DoorShelfOuterRoundness"],
            "InnerRoundness": group_input_2.outputs["DoorShelfInnerRoundness"],
        },
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": openroundcube_1,
            "Rotation": (0.0000, -1.5708, 0.0000),
        },
    )

    separate_xyz_4 = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": group_input_2.outputs["DoorShelfSize"]},
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": combine_xyz_1}
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_3.outputs["Z"],
            1: group_input_2.outputs["DoorShelfMargin"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz_4.outputs["X"],
            "Y": separate_xyz_3.outputs["Y"],
            "Z": subtract_2,
        },
    )

    sizeofobject = nw.new_node(
        nodegroup_size_of_object().name, input_kwargs={"Geometry": transform_geometry_4}
    )

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": 1.0000,
            "Y": 1.0000,
            "Z": group_input_2.outputs["DoorShelfNum"],
        },
    )

    multipleobjects = nw.new_node(
        nodegroup_multiple_objects().name,
        input_kwargs={
            "Object": transform_geometry_4,
            "SpaceSize": combine_xyz_2,
            "ObjectSize": sizeofobject,
            "ObjectNum": combine_xyz_9,
        },
    )

    movetoorigin_1 = nw.new_node(
        nodegroup_move_to_origin().name, input_kwargs={"Geometry": multipleobjects}
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": separate_xyz_4.outputs["Z"]}
    )

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_8, 1: (-0.5000, -0.5000, -0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": movetoorigin_1,
            "Translation": multiply_3.outputs["Vector"],
        },
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_3,
            "Material": group_input_2.outputs["DoorShelfMaterial"],
        },
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material_2, "Name": "joint2", "Value": 1},
        attrs={"data_type": "INT"},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_2.outputs["DoorHandleTopSize"]},
    )

    handle = nw.new_node(
        nodegroup_handle().name,
        input_kwargs={
            "TopSize": reroute_4,
            "TopThickness": group_input_2.outputs["DoorHandleTopThickness"],
            "TopRoundness": group_input_2.outputs["DoorHandleTopRoundness"],
            "SupportSize": group_input_2.outputs["DoorHandleSupportSize"],
            "SupportMargin": group_input_2.outputs["DoorHandleSupportMargin"],
            "Material": group_input_2.outputs["HandleMaterial"],
        },
    )

    separate_xyz_9 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": combine_xyz_1}
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_9.outputs["X"]},
        attrs={"operation": "MULTIPLY"},
    )

    separate_xyz_8 = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": group_input_2.outputs["DoorHandleSupportSize"]},
    )

    add_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_4, 1: separate_xyz_8.outputs["Z"]}
    )

    subtract_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_1, 1: reroute_4},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_3.outputs["Vector"]}
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_5, 1: group_input_2.outputs["DoorHandleMargin"]},
        attrs={"operation": "SUBTRACT"},
    )

    map_range_1 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": group_input_2.outputs["DoorOnRight"],
            3: 1.0000,
            4: -1.0000,
        },
    )

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_4, 1: map_range_1.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_3, "Y": multiply_6}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": handle,
            "Translation": combine_xyz_6,
            "Rotation": (1.5708, 1.5708, 1.5708),
        },
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_1, "Name": "joint2", "Value": 2},
        attrs={"data_type": "INT"},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                store_named_attribute_1,
                store_named_attribute_2,
                store_named_attribute_3,
            ]
        },
    )

    separate_xyz_10 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input_2.outputs["Size"]}
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_10.outputs["X"]},
        attrs={"operation": "MULTIPLY"},
    )

    separate_xyz_11 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": combine_xyz_1}
    )

    subtract_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_10.outputs["Y"], 1: separate_xyz_11.outputs["Y"]},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["DoorOnRight"]}
    )

    map_range_2 = nw.new_node(
        Nodes.MapRange, input_kwargs={"Value": reroute_2, 3: -1.0000}
    )

    multiply_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: map_range_2.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_5, 1: multiply_8},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_7, "Y": multiply_9}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_1, "Translation": combine_xyz_3},
    )

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry, "Label": "door"},
    )

    store_named_attribute_4 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_1,
            "Name": "joint1",
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    multiply_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_11.outputs["X"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_11 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_11.outputs["Y"]},
        attrs={"operation": "MULTIPLY"},
    )

    map_range = nw.new_node(
        Nodes.MapRange, input_kwargs={"Value": reroute_2, 3: -1.0000}
    )

    multiply_12 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_11, 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_10, "Y": multiply_12}
    )

    centerofgeometry = nw.new_node(
        nodegroup_center_of_geometry().name, input_kwargs={"Geometry": twomaterialcube}
    )

    centerofgeometry_1 = nw.new_node(
        nodegroup_center_of_geometry().name, input_kwargs={"Geometry": join_geometry_1}
    )

    subtract_6 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: centerofgeometry, 1: centerofgeometry_1},
        attrs={"operation": "SUBTRACT"},
    )

    add_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_4, 1: subtract_6.outputs["Vector"]},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_12})

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "door_hinge",
            "Parent": store_named_attribute_5,
            "Child": store_named_attribute_4,
            "Position": add_4.outputs["Vector"],
            "Axis": combine_xyz_5,
            "Max": 3.1400,
        },
    )

    store_named_attribute_6 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": hinge_joint.outputs["Geometry"],
            "Name": "joint0",
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    separate_xyz_5 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input_2.outputs["Size"]}
    )

    separate_xyz_12 = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": group_input_2.outputs["DoorShelfSize"]},
    )

    separate_xyz_6 = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": group_input_2.outputs["DrawerHandleTopSize"]},
    )

    separate_xyz_7 = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": group_input_2.outputs["DrawerHandleSupportSize"]},
    )

    add_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_6.outputs["Z"], 1: separate_xyz_7.outputs["Z"]},
    )

    add_6 = nw.new_node(
        Nodes.Math, input_kwargs={0: separate_xyz_12.outputs["Z"], 1: add_5}
    )

    multiply_13 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_2.outputs["WallThickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    add_7 = nw.new_node(Nodes.Math, input_kwargs={0: add_6, 1: multiply_13})

    subtract_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_5.outputs["X"], 1: add_7},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_5.outputs["Y"], 1: multiply_13},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_10 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": subtract_7,
            "Y": subtract_8,
            "Z": group_input_2.outputs["DrawerHeight"],
        },
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_2.outputs["DrawerMaterial"]}
    )

    multidrawer = nw.new_node(
        nodegroup_multi_drawer().name,
        input_kwargs={
            "Size": combine_xyz_10,
            "WallThickness": group_input_2.outputs["DrawerWallThickness"],
            "HandleMargin": group_input_2.outputs["DrawerHandleMargin"],
            "HandleTopSize": group_input_2.outputs["DrawerHandleTopSize"],
            "HandleTopThickness": group_input_2.outputs["DrawerHandleTopThickness"],
            "HandleTopRoundness": group_input_2.outputs["DrawerHandleTopRoundness"],
            "HandleSupportSize": group_input_2.outputs["DrawerHandleSupportSize"],
            "HandleSupportMargin": group_input_2.outputs["DrawerHandleSupportMargin"],
            "SlidingJointValue": group_input_2.outputs["DrawerSlidingJointValue"],
            "BodyRoundness": group_input_2.outputs["DrawerBodyRoundness"],
            "SlideRoundness": group_input_2.outputs["DrawerSlideRoundness"],
            "InnerRoundness": group_input_2.outputs["DrawernnerRoundness"],
            "InnerMaterial": reroute_3,
            "OuterMaterial": reroute_3,
            "HandleMaterial": group_input_2.outputs["DrawerHandleMaterial"],
            "DrawerNum": group_input_2.outputs["DrawerNum"],
        },
    )

    multiply_14 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_5.outputs["X"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_15 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_2.outputs["WallThickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    add_8 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_14, 1: multiply_15})

    multiply_16 = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract_7}, attrs={"operation": "MULTIPLY"}
    )

    add_9 = nw.new_node(Nodes.Math, input_kwargs={0: add_8, 1: multiply_16})

    subtract_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_5.outputs["Z"],
            1: group_input_2.outputs["DrawerHeight"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    multiply_17 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_9, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_17, 1: group_input_2.outputs["WallThickness"]},
    )

    combine_xyz_12 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_9, "Z": add_10}
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": multidrawer, "Translation": combine_xyz_12},
    )

    store_named_attribute_7 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_5, "Name": "switch1", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input_2.outputs["DrawerOnBottom"],
            "True": store_named_attribute_7,
        },
    )

    store_named_attribute_8 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch, "Name": "joint0", "Value": 2},
        attrs={"data_type": "INT"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                store_named_attribute,
                store_named_attribute_6,
                store_named_attribute_8,
            ]
        },
    )

    store_named_attribute_100 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": join_geometry,
            "Name": "jointr",
            "Value": group_input_2.outputs["Value"],
        },
        attrs={"data_type": "INT"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": store_named_attribute_100},
        attrs={"is_active_output": True},
    )


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


def sample_light_exterior():
    """Generate a light color for the lamp shade exterior"""
    # Light pastel colors - high value, moderate-low saturation
    h = np.random.uniform(0, 1)  # Any hue is possible
    s = np.random.uniform(0.1, 0.3)  # Low-moderate saturation for pastel effect
    v = np.random.uniform(0.8, 0.95)  # High value but not too bright

    # Convert to RGB
    rgb = hsv2rgba(h, s, v)
    return rgb


class SinglefridgeFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)
        self.sim_blueprint = blueprint_path_completion("singlefridge.json")

    def sample_parameters(self):
        # add code here to randomly sample from parameters
        return {}

    def create_asset(self, export=True, exporter="mjcf", asset_params=None, **kwargs):
        obj = butil.spawn_vert()
        butil.modify_mesh(
            obj,
            "NODES",
            apply=False,
            node_group=nodegroup_singlefridge(),
            ng_inputs=self.sample_parameters(),
        )

        return obj
