# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: Primary author

from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata", singleton=False, type="GeometryNodeTree"
)
def nodegroup_add_jointed_geometry_metadata(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

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


@node_utils.to_nodegroup("nodegroup_arrow", singleton=False, type="GeometryNodeTree")
def nodegroup_arrow(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Depth", 0.1000),
            ("NodeSocketVector", "Axis", (0.0000, 0.0000, 0.0000)),
        ],
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 1: 6.0000},
        attrs={"operation": "DIVIDE"},
    )

    cone = nw.new_node(
        "GeometryNodeMeshCone",
        input_kwargs={
            "Vertices": 5,
            "Radius Bottom": divide,
            "Depth": group_input.outputs["Depth"],
        },
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Depth"]}
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: (0.0000, 0.0000, 0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cone.outputs["Mesh"],
            "Translation": multiply.outputs["Vector"],
        },
    )

    divide_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: divide, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 6,
            "Radius": divide_1,
            "Depth": group_input.outputs["Depth"],
        },
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_1, cylinder.outputs["Mesh"]]},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_2,
            "Translation": multiply.outputs["Vector"],
        },
    )

    position = nw.new_node(Nodes.InputPosition)

    normalize = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Axis"]},
        attrs={"operation": "NORMALIZE"},
    )

    align_rotation_to_vector = nw.new_node(
        "FunctionNodeAlignRotationToVector",
        input_kwargs={"Vector": normalize.outputs["Vector"]},
    )

    rotate_vector = nw.new_node(
        "FunctionNodeRotateVector",
        input_kwargs={"Vector": position, "Rotation": align_rotation_to_vector},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": transform_geometry, "Position": rotate_vector},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_position},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_top_body", singleton=False, type="GeometryNodeTree")
def nodegroup_top_body(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    named_attribute_2 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    attribute_statistic_1 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Attribute": named_attribute_2.outputs["Attribute"],
        },
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_2.outputs["Attribute"],
            3: attribute_statistic_1.outputs["Min"],
        },
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    separate_geometry_3 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": group_input.outputs["Geometry"], "Selection": equal},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Selection": separate_geometry_3.outputs["Selection"],
            "Inverted": separate_geometry_3.outputs["Inverted"],
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_hinge_joint", singleton=False, type="GeometryNodeTree"
)
def nodegroup_hinge_joint(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketString", "Joint ID (do not set)", ""),
            ("NodeSocketString", "Joint Label", ""),
            ("NodeSocketGeometry", "Parent", None),
            ("NodeSocketGeometry", "Child", None),
            ("NodeSocketVector", "Position", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "Axis", (0.0000, 0.0000, 1.0000)),
            ("NodeSocketFloat", "Value", 0.0000),
            ("NodeSocketFloat", "Min", 0.0000),
            ("NodeSocketFloat", "Max", 0.0000),
            ("NodeSocketBool", "Show Joint", False),
        ],
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

    top_body = nw.new_node(
        nodegroup_top_body().name, input_kwargs={"Geometry": store_named_attribute}
    )

    named_attribute_11 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "is_jointed"},
        attrs={"data_type": "BOOLEAN"},
    )

    attribute_statistic_7 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": top_body.outputs["Selection"],
            "Attribute": named_attribute_11.outputs["Attribute"],
        },
    )

    greater_than = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: attribute_statistic_7.outputs["Sum"]},
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

    position_3 = nw.new_node(Nodes.InputPosition)

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

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: group_input.outputs["Min"], "Epsilon": 0.0000},
        attrs={"operation": "EQUAL"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: group_input.outputs["Max"], "Epsilon": 0.0000},
        attrs={"operation": "EQUAL"},
    )

    op_and = nw.new_node(Nodes.BooleanMath, input_kwargs={0: equal, 1: equal_1})

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
            "Axis": transform_direction,
            "Angle": reroute,
        },
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": store_named_attribute_2, "Position": vector_rotate},
    )

    named_attribute_7 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "is_jointed"},
        attrs={"data_type": "BOOLEAN"},
    )

    attribute_statistic_4 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": top_body.outputs["Selection"],
            "Attribute": named_attribute_7.outputs["Attribute"],
        },
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: attribute_statistic_4.outputs["Sum"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    position_4 = nw.new_node(Nodes.InputPosition)

    position_1 = nw.new_node(Nodes.InputPosition)

    named_attribute_13 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "axis_group"},
        attrs={"data_type": "INT"},
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: named_attribute_13.outputs["Attribute"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    separate_geometry_2 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": group_input.outputs["Parent"], "Selection": equal_3},
    )

    named_attribute = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "debugging"},
        attrs={"data_type": "INT"},
    )

    equal_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: named_attribute.outputs["Attribute"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    separate_geometry = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={
            "Geometry": separate_geometry_2.outputs["Selection"],
            "Selection": equal_4,
        },
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
            "Geometry": separate_geometry.outputs["Selection"],
            "Name": "part_id",
            "Value": switch_2,
        },
        attrs={"data_type": "INT"},
    )

    top_body_1 = nw.new_node(
        nodegroup_top_body().name, input_kwargs={"Geometry": store_named_attribute_1}
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": top_body_1.outputs["Selection"]}
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

    add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_1.outputs["Vector"], 1: group_input.outputs["Position"]},
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
        Nodes.SetPosition, input_kwargs={"Geometry": set_position_1, "Position": switch}
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_position, "Name": "is_jointed", "Value": True},
        attrs={"data_type": "BOOLEAN"},
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
            "Geometry": store_named_attribute_3,
            "Name": "body_transform",
            "Value": combine_matrix_1,
        },
        attrs={"data_type": "FLOAT4X4"},
    )

    string_5 = nw.new_node("FunctionNodeInputString", attrs={"string": "posparent"})

    reroute_4 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Joint ID (do not set)"]},
    )

    join_strings_5 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_5, reroute_4]},
    )

    store_named_attribute_9 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_4,
            "Name": join_strings_5,
            "Value": group_input.outputs["Position"],
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    string_1 = nw.new_node("FunctionNodeInputString", attrs={"string": "poschild"})

    join_strings_1 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_1, reroute_4]},
    )

    bounding_box_1 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": top_body.outputs["Selection"]}
    )

    add_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_1.outputs["Min"],
            1: bounding_box_1.outputs["Max"],
        },
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_3.outputs["Vector"], "Scale": -0.5000},
        attrs={"operation": "SCALE"},
    )

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_9,
            "Name": join_strings_1,
            "Value": scale.outputs["Vector"],
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

    bounding_box_2 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": top_body_1.outputs["Selection"]}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_2.outputs["Max"],
            1: bounding_box_2.outputs["Min"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    maximum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["X"], 1: separate_xyz_3.outputs["Y"]},
        attrs={"operation": "MAXIMUM"},
    )

    maximum_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: maximum, 1: separate_xyz_3.outputs["Z"]},
        attrs={"operation": "MAXIMUM"},
    )

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Axis"]}
    )

    arrow = nw.new_node(
        nodegroup_arrow().name, input_kwargs={"Depth": maximum_1, "Axis": reroute_5}
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Position"]}
    )

    add_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: attribute_statistic_2.outputs["Mean"], 1: reroute_3},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": arrow, "Translation": add_4.outputs["Vector"]},
    )

    store_named_attribute_11 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": transform_geometry_1,
            "Name": "debugging",
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_11, store_named_attribute_7]},
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Joint"],
            "False": store_named_attribute_7,
            "True": join_geometry_4,
        },
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": store_named_attribute_1}
    )

    named_attribute_1 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "debugging"},
        attrs={"data_type": "INT"},
    )

    equal_5 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: named_attribute_1.outputs["Attribute"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    separate_geometry_1 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": set_position, "Selection": equal_5},
    )

    top_body_2 = nw.new_node(
        nodegroup_top_body().name,
        input_kwargs={"Geometry": separate_geometry_1.outputs["Selection"]},
    )

    bounding_box_5 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": top_body_2.outputs["Selection"]}
    )

    add_5 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_5.outputs["Min"],
            1: bounding_box_5.outputs["Max"],
        },
    )

    scale_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_5.outputs["Vector"], "Scale": 0.5000},
        attrs={"operation": "SCALE"},
    )

    add_6 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: (0.0000, 0.0000, 1.0000), 1: scale_1.outputs["Vector"]},
    )

    points_1 = nw.new_node(
        "GeometryNodePoints",
        input_kwargs={"Position": add_6.outputs["Vector"], "Radius": 0.0000},
    )

    string_6 = nw.new_node("FunctionNodeInputString", attrs={"string": "zaxis"})

    join_strings_6 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_6, reroute_4]},
    )

    store_named_attribute_13 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": points_1, "Name": join_strings_6, "Value": 1},
        attrs={"data_type": "INT"},
    )

    add_7 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: (1.0000, 0.0000, 0.0000), 1: scale_1.outputs["Vector"]},
    )

    points_2 = nw.new_node(
        "GeometryNodePoints",
        input_kwargs={"Position": add_7.outputs["Vector"], "Radius": 0.0000},
    )

    string_7 = nw.new_node("FunctionNodeInputString", attrs={"string": "xaxis"})

    join_strings_7 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_7, reroute_4]},
    )

    store_named_attribute_16 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": points_2, "Name": join_strings_7, "Value": 1},
        attrs={"data_type": "INT"},
    )

    add_8 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: (0.0000, 1.0000, 0.0000), 1: scale_1.outputs["Vector"]},
    )

    points_3 = nw.new_node(
        "GeometryNodePoints",
        input_kwargs={"Position": add_8.outputs["Vector"], "Radius": 0.0000},
    )

    string_8 = nw.new_node("FunctionNodeInputString", attrs={"string": "yaxis"})

    join_strings_8 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_8, reroute_4]},
    )

    store_named_attribute_17 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": points_3, "Name": join_strings_8, "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                store_named_attribute_13,
                store_named_attribute_16,
                store_named_attribute_17,
            ]
        },
    )

    store_named_attribute_14 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_3, "Name": "axis_group", "Value": 1},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_15 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_14,
            "Name": "part_id",
            "Value": 999999,
        },
        attrs={"data_type": "INT"},
    )

    points_to_vertices_1 = nw.new_node(
        Nodes.PointsToVertices, input_kwargs={"Points": store_named_attribute_15}
    )

    join_geometry_5 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [switch_6, reroute_2, points_to_vertices_1]},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                join_geometry_5,
                separate_geometry.outputs["Inverted"],
                separate_geometry_2.outputs["Inverted"],
            ]
        },
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                reroute_2,
                separate_geometry.outputs["Inverted"],
                separate_geometry_2.outputs["Inverted"],
            ]
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [switch_6, points_to_vertices_1]}
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
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketString", "Joint ID (do not set)", ""),
            ("NodeSocketString", "Joint Label", ""),
            ("NodeSocketGeometry", "Parent", None),
            ("NodeSocketGeometry", "Child", None),
            ("NodeSocketVector", "Position", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "Axis", (0.0000, 0.0000, 1.0000)),
            ("NodeSocketFloat", "Value", 0.0000),
            ("NodeSocketFloat", "Min", 0.0000),
            ("NodeSocketFloat", "Max", 0.0000),
            ("NodeSocketBool", "Show Joint", False),
        ],
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

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_2.outputs["Attribute"],
            3: attribute_statistic_1.outputs["Min"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_3 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": store_named_attribute, "Selection": equal},
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
        input_kwargs={2: attribute_statistic_7.outputs["Sum"]},
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

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: attribute_statistic_4.outputs["Sum"]},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    position_4 = nw.new_node(Nodes.InputPosition)

    position_1 = nw.new_node(Nodes.InputPosition)

    named_attribute_12 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "axis_group"},
        attrs={"data_type": "INT"},
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: named_attribute_12.outputs["Attribute"]},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_5 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": group_input.outputs["Parent"], "Selection": equal_2},
    )

    named_attribute = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "debugging"},
        attrs={"data_type": "INT"},
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: named_attribute.outputs["Attribute"]},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={
            "Geometry": separate_geometry_5.outputs["Selection"],
            "Selection": equal_3,
        },
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
            "Geometry": separate_geometry.outputs["Selection"],
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

    equal_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_1.outputs["Attribute"],
            3: attribute_statistic.outputs["Min"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_2 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": store_named_attribute_1, "Selection": equal_4},
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

    add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: position_1, 1: attribute_statistic_2.outputs["Mean"]},
    )

    add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_1.outputs["Vector"], 1: group_input.outputs["Position"]},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_1,
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

    equal_5 = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: group_input.outputs["Min"], "Epsilon": 0.0000},
        attrs={"operation": "EQUAL"},
    )

    equal_6 = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: group_input.outputs["Max"], "Epsilon": 0.0000},
        attrs={"operation": "EQUAL"},
    )

    op_and = nw.new_node(Nodes.BooleanMath, input_kwargs={0: equal_5, 1: equal_6})

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

    string_4 = nw.new_node("FunctionNodeInputString", attrs={"string": "posparent"})

    reroute_2 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Joint ID (do not set)"]},
    )

    join_strings_4 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_4, reroute_2]},
    )

    store_named_attribute_15 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": set_position_2,
            "Name": join_strings_4,
            "Value": group_input.outputs["Position"],
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    string = nw.new_node("FunctionNodeInputString", attrs={"string": "poschild"})

    join_strings = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string, reroute_2]},
    )

    bounding_box_1 = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry_3.outputs["Selection"]},
    )

    add_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_1.outputs["Min"],
            1: bounding_box_1.outputs["Max"],
        },
    )

    scale_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_4.outputs["Vector"], "Scale": -0.5000},
        attrs={"operation": "SCALE"},
    )

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_15,
            "Name": join_strings,
            "Value": scale_1.outputs["Vector"],
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

    bounding_box_2 = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry_2.outputs["Selection"]},
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_2.outputs["Max"],
            1: bounding_box_2.outputs["Min"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    maximum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["X"], 1: separate_xyz_3.outputs["Y"]},
        attrs={"operation": "MAXIMUM"},
    )

    maximum_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: maximum, 1: separate_xyz_3.outputs["Z"]},
        attrs={"operation": "MAXIMUM"},
    )

    arrow = nw.new_node(
        nodegroup_arrow().name,
        input_kwargs={"Depth": maximum_1, "Axis": group_input.outputs["Axis"]},
    )

    add_5 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: attribute_statistic_2.outputs["Mean"],
            1: group_input.outputs["Position"],
        },
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": arrow, "Translation": add_5.outputs["Vector"]},
    )

    store_named_attribute_11 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": transform_geometry_1,
            "Name": "debugging",
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_11, store_named_attribute_7]},
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Joint"],
            "False": store_named_attribute_7,
            "True": join_geometry_4,
        },
    )

    named_attribute_6 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "debugging"},
        attrs={"data_type": "INT"},
    )

    equal_7 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: named_attribute_6.outputs["Attribute"]},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_4 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": set_position, "Selection": equal_7},
    )

    top_body = nw.new_node(
        nodegroup_top_body().name,
        input_kwargs={"Geometry": separate_geometry_4.outputs["Selection"]},
    )

    bounding_box_5 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": top_body.outputs["Selection"]}
    )

    add_6 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_5.outputs["Min"],
            1: bounding_box_5.outputs["Max"],
        },
    )

    scale_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_6.outputs["Vector"], "Scale": 0.5000},
        attrs={"operation": "SCALE"},
    )

    add_7 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: (0.0000, 0.0000, 1.0000), 1: scale_2.outputs["Vector"]},
    )

    points = nw.new_node(
        "GeometryNodePoints", input_kwargs={"Position": add_7.outputs["Vector"]}
    )

    string_5 = nw.new_node("FunctionNodeInputString", attrs={"string": "zaxis"})

    join_strings_5 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_5, reroute_2]},
    )

    store_named_attribute_9 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": points, "Name": join_strings_5, "Value": 1},
        attrs={"data_type": "INT"},
    )

    add_8 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: (1.0000, 0.0000, 0.0000), 1: scale_2.outputs["Vector"]},
    )

    points_1 = nw.new_node(
        "GeometryNodePoints", input_kwargs={"Position": add_8.outputs["Vector"]}
    )

    string_6 = nw.new_node("FunctionNodeInputString", attrs={"string": "xaxis"})

    join_strings_6 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_6, reroute_2]},
    )

    store_named_attribute_13 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": points_1, "Name": join_strings_6, "Value": 1},
        attrs={"data_type": "INT"},
    )

    add_9 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: (0.0000, 1.0000, 0.0000), 1: scale_2.outputs["Vector"]},
    )

    points_2 = nw.new_node(
        "GeometryNodePoints", input_kwargs={"Position": add_9.outputs["Vector"]}
    )

    string_7 = nw.new_node("FunctionNodeInputString", attrs={"string": "yaxis"})

    join_strings_7 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_7, reroute_2]},
    )

    store_named_attribute_14 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": points_2, "Name": join_strings_7, "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                store_named_attribute_9,
                store_named_attribute_13,
                store_named_attribute_14,
            ]
        },
    )

    store_named_attribute_10 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_3, "Name": "axis_group", "Value": 1},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_12 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_10,
            "Name": "part_id",
            "Value": 999999,
        },
        attrs={"data_type": "INT"},
    )

    points_to_vertices = nw.new_node(
        Nodes.PointsToVertices, input_kwargs={"Points": store_named_attribute_12}
    )

    join_geometry_5 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [switch_6, store_named_attribute_1, points_to_vertices]
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                join_geometry_5,
                separate_geometry.outputs["Inverted"],
                separate_geometry_5.outputs["Inverted"],
            ]
        },
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                store_named_attribute_1,
                separate_geometry.outputs["Inverted"],
                separate_geometry_5.outputs["Inverted"],
            ]
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [switch_6, points_to_vertices]}
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


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    cube = nw.new_node(Nodes.MeshCube)

    cube_1 = nw.new_node(Nodes.MeshCube)

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Translation": (0.0000, 1.0000, 0.0000),
        },
    )

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "sliding",
            "Parent": cube.outputs["Mesh"],
            "Child": transform_geometry,
            "Axis": (0.0000, 1.0000, 0.0000),
            "Max": 1.0000,
            "Show Joint": True,
        },
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": sliding_joint.outputs["Geometry"],
            "Translation": (0.0000, 0.0000, 0.5000),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_duplicate_joints_on_parent", singleton=False, type="GeometryNodeTree"
)
def nodegroup_duplicate_joints_on_parent(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

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
    "nodegroup_distance_from_center", singleton=False, type="GeometryNodeTree"
)
def nodegroup_distance_from_center(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    named_attribute = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    attribute_statistic = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Attribute": named_attribute.outputs["Attribute"],
        },
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: attribute_statistic.outputs["Min"],
            3: named_attribute.outputs["Attribute"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": group_input.outputs["Geometry"], "Selection": equal},
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry.outputs["Selection"]},
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    attribute_statistic_2 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz.outputs["X"],
        },
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_2.outputs["Max"],
            1: attribute_statistic_2.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_1})

    attribute_statistic_3 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz_1.outputs["Y"],
        },
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_3.outputs["Max"],
            1: attribute_statistic_3.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    position_2 = nw.new_node(Nodes.InputPosition)

    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_2})

    attribute_statistic_4 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz_2.outputs["Z"],
        },
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_4.outputs["Max"],
            1: attribute_statistic_4.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": subtract_1, "Z": subtract_2}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Vector": combine_xyz},
        attrs={"is_active_output": True},
    )
