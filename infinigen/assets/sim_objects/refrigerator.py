# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: Version 2
# - Hongyu Wen: Version 1
# - Max Gonzalez Saez-Diez: Updates for sim

import gin
from numpy.random import randint, uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.joints import (
    nodegroup_add_jointed_geometry_metadata,
    nodegroup_hinge_joint,
    nodegroup_sliding_joint,
)
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import weighted_sample


@node_utils.to_nodegroup(
    "nodegroup_rounded_quad_029", singleton=False, type="GeometryNodeTree"
)
def nodegroup_rounded_quad_029(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": reroute, "Height": reroute_1},
    )

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute, 1: reroute_1},
        attrs={"operation": "MINIMUM"},
    )

    map_range = nw.new_node(Nodes.MapRange, input_kwargs={"Value": 0.5000, 4: 0.5000})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum, 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={"Curve": quadrilateral, "Count": 3, "Radius": multiply},
        attrs={"mode": "POLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Curve": fillet_curve},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_index_select_119", singleton=False, type="GeometryNodeTree"
)
def nodegroup_index_select_119(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "Index", 0)]
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: group_input.outputs["Index"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Result": equal},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_index_select_118", singleton=False, type="GeometryNodeTree"
)
def nodegroup_index_select_118(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "Index", 0)]
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: group_input.outputs["Index"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Result": equal},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_index_select_117", singleton=False, type="GeometryNodeTree"
)
def nodegroup_index_select_117(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "Index", 0)]
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: group_input.outputs["Index"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Result": equal},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_index_select_116", singleton=False, type="GeometryNodeTree"
)
def nodegroup_index_select_116(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "Index", 0)]
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: group_input.outputs["Index"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Result": equal},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_rounded_quad_025", singleton=False, type="GeometryNodeTree"
)
def nodegroup_rounded_quad_025(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": reroute, "Height": reroute_1},
    )

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute, 1: reroute_1},
        attrs={"operation": "MINIMUM"},
    )

    map_range = nw.new_node(Nodes.MapRange, input_kwargs={"Value": 0.5000, 4: 0.5000})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum, 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={"Curve": quadrilateral, "Count": 3, "Radius": multiply},
        attrs={"mode": "POLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Curve": fillet_curve},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_index_select_103", singleton=False, type="GeometryNodeTree"
)
def nodegroup_index_select_103(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "Index", 0)]
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: group_input.outputs["Index"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Result": equal},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_index_select_102", singleton=False, type="GeometryNodeTree"
)
def nodegroup_index_select_102(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "Index", 0)]
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: group_input.outputs["Index"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Result": equal},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_index_select_101", singleton=False, type="GeometryNodeTree"
)
def nodegroup_index_select_101(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "Index", 0)]
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: group_input.outputs["Index"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Result": equal},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_index_select_100", singleton=False, type="GeometryNodeTree"
)
def nodegroup_index_select_100(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "Index", 0)]
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: group_input.outputs["Index"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Result": equal},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_rounded_quad_031", singleton=False, type="GeometryNodeTree"
)
def nodegroup_rounded_quad_031(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": reroute, "Height": reroute_1},
    )

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute, 1: reroute_1},
        attrs={"operation": "MINIMUM"},
    )

    map_range = nw.new_node(Nodes.MapRange, input_kwargs={"Value": 0.5000, 4: 0.5000})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum, 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={"Curve": quadrilateral, "Count": 3, "Radius": multiply},
        attrs={"mode": "POLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Curve": fillet_curve},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_index_select_127", singleton=False, type="GeometryNodeTree"
)
def nodegroup_index_select_127(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "Index", 0)]
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: group_input.outputs["Index"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Result": equal},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_index_select_126", singleton=False, type="GeometryNodeTree"
)
def nodegroup_index_select_126(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "Index", 0)]
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: group_input.outputs["Index"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Result": equal},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_index_select_125", singleton=False, type="GeometryNodeTree"
)
def nodegroup_index_select_125(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "Index", 0)]
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: group_input.outputs["Index"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Result": equal},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_index_select_124", singleton=False, type="GeometryNodeTree"
)
def nodegroup_index_select_124(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "Index", 0)]
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: group_input.outputs["Index"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Result": equal},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_rounded_quad_030", singleton=False, type="GeometryNodeTree"
)
def nodegroup_rounded_quad_030(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": reroute, "Height": reroute_1},
    )

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute, 1: reroute_1},
        attrs={"operation": "MINIMUM"},
    )

    map_range = nw.new_node(Nodes.MapRange, input_kwargs={"Value": 0.5000, 4: 0.5000})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum, 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={"Curve": quadrilateral, "Count": 3, "Radius": multiply},
        attrs={"mode": "POLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Curve": fillet_curve},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_index_select_123", singleton=False, type="GeometryNodeTree"
)
def nodegroup_index_select_123(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "Index", 0)]
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: group_input.outputs["Index"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Result": equal},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_index_select_122", singleton=False, type="GeometryNodeTree"
)
def nodegroup_index_select_122(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "Index", 0)]
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: group_input.outputs["Index"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Result": equal},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_index_select_121", singleton=False, type="GeometryNodeTree"
)
def nodegroup_index_select_121(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "Index", 0)]
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: group_input.outputs["Index"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Result": equal},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_index_select_120", singleton=False, type="GeometryNodeTree"
)
def nodegroup_index_select_120(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    index = nw.new_node(Nodes.Index)

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "Index", 0)]
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: group_input.outputs["Index"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Result": equal},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_fridge_shelf_003", singleton=False, type="GeometryNodeTree"
)
def nodegroup_fridge_shelf_003(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Fridge Shelf Thickness", 0.0000),
            ("NodeSocketBool", "Has Grated Shelves", False),
            ("NodeSocketMaterial", "Interior Shelf Material Border", None),
            ("NodeSocketMaterial", "Interior Shelf Material Inside", None),
        ],
    )

    cube = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": group_input.outputs["Size"]}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Fridge Shelf Thickness"],
            "Y": group_input.outputs["Fridge Shelf Thickness"],
        },
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: combine_xyz},
        attrs={"operation": "SUBTRACT"},
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": subtract.outputs["Vector"]}
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": cube.outputs["Mesh"], "Mesh 2": cube_1.outputs["Mesh"]},
        attrs={"solver": "EXACT"},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": difference.outputs["Mesh"],
            "Material": group_input.outputs["Interior Shelf Material Border"],
        },
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Material": group_input.outputs["Interior Shelf Material Inside"],
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, set_material_1]}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Size"]}
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute, 1: (0.0000, -0.5000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute, 1: (0.0000, 0.5000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    mesh_line = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={
            "Count": 24,
            "Start Location": multiply.outputs["Vector"],
            "Offset": multiply_1.outputs["Vector"],
        },
        attrs={"mode": "END_POINTS"},
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0020

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": reroute})

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 8,
            "Radius": value,
            "Depth": separate_xyz.outputs["X"],
        },
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={"Points": mesh_line, "Instance": transform_geometry},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points}
    )

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 8,
            "Radius": value,
            "Depth": separate_xyz.outputs["Y"],
        },
    )

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute, 1: (0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": multiply_2.outputs["Vector"],
            "Rotation": (0.0000, 1.5708, 1.5708),
        },
    )

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute, 1: (-0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": multiply_3.outputs["Vector"],
            "Rotation": (0.0000, 1.5708, 1.5708),
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [realize_instances, transform_geometry_1, transform_geometry_2]
        },
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Material": group_input.outputs["Interior Shelf Material Border"],
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Has Grated Shelves"],
            "False": join_geometry,
            "True": set_material_2,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": switch},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_drawer_010", singleton=False, type="GeometryNodeTree"
)
def nodegroup_drawer_010(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Thickness", 0.0000),
            ("NodeSocketMaterial", "Drawer Material", None),
        ],
    )

    cube = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": group_input.outputs["Size"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply, "Y": multiply}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: combine_xyz},
        attrs={"operation": "SUBTRACT"},
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": subtract.outputs["Vector"]}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Thickness"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube_1.outputs["Mesh"], "Translation": combine_xyz_1},
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": cube.outputs["Mesh"], "Mesh 2": transform_geometry},
        attrs={"solver": "EXACT"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": 0.0200, "Y": separate_xyz.outputs["Y"], "Z": 0.0050},
    )

    cube_2 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_2})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_2.outputs["Mesh"],
            "Translation": (0.0075, 0.0000, -0.0075),
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    cube_3 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_2})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_1, cube_3.outputs["Mesh"]]},
    )

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: (0.5000, 0.0000, 0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_1.outputs["Vector"], 1: (0.0100, 0.0000, -0.0025)},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Translation": add.outputs["Vector"]},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [difference.outputs["Mesh"], transform_geometry_2]},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Material": group_input.outputs["Drawer Material"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": set_material},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_shell_boolean_size_006", singleton=False, type="GeometryNodeTree"
)
def nodegroup_shell_boolean_size_006(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Outer size", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "Thickness", 0.5000),
            ("NodeSocketFloat", "Thickness offset", 0.5000),
        ],
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Thickness"],
            1: group_input.outputs["Thickness offset"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": multiply, "Z": multiply}
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Outer size"], 1: combine_xyz},
        attrs={"operation": "SUBTRACT"},
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Boolean size": subtract_1.outputs["Vector"],
            "Boolean translate": combine_xyz_1,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_shell_boolean_size_007", singleton=False, type="GeometryNodeTree"
)
def nodegroup_shell_boolean_size_007(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Outer size", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "Thickness", 0.5000),
            ("NodeSocketFloat", "Thickness offset", 0.5000),
        ],
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Thickness"],
            1: group_input.outputs["Thickness offset"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": multiply, "Z": multiply}
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Outer size"], 1: combine_xyz},
        attrs={"operation": "SUBTRACT"},
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Boolean size": subtract_1.outputs["Vector"],
            "Boolean translate": combine_xyz_1,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_handle_037", singleton=False, type="GeometryNodeTree"
)
def nodegroup_handle_037(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "Handle Type", 0),
            ("NodeSocketFloat", "Handle Length", 0.0000),
            ("NodeSocketFloat", "Handle Radius", 0.0100),
            ("NodeSocketFloat", "Handle Protrude", 0.0300),
            ("NodeSocketFloat", "Handle Width", 0.0000),
            ("NodeSocketFloat", "Handle Height", 0.0000),
        ],
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Handle Type"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Handle Type"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    mesh_line = nw.new_node(Nodes.MeshLine, input_kwargs={"Count": 4})

    index_select_116 = nw.new_node(nodegroup_index_select_116().name)

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide, 1: group_input.outputs["Handle Radius"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": subtract})

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": mesh_line,
            "Selection": index_select_116,
            "Position": combine_xyz,
        },
    )

    index_select_117 = nw.new_node(
        nodegroup_index_select_117().name, input_kwargs={"Index": 1}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": group_input.outputs["Handle Protrude"], "Z": subtract},
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position,
            "Selection": index_select_117,
            "Position": combine_xyz_1,
        },
    )

    index_select_118 = nw.new_node(
        nodegroup_index_select_118().name, input_kwargs={"Index": 2}
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: divide_1, 1: group_input.outputs["Handle Radius"]}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": group_input.outputs["Handle Protrude"], "Z": add},
    )

    set_position_2 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position_1,
            "Selection": index_select_118,
            "Position": combine_xyz_2,
        },
    )

    index_select_119 = nw.new_node(
        nodegroup_index_select_119().name, input_kwargs={"Index": 3}
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add})

    set_position_3 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position_2,
            "Selection": index_select_119,
            "Position": combine_xyz_3,
        },
    )

    mesh_to_curve = nw.new_node(
        Nodes.MeshToCurve, input_kwargs={"Mesh": set_position_3}
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": mesh_to_curve,
            "Count": 6,
            "Radius": group_input.outputs["Handle Radius"],
        },
        attrs={"mode": "POLY"},
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={"Resolution": 16, "Radius": group_input.outputs["Handle Radius"]},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": fillet_curve,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Protrude"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_2})

    quadratic_b_zier = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Resolution": 10,
            "Start": combine_xyz_4,
            "Middle": combine_xyz_5,
            "End": combine_xyz_6,
        },
    )

    rounded_quad_029 = nw.new_node(
        nodegroup_rounded_quad_029().name,
        input_kwargs={
            "Width": group_input.outputs["Handle Width"],
            "Height": group_input.outputs["Handle Height"],
        },
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": quadratic_b_zier,
            "Profile Curve": rounded_quad_029,
            "Fill Caps": True,
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_1,
            "False": curve_to_mesh,
            "True": curve_to_mesh_1,
        },
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Length"],
        },
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input.outputs["Handle Protrude"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_7,
        },
    )

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Protrude"],
        },
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Protrude"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: 0.3500},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_2, "Z": multiply_3}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": combine_xyz_8,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    cylinder_2 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Protrude"],
        },
    )

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Protrude"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_3, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_3, "Z": multiply_4}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_2.outputs["Mesh"],
            "Translation": combine_xyz_9,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [transform_geometry, transform_geometry_1, transform_geometry_2]
        },
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": switch, "True": join_geometry},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": switch_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_254",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_254(nw: NodeWrangler):
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


@node_utils.to_nodegroup(
    "nodegroup_door_shelf_008", singleton=False, type="GeometryNodeTree"
)
def nodegroup_door_shelf_008(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Shelf Thickness", 0.0000),
        ],
    )

    cube = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": group_input.outputs["Size"]}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Shelf Thickness"]}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute, "Y": reroute}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: combine_xyz},
        attrs={"operation": "SUBTRACT"},
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": subtract.outputs["Vector"]}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Shelf Thickness"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube_1.outputs["Mesh"], "Translation": combine_xyz_1},
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": cube.outputs["Mesh"], "Mesh 2": transform_geometry},
        attrs={"solver": "EXACT"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": difference.outputs["Mesh"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_drawer_003", singleton=False, type="GeometryNodeTree"
)
def nodegroup_drawer_003(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Thickness", 0.0000),
            ("NodeSocketMaterial", "Drawer Material", None),
        ],
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Size"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": reroute_6})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply, "Y": multiply}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_3, 1: combine_xyz},
        attrs={"operation": "SUBTRACT"},
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": subtract.outputs["Vector"]}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Thickness"]}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube_1.outputs["Mesh"], "Translation": combine_xyz_1},
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": cube.outputs["Mesh"], "Mesh 2": transform_geometry},
        attrs={"solver": "EXACT"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": 0.0200, "Y": separate_xyz.outputs["Y"], "Z": 0.0050},
    )

    cube_2 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_2})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_2.outputs["Mesh"],
            "Translation": (0.0075, 0.0000, -0.0075),
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_2})

    cube_3 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": reroute_4})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_1, cube_3.outputs["Mesh"]]},
    )

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_5, 1: (0.5000, 0.0000, 0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_1.outputs["Vector"], 1: (0.0100, 0.0000, -0.0025)},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Translation": add.outputs["Vector"]},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [difference.outputs["Mesh"], transform_geometry_2]},
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Material"]}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": join_geometry_1, "Material": reroute_1},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": set_material},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_shell_boolean_size_004", singleton=False, type="GeometryNodeTree"
)
def nodegroup_shell_boolean_size_004(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Outer size", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "Thickness", 0.5000),
            ("NodeSocketFloat", "Thickness offset", 0.5000),
        ],
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Thickness"],
            1: group_input.outputs["Thickness offset"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": multiply, "Z": multiply}
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Outer size"], 1: combine_xyz},
        attrs={"operation": "SUBTRACT"},
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Boolean size": subtract_1.outputs["Vector"],
            "Boolean translate": combine_xyz_1,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_shell_boolean_size_005", singleton=False, type="GeometryNodeTree"
)
def nodegroup_shell_boolean_size_005(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Outer size", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "Thickness", 0.5000),
            ("NodeSocketFloat", "Thickness offset", 0.5000),
        ],
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Thickness"],
            1: group_input.outputs["Thickness offset"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": multiply, "Z": multiply}
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Outer size"], 1: combine_xyz},
        attrs={"operation": "SUBTRACT"},
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Boolean size": subtract_1.outputs["Vector"],
            "Boolean translate": combine_xyz_1,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_fridge_shelf_002", singleton=False, type="GeometryNodeTree"
)
def nodegroup_fridge_shelf_002(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Fridge Shelf Thickness", 0.0000),
            ("NodeSocketBool", "Has Grated Shelves", False),
            ("NodeSocketMaterial", "Interior Shelf Material Border", None),
            ("NodeSocketMaterial", "Interior Shelf Material Inside", None),
        ],
    )

    cube = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": group_input.outputs["Size"]}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Fridge Shelf Thickness"],
            "Y": group_input.outputs["Fridge Shelf Thickness"],
        },
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: combine_xyz},
        attrs={"operation": "SUBTRACT"},
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": subtract.outputs["Vector"]}
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": cube.outputs["Mesh"], "Mesh 2": cube_1.outputs["Mesh"]},
        attrs={"solver": "EXACT"},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": difference.outputs["Mesh"],
            "Material": group_input.outputs["Interior Shelf Material Border"],
        },
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Material": group_input.outputs["Interior Shelf Material Inside"],
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, set_material_1]}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Size"]}
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute, 1: (0.0000, -0.5000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute, 1: (0.0000, 0.5000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    mesh_line = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={
            "Count": 24,
            "Start Location": multiply.outputs["Vector"],
            "Offset": multiply_1.outputs["Vector"],
        },
        attrs={"mode": "END_POINTS"},
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0020

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": reroute})

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 8,
            "Radius": value,
            "Depth": separate_xyz.outputs["X"],
        },
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={"Points": mesh_line, "Instance": transform_geometry},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points}
    )

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 8,
            "Radius": value,
            "Depth": separate_xyz.outputs["Y"],
        },
    )

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute, 1: (0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": multiply_2.outputs["Vector"],
            "Rotation": (0.0000, 1.5708, 1.5708),
        },
    )

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute, 1: (-0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": multiply_3.outputs["Vector"],
            "Rotation": (0.0000, 1.5708, 1.5708),
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [realize_instances, transform_geometry_1, transform_geometry_2]
        },
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Material": group_input.outputs["Interior Shelf Material Border"],
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Has Grated Shelves"],
            "False": join_geometry,
            "True": set_material_2,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": switch},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_handle_033", singleton=False, type="GeometryNodeTree"
)
def nodegroup_handle_033(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "Handle Type", 0),
            ("NodeSocketFloat", "Handle Length", 0.0000),
            ("NodeSocketFloat", "Handle Radius", 0.0100),
            ("NodeSocketFloat", "Handle Protrude", 0.0300),
            ("NodeSocketFloat", "Handle Width", 0.0000),
            ("NodeSocketFloat", "Handle Height", 0.0000),
        ],
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Type"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_21},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_5, 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    mesh_line = nw.new_node(Nodes.MeshLine, input_kwargs={"Count": 4})

    index_select_100 = nw.new_node(nodegroup_index_select_100().name)

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Radius"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide, 1: reroute_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": subtract})

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": mesh_line,
            "Selection": index_select_100,
            "Position": combine_xyz,
        },
    )

    index_select_101 = nw.new_node(
        nodegroup_index_select_101().name, input_kwargs={"Index": 1}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Protrude"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_14, "Z": subtract}
    )

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_1})

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position,
            "Selection": index_select_101,
            "Position": reroute_19,
        },
    )

    index_select_102 = nw.new_node(
        nodegroup_index_select_102().name, input_kwargs={"Index": 2}
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: divide_1, 1: reroute_1})

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_14, "Z": add}
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_2})

    set_position_2 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position_1,
            "Selection": index_select_102,
            "Position": reroute_18,
        },
    )

    index_select_103 = nw.new_node(
        nodegroup_index_select_103().name, input_kwargs={"Index": 3}
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add})

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_15})

    set_position_3 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position_2,
            "Selection": index_select_103,
            "Position": combine_xyz_3,
        },
    )

    mesh_to_curve = nw.new_node(
        Nodes.MeshToCurve, input_kwargs={"Mesh": set_position_3}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={"Curve": mesh_to_curve, "Count": 6, "Radius": reroute_12},
        attrs={"mode": "POLY"},
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 16, "Radius": reroute_12}
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": fillet_curve,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Protrude"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_2})

    quadratic_b_zier = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Resolution": 10,
            "Start": combine_xyz_4,
            "Middle": combine_xyz_5,
            "End": combine_xyz_6,
        },
    )

    rounded_quad_025 = nw.new_node(
        nodegroup_rounded_quad_025().name,
        input_kwargs={
            "Width": group_input.outputs["Handle Width"],
            "Height": group_input.outputs["Handle Height"],
        },
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": rounded_quad_025})

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": quadratic_b_zier,
            "Profile Curve": reroute_9,
            "Fill Caps": True,
        },
    )

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": curve_to_mesh_1})

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal_1, "False": curve_to_mesh, "True": reroute_17},
    )

    cylinder_2 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Protrude"],
        },
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cylinder_2.outputs["Mesh"]}
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_3, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: 0.3500},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_3, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_2, "Z": multiply_4}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_8,
            "Translation": combine_xyz_9,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Length"],
        },
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input.outputs["Handle Protrude"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_7,
        },
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Protrude"],
        },
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cylinder_1.outputs["Mesh"]}
    )

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Protrude"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_3, "Z": multiply_3}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_6,
            "Translation": combine_xyz_8,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_1}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_2, reroute_10, reroute_16]},
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry})

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": switch, "True": reroute_20},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": switch_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_108",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_108(nw: NodeWrangler):
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


@node_utils.to_nodegroup(
    "nodegroup_door_shelf_007", singleton=False, type="GeometryNodeTree"
)
def nodegroup_door_shelf_007(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Shelf Thickness", 0.0000),
        ],
    )

    cube = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": group_input.outputs["Size"]}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Shelf Thickness"]}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute, "Y": reroute}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: combine_xyz},
        attrs={"operation": "SUBTRACT"},
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": subtract.outputs["Vector"]}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Shelf Thickness"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube_1.outputs["Mesh"], "Translation": combine_xyz_1},
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": cube.outputs["Mesh"], "Mesh 2": transform_geometry},
        attrs={"solver": "EXACT"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": difference.outputs["Mesh"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_handle_039", singleton=False, type="GeometryNodeTree"
)
def nodegroup_handle_039(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "Handle Type", 0),
            ("NodeSocketFloat", "Handle Length", 0.0000),
            ("NodeSocketFloat", "Handle Radius", 0.0100),
            ("NodeSocketFloat", "Handle Protrude", 0.0300),
            ("NodeSocketFloat", "Handle Width", 0.0000),
            ("NodeSocketFloat", "Handle Height", 0.0000),
        ],
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Handle Type"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Handle Type"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    mesh_line = nw.new_node(Nodes.MeshLine, input_kwargs={"Count": 4})

    index_select_124 = nw.new_node(nodegroup_index_select_124().name)

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide, 1: group_input.outputs["Handle Radius"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": subtract})

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": mesh_line,
            "Selection": index_select_124,
            "Position": combine_xyz,
        },
    )

    index_select_125 = nw.new_node(
        nodegroup_index_select_125().name, input_kwargs={"Index": 1}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": group_input.outputs["Handle Protrude"], "Z": subtract},
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position,
            "Selection": index_select_125,
            "Position": combine_xyz_1,
        },
    )

    index_select_126 = nw.new_node(
        nodegroup_index_select_126().name, input_kwargs={"Index": 2}
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: divide_1, 1: group_input.outputs["Handle Radius"]}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": group_input.outputs["Handle Protrude"], "Z": add},
    )

    set_position_2 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position_1,
            "Selection": index_select_126,
            "Position": combine_xyz_2,
        },
    )

    index_select_127 = nw.new_node(
        nodegroup_index_select_127().name, input_kwargs={"Index": 3}
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add})

    set_position_3 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position_2,
            "Selection": index_select_127,
            "Position": combine_xyz_3,
        },
    )

    mesh_to_curve = nw.new_node(
        Nodes.MeshToCurve, input_kwargs={"Mesh": set_position_3}
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": mesh_to_curve,
            "Count": 6,
            "Radius": group_input.outputs["Handle Radius"],
        },
        attrs={"mode": "POLY"},
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={"Resolution": 16, "Radius": group_input.outputs["Handle Radius"]},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": fillet_curve,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Protrude"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_2})

    quadratic_b_zier = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Resolution": 10,
            "Start": combine_xyz_4,
            "Middle": combine_xyz_5,
            "End": combine_xyz_6,
        },
    )

    rounded_quad_031 = nw.new_node(
        nodegroup_rounded_quad_031().name,
        input_kwargs={
            "Width": group_input.outputs["Handle Width"],
            "Height": group_input.outputs["Handle Height"],
        },
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": quadratic_b_zier,
            "Profile Curve": rounded_quad_031,
            "Fill Caps": True,
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_1,
            "False": curve_to_mesh,
            "True": curve_to_mesh_1,
        },
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Length"],
        },
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input.outputs["Handle Protrude"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_7,
        },
    )

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Protrude"],
        },
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Protrude"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: 0.3500},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_2, "Z": multiply_3}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": combine_xyz_8,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    cylinder_2 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Protrude"],
        },
    )

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Protrude"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_3, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_3, "Z": multiply_4}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_2.outputs["Mesh"],
            "Translation": combine_xyz_9,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [transform_geometry, transform_geometry_1, transform_geometry_2]
        },
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": switch, "True": join_geometry},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": switch_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_handle_038", singleton=False, type="GeometryNodeTree"
)
def nodegroup_handle_038(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "Handle Type", 0),
            ("NodeSocketFloat", "Handle Length", 0.0000),
            ("NodeSocketFloat", "Handle Radius", 0.0100),
            ("NodeSocketFloat", "Handle Protrude", 0.0300),
            ("NodeSocketFloat", "Handle Width", 0.0000),
            ("NodeSocketFloat", "Handle Height", 0.0000),
        ],
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Handle Type"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Handle Type"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    mesh_line = nw.new_node(Nodes.MeshLine, input_kwargs={"Count": 4})

    index_select_120 = nw.new_node(nodegroup_index_select_120().name)

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide, 1: group_input.outputs["Handle Radius"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": subtract})

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": mesh_line,
            "Selection": index_select_120,
            "Position": combine_xyz,
        },
    )

    index_select_121 = nw.new_node(
        nodegroup_index_select_121().name, input_kwargs={"Index": 1}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": group_input.outputs["Handle Protrude"], "Z": subtract},
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position,
            "Selection": index_select_121,
            "Position": combine_xyz_1,
        },
    )

    index_select_122 = nw.new_node(
        nodegroup_index_select_122().name, input_kwargs={"Index": 2}
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: divide_1, 1: group_input.outputs["Handle Radius"]}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": group_input.outputs["Handle Protrude"], "Z": add},
    )

    set_position_2 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position_1,
            "Selection": index_select_122,
            "Position": combine_xyz_2,
        },
    )

    index_select_123 = nw.new_node(
        nodegroup_index_select_123().name, input_kwargs={"Index": 3}
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add})

    set_position_3 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position_2,
            "Selection": index_select_123,
            "Position": combine_xyz_3,
        },
    )

    mesh_to_curve = nw.new_node(
        Nodes.MeshToCurve, input_kwargs={"Mesh": set_position_3}
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": mesh_to_curve,
            "Count": 6,
            "Radius": group_input.outputs["Handle Radius"],
        },
        attrs={"mode": "POLY"},
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={"Resolution": 16, "Radius": group_input.outputs["Handle Radius"]},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": fillet_curve,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Protrude"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_2})

    quadratic_b_zier = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Resolution": 10,
            "Start": combine_xyz_4,
            "Middle": combine_xyz_5,
            "End": combine_xyz_6,
        },
    )

    rounded_quad_030 = nw.new_node(
        nodegroup_rounded_quad_030().name,
        input_kwargs={
            "Width": group_input.outputs["Handle Width"],
            "Height": group_input.outputs["Handle Height"],
        },
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": quadratic_b_zier,
            "Profile Curve": rounded_quad_030,
            "Fill Caps": True,
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_1,
            "False": curve_to_mesh,
            "True": curve_to_mesh_1,
        },
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Length"],
        },
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input.outputs["Handle Protrude"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_7,
        },
    )

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Protrude"],
        },
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Protrude"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: 0.3500},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_2, "Z": multiply_3}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": combine_xyz_8,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    cylinder_2 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Protrude"],
        },
    )

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Protrude"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_3, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_3, "Z": multiply_4}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_2.outputs["Mesh"],
            "Translation": combine_xyz_9,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [transform_geometry, transform_geometry_1, transform_geometry_2]
        },
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": switch, "True": join_geometry},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": switch_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_fridge_body_001", singleton=False, type="GeometryNodeTree"
)
def nodegroup_fridge_body_001(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Border Thickness", 0.0250),
            ("NodeSocketInt", "Num Drawers", 0),
            ("NodeSocketFloat", "Drawer Height", 0.5000),
            ("NodeSocketFloat", "Drawer Depth", 0.1000),
            ("NodeSocketFloat", "Drawer Thickness", 0.0000),
            ("NodeSocketFloat", "Shelf Height", 0.0100),
            ("NodeSocketInt", "Num Shelves", 10),
            ("NodeSocketFloat", "Fridge Shelf Thickness", 0.0000),
            ("NodeSocketBool", "Has Grated Shelves", False),
            ("NodeSocketMaterial", "Outer Shell Material", None),
            ("NodeSocketMaterial", "Inner Shell Material", None),
            ("NodeSocketMaterial", "Interior Shelf Material Border", None),
            ("NodeSocketMaterial", "Interior Shelf Material Inside", None),
            ("NodeSocketMaterial", "Drawer Material", None),
        ],
    )

    reroute_25 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Drawers"]}
    )

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_25})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0050

    shell_boolean_size_006 = nw.new_node(
        nodegroup_shell_boolean_size_006().name,
        input_kwargs={
            "Outer size": group_input.outputs["Size"],
            "Thickness": group_input.outputs["Border Thickness"],
            "Thickness offset": value,
        },
    )

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": value})

    shell_boolean_size_007 = nw.new_node(
        nodegroup_shell_boolean_size_007().name,
        input_kwargs={
            "Outer size": shell_boolean_size_006.outputs["Boolean size"],
            "Thickness": reroute_29,
            "Thickness offset": 0.0000,
        },
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Num Drawers"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    vector = nw.new_node(Nodes.Vector)
    vector.vector = (0.0000, -0.2500, 0.0000)

    vector_1 = nw.new_node(Nodes.Vector)
    vector_1.vector = (0.0000, 0.0000, 0.0000)

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": vector, "True": vector_1},
        attrs={"input_type": "VECTOR"},
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: shell_boolean_size_007.outputs["Boolean size"], 1: switch},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: shell_boolean_size_007.outputs["Boolean size"],
            1: (0.0000, 0.2500, 0.0000),
        },
        attrs={"operation": "MULTIPLY"},
    )

    mesh_line = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={
            "Count": reroute_26,
            "Start Location": multiply.outputs["Vector"],
            "Offset": multiply_1.outputs["Vector"],
        },
        attrs={"mode": "END_POINTS"},
    )

    reroute_38 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": mesh_line})

    reroute_39 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_38})

    reroute_49 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_39})

    cube_1 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={"Size": shell_boolean_size_006.outputs["Boolean size"]},
    )

    reroute_30 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": shell_boolean_size_006.outputs["Boolean translate"]},
    )

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_30})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube_1.outputs["Mesh"], "Translation": reroute_31},
    )

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    cube_2 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={"Size": shell_boolean_size_007.outputs["Boolean size"]},
    )

    add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: shell_boolean_size_007.outputs["Boolean translate"],
            1: reroute_31,
        },
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_2.outputs["Mesh"],
            "Translation": add.outputs["Vector"],
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": reroute_37, "Mesh 2": transform_geometry_1},
        attrs={"solver": "EXACT"},
    )

    reroute_23 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Inner Shell Material"]},
    )

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": difference.outputs["Mesh"], "Material": reroute_24},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Size"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": reroute_5})

    difference_1 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": cube.outputs["Mesh"], "Mesh 2": transform_geometry},
        attrs={"solver": "EXACT"},
    )

    reroute_21 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Outer Shell Material"]},
    )

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": difference_1.outputs["Mesh"], "Material": reroute_22},
    )

    reroute_41 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material_1, reroute_41]}
    )

    reroute_48 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry})

    reroute_27 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Depth"]}
    )

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_27})

    reroute_36 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_28})

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": shell_boolean_size_007.outputs["Boolean size"]},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: reroute_26},
        attrs={"operation": "DIVIDE"},
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Height"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_36, "Y": divide, "Z": reroute_7}
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Thickness"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Material"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    drawer_010 = nw.new_node(
        nodegroup_drawer_010().name,
        input_kwargs={
            "Size": combine_xyz,
            "Thickness": reroute_9,
            "Drawer Material": reroute_11,
        },
    )

    reroute_45 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": drawer_010})

    reroute_52 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_45})

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Shelves"]}
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Border Thickness"],
            1: group_input.outputs["Drawer Height"],
        },
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Shelf Height"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: add_1, 1: reroute_3})

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: add_2, 1: multiply_2})

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add_3})

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.5000

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_3, "Scale": value_1},
        attrs={"operation": "SCALE"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"]},
        attrs={"operation": "MULTIPLY"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Border Thickness"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_32 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_3, 1: reroute_32},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": subtract})

    scale_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_4, "Scale": value_1},
        attrs={"operation": "SCALE"},
    )

    mesh_line_1 = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={
            "Count": reroute_13,
            "Start Location": scale.outputs["Vector"],
            "Offset": scale_1.outputs["Vector"],
        },
        attrs={"mode": "END_POINTS"},
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": shell_boolean_size_007.outputs["Boolean size"]},
    )

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    reroute_34 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_33})

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": reroute_28,
            "Y": separate_xyz_3.outputs["Y"],
            "Z": reroute_34,
        },
    )

    reroute_14 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Fridge Shelf Thickness"]},
    )

    reroute_15 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Has Grated Shelves"]}
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_15})

    reroute_17 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Interior Shelf Material Border"]},
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_17})

    reroute_19 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Interior Shelf Material Inside"]},
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    fridge_shelf_003 = nw.new_node(
        nodegroup_fridge_shelf_003().name,
        input_kwargs={
            "Size": combine_xyz_5,
            "Fridge Shelf Thickness": reroute_14,
            "Has Grated Shelves": reroute_16,
            "Interior Shelf Material Border": reroute_18,
            "Interior Shelf Material Inside": reroute_20,
        },
    )

    reroute_44 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": fridge_shelf_003})

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={"Points": mesh_line_1, "Instance": reroute_44},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points}
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Drawer Depth"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_1, 1: divide_1})

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_4, 1: divide_2},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": subtract_1})

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Drawer Height"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide_3})

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: shell_boolean_size_007.outputs["Boolean size"],
            1: (0.0000, 0.0000, -0.5000),
            2: combine_xyz_2,
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_35 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply_add.outputs["Vector"]}
    )

    add_5 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: combine_xyz_1, 1: reroute_35}
    )

    multiply_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_5.outputs["Vector"], 1: (1.0000, 0.0000, 1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_5 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_4.outputs["Vector"], 1: (1.0000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_46 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply_5.outputs["Vector"]}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": realize_instances, "Translation": reroute_46},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": transform_geometry_2}
    )

    reroute_50 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_1})

    reroute_42 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_5.outputs["Vector"]}
    )

    reroute_43 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_42})

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_43})

    reroute_40 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_36})

    reroute_47 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_40})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Top Freezer Points Duplicate": reroute_49,
            "Top Freezer Drawer 1 Body": reroute_48,
            "Top Freezer Drawer 1": reroute_52,
            "Top Freezer Body": reroute_50,
            "Top Freezer 1 Position": reroute_51,
            "Top Freezer 1 Max Sliding": reroute_47,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_fridge_door_003", singleton=False, type="GeometryNodeTree"
)
def nodegroup_fridge_door_003(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Border Thickness", 0.0000),
            ("NodeSocketBool", "Has Handles", False),
            ("NodeSocketInt", "Handle Type", 0),
            ("NodeSocketInt", "Handle Location", 0),
            ("NodeSocketInt", "Handle Orientation", 0),
            ("NodeSocketFloat", "Handle Length", 0.0000),
            ("NodeSocketFloat", "Handle Y Offset", 0.0000),
            ("NodeSocketFloat", "Handle Z Offset", 0.0000),
            ("NodeSocketFloat", "Handle Radius", 0.0000),
            ("NodeSocketFloat", "Handle Protrude", 0.0300),
            ("NodeSocketFloat", "Handle Width", 0.0500),
            ("NodeSocketFloat", "Handle Height", 0.0200),
            ("NodeSocketInt", "Num Door Shelves", 2),
            ("NodeSocketFloat", "Door Shelf Thickness", 0.0300),
            ("NodeSocketFloat", "Door Shelf Height", 0.1000),
            ("NodeSocketFloat", "Door Shelf Length", 0.2000),
            ("NodeSocketMaterial", "Handle Material", None),
            ("NodeSocketMaterial", "Door Material", None),
            ("NodeSocketMaterial", "Inner Material", None),
            ("NodeSocketMaterial", "Door Shelf Material", None),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Door Shelves"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_2},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Num Door Shelves"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0050

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": value})

    vector = nw.new_node(Nodes.Vector)
    vector.vector = (1.0000, 1.0000, 0.9900)

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: vector},
        attrs={"operation": "MULTIPLY"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": multiply.outputs["Vector"]}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Border Thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_3, "Y": subtract, "Z": subtract_1}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz})

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_5, 1: (0.0000, 0.0000, -0.3500)},
        attrs={"operation": "MULTIPLY"},
    )

    vector_1 = nw.new_node(Nodes.Vector)
    vector_1.vector = (0.0000, 0.0000, 0.0000)

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_1,
            "False": multiply_2.outputs["Vector"],
            "True": vector_1,
        },
        attrs={"input_type": "VECTOR"},
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Num Door Shelves"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_5, 1: (0.0000, 0.0000, 0.3500)},
        attrs={"operation": "MULTIPLY"},
    )

    vector_2 = nw.new_node(Nodes.Vector)
    vector_2.vector = (0.0000, 0.0000, 0.0000)

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_2,
            "False": multiply_3.outputs["Vector"],
            "True": vector_2,
        },
        attrs={"input_type": "VECTOR"},
    )

    mesh_line = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={"Count": reroute_1, "Start Location": switch, "Offset": switch_1},
        attrs={"mode": "END_POINTS"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": combine_xyz}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Door Shelf Length"],
            "Y": separate_xyz_1.outputs["Y"],
            "Z": group_input.outputs["Door Shelf Height"],
        },
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Shelf Thickness"]},
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    door_shelf_008 = nw.new_node(
        nodegroup_door_shelf_008().name,
        input_kwargs={"Size": combine_xyz_1, "Shelf Thickness": reroute_7},
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Shelf Material"]},
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": door_shelf_008, "Material": reroute_9},
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={"Points": mesh_line, "Instance": set_material},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points}
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_10, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Door Shelf Length"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_4, 1: multiply_5})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": add})

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_2})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": realize_instances, "Translation": reroute_11},
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: value, 1: -2.0000}, attrs={"operation": "DIVIDE"}
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide})

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_3})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Translation": reroute_12},
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Inner Material"]}
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_1, "Material": reroute_14},
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_1})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_geometry, reroute_15]}
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_15})

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": join_geometry, "True": reroute_17},
    )

    reroute_18 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Has Handles"]}
    )

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    reroute_20 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Protrude"]}
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    handle_037 = nw.new_node(
        nodegroup_handle_037().name,
        input_kwargs={
            "Handle Type": group_input.outputs["Handle Type"],
            "Handle Length": group_input.outputs["Handle Length"],
            "Handle Radius": group_input.outputs["Handle Radius"],
            "Handle Protrude": reroute_21,
            "Handle Width": group_input.outputs["Handle Width"],
            "Handle Height": group_input.outputs["Handle Height"],
        },
    )

    add_jointed_geometry_metadata_254 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_254().name,
        input_kwargs={"Geometry": handle_037, "Label": "handle"},
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_19, "True": add_jointed_geometry_metadata_254},
    )

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_3})

    reroute_23 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Location"]}
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_23},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    multiply_6 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: vector},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_24 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply_6.outputs["Vector"]}
    )

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_24})

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_25})

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_26})

    multiply_7 = nw.new_node(
        Nodes.Math, input_kwargs={0: value, 1: -1.0000}, attrs={"operation": "MULTIPLY"}
    )

    reroute_28 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Y Offset"]}
    )

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_28})

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_7, "Y": reroute_29}
    )

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_27, 1: (1.0000, -0.5000, 0.0000), 2: combine_xyz_4},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    multiply_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Y Offset"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_7, "Y": multiply_8}
    )

    multiply_add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_27, 1: (1.0000, 0.5000, 0.0000), 2: combine_xyz_5},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_3,
            "False": multiply_add.outputs["Vector"],
            "True": multiply_add_1.outputs["Vector"],
        },
        attrs={"input_type": "VECTOR"},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": reroute_22, "Translation": switch_4}
    )

    reroute_30 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Z Offset"]}
    )

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_30})

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_31})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_2, "Translation": combine_xyz_6},
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_3,
            "Material": group_input.outputs["Handle Material"],
        },
    )

    reroute_32 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_2})

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": value})

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": reroute_33})

    subtract_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_25, 1: combine_xyz_7},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_34 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": subtract_2.outputs["Vector"]}
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": reroute_34})

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_2.outputs["Vector"]}
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide_1})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube_1.outputs["Mesh"], "Translation": combine_xyz_8},
    )

    reroute_35 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Door Material"]}
    )

    reroute_36 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_35})

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_4, "Material": reroute_36},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [switch_2, reroute_32, set_material_3]},
    )

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_27})

    reroute_38 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_37})

    equal_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Handle Location"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    vector_3 = nw.new_node(Nodes.Vector)
    vector_3.vector = (0.0000, -0.5000, 0.0000)

    vector_4 = nw.new_node(Nodes.Vector)
    vector_4.vector = (0.0000, 0.5000, 0.0000)

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal_4, "False": vector_3, "True": vector_4},
        attrs={"input_type": "VECTOR"},
    )

    multiply_9 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_38, 1: switch_5},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Translation": multiply_9.outputs["Vector"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": transform_geometry_5},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_fridge_body_002", singleton=False, type="GeometryNodeTree"
)
def nodegroup_fridge_body_002(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Border Thickness", 0.0250),
            ("NodeSocketInt", "Num Drawers", 0),
            ("NodeSocketFloat", "Drawer Height", 0.5000),
            ("NodeSocketFloat", "Drawer Depth", 0.1000),
            ("NodeSocketFloat", "Drawer Thickness", 0.0000),
            ("NodeSocketFloat", "Shelf Height", 0.0100),
            ("NodeSocketInt", "Num Shelves", 10),
            ("NodeSocketFloat", "Fridge Shelf Thickness", 0.0000),
            ("NodeSocketBool", "Has Grated Shelves", False),
            ("NodeSocketMaterial", "Outer Shell Material", None),
            ("NodeSocketMaterial", "Inner Shell Material", None),
            ("NodeSocketMaterial", "Interior Shelf Material Border", None),
            ("NodeSocketMaterial", "Interior Shelf Material Inside", None),
            ("NodeSocketMaterial", "Drawer Material", None),
        ],
    )

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Shelves"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_34 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_34})

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Border Thickness"],
            1: group_input.outputs["Drawer Height"],
        },
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Shelf Height"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: reroute_3})

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: add_1, 1: multiply})

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add_2})

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.5000

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_3, "Scale": value_1},
        attrs={"operation": "SCALE"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"]},
        attrs={"operation": "MULTIPLY"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Border Thickness"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_32 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: reroute_32},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": subtract})

    scale_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_4, "Scale": value_1},
        attrs={"operation": "SCALE"},
    )

    mesh_line_1 = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={
            "Count": reroute_35,
            "Start Location": scale.outputs["Vector"],
            "Offset": scale_1.outputs["Vector"],
        },
        attrs={"mode": "END_POINTS"},
    )

    add_jointed_geometry_metadata_106 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_106().name,
        input_kwargs={"Geometry": mesh_line_1, "Label": "shelves"},
    )

    reroute_26 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Depth"]}
    )

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_26})

    reroute_45 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_27})

    reroute_46 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_45})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0050

    shell_boolean_size_004 = nw.new_node(
        nodegroup_shell_boolean_size_004().name,
        input_kwargs={
            "Outer size": group_input.outputs["Size"],
            "Thickness": group_input.outputs["Border Thickness"],
            "Thickness offset": value,
        },
    )

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": value})

    shell_boolean_size_005 = nw.new_node(
        nodegroup_shell_boolean_size_005().name,
        input_kwargs={
            "Outer size": shell_boolean_size_004.outputs["Boolean size"],
            "Thickness": reroute_28,
            "Thickness offset": 0.0000,
        },
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": shell_boolean_size_005.outputs["Boolean size"]},
    )

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": reroute_46,
            "Y": separate_xyz_3.outputs["Y"],
            "Z": reroute_33,
        },
    )

    reroute_12 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Fridge Shelf Thickness"]},
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    reroute_36 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_36})

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Has Grated Shelves"]}
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

    reroute_16 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Interior Shelf Material Border"]},
    )

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    reroute_18 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Interior Shelf Material Inside"]},
    )

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    reroute_38 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    reroute_39 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_38})

    fridge_shelf_002 = nw.new_node(
        nodegroup_fridge_shelf_002().name,
        input_kwargs={
            "Size": combine_xyz_5,
            "Fridge Shelf Thickness": reroute_37,
            "Has Grated Shelves": reroute_15,
            "Interior Shelf Material Border": reroute_17,
            "Interior Shelf Material Inside": reroute_39,
        },
    )

    reroute_59 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": fridge_shelf_002})

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": add_jointed_geometry_metadata_106,
            "Instance": reroute_59,
        },
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Drawer Depth"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_1, 1: divide})

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_3, 1: divide_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": subtract_1})

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Drawer Height"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide_2})

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: shell_boolean_size_005.outputs["Boolean size"],
            1: (0.0000, 0.0000, -0.5000),
            2: combine_xyz_2,
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_48 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply_add.outputs["Vector"]}
    )

    add_4 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: combine_xyz_1, 1: reroute_48}
    )

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_4.outputs["Vector"], 1: (1.0000, 0.0000, 1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_61 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply_2.outputs["Vector"]}
    )

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_61, 1: (1.0000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": realize_instances,
            "Translation": multiply_3.outputs["Vector"],
        },
    )

    reroute_24 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Drawers"]}
    )

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_24})

    reroute_43 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_25})

    reroute_44 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_43})

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_44})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_51},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    add_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Drawer Height"],
            1: group_input.outputs["Shelf Height"],
        },
    )

    divide_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_5, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide_3})

    reroute_40 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_6})

    add_6 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: multiply_2.outputs["Vector"], 1: reroute_40}
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_59, "Translation": add_6.outputs["Vector"]},
    )

    switch_1 = nw.new_node(
        Nodes.Switch, input_kwargs={"Switch": equal, "False": transform_geometry_4}
    )

    reroute_63 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_1})

    reroute_64 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_63})

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_3, reroute_64]},
    )

    reroute_54 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_1})

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Size"]}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": reroute_4})

    reroute_31 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": shell_boolean_size_004.outputs["Boolean size"]},
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": reroute_31})

    reroute_29 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": shell_boolean_size_004.outputs["Boolean translate"]},
    )

    reroute_30 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_29})

    reroute_47 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_30})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube_1.outputs["Mesh"], "Translation": reroute_47},
    )

    add_7 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: shell_boolean_size_004.outputs["Boolean size"],
            1: (0.1000, 0.0000, 0.0000),
        },
    )

    cube_2 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": add_7.outputs["Vector"]})

    add_8 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: reroute_30, 1: (0.0500, 0.0000, 0.0000)}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_2.outputs["Mesh"],
            "Translation": add_8.outputs["Vector"],
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 1": cube.outputs["Mesh"],
            "Mesh 2": [transform_geometry, transform_geometry_1],
        },
        attrs={"solver": "EXACT"},
    )

    reroute_55 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": difference.outputs["Mesh"]}
    )

    reroute_20 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Outer Shell Material"]},
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    set_material = nw.new_node(
        Nodes.SetMaterial, input_kwargs={"Geometry": reroute_55, "Material": reroute_21}
    )

    cube_3 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={"Size": shell_boolean_size_005.outputs["Boolean size"]},
    )

    add_9 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: shell_boolean_size_005.outputs["Boolean translate"],
            1: reroute_30,
        },
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_3.outputs["Mesh"],
            "Translation": add_9.outputs["Vector"],
        },
    )

    difference_1 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": transform_geometry, "Mesh 2": transform_geometry_2},
        attrs={"solver": "EXACT"},
    )

    reroute_22 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Inner Shell Material"]},
    )

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": difference_1.outputs["Mesh"], "Material": reroute_23},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, set_material_1]}
    )

    reroute_60 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry})

    reroute_52 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_46})

    reroute_53 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_52})

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": shell_boolean_size_005.outputs["Boolean size"]},
    )

    divide_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: reroute_44},
        attrs={"operation": "DIVIDE"},
    )

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Height"]}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_53, "Y": divide_4, "Z": reroute_5}
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Thickness"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Material"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    reroute_41 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    reroute_42 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_41})

    reroute_49 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_42})

    reroute_50 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_49})

    reroute_58 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_50})

    drawer_003 = nw.new_node(
        nodegroup_drawer_003().name,
        input_kwargs={
            "Size": combine_xyz,
            "Thickness": reroute_7,
            "Drawer Material": reroute_58,
        },
    )

    reroute_57 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": drawer_003})

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Num Drawers"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    vector = nw.new_node(Nodes.Vector)
    vector.vector = (0.0000, -0.2500, 0.0000)

    vector_1 = nw.new_node(Nodes.Vector)
    vector_1.vector = (0.0000, 0.0000, 0.0000)

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal_1, "False": vector, "True": vector_1},
        attrs={"input_type": "VECTOR"},
    )

    multiply_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: shell_boolean_size_005.outputs["Boolean size"], 1: switch},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_5 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: shell_boolean_size_005.outputs["Boolean size"],
            1: (0.0000, 0.2500, 0.0000),
        },
        attrs={"operation": "MULTIPLY"},
    )

    mesh_line = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={
            "Count": reroute_44,
            "Start Location": multiply_4.outputs["Vector"],
            "Offset": multiply_5.outputs["Vector"],
        },
        attrs={"mode": "END_POINTS"},
    )

    reroute_62 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": mesh_line})

    reroute_56 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_4.outputs["Vector"]}
    )

    reroute_65 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_56})

    reroute_66 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_53})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Fridge Body": reroute_54,
            "Fridge Body Drawer Body 2": reroute_60,
            "Fridge Body Drawer": reroute_57,
            "Fridge Body Duplicate Points": reroute_62,
            "Fridge Body Drawer Position": reroute_65,
            "Fridge Body Drawer Max": reroute_66,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_fridge_door_002", singleton=False, type="GeometryNodeTree"
)
def nodegroup_fridge_door_002(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Border Thickness", 0.0000),
            ("NodeSocketBool", "Has Handles", False),
            ("NodeSocketInt", "Handle Type", 0),
            ("NodeSocketInt", "Handle Location", 0),
            ("NodeSocketInt", "Handle Orientation", 0),
            ("NodeSocketFloat", "Handle Length", 0.0000),
            ("NodeSocketFloat", "Handle Y Offset", 0.0000),
            ("NodeSocketFloat", "Handle Z Offset", 0.0000),
            ("NodeSocketFloat", "Handle Radius", 0.0000),
            ("NodeSocketFloat", "Handle Protrude", 0.0300),
            ("NodeSocketFloat", "Handle Width", 0.0500),
            ("NodeSocketFloat", "Handle Height", 0.0200),
            ("NodeSocketInt", "Num Door Shelves", 2),
            ("NodeSocketFloat", "Door Shelf Thickness", 0.0300),
            ("NodeSocketFloat", "Door Shelf Height", 0.1000),
            ("NodeSocketFloat", "Door Shelf Length", 0.2000),
            ("NodeSocketMaterial", "Handle Material", None),
            ("NodeSocketMaterial", "Door Material", None),
            ("NodeSocketMaterial", "Inner Material", None),
            ("NodeSocketMaterial", "Door Shelf Material", None),
        ],
    )

    reroute_20 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Door Shelves"]}
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    reroute_32 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_32})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_33},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_21, 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0050

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": value})

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    vector = nw.new_node(Nodes.Vector)
    vector.vector = (1.0000, 1.0000, 0.9900)

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: vector},
        attrs={"operation": "MULTIPLY"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": multiply.outputs["Vector"]}
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Border Thickness"]}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_27, "Y": subtract, "Z": subtract_1}
    )

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: (0.0000, 0.0000, -0.3500)},
        attrs={"operation": "MULTIPLY"},
    )

    vector_1 = nw.new_node(Nodes.Vector)
    vector_1.vector = (0.0000, 0.0000, 0.0000)

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_1,
            "False": multiply_2.outputs["Vector"],
            "True": vector_1,
        },
        attrs={"input_type": "VECTOR"},
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_21, 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: (0.0000, 0.0000, 0.3500)},
        attrs={"operation": "MULTIPLY"},
    )

    vector_2 = nw.new_node(Nodes.Vector)
    vector_2.vector = (0.0000, 0.0000, 0.0000)

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_2,
            "False": multiply_3.outputs["Vector"],
            "True": vector_2,
        },
        attrs={"input_type": "VECTOR"},
    )

    mesh_line = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={
            "Count": reroute_32,
            "Start Location": switch,
            "Offset": switch_1,
        },
        attrs={"mode": "END_POINTS"},
    )

    reroute_38 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": mesh_line})

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Door Shelf Length"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": combine_xyz}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Door Shelf Height"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": reroute_5, "Y": separate_xyz_1.outputs["Y"], "Z": reroute_3},
    )

    reroute_14 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Shelf Thickness"]},
    )

    door_shelf_007 = nw.new_node(
        nodegroup_door_shelf_007().name,
        input_kwargs={"Size": combine_xyz_1, "Shelf Thickness": reroute_14},
    )

    reroute_19 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Shelf Material"]},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": door_shelf_007, "Material": reroute_19},
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={"Points": reroute_38, "Instance": set_material},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points}
    )

    multiply_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: value, 1: -1.0000}, attrs={"operation": "MULTIPLY"}
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Door Shelf Length"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_4, 1: multiply_5})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": add})

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_2})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": realize_instances, "Translation": reroute_28},
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_27, 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Translation": combine_xyz_3},
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Inner Material"]}
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_17})

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_25})

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_1, "Material": reroute_26},
    )

    reroute_36 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_1})

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_36})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_geometry, reroute_37]}
    )

    reroute_39 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_37})

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": join_geometry, "True": reroute_39},
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Has Handles"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    handle_033 = nw.new_node(
        nodegroup_handle_033().name,
        input_kwargs={
            "Handle Type": group_input.outputs["Handle Type"],
            "Handle Length": group_input.outputs["Handle Length"],
            "Handle Radius": group_input.outputs["Handle Radius"],
            "Handle Protrude": group_input.outputs["Handle Protrude"],
            "Handle Width": group_input.outputs["Handle Width"],
            "Handle Height": group_input.outputs["Handle Height"],
        },
    )

    add_jointed_geometry_metadata_108 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_108().name,
        input_kwargs={"Geometry": handle_033, "Label": "handle"},
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_9, "True": add_jointed_geometry_metadata_108},
    )

    reroute_30 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_3})

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Location"]}
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_10},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Size"]}
    )

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": vector})

    multiply_6 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute, 1: reroute_24},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_7 = nw.new_node(
        Nodes.Math, input_kwargs={0: value, 1: -1.0000}, attrs={"operation": "MULTIPLY"}
    )

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Y Offset"]}
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_7, "Y": reroute_12}
    )

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: multiply_6.outputs["Vector"],
            1: (1.0000, -0.5000, 0.0000),
            2: combine_xyz_4,
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    multiply_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Y Offset"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_7, "Y": multiply_8}
    )

    multiply_add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: multiply_6.outputs["Vector"],
            1: (1.0000, 0.5000, 0.0000),
            2: combine_xyz_5,
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_3,
            "False": multiply_add.outputs["Vector"],
            "True": multiply_add_1.outputs["Vector"],
        },
        attrs={"input_type": "VECTOR"},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": reroute_30, "Translation": switch_4}
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Z Offset"]}
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_13})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_2, "Translation": combine_xyz_6},
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Material"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_3, "Material": reroute_7},
    )

    reroute_34 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_2})

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": reroute_23})

    subtract_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_6.outputs["Vector"], 1: combine_xyz_7},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_31 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": subtract_2.outputs["Vector"]}
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": reroute_31})

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_2.outputs["Vector"]}
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide_1})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube_1.outputs["Mesh"], "Translation": combine_xyz_8},
    )

    reroute_15 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Door Material"]}
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_15})

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_4, "Material": reroute_16},
    )

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_3})

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [switch_2, reroute_34, reroute_35]},
    )

    equal_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Handle Location"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    vector_3 = nw.new_node(Nodes.Vector)
    vector_3.vector = (0.0000, -0.5000, 0.0000)

    vector_4 = nw.new_node(Nodes.Vector)
    vector_4.vector = (0.0000, 0.5000, 0.0000)

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal_4, "False": vector_3, "True": vector_4},
        attrs={"input_type": "VECTOR"},
    )

    multiply_9 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_6.outputs["Vector"], 1: switch_5},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_29 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply_9.outputs["Vector"]}
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_1, "Translation": reroute_29},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": transform_geometry_5},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_bottom_freezer_003", singleton=False, type="GeometryNodeTree"
)
def nodegroup_bottom_freezer_003(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Main Fridge Dimensions", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketBool", "Has Handle", False),
            ("NodeSocketFloat", "Exterior Drawer Height", 0.0000),
            ("NodeSocketFloat", "Fridge Door Thickness", 0.0000),
            ("NodeSocketInt", "Handle Type", 0),
            ("NodeSocketFloat", "Handle Radius", 0.0100),
            ("NodeSocketFloat", "Handle Protrude", 0.0300),
            ("NodeSocketFloat", "Handle Width", 0.0000),
            ("NodeSocketFloat", "Handle Height", 0.0000),
            ("NodeSocketMaterial", "Exterior Drawer Shell Material", None),
            ("NodeSocketMaterial", "Exterior Drawer Inside Material", None),
            ("NodeSocketMaterial", "Exterior Drawer Material", None),
            ("NodeSocketMaterial", "Exterior Drawer Handle Material", None),
            ("NodeSocketMaterial", "Exterior Drawer Door Material", None),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Main Fridge Dimensions"]},
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"Z": group_input.outputs["Exterior Drawer Height"]},
    )

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_1, 1: (1.0000, 1.0000, 0.0000), 2: combine_xyz},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_28 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply_add.outputs["Vector"]}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": reroute_28})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0500

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": value})

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_add.outputs["Vector"], 1: combine_xyz_1},
        attrs={"operation": "SUBTRACT"},
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": subtract.outputs["Vector"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Translation": (0.0250, 0.0000, 0.0000),
            "Scale": (1.0000, 0.9000, 0.9000),
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": cube.outputs["Mesh"], "Mesh 2": transform_geometry},
        attrs={"solver": "EXACT"},
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Exterior Drawer Shell Material"]},
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": difference.outputs["Mesh"], "Material": reroute_14},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": set_material}
    )

    reroute_38 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_1}
    )

    reroute_33 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cube_1.outputs["Mesh"]}
    )

    reroute_34 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_33})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Translation": (0.0000, 0.0000, 0.0500),
            "Scale": (0.9000, 0.9000, 1.0000),
        },
    )

    difference_1 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": reroute_34, "Mesh 2": transform_geometry_4},
        attrs={"solver": "EXACT"},
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": difference_1.outputs["Mesh"],
            "Scale": (0.9900, 0.9900, 0.9900),
        },
    )

    reroute_15 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Exterior Drawer Inside Material"]},
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_15})

    set_material_4 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_5, "Material": reroute_16},
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Exterior Drawer Material"]},
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_17})

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": difference_1.outputs["Mesh"], "Material": reroute_18},
    )

    reroute_36 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_3})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material_4, reroute_36]}
    )

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": value})

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_25})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_26, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide})

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry,
            "Translation": combine_xyz_5,
            "Scale": (1.0000, 0.9000, 0.9000),
        },
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Fridge Door Thickness"],
            "Z": group_input.outputs["Exterior Drawer Height"],
        },
    )

    multiply_add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_1, 1: (0.0000, 1.0000, 0.0000), 2: combine_xyz_2},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    cube_2 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": multiply_add_1.outputs["Vector"]}
    )

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Fridge Door Thickness"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide_1})

    multiply_add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_27, 1: (0.5000, 0.0000, 0.0000), 2: combine_xyz_3},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_2.outputs["Mesh"],
            "Translation": multiply_add_2.outputs["Vector"],
        },
    )

    reroute_21 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Exterior Drawer Door Material"]},
    )

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_2, "Material": reroute_22},
    )

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_1})

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Has Handle"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Type"]}
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": group_input.outputs["Main Fridge Dimensions"]},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: 0.8000},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Radius"]}
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Protrude"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Width"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Height"]}
    )

    handle_039 = nw.new_node(
        nodegroup_handle_039().name,
        input_kwargs={
            "Handle Type": reroute_6,
            "Handle Length": multiply,
            "Handle Radius": reroute_7,
            "Handle Protrude": reroute_9,
            "Handle Width": reroute_11,
            "Handle Height": reroute_12,
        },
    )

    reroute_32 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": handle_039})

    reroute_4 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Fridge Door Thickness"]},
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: reroute_5, 1: divide_2})

    reroute_23 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Exterior Drawer Height"]},
    )

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_24, 1: 0.4000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Z": multiply_1}
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_32,
            "Translation": combine_xyz_4,
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    reroute_19 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Exterior Drawer Handle Material"]},
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_3, "Material": reroute_20},
    )

    switch = nw.new_node(
        Nodes.Switch, input_kwargs={"Switch": reroute_3, "True": set_material_2}
    )

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch})

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_6, reroute_35, reroute_37]},
    )

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_24})

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_29, 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide_3})

    reroute_30 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": subtract.outputs["Vector"]}
    )

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_30})

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_31, 1: (1.0000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": multiply_2.outputs["Vector"]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Bottom Freezer Body": reroute_38,
            "Bottom Freezer Drawer": join_geometry_1,
            "Bottom Freezer Translation": combine_xyz_6,
            "Sliding Joint Max": separate_xyz_1.outputs["X"],
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_bottom_freezer_002", singleton=False, type="GeometryNodeTree"
)
def nodegroup_bottom_freezer_002(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Main Fridge Dimensions", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketBool", "Has Handle", False),
            ("NodeSocketFloat", "Exterior Drawer Height", 0.0000),
            ("NodeSocketFloat", "Fridge Door Thickness", 0.0000),
            ("NodeSocketInt", "Handle Type", 0),
            ("NodeSocketFloat", "Handle Radius", 0.0100),
            ("NodeSocketFloat", "Handle Protrude", 0.0300),
            ("NodeSocketFloat", "Handle Width", 0.0000),
            ("NodeSocketFloat", "Handle Height", 0.0000),
            ("NodeSocketMaterial", "Exterior Drawer Shell Material", None),
            ("NodeSocketMaterial", "Exterior Drawer Inside Material", None),
            ("NodeSocketMaterial", "Exterior Drawer Material", None),
            ("NodeSocketMaterial", "Exterior Drawer Handle Material", None),
            ("NodeSocketMaterial", "Exterior Drawer Door Material", None),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Main Fridge Dimensions"]},
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"Z": group_input.outputs["Exterior Drawer Height"]},
    )

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_1, 1: (1.0000, 1.0000, 0.0000), 2: combine_xyz},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_28 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply_add.outputs["Vector"]}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": reroute_28})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0500

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": value})

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_add.outputs["Vector"], 1: combine_xyz_1},
        attrs={"operation": "SUBTRACT"},
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": subtract.outputs["Vector"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Translation": (0.0250, 0.0000, 0.0000),
            "Scale": (1.0000, 0.9000, 0.9000),
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": cube.outputs["Mesh"], "Mesh 2": transform_geometry},
        attrs={"solver": "EXACT"},
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Exterior Drawer Shell Material"]},
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": difference.outputs["Mesh"], "Material": reroute_14},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": set_material}
    )

    reroute_38 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_1}
    )

    reroute_33 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cube_1.outputs["Mesh"]}
    )

    reroute_34 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_33})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Translation": (0.0000, 0.0000, 0.0500),
            "Scale": (0.9000, 0.9000, 1.0000),
        },
    )

    difference_1 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": reroute_34, "Mesh 2": transform_geometry_4},
        attrs={"solver": "EXACT"},
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": difference_1.outputs["Mesh"],
            "Scale": (0.9900, 0.9900, 0.9900),
        },
    )

    reroute_15 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Exterior Drawer Inside Material"]},
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_15})

    set_material_4 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_5, "Material": reroute_16},
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Exterior Drawer Material"]},
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_17})

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": difference_1.outputs["Mesh"], "Material": reroute_18},
    )

    reroute_36 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_3})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material_4, reroute_36]}
    )

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": value})

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_25})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_26, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide})

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry,
            "Translation": combine_xyz_5,
            "Scale": (1.0000, 0.9000, 0.9000),
        },
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Fridge Door Thickness"],
            "Z": group_input.outputs["Exterior Drawer Height"],
        },
    )

    multiply_add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_1, 1: (0.0000, 1.0000, 0.0000), 2: combine_xyz_2},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    cube_2 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": multiply_add_1.outputs["Vector"]}
    )

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Fridge Door Thickness"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide_1})

    multiply_add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_27, 1: (0.5000, 0.0000, 0.0000), 2: combine_xyz_3},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_2.outputs["Mesh"],
            "Translation": multiply_add_2.outputs["Vector"],
        },
    )

    reroute_21 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Exterior Drawer Door Material"]},
    )

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_2, "Material": reroute_22},
    )

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_1})

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Has Handle"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Type"]}
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": group_input.outputs["Main Fridge Dimensions"]},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: 0.8000},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Radius"]}
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Protrude"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Width"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Height"]}
    )

    handle_038 = nw.new_node(
        nodegroup_handle_038().name,
        input_kwargs={
            "Handle Type": reroute_6,
            "Handle Length": multiply,
            "Handle Radius": reroute_7,
            "Handle Protrude": reroute_9,
            "Handle Width": reroute_11,
            "Handle Height": reroute_12,
        },
    )

    reroute_32 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": handle_038})

    reroute_4 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Fridge Door Thickness"]},
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: reroute_5, 1: divide_2})

    reroute_23 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Exterior Drawer Height"]},
    )

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_24, 1: 0.4000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Z": multiply_1}
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_32,
            "Translation": combine_xyz_4,
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    reroute_19 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Exterior Drawer Handle Material"]},
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_3, "Material": reroute_20},
    )

    switch = nw.new_node(
        Nodes.Switch, input_kwargs={"Switch": reroute_3, "True": set_material_2}
    )

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch})

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_6, reroute_35, reroute_37]},
    )

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_24})

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_29, 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide_3})

    reroute_30 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": subtract.outputs["Vector"]}
    )

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_30})

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_31, 1: (1.0000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": multiply_2.outputs["Vector"]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Bottom Freezer 2 Body": reroute_38,
            "Bottom Freezer 2 Drawer": join_geometry_1,
            "Bottom Freezer 2 Translation": combine_xyz_6,
            "Bottom Freezer 2 Max Sliding": separate_xyz_1.outputs["X"],
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_464",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_464(nw: NodeWrangler):
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


@node_utils.to_nodegroup(
    "nodegroup_duplicate_joints_on_parent_034", singleton=False, type="GeometryNodeTree"
)
def nodegroup_duplicate_joints_on_parent_034(nw: NodeWrangler):
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
    "nodegroup_top_freezer_001", singleton=False, type="GeometryNodeTree"
)
def nodegroup_top_freezer_001(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Dimensions", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketInt", "Num Doors", 0),
            ("NodeSocketInt", "Handle Location", 0),
            ("NodeSocketFloat", "Border Thickness", 0.0000),
            ("NodeSocketInt", "Num Drawers", 0),
            ("NodeSocketFloat", "Drawer Height", 0.0000),
            ("NodeSocketFloat", "Drawer Depth", 0.0000),
            ("NodeSocketFloat", "Drawer Thickness", 0.0000),
            ("NodeSocketFloat", "Shelf Height", 0.0000),
            ("NodeSocketInt", "Num Shelves", 0),
            ("NodeSocketFloat", "Fridge Door Thickness", 0.0000),
            ("NodeSocketBool", "Has Handles", False),
            ("NodeSocketInt", "Handle Type", 0),
            ("NodeSocketInt", "Handle Orientation", 0),
            ("NodeSocketFloat", "Handle Length", 0.0000),
            ("NodeSocketFloat", "Handle Y Offset", 0.0000),
            ("NodeSocketFloat", "Handle Z Offset", 0.0000),
            ("NodeSocketFloat", "Handle Protrude", 0.0000),
            ("NodeSocketFloat", "Handle Radius", 0.0000),
            ("NodeSocketFloat", "Handle Width", 0.0000),
            ("NodeSocketFloat", "Handle Height", 0.0000),
            ("NodeSocketInt", "Num Door Shelves", 0),
            ("NodeSocketFloat", "Door Shelf Thickness", 0.0000),
            ("NodeSocketFloat", "Door Shelf Height", 0.1000),
            ("NodeSocketFloat", "Door Shelf Length", 0.2000),
            ("NodeSocketFloat", "Fridge Shelf Thickness", 0.0000),
            ("NodeSocketBool", "Has Grated Shelves", False),
            ("NodeSocketMaterial", "Outer Shell Material", None),
            ("NodeSocketMaterial", "Inner Shell Material", None),
            ("NodeSocketMaterial", "Handle Material", None),
            ("NodeSocketMaterial", "Door Material", None),
            ("NodeSocketMaterial", "Door Inner Material", None),
            ("NodeSocketMaterial", "Door Shelf Material", None),
            ("NodeSocketMaterial", "Interior Shelf Material Border", None),
            ("NodeSocketMaterial", "Interior Shelf Material Inside", None),
            ("NodeSocketMaterial", "Drawer Material", None),
        ],
    )

    reroute_28 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Fridge Door Thickness"]},
    )

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_28})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 1.0000, "Y": 1.0000, "Z": 1.0000}
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Dimensions"], 1: combine_xyz},
        attrs={"operation": "MULTIPLY"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": multiply.outputs["Vector"]}
    )

    reroute_65 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Doors"]}
    )

    reroute_66 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_65})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: reroute_66},
        attrs={"operation": "DIVIDE"},
    )

    reroute_70 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["Z"]}
    )

    reroute_71 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_70})

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_29, "Y": divide, "Z": reroute_71}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Border Thickness"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_69 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    reroute_30 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Has Handles"]}
    )

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_30})

    reroute_32 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Type"]}
    )

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_32})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Num Doors"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    integer = nw.new_node(Nodes.Integer)
    integer.integer = 0

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Location"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": integer, "True": reroute_1},
        attrs={"input_type": "INT"},
    )

    reroute_72 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch})

    reroute_34 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Orientation"]}
    )

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_34})

    reroute_36 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Length"]}
    )

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_36})

    reroute_38 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Y Offset"]}
    )

    reroute_39 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_38})

    reroute_40 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Z Offset"]}
    )

    reroute_41 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_40})

    reroute_44 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Radius"]}
    )

    reroute_45 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_44})

    reroute_42 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Protrude"]}
    )

    reroute_43 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_42})

    reroute_46 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Width"]}
    )

    reroute_47 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_46})

    reroute_48 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Height"]}
    )

    reroute_49 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_48})

    reroute_50 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Door Shelves"]}
    )

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_50})

    reroute_52 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Shelf Thickness"]},
    )

    reroute_53 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_52})

    reroute_54 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Door Shelf Height"]}
    )

    reroute_55 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Door Shelf Length"]}
    )

    reroute_56 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_55})

    reroute_57 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Material"]}
    )

    reroute_58 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Door Material"]}
    )

    reroute_59 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_58})

    reroute_60 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Inner Material"]},
    )

    reroute_61 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Shelf Material"]},
    )

    reroute_62 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_61})

    fridge_door_003 = nw.new_node(
        nodegroup_fridge_door_003().name,
        input_kwargs={
            "Size": combine_xyz_1,
            "Border Thickness": reroute_69,
            "Has Handles": reroute_31,
            "Handle Type": reroute_33,
            "Handle Location": reroute_72,
            "Handle Orientation": reroute_35,
            "Handle Length": reroute_37,
            "Handle Y Offset": reroute_39,
            "Handle Z Offset": reroute_41,
            "Handle Radius": reroute_45,
            "Handle Protrude": reroute_43,
            "Handle Width": reroute_47,
            "Handle Height": reroute_49,
            "Num Door Shelves": reroute_51,
            "Door Shelf Thickness": reroute_53,
            "Door Shelf Height": reroute_54,
            "Door Shelf Length": reroute_56,
            "Handle Material": reroute_57,
            "Door Material": reroute_59,
            "Inner Material": reroute_60,
            "Door Shelf Material": reroute_62,
        },
    )

    reroute_77 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": fridge_door_003})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": fridge_door_003, "Scale": (1.0000, -1.0000, 1.0000)},
    )

    flip_faces = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": transform_geometry})

    reroute_74 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_66})

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_74, 3: 2},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    reroute_68 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_68},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_or = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: equal_1, 1: equal_2},
        attrs={"operation": "OR"},
    )

    reroute_63 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Dimensions"]}
    )

    reroute_64 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_63})

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_64, 1: (0.5000, 0.5000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_64, 1: (0.5000, -0.5000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": op_or,
            "False": multiply_1.outputs["Vector"],
            "True": multiply_2.outputs["Vector"],
        },
        attrs={"input_type": "VECTOR"},
    )

    vector = nw.new_node(Nodes.Vector)
    vector.vector = (0.0000, 0.0000, 1.0000)

    vector_1 = nw.new_node(Nodes.Vector)
    vector_1.vector = (0.0000, 0.0000, -1.0000)

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": op_or, "False": vector, "True": vector_1},
        attrs={"input_type": "VECTOR"},
    )

    reroute_78 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply_1.outputs["Vector"]}
    )

    reroute_79 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_78})

    reroute_67 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": vector})

    reroute_80 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": equal_1})

    reroute_75 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_64})

    reroute_76 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_75})

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Drawers"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Height"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Depth"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Thickness"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Shelf Height"]}
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Shelves"]}
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

    reroute_16 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Fridge Shelf Thickness"]},
    )

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    reroute_18 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Has Grated Shelves"]}
    )

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    reroute_20 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Outer Shell Material"]},
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    reroute_22 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Inner Shell Material"]},
    )

    reroute_23 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Interior Shelf Material Border"]},
    )

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    reroute_25 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Interior Shelf Material Inside"]},
    )

    reroute_26 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Material"]}
    )

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_26})

    fridge_body_001 = nw.new_node(
        nodegroup_fridge_body_001().name,
        input_kwargs={
            "Size": multiply.outputs["Vector"],
            "Border Thickness": reroute_3,
            "Num Drawers": reroute_5,
            "Drawer Height": reroute_7,
            "Drawer Depth": reroute_9,
            "Drawer Thickness": reroute_11,
            "Shelf Height": reroute_13,
            "Num Shelves": reroute_15,
            "Fridge Shelf Thickness": reroute_17,
            "Has Grated Shelves": reroute_19,
            "Outer Shell Material": reroute_21,
            "Inner Shell Material": reroute_22,
            "Interior Shelf Material Border": reroute_24,
            "Interior Shelf Material Inside": reroute_25,
            "Drawer Material": reroute_27,
        },
    )

    reroute_84 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": fridge_body_001.outputs["Top Freezer Points Duplicate"]},
    )

    reroute_82 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": fridge_body_001.outputs["Top Freezer Drawer 1 Body"]},
    )

    reroute_86 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": fridge_body_001.outputs["Top Freezer Drawer 1"]},
    )

    reroute_83 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": fridge_body_001.outputs["Top Freezer Body"]},
    )

    reroute_81 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": fridge_body_001.outputs["Top Freezer 1 Position"]},
    )

    reroute_85 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": fridge_body_001.outputs["Top Freezer 1 Max Sliding"]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Top Freezer Door 1": reroute_77,
            "Top Freezer Door 2": flip_faces,
            "Top Freezer Position": switch_1,
            "Top Freezer Axis": switch_2,
            "Top Freezer Position 2": reroute_79,
            "Top Freezer Axis 2": reroute_67,
            "Multidoor Freezer": reroute_80,
            "Translation Freezer": reroute_76,
            "Top Freezer Points Duplicate": reroute_84,
            "Top Freezer Drawer 1 Body": reroute_82,
            "Top Freezer Drawer 1": reroute_86,
            "Top Freezer Body": reroute_83,
            "Top Freezer 1 Position": reroute_81,
            "Top Freezer 1 Max Sliding": reroute_85,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_255",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_255(nw: NodeWrangler):
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


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_110",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_110(nw: NodeWrangler):
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


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_109",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_109(nw: NodeWrangler):
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


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_106",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_106(nw: NodeWrangler):
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


@node_utils.to_nodegroup(
    "nodegroup_duplicate_joints_on_parent_033", singleton=False, type="GeometryNodeTree"
)
def nodegroup_duplicate_joints_on_parent_033(nw: NodeWrangler):
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
    "nodegroup_fridge_001", singleton=False, type="GeometryNodeTree"
)
def nodegroup_fridge_001(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Dimensions", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketInt", "Num Doors", 0),
            ("NodeSocketInt", "Handle Location", 0),
            ("NodeSocketFloat", "Border Thickness", 0.0000),
            ("NodeSocketInt", "Num Drawers", 0),
            ("NodeSocketFloat", "Drawer Height", 0.0000),
            ("NodeSocketFloat", "Drawer Depth", 0.0000),
            ("NodeSocketFloat", "Drawer Thickness", 0.0000),
            ("NodeSocketFloat", "Shelf Height", 0.0000),
            ("NodeSocketInt", "Num Shelves", 0),
            ("NodeSocketFloat", "Fridge Door Thickness", 0.0000),
            ("NodeSocketBool", "Has Handles", False),
            ("NodeSocketInt", "Handle Type", 0),
            ("NodeSocketInt", "Handle Orientation", 0),
            ("NodeSocketFloat", "Handle Length", 0.0000),
            ("NodeSocketFloat", "Handle Y Offset", 0.0000),
            ("NodeSocketFloat", "Handle Z Offset", 0.0000),
            ("NodeSocketFloat", "Handle Protrude", 0.0000),
            ("NodeSocketFloat", "Handle Radius", 0.0000),
            ("NodeSocketFloat", "Handle Width", 0.0000),
            ("NodeSocketFloat", "Handle Height", 0.0000),
            ("NodeSocketInt", "Num Door Shelves", 0),
            ("NodeSocketFloat", "Door Shelf Thickness", 0.0000),
            ("NodeSocketFloat", "Door Shelf Height", 0.1000),
            ("NodeSocketFloat", "Door Shelf Length", 0.2000),
            ("NodeSocketFloat", "Fridge Shelf Thickness", 0.0000),
            ("NodeSocketBool", "Has Grated Shelves", False),
            ("NodeSocketMaterial", "Outer Shell Material", None),
            ("NodeSocketMaterial", "Inner Shell Material", None),
            ("NodeSocketMaterial", "Handle Material", None),
            ("NodeSocketMaterial", "Door Material", None),
            ("NodeSocketMaterial", "Door Inner Material", None),
            ("NodeSocketMaterial", "Door Shelf Material", None),
            ("NodeSocketMaterial", "Interior Shelf Material Border", None),
            ("NodeSocketMaterial", "Interior Shelf Material Inside", None),
            ("NodeSocketMaterial", "Drawer Material", None),
        ],
    )

    reroute_30 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Fridge Door Thickness"]},
    )

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_30})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 1.0000, "Y": 1.0000, "Z": 1.0000}
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Dimensions"], 1: combine_xyz},
        attrs={"operation": "MULTIPLY"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": multiply.outputs["Vector"]}
    )

    reroute_65 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Doors"]}
    )

    reroute_66 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_65})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: reroute_66},
        attrs={"operation": "DIVIDE"},
    )

    reroute_71 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["Z"]}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_31, "Y": divide, "Z": reroute_71}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Border Thickness"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_70 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    reroute_32 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Has Handles"]}
    )

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_32})

    reroute_34 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Type"]}
    )

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_34})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Num Doors"], 3: 1},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    integer = nw.new_node(Nodes.Integer)
    integer.integer = 0

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Location"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": integer, "True": reroute_1},
        attrs={"input_type": "INT"},
    )

    reroute_72 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch})

    reroute_36 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Orientation"]}
    )

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_36})

    reroute_38 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Length"]}
    )

    reroute_39 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_38})

    reroute_40 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Y Offset"]}
    )

    reroute_41 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_40})

    reroute_42 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Z Offset"]}
    )

    reroute_43 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_42})

    reroute_46 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Radius"]}
    )

    reroute_47 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_46})

    reroute_44 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Protrude"]}
    )

    reroute_45 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_44})

    reroute_48 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Width"]}
    )

    reroute_49 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_48})

    reroute_50 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Height"]}
    )

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_50})

    reroute_52 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Door Shelves"]}
    )

    reroute_53 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Shelf Thickness"]},
    )

    reroute_54 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_53})

    reroute_55 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Door Shelf Height"]}
    )

    reroute_56 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Door Shelf Length"]}
    )

    reroute_57 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Material"]}
    )

    reroute_58 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_57})

    reroute_59 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Door Material"]}
    )

    reroute_60 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Inner Material"]},
    )

    reroute_61 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_60})

    reroute_62 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Shelf Material"]},
    )

    fridge_door_002 = nw.new_node(
        nodegroup_fridge_door_002().name,
        input_kwargs={
            "Size": combine_xyz_1,
            "Border Thickness": reroute_70,
            "Has Handles": reroute_33,
            "Handle Type": reroute_35,
            "Handle Location": reroute_72,
            "Handle Orientation": reroute_37,
            "Handle Length": reroute_39,
            "Handle Y Offset": reroute_41,
            "Handle Z Offset": reroute_43,
            "Handle Radius": reroute_47,
            "Handle Protrude": reroute_45,
            "Handle Width": reroute_49,
            "Handle Height": reroute_51,
            "Num Door Shelves": reroute_52,
            "Door Shelf Thickness": reroute_54,
            "Door Shelf Height": reroute_55,
            "Door Shelf Length": reroute_56,
            "Handle Material": reroute_58,
            "Door Material": reroute_59,
            "Inner Material": reroute_61,
            "Door Shelf Material": reroute_62,
        },
    )

    reroute_77 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": fridge_door_002})

    reroute_78 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_77})

    reroute_74 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_66})

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_74, 3: 2},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    reroute_69 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_69},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    op_or = nw.new_node(
        Nodes.BooleanMath,
        input_kwargs={0: equal_1, 1: equal_2},
        attrs={"operation": "OR"},
    )

    reroute_63 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Dimensions"]}
    )

    reroute_64 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_63})

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_64, 1: (0.5000, 0.5000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_64, 1: (0.5000, -0.5000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": op_or,
            "False": multiply_1.outputs["Vector"],
            "True": multiply_2.outputs["Vector"],
        },
        attrs={"input_type": "VECTOR"},
    )

    vector = nw.new_node(Nodes.Vector)
    vector.vector = (0.0000, 0.0000, 1.0000)

    vector_1 = nw.new_node(Nodes.Vector)
    vector_1.vector = (0.0000, 0.0000, -1.0000)

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": op_or, "False": vector, "True": vector_1},
        attrs={"input_type": "VECTOR"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": fridge_door_002, "Scale": (1.0000, -1.0000, 1.0000)},
    )

    flip_faces = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": transform_geometry})

    reroute_67 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": vector})

    reroute_68 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_67})

    reroute_75 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_64})

    reroute_76 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_75})

    reroute_82 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_76})

    reroute_79 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply_1.outputs["Vector"]}
    )

    reroute_80 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_79})

    reroute_81 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": equal_1})

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Drawers"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Height"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Depth"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Thickness"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Shelf Height"]}
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Num Shelves"]}
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

    reroute_16 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Fridge Shelf Thickness"]},
    )

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    reroute_18 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Has Grated Shelves"]}
    )

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    reroute_20 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Outer Shell Material"]},
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    reroute_22 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Inner Shell Material"]},
    )

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    reroute_24 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Interior Shelf Material Border"]},
    )

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_24})

    reroute_26 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Interior Shelf Material Inside"]},
    )

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_26})

    reroute_28 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Material"]}
    )

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_28})

    fridge_body_002 = nw.new_node(
        nodegroup_fridge_body_002().name,
        input_kwargs={
            "Size": multiply.outputs["Vector"],
            "Border Thickness": reroute_3,
            "Num Drawers": reroute_5,
            "Drawer Height": reroute_7,
            "Drawer Depth": reroute_9,
            "Drawer Thickness": reroute_11,
            "Shelf Height": reroute_13,
            "Num Shelves": reroute_15,
            "Fridge Shelf Thickness": reroute_17,
            "Has Grated Shelves": reroute_19,
            "Outer Shell Material": reroute_21,
            "Inner Shell Material": reroute_23,
            "Interior Shelf Material Border": reroute_25,
            "Interior Shelf Material Inside": reroute_27,
            "Drawer Material": reroute_29,
        },
    )

    reroute_83 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": fridge_body_002.outputs["Fridge Body"]}
    )

    reroute_85 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": fridge_body_002.outputs["Fridge Body Drawer Body 2"]},
    )

    reroute_86 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": fridge_body_002.outputs["Fridge Body Drawer"]},
    )

    reroute_84 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": fridge_body_002.outputs["Fridge Body Duplicate Points"]},
    )

    reroute_73 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": fridge_body_002.outputs["Fridge Body Drawer Position"]},
    )

    reroute_87 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": fridge_body_002.outputs["Fridge Body Drawer Max"]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Fridge Door 1 Geom": reroute_78,
            "Fridge Position": switch_1,
            "Fridge Door 1 Axus": switch_2,
            "Fridge Door 2 Geom": flip_faces,
            "Frdige Door 2 Axis": reroute_68,
            "Fridge Door 2 Position": reroute_82,
            "Fridge Translation": reroute_80,
            "MulitDoor Frdige": reroute_81,
            "Fridge Body": reroute_83,
            "Fridge Body Drawer Body 2": reroute_85,
            "Fridge Body Drawer": reroute_86,
            "Fridge Body Duplicate Points": reroute_84,
            "Fridge Body Drawer Position": reroute_73,
            "Fridge Body Drawer Max": reroute_87,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_111",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_111(nw: NodeWrangler):
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


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketBool", "Has Top Freezer", False),
            ("NodeSocketBool", "Has Bottom Freezer", False),
            ("NodeSocketBool", "Has Handles", False),
            ("NodeSocketVector", "Main Fridge Dimensions", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "Top Freezer Height", 0.0000),
            ("NodeSocketFloat", "Bottom Freezer 1 Height", 0.0000),
            ("NodeSocketFloat", "Bottom Freezer 2 Height", 0.0000),
            ("NodeSocketInt", "Main Fridge Num Doors", 1),
            ("NodeSocketInt", "Bottom Freezer Num Freezers", 0),
            ("NodeSocketFloat", "Main Fridge Border Thickness", 0.0300),
            ("NodeSocketInt", "Main Fridge Num Shelves", 2),
            ("NodeSocketInt", "Main Fridge Num Door Shelves", 2),
            ("NodeSocketInt", "Main Fridge Num Internal Drawers", 0),
            ("NodeSocketFloat", "Main Fridge Drawer Height", 0.0000),
            ("NodeSocketFloat", "Main Fridge Drawer Shelf Depth", 0.1000),
            ("NodeSocketFloat", "Main Fridge Drawer Thickness", 0.0000),
            ("NodeSocketFloat", "Top Freezer Handle Length", 0.0000),
            ("NodeSocketFloat", "Top Freezer Handle Y Offset", 0.0000),
            ("NodeSocketFloat", "Top Freezer Handle Z Offset", 0.0000),
            ("NodeSocketInt", "Top Freezer Num Shelves", 0),
            ("NodeSocketInt", "Top Freezer Num Door Shelves", 0),
            ("NodeSocketInt", "Bottom Freezer Num Drawers", 0),
            ("NodeSocketInt", "Handle Type", 0),
            ("NodeSocketInt", "Handle Location", 0),
            ("NodeSocketInt", "Handle Orientation", 0),
            ("NodeSocketFloat", "Handle Length", 0.0000),
            ("NodeSocketFloat", "Handle Y Offset", 0.0000),
            ("NodeSocketFloat", "Handle Z Offset", 0.0000),
            ("NodeSocketFloat", "Handle Protrude", 0.0300),
            ("NodeSocketFloat", "Handle Radius", 0.0100),
            ("NodeSocketFloat", "Handle Width", 0.0500),
            ("NodeSocketFloat", "Handle Height", 0.0200),
            ("NodeSocketFloat", "Door Thickness", 0.0300),
            ("NodeSocketFloat", "Shelf Height", 0.0100),
            ("NodeSocketFloat", "Shelf Border Thickness", 0.0000),
            ("NodeSocketFloat", "Door Shelf Height", 0.1000),
            ("NodeSocketFloat", "Door Shelf Depth", 0.2000),
            ("NodeSocketFloat", "Door Shelf Thickness", 0.0300),
            ("NodeSocketBool", "Main Fridge Has Grated Shelves", False),
            ("NodeSocketBool", "Top Freezer Has Grated Shelves", False),
            ("NodeSocketMaterial", "Outer Shell Material", None),
            ("NodeSocketMaterial", "Inner Shell Material", None),
            ("NodeSocketMaterial", "Handle Material", None),
            ("NodeSocketMaterial", "Door Material", None),
            ("NodeSocketMaterial", "Door Inner Material", None),
            ("NodeSocketMaterial", "Door Shelf Material", None),
            ("NodeSocketMaterial", "Interior Shelf Material Border", None),
            ("NodeSocketMaterial", "Interior Shelf Material Inside", None),
            ("NodeSocketMaterial", "Drawer Material", None),
            ("NodeSocketMaterial", "Exterior Drawer Shell Material", None),
            ("NodeSocketMaterial", "Exterior Drawer Inside Material", None),
            ("NodeSocketMaterial", "Exterior Drawer Handle Material", None),
            ("NodeSocketMaterial", "Exterior Drawer Material", None),
            ("NodeSocketMaterial", "Exterior Drawer Door Material", None),
        ],
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Has Bottom Freezer"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Has Top Freezer"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    fridge_001 = nw.new_node(
        nodegroup_fridge_001().name,
        input_kwargs={
            "Dimensions": group_input.outputs["Main Fridge Dimensions"],
            "Num Doors": group_input.outputs["Main Fridge Num Doors"],
            "Handle Location": group_input.outputs["Handle Location"],
            "Border Thickness": group_input.outputs["Main Fridge Border Thickness"],
            "Num Drawers": group_input.outputs["Main Fridge Num Internal Drawers"],
            "Drawer Height": group_input.outputs["Main Fridge Drawer Height"],
            "Drawer Depth": group_input.outputs["Main Fridge Drawer Shelf Depth"],
            "Drawer Thickness": group_input.outputs["Main Fridge Drawer Thickness"],
            "Shelf Height": group_input.outputs["Shelf Height"],
            "Num Shelves": group_input.outputs["Main Fridge Num Shelves"],
            "Fridge Door Thickness": group_input.outputs["Door Thickness"],
            "Has Handles": group_input.outputs["Has Handles"],
            "Handle Type": group_input.outputs["Handle Type"],
            "Handle Orientation": group_input.outputs["Handle Orientation"],
            "Handle Length": group_input.outputs["Handle Length"],
            "Handle Y Offset": group_input.outputs["Handle Y Offset"],
            "Handle Z Offset": group_input.outputs["Handle Z Offset"],
            "Handle Protrude": group_input.outputs["Handle Protrude"],
            "Handle Radius": group_input.outputs["Handle Radius"],
            "Handle Width": group_input.outputs["Handle Width"],
            "Handle Height": group_input.outputs["Handle Height"],
            "Num Door Shelves": group_input.outputs["Main Fridge Num Door Shelves"],
            "Door Shelf Thickness": group_input.outputs["Door Shelf Thickness"],
            "Door Shelf Height": group_input.outputs["Door Shelf Height"],
            "Door Shelf Length": group_input.outputs["Door Shelf Depth"],
            "Fridge Shelf Thickness": group_input.outputs["Shelf Border Thickness"],
            "Has Grated Shelves": group_input.outputs["Main Fridge Has Grated Shelves"],
            "Outer Shell Material": group_input.outputs["Outer Shell Material"],
            "Inner Shell Material": group_input.outputs["Inner Shell Material"],
            "Handle Material": group_input.outputs["Handle Material"],
            "Door Material": group_input.outputs["Door Material"],
            "Door Inner Material": group_input.outputs["Door Inner Material"],
            "Door Shelf Material": group_input.outputs["Door Shelf Material"],
            "Interior Shelf Material Border": group_input.outputs[
                "Interior Shelf Material Border"
            ],
            "Interior Shelf Material Inside": group_input.outputs[
                "Interior Shelf Material Inside"
            ],
            "Drawer Material": group_input.outputs["Drawer Material"],
        },
    )

    reroute_94 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": fridge_001.outputs["MulitDoor Frdige"]}
    )

    reroute_95 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_94})

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": fridge_001.outputs["Fridge Body Drawer Body 2"],
            "Label": "drawer_body2",
        },
    )

    add_jointed_geometry_metadata_106 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_106().name,
        input_kwargs={
            "Geometry": fridge_001.outputs["Fridge Body Drawer"],
            "Label": "drawer2",
        },
    )

    reroute_99 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": fridge_001.outputs["Fridge Body Drawer Position"]},
    )

    reroute_100 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_99})

    reroute_101 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": fridge_001.outputs["Fridge Body Drawer Max"]},
    )

    reroute_102 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_101})

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "drawer_joint_2",
            "Parent": add_jointed_geometry_metadata,
            "Child": add_jointed_geometry_metadata_106,
            "Position": reroute_100,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Max": reroute_102,
        },
        label="Sliding Joint",
    )

    reroute_97 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": fridge_001.outputs["Fridge Body Duplicate Points"]},
    )

    reroute_98 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_97})

    duplicate_joints_on_parent_033 = nw.new_node(
        nodegroup_duplicate_joints_on_parent_033().name,
        input_kwargs={
            "Parent": sliding_joint.outputs["Parent"],
            "Child": sliding_joint.outputs["Child"],
            "Points": reroute_98,
        },
    )

    reroute_96 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": fridge_001.outputs["Fridge Body"]}
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [duplicate_joints_on_parent_033, reroute_96]},
    )

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": join_geometry_3, "Label": "body"},
    )

    reroute_83 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": fridge_001.outputs["Fridge Door 1 Geom"]}
    )

    add_jointed_geometry_metadata_109 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_109().name,
        input_kwargs={"Geometry": reroute_83, "Label": "door1"},
    )

    reroute_84 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": fridge_001.outputs["Fridge Position"]}
    )

    reroute_85 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_84})

    reroute_86 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": fridge_001.outputs["Fridge Door 1 Axus"]}
    )

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "door_hinge_3",
            "Parent": add_jointed_geometry_metadata_1,
            "Child": add_jointed_geometry_metadata_109,
            "Position": reroute_85,
            "Axis": reroute_86,
            "Max": 3.1416,
        },
        label="Hinge Joint",
    )

    reroute_140 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint.outputs["Geometry"]}
    )

    add_jointed_geometry_metadata_110 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_110().name,
        input_kwargs={
            "Geometry": hinge_joint.outputs["Geometry"],
            "Label": "body_with_door1",
        },
    )

    reroute_87 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": fridge_001.outputs["Fridge Door 2 Geom"]}
    )

    add_jointed_geometry_metadata_110_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_110().name,
        input_kwargs={"Geometry": reroute_87, "Label": "door2"},
    )

    reroute_90 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": fridge_001.outputs["Fridge Translation"]}
    )

    reroute_91 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_90})

    reroute_88 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": fridge_001.outputs["Frdige Door 2 Axis"]}
    )

    reroute_89 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_88})

    hinge_joint_1 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "door_hinge_4",
            "Parent": add_jointed_geometry_metadata_110,
            "Child": add_jointed_geometry_metadata_110_1,
            "Position": reroute_91,
            "Axis": reroute_89,
            "Max": 3.1416,
        },
        label="Hinge Joint",
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_95,
            "False": reroute_140,
            "True": hinge_joint_1.outputs["Geometry"],
        },
    )

    reroute_92 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": fridge_001.outputs["Fridge Door 2 Position"]},
    )

    reroute_93 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_92})

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_93, 1: (0.0000, 0.0000, 0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": switch_3, "Translation": multiply.outputs["Vector"]},
    )

    reroute_144 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_3}
    )

    add_jointed_geometry_metadata_111 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_111().name,
        input_kwargs={"Geometry": reroute_144, "Label": "main_fridge"},
    )

    reroute_145 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_111}
    )

    reroute = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Main Fridge Dimensions"]},
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Top Freezer Height"]}
    )

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_1, 1: (1.0000, 1.0000, 0.0000), 2: combine_xyz},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_23 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Location"]}
    )

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    reroute_8 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Main Fridge Border Thickness"]},
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    reroute_10 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Main Fridge Drawer Shelf Depth"]},
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_36 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Shelf Height"]}
    )

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_36})

    reroute_17 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Top Freezer Num Shelves"]},
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_17})

    reroute_34 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Door Thickness"]}
    )

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_34})

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Has Handles"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    reroute_21 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Type"]}
    )

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    reroute_25 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Orientation"]}
    )

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_25})

    reroute_12 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Top Freezer Handle Length"]},
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    reroute_14 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Top Freezer Handle Y Offset"]},
    )

    reroute_15 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Top Freezer Handle Z Offset"]},
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_15})

    reroute_27 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Protrude"]}
    )

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_27})

    reroute_29 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Radius"]}
    )

    reroute_30 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Width"]}
    )

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_30})

    reroute_32 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Height"]}
    )

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_32})

    reroute_19 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Top Freezer Num Door Shelves"]},
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    reroute_44 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Shelf Thickness"]},
    )

    reroute_45 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_44})

    reroute_40 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Door Shelf Height"]}
    )

    reroute_41 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_40})

    reroute_42 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Door Shelf Depth"]}
    )

    reroute_43 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_42})

    reroute_38 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Shelf Border Thickness"]},
    )

    reroute_39 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_38})

    reroute_46 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Top Freezer Has Grated Shelves"]},
    )

    reroute_47 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_46})

    reroute_48 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Outer Shell Material"]},
    )

    reroute_49 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_48})

    reroute_50 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Inner Shell Material"]},
    )

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_50})

    reroute_52 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Material"]}
    )

    reroute_53 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_52})

    reroute_54 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Door Material"]}
    )

    reroute_55 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_54})

    reroute_112 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_55})

    reroute_56 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Inner Material"]},
    )

    reroute_57 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_56})

    reroute_113 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_57})

    reroute_58 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Shelf Material"]},
    )

    reroute_59 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_58})

    reroute_114 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_59})

    reroute_60 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Interior Shelf Material Border"]},
    )

    reroute_61 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_60})

    reroute_115 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_61})

    reroute_62 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Interior Shelf Material Inside"]},
    )

    reroute_63 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_62})

    reroute_116 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_63})

    reroute_64 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Material"]}
    )

    reroute_65 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_64})

    reroute_117 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_65})

    top_freezer_001 = nw.new_node(
        nodegroup_top_freezer_001().name,
        input_kwargs={
            "Dimensions": multiply_add.outputs["Vector"],
            "Num Doors": 1,
            "Handle Location": reroute_24,
            "Border Thickness": reroute_9,
            "Drawer Depth": reroute_11,
            "Shelf Height": reroute_37,
            "Num Shelves": reroute_18,
            "Fridge Door Thickness": reroute_35,
            "Has Handles": reroute_7,
            "Handle Type": reroute_22,
            "Handle Orientation": reroute_26,
            "Handle Length": reroute_13,
            "Handle Y Offset": reroute_14,
            "Handle Z Offset": reroute_16,
            "Handle Protrude": reroute_28,
            "Handle Radius": reroute_29,
            "Handle Width": reroute_31,
            "Handle Height": reroute_33,
            "Num Door Shelves": reroute_20,
            "Door Shelf Thickness": reroute_45,
            "Door Shelf Height": reroute_41,
            "Door Shelf Length": reroute_43,
            "Fridge Shelf Thickness": reroute_39,
            "Has Grated Shelves": reroute_47,
            "Outer Shell Material": reroute_49,
            "Inner Shell Material": reroute_51,
            "Handle Material": reroute_53,
            "Door Material": reroute_112,
            "Door Inner Material": reroute_113,
            "Door Shelf Material": reroute_114,
            "Interior Shelf Material Border": reroute_115,
            "Interior Shelf Material Inside": reroute_116,
            "Drawer Material": reroute_117,
        },
    )

    reroute_131 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": top_freezer_001.outputs["Multidoor Freezer"]},
    )

    add_jointed_geometry_metadata_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": top_freezer_001.outputs["Top Freezer Drawer 1 Body"],
            "Label": "drawer1_parent",
        },
    )

    add_jointed_geometry_metadata_3 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": top_freezer_001.outputs["Top Freezer Drawer 1"],
            "Label": "drawer1_body",
        },
    )

    reroute_138 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": top_freezer_001.outputs["Top Freezer 1 Position"]},
    )

    reroute_139 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": top_freezer_001.outputs["Top Freezer 1 Max Sliding"]},
    )

    sliding_joint_1 = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "drawer_joint_1",
            "Parent": add_jointed_geometry_metadata_2,
            "Child": add_jointed_geometry_metadata_3,
            "Position": reroute_138,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Max": reroute_139,
        },
        label="Sliding Joint",
    )

    reroute_134 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": top_freezer_001.outputs["Top Freezer Points Duplicate"]},
    )

    reroute_135 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_134})

    duplicate_joints_on_parent_034 = nw.new_node(
        nodegroup_duplicate_joints_on_parent_034().name,
        input_kwargs={
            "Parent": sliding_joint_1.outputs["Parent"],
            "Child": sliding_joint_1.outputs["Child"],
            "Points": reroute_135,
        },
    )

    reroute_136 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": top_freezer_001.outputs["Top Freezer Body"]},
    )

    reroute_137 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_136})

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [duplicate_joints_on_parent_034, reroute_137]},
    )

    add_jointed_geometry_metadata_4 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": join_geometry_4, "Label": "top_freezer_body"},
    )

    reroute_121 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": top_freezer_001.outputs["Top Freezer Door 1"]},
    )

    add_jointed_geometry_metadata_464 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_464().name,
        input_kwargs={"Geometry": reroute_121, "Label": "top_freezer_door1"},
    )

    reroute_123 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": top_freezer_001.outputs["Top Freezer Position"]},
    )

    reroute_124 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_123})

    reroute_125 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": top_freezer_001.outputs["Top Freezer Axis"]},
    )

    reroute_126 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_125})

    hinge_joint_2 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "door_hinge_1",
            "Parent": add_jointed_geometry_metadata_4,
            "Child": add_jointed_geometry_metadata_464,
            "Position": reroute_124,
            "Axis": reroute_126,
            "Max": 3.1416,
        },
        label="Hinge Joint",
    )

    reroute_143 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_2.outputs["Geometry"]}
    )

    add_jointed_geometry_metadata_5 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": hinge_joint_2.outputs["Geometry"],
            "Label": "top_freezer_with_door1",
        },
    )

    reroute_122 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": top_freezer_001.outputs["Top Freezer Door 2"]},
    )

    add_jointed_geometry_metadata_6 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_122, "Label": "top_freezer_door2"},
    )

    reroute_127 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": top_freezer_001.outputs["Top Freezer Position 2"]},
    )

    reroute_128 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_127})

    reroute_129 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": top_freezer_001.outputs["Top Freezer Axis 2"]},
    )

    reroute_130 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_129})

    hinge_joint_3 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "door_hinge_2",
            "Parent": add_jointed_geometry_metadata_5,
            "Child": add_jointed_geometry_metadata_6,
            "Position": reroute_128,
            "Axis": reroute_130,
            "Max": 3.1416,
        },
        label="Hinge Joint",
    )

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_131,
            "False": reroute_143,
            "True": hinge_joint_3.outputs["Geometry"],
        },
    )

    reroute_132 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": top_freezer_001.outputs["Translation Freezer"]},
    )

    reroute_133 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_132})

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_133, 1: (0.0000, 0.0000, 0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": switch_4,
            "Translation": multiply_1.outputs["Vector"],
        },
    )

    reroute_107 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_107, 1: (0.0000, 0.0000, 1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_4,
            "Translation": multiply_2.outputs["Vector"],
        },
    )

    add_jointed_geometry_metadata_255 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_255().name,
        input_kwargs={"Geometry": transform_geometry, "Label": "top_freezer"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                add_jointed_geometry_metadata_111,
                add_jointed_geometry_metadata_255,
            ]
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_3, "False": reroute_145, "True": join_geometry},
    )

    reroute_146 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch})

    reroute_67 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Bottom Freezer Num Freezers"]},
    )

    reroute_68 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_67})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_68, 3: 2},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    bottom_freezer_002 = nw.new_node(
        nodegroup_bottom_freezer_002().name,
        input_kwargs={
            "Main Fridge Dimensions": group_input.outputs["Main Fridge Dimensions"],
            "Has Handle": group_input.outputs["Has Handles"],
            "Exterior Drawer Height": group_input.outputs["Bottom Freezer 1 Height"],
            "Fridge Door Thickness": group_input.outputs["Door Thickness"],
            "Handle Type": group_input.outputs["Handle Type"],
            "Handle Radius": group_input.outputs["Handle Radius"],
            "Handle Protrude": group_input.outputs["Handle Protrude"],
            "Handle Width": group_input.outputs["Handle Width"],
            "Handle Height": group_input.outputs["Handle Height"],
            "Exterior Drawer Shell Material": group_input.outputs[
                "Exterior Drawer Shell Material"
            ],
            "Exterior Drawer Inside Material": group_input.outputs[
                "Exterior Drawer Inside Material"
            ],
            "Exterior Drawer Material": group_input.outputs["Exterior Drawer Material"],
            "Exterior Drawer Handle Material": group_input.outputs[
                "Exterior Drawer Handle Material"
            ],
            "Exterior Drawer Door Material": group_input.outputs[
                "Exterior Drawer Door Material"
            ],
        },
    )

    add_jointed_geometry_metadata_7 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": bottom_freezer_002.outputs["Bottom Freezer 2 Body"],
            "Label": "freezer_body",
        },
    )

    add_jointed_geometry_metadata_8 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": bottom_freezer_002.outputs["Bottom Freezer 2 Drawer"],
            "Label": "freezer_drawer",
        },
    )

    reroute_105 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={
            "Input": bottom_freezer_002.outputs["Bottom Freezer 2 Max Sliding"]
        },
    )

    reroute_106 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_105})

    sliding_joint_2 = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "bottom_freezer",
            "Parent": add_jointed_geometry_metadata_7,
            "Child": add_jointed_geometry_metadata_8,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Max": reroute_106,
        },
        label="Sliding Joint",
    )

    reroute_103 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={
            "Input": bottom_freezer_002.outputs["Bottom Freezer 2 Translation"]
        },
    )

    reroute_104 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_103})

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": sliding_joint_2.outputs["Geometry"],
            "Translation": reroute_104,
        },
    )

    reroute_119 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_7}
    )

    reroute_120 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_119})

    reroute_141 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_120})

    reroute_80 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Main Fridge Dimensions"]},
    )

    reroute_79 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Has Handles"]}
    )

    reroute_81 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Bottom Freezer 2 Height"]},
    )

    reroute_73 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Door Thickness"]}
    )

    reroute_82 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Type"]}
    )

    reroute_70 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Radius"]}
    )

    reroute_69 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Protrude"]}
    )

    reroute_71 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Width"]}
    )

    reroute_72 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Height"]}
    )

    reroute_74 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Exterior Drawer Shell Material"]},
    )

    reroute_75 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Exterior Drawer Inside Material"]},
    )

    reroute_77 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Exterior Drawer Material"]},
    )

    reroute_76 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Exterior Drawer Handle Material"]},
    )

    reroute_78 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Exterior Drawer Door Material"]},
    )

    bottom_freezer_003 = nw.new_node(
        nodegroup_bottom_freezer_003().name,
        input_kwargs={
            "Main Fridge Dimensions": reroute_80,
            "Has Handle": reroute_79,
            "Exterior Drawer Height": reroute_81,
            "Fridge Door Thickness": reroute_73,
            "Handle Type": reroute_82,
            "Handle Radius": reroute_70,
            "Handle Protrude": reroute_69,
            "Handle Width": reroute_71,
            "Handle Height": reroute_72,
            "Exterior Drawer Shell Material": reroute_74,
            "Exterior Drawer Inside Material": reroute_75,
            "Exterior Drawer Material": reroute_77,
            "Exterior Drawer Handle Material": reroute_76,
            "Exterior Drawer Door Material": reroute_78,
        },
    )

    add_jointed_geometry_metadata_9 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": bottom_freezer_003.outputs["Bottom Freezer Body"],
            "Label": "bottom_freezer_body",
        },
    )

    reroute_118 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_9}
    )

    add_jointed_geometry_metadata_10 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": bottom_freezer_003.outputs["Bottom Freezer Drawer"],
            "Label": "bottom_freezer_drawer",
        },
    )

    reroute_110 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": bottom_freezer_003.outputs["Sliding Joint Max"]},
    )

    reroute_111 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_110})

    sliding_joint_3 = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "bottom_fridge_drawer_1",
            "Parent": reroute_118,
            "Child": add_jointed_geometry_metadata_10,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Max": reroute_111,
        },
        label="Sliding Joint",
    )

    reroute_108 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={
            "Input": bottom_freezer_003.outputs["Bottom Freezer Translation"]
        },
    )

    reroute_109 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_108})

    transform_geometry_8 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": sliding_joint_3.outputs["Geometry"],
            "Translation": reroute_109,
        },
    )

    add_jointed_geometry_metadata_11 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_8, "Label": "bottom_freezer"},
    )

    reroute_66 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Bottom Freezer 1 Height"]},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_66, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_3})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_11,
            "Translation": combine_xyz_1,
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_1, reroute_120]},
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": reroute_141, "True": join_geometry_1},
    )

    reroute_142 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_1})

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [switch, reroute_142]}
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_5,
            "False": reroute_146,
            "True": join_geometry_2,
        },
    )

    reroute_147 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_2})

    bounding_box = nw.new_node(Nodes.BoundingBox, input_kwargs={"Geometry": switch_2})

    multiply_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Min"], 1: (0.0000, 0.0000, -1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_147,
            "Translation": multiply_4.outputs["Vector"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_2},
        attrs={"is_active_output": True},
    )


class RefrigeratorFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    @gin.configurable(module="RefrigeratorFactory")
    def sample_joint_parameters(
        cls,
        door_hinge_stiffness_min: float = 0.0,
        door_hinge_stiffness_max: float = 3.0,
        door_hinge_damping_min: float = 10.0,
        door_hinge_damping_max: float = 30.0,
        bottom_fridge_drawer_stiffness_min: float = 0.0,
        bottom_fridge_drawer_stiffness_max: float = 3.0,
        bottom_fridge_drawer_damping_min: float = 10.0,
        bottom_fridge_drawer_damping_max: float = 30.0,
        drawer_joint_stiffness_min: float = 0.0,
        drawer_joint_stiffness_max: float = 3.0,
        drawer_joint_damping_min: float = 5.0,
        drawer_joint_damping_max: float = 15.0,
        bottom_freezer_stiffness_min: float = 0.0,
        bottom_freezer_stiffness_max: float = 3.0,
        bottom_freezer_damping_min: float = 10.0,
        bottom_freezer_damping_max: float = 30.0,
    ):
        return {
            "door_hinge_3": {
                "stiffness": uniform(
                    door_hinge_stiffness_min, door_hinge_stiffness_max
                ),
                "damping": uniform(door_hinge_damping_min, door_hinge_damping_max),
            },
            "bottom_fridge_drawer_1": {
                "stiffness": uniform(
                    bottom_fridge_drawer_stiffness_min,
                    bottom_fridge_drawer_stiffness_max,
                ),
                "damping": uniform(
                    bottom_fridge_drawer_damping_min,
                    bottom_fridge_drawer_damping_max,
                ),
            },
            "drawer_joint_1": {
                "stiffness": uniform(
                    drawer_joint_stiffness_min, drawer_joint_stiffness_max
                ),
                "damping": uniform(drawer_joint_damping_min, drawer_joint_damping_max),
            },
            "drawer_joint_2": {
                "stiffness": uniform(
                    drawer_joint_stiffness_min, drawer_joint_stiffness_max
                ),
                "damping": uniform(drawer_joint_damping_min, drawer_joint_damping_max),
            },
            "bottom_freezer": {
                "stiffness": uniform(
                    bottom_freezer_stiffness_min, bottom_freezer_stiffness_max
                ),
                "damping": uniform(
                    bottom_freezer_damping_min, bottom_freezer_damping_max
                ),
            },
            "door_hinge_1": {
                "stiffness": uniform(
                    door_hinge_stiffness_min, door_hinge_stiffness_max
                ),
                "damping": uniform(door_hinge_damping_min, door_hinge_damping_max),
            },
            "door_hinge_2": {
                "stiffness": uniform(
                    door_hinge_stiffness_min, door_hinge_stiffness_max
                ),
                "damping": uniform(door_hinge_damping_min, door_hinge_damping_max),
            },
            "door_hinge_4": {
                "stiffness": uniform(
                    door_hinge_stiffness_min, door_hinge_stiffness_max
                ),
                "damping": uniform(door_hinge_damping_min, door_hinge_damping_max),
            },
        }

    def sample_parameters(self):
        # add code here to randomly sample from parameters

        import numpy as np

        from infinigen.assets.materials.plastic import plastic_rough
        from infinigen.core.util.color import hsv2rgba

        is_double_door = np.random.choice([True, False])
        if is_double_door:
            n_doors = 2
            has_handles = True
            has_top_freezer = False
            width = uniform(0.8, 1.1)
        else:
            n_doors = 1
            has_handles = np.random.choice([True, False])
            has_top_freezer = np.random.choice([True, False])
            width = uniform(0.5, 0.8)

        if has_top_freezer:
            has_bottom_freezer = False
            num_freezers = 0
        else:
            has_bottom_freezer = True

        if has_top_freezer:
            has_bottom_freezer = False
            num_freezers = 0
            top_freezer_height = uniform(0.3, 0.5)
        else:
            has_bottom_freezer = np.random.choice([True, False])
            num_freezers = np.random.choice(
                [1, 2], p=[0.8, 0.2]
            )  # TODO: sample from here once freezers are added
            top_freezer_height = 0

        depth = uniform(0.5, 0.8)

        if has_top_freezer:
            height = uniform(0.8, 1.2)  # no drawer, top freezer
        else:
            if num_freezers == 0:
                height = uniform(0.85, 2.0)
            elif num_freezers == 1:
                height = uniform(1.0, 1.5)
            else:
                height = uniform(0.7, 1.2)

        freezer_height_1 = 0
        freezer_height_2 = 0
        if has_bottom_freezer:
            if num_freezers == 1:
                freezer_height_1 = uniform(0.4, 0.6)
            else:
                freezer_height_1 = uniform(0.1, 0.15)
                freezer_height_2 = uniform(0.3, 0.5)

        height = min(height, 3 * width)

        main_dimensions = (depth, width, height)
        border_thickness = uniform(0.01, 0.07)

        door_to_shelf_gap = uniform(0.06, 0.12)
        drawer_depth = depth - border_thickness - door_to_shelf_gap

        handle_type = randint(0, 3)
        handle_radius = uniform(0.002, 0.02)
        handle_width = uniform(0.02, 0.1)
        handle_height = uniform(0.01, 0.03)

        fridge_shelf_height = uniform(0.01, 0.04)

        door_shelf_thickness = uniform(0.005, 0.03)
        door_shelf_height = uniform(0.05, 0.1)

        drawer_height = uniform(0.15, 0.2)
        drawer_thickness = uniform(0.002, 0.01)

        main_handle_length = uniform(0.3, height)

        if handle_type == 0:
            main_handle_y_offset = uniform(
                handle_radius, max(handle_radius, 0.25 * width)
            )
        else:
            main_handle_y_offset = uniform(
                handle_width, max(handle_width, 0.25 * width)
            )

        main_handle_z_offset = uniform(0, height / 2 - main_handle_length / 2)

        top_freezer_handle_length = uniform(0.1, top_freezer_height)
        top_freezer_handle_z_offset = -uniform(
            0, top_freezer_height / 2 - top_freezer_handle_length / 2
        )

        outer_body_material = weighted_sample(
            material_assignments.kitchen_appliance_hard
        )()()
        color_exterior = np.random.choice([0, 1, 2, 3])
        hsv = None
        if color_exterior < 3:
            h = np.random.uniform(0, 1)
            s = np.random.uniform(0, 0.1)
            if color_exterior == 0:
                v = np.random.uniform(0.9, 1.0)
            elif color_exterior == 1:
                v = np.random.uniform(0.05, 0.15)
            else:
                v = 0
            hsv = (h, s, v)
            rgba = hsv2rgba(hsv)
            outer_body_material = plastic_rough.PlasticRough().generate(base_color=rgba)

        inner_body_material = weighted_sample(material_assignments.metal_neutral)()()
        light_interior = np.random.choice([True, False])
        hsv = None
        if light_interior:
            h = np.random.uniform(0, 1)
            s = np.random.uniform(0, 0.1)
            v = np.random.uniform(0.9, 1.0)
            hsv = (h, s, v)
            rgba = hsv2rgba(hsv)
            inner_body_material = plastic_rough.PlasticRough().generate(base_color=rgba)

        shelf_material = weighted_sample(material_assignments.fridge_shelf)()()
        drawer_material = weighted_sample(material_assignments.glasses)()()
        handle_material = weighted_sample(material_assignments.hard_materials)()()

        # add code here to randomly sample from parameters
        params = {
            "Has Top Freezer": has_top_freezer,
            "Has Bottom Freezer": has_bottom_freezer,
            "Has Handles": has_handles,
            "Main Fridge Dimensions": main_dimensions,
            "Top Freezer Height": top_freezer_height,
            "Bottom Freezer 1 Height": freezer_height_1,
            "Bottom Freezer 2 Height": freezer_height_2,
            "Main Fridge Num Doors": n_doors,
            "Bottom Freezer Num Freezers": num_freezers,
            "Main Fridge Border Thickness": border_thickness,
            "Main Fridge Num Shelves": randint(1, 4),
            "Main Fridge Num Door Shelves": randint(1, 4),
            "Main Fridge Num Internal Drawers": randint(0, 3),
            "Main Fridge Drawer Height": drawer_height,
            "Main Fridge Drawer Shelf Depth": drawer_depth,
            "Main Fridge Drawer Thickness": drawer_thickness,
            "Top Freezer Handle Length": top_freezer_handle_length,
            "Top Freezer Handle Y Offset": main_handle_y_offset,
            "Top Freezer Handle Z Offset": top_freezer_handle_z_offset,
            "Top Freezer Num Shelves": randint(0, 2),
            "Top Freezer Num Door Shelves": randint(0, 2),
            "Bottom Freezer Num Drawers": num_freezers,
            "Handle Type": handle_type,
            "Handle Location": randint(0, 2),
            "Handle Orientation": 0,
            "Handle Length": main_handle_length,
            "Handle Y Offset": main_handle_y_offset,
            "Handle Z Offset": main_handle_z_offset,
            "Handle Protrude": uniform(0.01, 0.1),
            "Handle Radius": handle_radius,
            "Handle Width": handle_width,
            "Handle Height": handle_height,
            "Door Thickness": uniform(0.03, 0.1),
            "Shelf Height": fridge_shelf_height,
            "Shelf Border Thickness": uniform(0.005, 0.02),
            "Door Shelf Height": door_shelf_height,
            "Door Shelf Depth": max(0.03, door_to_shelf_gap - 0.03),
            "Door Shelf Thickness": door_shelf_thickness,
            "Main Fridge Has Grated Shelves": np.random.choice([True, False]),
            "Top Freezer Has Grated Shelves": np.random.choice([True, False]),
            "Outer Shell Material": outer_body_material,
            "Inner Shell Material": inner_body_material,
            "Handle Material": handle_material,
            "Door Material": outer_body_material,
            "Door Inner Material": inner_body_material
            if uniform() < 0.8
            else outer_body_material,
            "Door Shelf Material": shelf_material,
            "Interior Shelf Material Border": inner_body_material,
            "Interior Shelf Material Inside": shelf_material,
            "Drawer Material": drawer_material
            if uniform() < 0.2
            else inner_body_material,
            "Exterior Drawer Shell Material": outer_body_material,
            "Exterior Drawer Inside Material": inner_body_material,
            "Exterior Drawer Material": outer_body_material,
            "Exterior Drawer Handle Material": handle_material,
            "Exterior Drawer Door Material": outer_body_material,
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
