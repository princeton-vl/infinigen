# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Yiming Zuo: primary author
# - Abhishek Joshi: updates for sim integration

import numpy as np
from numpy.random import uniform
import gin

from infinigen.assets.materials import metal, plastic
from infinigen.assets.objects.elements.doors.joint_utils import (
    nodegroup_hinge_joint,
    nodegroup_sliding_joint,
)
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import weighted_sample


def nodegroup_carriage_flat(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    curve_line_1 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (0.0000, 0.0000, 0.0500),
            "End": (0.0000, 0.0000, -0.0500),
        },
    )

    quadrilateral = nw.new_node("GeometryNodeCurvePrimitiveQuadrilateral")

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": quadrilateral,
            "Translation": (1.0000, 0.0000, 0.0000),
        },
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": transform_geometry_1, "Count": 16}
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": resample_curve,
            "Count": 8,
            "Radius": 0.1000,
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    resample_curve_1 = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": fillet_curve, "Count": 64}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": resample_curve_1}
    )

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry,
            "Scale": (0.7500, 0.7500, 1.0000),
        },
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line_1,
            "Profile Curve": transform_geometry_7,
            "Fill Caps": True,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": curve_to_mesh_1},
        attrs={"is_active_output": True},
    )


def nodegroup_carriage_sphere(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    uv_sphere = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Segments": 64, "Rings": 32}
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere.outputs["Mesh"],
            "Translation": (0.6000, 0.0000, 0.0000),
            "Scale": (0.6000, 0.6000, 0.6000),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_3},
        attrs={"is_active_output": True},
    )


def nodegroup_carriage_eroded_sphere(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    uv_sphere = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Segments": 64, "Rings": 32}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere.outputs["Mesh"],
            "Translation": (0.1800, 0.0000, 0.0000),
            "Scale": (0.8000, 0.6000, 0.5000),
        },
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (1.0000, 2.0000, 2.0000)})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Translation": (-0.5000, 0.0000, 0.0000),
        },
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere.outputs["Mesh"],
            "Translation": (0.9600, 0.0000, 1.0600),
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 1": transform_geometry,
            "Mesh 2": [transform_geometry_1, transform_geometry_2],
        },
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": difference.outputs["Mesh"],
            "Scale": (1.0000, 1.2000, 0.6000),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_3},
        attrs={"is_active_output": True},
    )


def nodegroup_carriage_cylider(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 16, "Side Segments": 12, "Fill Segments": 2},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": (0.2000, 0.0000, 0.0000),
            "Rotation": (1.5708, 0.0000, 0.0000),
            "Scale": (0.2000, 0.2000, 0.8000),
        },
    )

    subdivision_surface = nw.new_node(
        Nodes.SubdivisionSurface, input_kwargs={"Mesh": transform_geometry}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": subdivision_surface},
        attrs={"is_active_output": True},
    )


def nodegroup_carriage_half_cylinder(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Side Segments": 12, "Fill Segments": 2},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": (0.5000, 0.0000, 0.0000),
            "Scale": (0.5000, 0.5000, 0.0800),
        },
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": separate_xyz.outputs["X"], 3: 1.0000, 4: 0.4000},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": 1.0000, "Y": 1.0000, "Z": map_range.outputs["Result"]},
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: position, 1: combine_xyz},
        attrs={"operation": "MULTIPLY"},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": transform_geometry,
            "Position": multiply.outputs["Vector"],
        },
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": set_position,
            "Translation": (-0.5100, 0.0000, 0.0000),
            "Scale": (1.5000, 1.5000, 2.0000),
        },
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (1.0000, 2.0000, 2.0000)})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Translation": (-0.5000, 0.0000, 0.0000),
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": transform_geometry_1, "Mesh 2": transform_geometry_2},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": difference.outputs["Mesh"]},
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

    instance_on_points_1 = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": group_input.outputs["Points"],
            "Instance": group_input.outputs["Child"],
        },
    )

    reroute = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Duplicate ID (do not set)"]},
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    index_1 = nw.new_node(Nodes.Index)

    add = nw.new_node(Nodes.Math, input_kwargs={0: index_1, 1: 1.0000})

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": instance_on_points_1,
            "Name": reroute_1,
            "Value": add,
        },
        attrs={"data_type": "INT", "domain": "INSTANCE"},
    )

    realize_instances_1 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": store_named_attribute_1}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Parent"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [realize_instances_1, reroute_3]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_toaster", singleton=False, type="GeometryNodeTree")
def nodegroup_toaster(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    string = nw.new_node("FunctionNodeInputString", attrs={"string": "duplicate0"})

    string_1 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint0"})

    string_2 = nw.new_node("FunctionNodeInputString", attrs={"string": "duplicate1"})

    string_3 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint1"})

    string_4 = nw.new_node("FunctionNodeInputString", attrs={"string": "duplicate2"})

    string_5 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint3"})

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "carriage_object", None),
            ("NodeSocketInt", "num_slots", 2),
            ("NodeSocketVector", "carriage_dimensions", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "slot width", 0.0000),
            ("NodeSocketFloat", "slot length", 0.0000),
            ("NodeSocketFloat", "slot depth", 0.0000),
            ("NodeSocketBool", "double slots", False),
            ("NodeSocketFloat", "toaster length", 0.0000),
            ("NodeSocketFloat", "knob vertical offset", -0.6500),
            ("NodeSocketFloat", "knob horizontal offset", 0.0000),
            ("NodeSocketFloat", "knob size", 0.0000),
            ("NodeSocketFloat", "button size", 0.0000),
            ("NodeSocketFloat", "button width", 0.0000),
            ("NodeSocketInt", "num_buttons", 0),
            ("NodeSocketFloat", "button vertical interval", 0.5000),
            ("NodeSocketFloat", "button horizontal offset", 0.0000),
            ("NodeSocketFloat", "button vertical offset", -0.4000),
            ("NodeSocketBool", "base alternative style", False),
            ("NodeSocketFloat", "base side shape param", 0.4600),
            ("NodeSocketMaterial", "body mat 1", None),
            ("NodeSocketMaterial", "body mat 2", None),
            ("NodeSocketMaterial", "button mat", None),
            ("NodeSocketMaterial", "knob mat", None),
            ("NodeSocketMaterial", "carriage mat", None),
        ],
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["num_slots"]}
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_3}, attrs={"operation": "MULTIPLY"}
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: 0.2500})

    map_range_4 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": group_input.outputs["base alternative style"],
            3: add,
            4: 1.6000,
        },
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": map_range_4.outputs["Result"]}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": reroute_1})

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_2, "End": combine_xyz_1}
    )

    resample_curve_2 = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": curve_line, "Count": 128}
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_1})

    absolute = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"]},
        attrs={"operation": "ABSOLUTE"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: reroute_1},
        attrs={"operation": "SUBTRACT"},
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: absolute, 1: subtract}, attrs={"use_clamp": True}
    )

    float_curve = nw.new_node(Nodes.FloatCurve, input_kwargs={"Value": add_1})
    node_utils.assign_curve(
        float_curve.mapping.curves[0],
        [(0.0000, 1.0000), (0.8886, 0.9375), (1.0000, 0.0281)],
    )

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius,
        input_kwargs={"Curve": resample_curve_2, "Radius": float_curve},
    )

    quadrilateral = nw.new_node("GeometryNodeCurvePrimitiveQuadrilateral")

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": quadrilateral,
            "Translation": (1.0000, 0.0000, 0.0000),
        },
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": transform_geometry_1, "Count": 16}
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": resample_curve,
            "Count": 8,
            "Radius": 0.1000,
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    resample_curve_1 = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": fillet_curve, "Count": 128}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": resample_curve_1}
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    power = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz.outputs["X"],
            1: group_input.outputs["base side shape param"],
        },
        attrs={"operation": "POWER"},
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": power,
            3: 1.0000,
            4: group_input.outputs["toaster length"],
        },
        attrs={"clamp": False},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": 1.0000, "Y": map_range.outputs["Result"], "Z": 1.0000},
    )

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: position},
        attrs={"operation": "MULTIPLY"},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": transform_geometry,
            "Position": multiply_2.outputs["Vector"],
        },
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": set_position,
            "Translation": (0.0000, -2.0000, 0.0000),
            "Rotation": (0.0000, 0.0000, 1.5708),
        },
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["num_slots"], 1: 0.4500},
        attrs={"operation": "MULTIPLY"},
    )

    map_range_3 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": group_input.outputs["base alternative style"],
            3: 1.0000,
            4: multiply_3,
        },
    )

    combine_xyz_12 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": map_range_3.outputs["Result"], "Y": 1.0000, "Z": 1.0000},
    )

    transform_geometry_13 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_2, "Scale": combine_xyz_12},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius,
            "Profile Curve": transform_geometry_13,
            "Fill Caps": True,
        },
    )

    store_named_attribute_6 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": curve_to_mesh, "Name": "switch1"},
        attrs={"data_type": "INT"},
    )

    transform_geometry_12 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": curve_to_mesh, "Rotation": (0.0000, 0.0000, 1.5708)},
    )

    store_named_attribute_7 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_12, "Name": "switch1", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["base alternative style"],
            "False": curve_to_mesh,
            "True": transform_geometry_12,
        },
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add_1})

    position_2 = nw.new_node(Nodes.InputPosition)

    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_2})

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: 0.6000},
        attrs={"operation": "MULTIPLY"},
    )

    absolute_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_4}, attrs={"operation": "ABSOLUTE"}
    )

    map_range_7 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": group_input.outputs["base alternative style"],
            3: reroute_5,
            4: absolute_1,
        },
    )

    less_than = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: map_range_7.outputs["Result"], 1: 0.8000},
        attrs={"operation": "LESS_THAN"},
    )

    set_material_4 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": switch_1,
            "Selection": less_than,
            "Material": group_input.outputs["body mat 1"],
        },
    )

    op_not = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: less_than}, attrs={"operation": "NOT"}
    )

    set_material_5 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": set_material_4,
            "Selection": op_not,
            "Material": group_input.outputs["body mat 2"],
        },
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["double slots"]}
    )

    index = nw.new_node(Nodes.Index)

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_3, 2: -0.5000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: index, 1: multiply_add},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": subtract_1})

    points = nw.new_node(
        "GeometryNodePoints",
        input_kwargs={"Count": reroute_3, "Position": combine_xyz_3},
    )

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": points, "Name": "switch0"},
        attrs={"data_type": "INT"},
    )

    index_2 = nw.new_node(Nodes.Index)

    map_range_1 = nw.new_node(
        Nodes.MapRange, input_kwargs={"Value": index_2, 3: -0.2500, 4: 0.2500}
    )

    combine_xyz_11 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": map_range_1.outputs["Result"]}
    )

    points_2 = nw.new_node(
        "GeometryNodePoints", input_kwargs={"Count": 2, "Position": combine_xyz_11}
    )

    instance_on_points_3 = nw.new_node(
        Nodes.InstanceOnPoints, input_kwargs={"Points": points, "Instance": points_2}
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points_3}
    )

    store_named_attribute_4 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": realize_instances, "Name": "switch0", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_4,
            "False": points,
            "True": realize_instances,
        },
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": switch, "Translation": (0.0000, 0.0000, 2.0000)},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["slot depth"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["slot width"],
            "Y": group_input.outputs["slot length"],
            "Z": multiply_5,
        },
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_4,
            "Vertices X": 10,
            "Vertices Y": 10,
            "Vertices Z": 10,
        },
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={"Points": transform_geometry_3, "Instance": cube.outputs["Mesh"]},
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": set_material_5, "Mesh 2": instance_on_points},
    )

    map_range_6 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": group_input.outputs["base alternative style"],
            3: group_input.outputs["toaster length"],
            4: 1.5000,
        },
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": map_range_6.outputs["Result"], "Z": 1.0000}
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": points,
            "Translation": combine_xyz_7,
            "Scale": (1.0000, 1.0000, 0.0000),
        },
    )

    carriage_slit_width = nw.new_node(Nodes.Value, label="carriage slit width")
    carriage_slit_width.outputs[0].default_value = 0.0650

    carriage_slit_depth = nw.new_node(Nodes.Value, label="carriage slit depth")
    carriage_slit_depth.outputs[0].default_value = 0.2000

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: carriage_slit_depth, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    carriage_slit_height = nw.new_node(Nodes.Value, label="carriage slit height")
    carriage_slit_height.outputs[0].default_value = 1.2000

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": carriage_slit_width,
            "Y": multiply_6,
            "Z": carriage_slit_height,
        },
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_5,
            "Vertices X": 10,
            "Vertices Y": 10,
            "Vertices Z": 10,
        },
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Translation": (0.0000, 0.0000, 0.2000),
        },
    )

    instance_on_points_1 = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={"Points": transform_geometry_4, "Instance": transform_geometry_5},
    )

    difference_1 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 1": difference.outputs["Mesh"],
            "Mesh 2": instance_on_points_1,
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": difference_1.outputs["Mesh"],
            "Label": "toaster_body",
        },
    )

    store_named_attribute_8 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": add_jointed_geometry_metadata, "Name": "joint3"},
        attrs={"data_type": "INT"},
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 16, "Side Segments": 32, "Radius": 0.1000},
    )

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": (1.0000, 0.0000, 0.0000),
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    store_named_attribute_9 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_6, "Name": "joint4"},
        attrs={"data_type": "INT"},
    )

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": group_input.outputs["carriage_object"],
            "Translation": (2.0000, 0.0000, 0.0000),
            "Scale": group_input.outputs["carriage_dimensions"],
        },
    )

    store_named_attribute_10 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_7, "Name": "joint4", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_6, transform_geometry_7]},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": map_range_6.outputs["Result"]}
    )

    subtract_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_2}, attrs={"operation": "SUBTRACT"}
    )

    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": subtract_2})

    transform_geometry_8 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry,
            "Translation": combine_xyz_9,
            "Rotation": (0.0000, 0.0000, 1.5708),
            "Scale": (0.3000, 0.3000, 0.3000),
        },
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_8,
            "Material": group_input.outputs["carriage mat"],
        },
    )

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_2, "Label": "slider"},
    )

    store_named_attribute_11 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_1,
            "Name": "joint3",
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    hinge_joint_005 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "slider_joint",
            "Parent": add_jointed_geometry_metadata,
            "Child": add_jointed_geometry_metadata_1,
            "Position": (0.0000, -2.0000, 0.0000),
            "Axis": (-1.0000, 0.0000, 0.0000),
            "Min": -0.300,
            "Max": 0.15000,
        },
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": points})

    duplicate_joints_on_parent = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Parent": hinge_joint_005.outputs["Parent"],
            "Child": hinge_joint_005.outputs["Child"],
            "Points": reroute,
        },
    )

    store_named_attribute_12 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": duplicate_joints_on_parent, "Name": "joint1"},
        attrs={"data_type": "INT"},
    )

    cylinder_1 = nw.new_node("GeometryNodeMeshCylinder", input_kwargs={"Vertices": 64})

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": cylinder_1.outputs["Mesh"], "Name": "joint2"},
        attrs={"data_type": "INT"},
    )

    cube_2 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": (0.2000, 1.0000, 0.5000),
            "Vertices X": 3,
            "Vertices Y": 10,
            "Vertices Z": 5,
        },
    )

    transform_geometry_9 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_2.outputs["Mesh"],
            "Translation": (0.0000, 0.4800, 0.9500),
        },
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_9, "Name": "joint2", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [cylinder_1.outputs["Mesh"], transform_geometry_9]},
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["knob horizontal offset"],
            "Y": reroute_2,
            "Z": group_input.outputs["knob vertical offset"],
        },
    )

    transform_geometry_10 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Translation": combine_xyz_8,
            "Rotation": (1.5708, 0.0000, 3.1416),
            "Scale": group_input.outputs["knob size"],
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_10,
            "Material": group_input.outputs["knob mat"],
        },
    )

    add_jointed_geometry_metadata_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material, "Label": "knob"},
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_2,
            "Name": "joint1",
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    hinge_joint_005_1 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "knob_joint",
            "Parent": duplicate_joints_on_parent,
            "Child": add_jointed_geometry_metadata_2,
            "Axis": (0.0000, 1.0000, 0.0000),
            "Min": -2.5000,
            "Max": 2.5000,
        },
    )

    duplicate_joints_on_parent_1 = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Parent": hinge_joint_005_1.outputs["Parent"],
            "Child": hinge_joint_005_1.outputs["Child"],
            "Points": reroute,
        },
    )

    store_named_attribute_13 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": duplicate_joints_on_parent_1, "Name": "joint0"},
        attrs={"data_type": "INT"},
    )

    curve_line_2 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (0.0000, 0.0000, -2.0000),
            "End": (0.0000, 0.0000, 2.5000),
        },
    )

    resample_curve_3 = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": curve_line_2, "Count": 128}
    )

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    float_curve_1 = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": spline_parameter.outputs["Factor"]}
    )
    node_utils.assign_curve(
        float_curve_1.mapping.curves[0],
        [(0.0000, 1.0000), (0.9250, 0.9281), (1.0000, 0.0000)],
    )

    set_curve_radius_1 = nw.new_node(
        Nodes.SetCurveRadius,
        input_kwargs={"Curve": resample_curve_3, "Radius": float_curve_1},
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["button width"]}
    )

    quadrilateral_1 = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral", input_kwargs={"Width": reroute_6}
    )

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_6, 1: 2.0000})

    multiply_7 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_2, 1: 2.0000}, attrs={"operation": "MULTIPLY"}
    )

    resample_curve_4 = nw.new_node(
        Nodes.ResampleCurve,
        input_kwargs={"Curve": quadrilateral_1, "Count": multiply_7},
    )

    fillet_curve_1 = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": resample_curve_4,
            "Count": 8,
            "Radius": 1.0000,
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    resample_curve_5 = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": fillet_curve_1, "Count": 64}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 2.0000, 1: reroute_6},
        attrs={"operation": "DIVIDE"},
    )

    transform_geometry_14 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": resample_curve_5, "Scale": divide}
    )

    curve_to_mesh_2 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius_1,
            "Profile Curve": transform_geometry_14,
            "Fill Caps": True,
        },
    )

    combine_xyz_10 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": reroute_2})

    transform_geometry_11 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_to_mesh_2,
            "Translation": combine_xyz_10,
            "Rotation": (-1.5708, 0.0000, 0.0000),
            "Scale": group_input.outputs["button size"],
        },
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_11,
            "Material": group_input.outputs["button mat"],
        },
    )

    add_jointed_geometry_metadata_3 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_1, "Label": "button"},
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_3,
            "Name": "joint0",
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    sliding_joint_003 = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "button_joint",
            "Parent": duplicate_joints_on_parent_1,
            "Child": add_jointed_geometry_metadata_3,
            "Axis": (0.0000, -1.0000, 0.0000),
            "Max": 0.1000,
        },
    )

    index_1 = nw.new_node(Nodes.Index)

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: map_range_6.outputs["Result"], 1: 1.0000},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["base side shape param"], 1: -0.3500},
        attrs={"operation": "MULTIPLY"},
    )

    map_range_5 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": group_input.outputs["base alternative style"],
            3: multiply_8,
            4: 0.0000,
        },
    )

    multiply_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_3, 1: map_range_5.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: index_1, 1: multiply_9},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_11 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: index_1, 1: group_input.outputs["button vertical interval"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_11, 1: group_input.outputs["button vertical offset"]},
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["button horizontal offset"],
            "Y": multiply_10,
            "Z": add_3,
        },
    )

    points_1 = nw.new_node(
        "GeometryNodePoints",
        input_kwargs={
            "Count": group_input.outputs["num_buttons"],
            "Position": combine_xyz_6,
        },
    )

    instance_on_points_2 = nw.new_node(
        Nodes.InstanceOnPoints, input_kwargs={"Points": points_1, "Instance": reroute}
    )

    realize_instances_1 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points_2}
    )

    duplicate_joints_on_parent_2 = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Parent": sliding_joint_003.outputs["Parent"],
            "Child": sliding_joint_003.outputs["Child"],
            "Points": realize_instances_1,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": duplicate_joints_on_parent_2},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "num_slots", 2),
            ("NodeSocketFloat", "slot width", 0.0000),
            ("NodeSocketFloat", "slot length", 0.0000),
            ("NodeSocketFloat", "slot depth", 0.0000),
            ("NodeSocketBool", "double slots", False),
            ("NodeSocketFloat", "toaster length", 0.0000),
            ("NodeSocketFloat", "knob vertical offset", -0.6500),
            ("NodeSocketFloat", "knob size", 0.0000),
            ("NodeSocketFloat", "button size", 0.0000),
            ("NodeSocketInt", "num_buttons", 0),
            ("NodeSocketFloat", "button vertical interval", 0.5000),
            ("NodeSocketFloat", "button horizontal offset", 0.0000),
            ("NodeSocketFloat", "button vertical offset", -0.4000),
            ("NodeSocketBool", "base alternative style", False),
            ("NodeSocketFloat", "base side shape param", 0.4600),
            ("NodeSocketMaterial", "body mat", None),
            ("NodeSocketMaterial", "button mat", None),
            ("NodeSocketMaterial", "knob mat", None),
            ("NodeSocketMaterial", "carriage mat", None),
        ],
    )

    toaster = nw.new_node(
        nodegroup_toaster().name,
        input_kwargs={
            "num_slots": 3,
            "slot width": 0.2500,
            "slot length": 1.6600,
            "slot depth": 1.1600,
            "toaster length": 1.2000,
            "knob vertical offset": -0.6600,
            "knob size": 0.2000,
            "button size": 0.1100,
            "num_buttons": 3,
            "button vertical interval": 0.4000,
            "button horizontal offset": -0.3500,
            "button vertical offset": -0.2000,
            "base alternative style": True,
            "base side shape param": 0.2800,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": toaster},
        attrs={"is_active_output": True},
    )


class ToasterFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False, use_transparent_mat=False):
        super().__init__(factory_seed=factory_seed, coarse=coarse)
        self.use_transparent_mat = use_transparent_mat

    @classmethod
    @gin.configurable(module='ToasterFactory')
    def sample_joint_parameters(
        cls,
        slider_joint_stiffness_min: float = 8000.0,
        slider_joint_stiffness_max: float = 12000.0,
        slider_joint_damping_min: float = 1500.0,
        slider_joint_damping_max: float = 2500.0,
        knob_joint_stiffness_min: float = 0.0,
        knob_joint_stiffness_max: float = 0.0,
        knob_joint_damping_min: float = 0.0,
        knob_joint_damping_max: float = 10.0,
        button_joint_stiffness_min: float = 100.0,
        button_joint_stiffness_max: float = 150.0,
        button_joint_damping_min: float = 100.0,
        button_joint_damping_max: float = 150.0
    ):
        return {
            "slider_joint": {
                "stiffness": uniform(slider_joint_stiffness_min, slider_joint_stiffness_max),
                "damping": uniform(slider_joint_damping_min, slider_joint_damping_max),
            },
            "knob_joint": {
                "stiffness": uniform(knob_joint_stiffness_min, knob_joint_stiffness_max),
                "damping": uniform(knob_joint_damping_min, knob_joint_damping_max),
            },
            "button_joint": {
                "stiffness": uniform(button_joint_stiffness_min, button_joint_stiffness_max),
                "damping": uniform(button_joint_damping_min, button_joint_damping_max),
            },
        }

    def sample_parameters(self):
        # add code here to randomly sample from parameters
        with FixedSeed(self.factory_seed):
            toaster_length = uniform(1.2, 1.6)

            toaster_materials = (
                (metal.MetalBasic, 2.0),
                (metal.BrushedMetal, 2.0),
                (metal.GalvanizedMetal, 2.0),
                (metal.BrushedBlackMetal, 2.0),
                (plastic.Plastic, 1.0),
                (plastic.PlasticRough, 1.0),
            )

            self.body_shader_1 = weighted_sample(toaster_materials)()
            self.body_shader_2 = weighted_sample(toaster_materials)()
            self.button_shader = weighted_sample(toaster_materials)()
            self.knob_shader = weighted_sample(toaster_materials)()
            self.carriage_shader = weighted_sample(toaster_materials)()

            body_mat_1 = self.body_shader_1.generate()
            body_mat_2 = body_mat_1
            if uniform() < 0.5:
                body_mat_2 = self.body_shader_2.generate()
            button_mat = self.button_shader.generate()
            knob_mat = button_mat
            carriage_mat = button_mat
            if uniform() < 0.5:
                knob_mat = self.knob_shader.generate()
                carriage_mat = self.carriage_shader.generate()

            button_side = np.random.choice([-1.0, 1.0])
            knob_side = np.random.choice(
                [0.0, -button_side]
            )  # should be different side or center

            # create carriage obj
            carriage_style = np.random.choice(
                [
                    nodegroup_carriage_cylider,
                    nodegroup_carriage_half_cylinder,
                    nodegroup_carriage_eroded_sphere,
                    nodegroup_carriage_sphere,
                    nodegroup_carriage_flat,
                ]
            )

            self.carriage_obj = butil.spawn_vert()
            surface.add_geomod(
                self.carriage_obj, carriage_style, apply=True, input_kwargs={}
            )

            return {
                "num_slots": np.random.choice([1, 2, 3], p=[0.3, 0.6, 0.1]),
                "carriage_dimensions": (
                    uniform(0.8, 1.2),
                    0.6 * uniform(0.8, 1.2),
                    uniform(0.8, 1.2),
                ),
                "slot width": uniform(0.25, 0.4),
                "slot length": uniform(1.2, 1.5) * toaster_length,
                "slot depth": uniform(1.2, 1.8),
                "double slots": np.random.choice([True, False]),
                "toaster length": toaster_length,
                "knob vertical offset": uniform(-0.63, -0.67)
                + np.abs(knob_side) * uniform(0.0, 1.1),
                "knob horizontal offset": uniform(0.28, 0.35) * knob_side,
                "knob size": uniform(0.12, 0.15),
                "button size": uniform(0.09, 0.11),
                "button width": uniform(2.0, 5.0),
                "num_buttons": np.random.choice([1, 2, 3]),
                "button vertical interval": uniform(0.28, 0.4),
                "button horizontal offset": uniform(0.28, 0.35) * button_side,
                "button vertical offset": uniform(-0.65, -0.20),
                "base alternative style": np.random.choice([True, False]),
                "base side shape param": uniform(0.1, 0.5),
                "body mat 1": body_mat_1,
                "body mat 2": body_mat_2,
                "button mat": button_mat,
                "knob mat": knob_mat,
                "carriage mat": carriage_mat,
            }

    def create_asset(self, export=True, exporter="mjcf", asset_params=None, **kwargs):
        ng_input = self.sample_parameters()
        obj = self.carriage_obj

        butil.modify_mesh(
            obj,
            "NODES",
            apply=False,
            node_group=nodegroup_toaster(),
            ng_inputs=ng_input,
        )

        return obj
