# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Jack Nugent: primary author
# - Abhishek Joshi: Updates for sim
# - Max Gonzalez Saez-Diez: Updates for sim
# - Hongyu Wen: developed original lamp

import functools

import gin
import numpy as np

from infinigen.assets.composition import material_assignments
from infinigen.assets.materials import ceramic, fabric, metal, plastic
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
    "nodegroup_bulb_003_gp", singleton=False, type="GeometryNodeTree"
)
def nodegroup_bulb_003_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    curve_line = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (0.0000, 0.0000, -0.2000),
            "End": (0.0000, 0.0000, 0.0000),
        },
    )

    integer = nw.new_node(Nodes.Integer)
    integer.integer = 8

    curve_circle = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": integer, "Radius": 0.1500}
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    spiral = nw.new_node(
        "GeometryNodeCurveSpiral",
        input_kwargs={
            "Resolution": integer,
            "Rotations": 5.0000,
            "Start Radius": 0.1500,
            "End Radius": 0.1500,
            "Height": 0.2000,
        },
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": spiral, "Translation": (0.0000, 0.0000, -0.2000)},
    )

    curve_circle_1 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": integer, "Radius": 0.0150}
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": transform_geometry,
            "Profile Curve": curve_circle_1.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    curve_line_1 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (0.0000, 0.0000, -0.2000),
            "End": (0.0000, 0.0000, -0.3000),
        },
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": curve_line_1, "Count": integer}
    )

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    float_curve = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": spline_parameter.outputs["Factor"]}
    )
    node_utils.assign_curve(
        float_curve.mapping.curves[0],
        [(0.0000, 1.0000), (0.4432, 0.5500), (1.0000, 0.2750)],
        handles=["AUTO", "VECTOR", "AUTO"],
    )

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius,
        input_kwargs={"Curve": resample_curve, "Radius": float_curve},
    )

    curve_circle_2 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": integer, "Radius": 0.1500}
    )

    curve_to_mesh_2 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius,
            "Profile Curve": curve_circle_2.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [curve_to_mesh, curve_to_mesh_1, curve_to_mesh_2]},
    )

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketMaterial", "LampshadeMaterial", None),
            ("NodeSocketMaterial", "MetalMaterial", None),
        ],
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": join_geometry,
            "Material": group_input.outputs["MetalMaterial"],
        },
    )

    curve_line_2 = nw.new_node(Nodes.CurveLine)

    resample_curve_1 = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": curve_line_2, "Count": integer}
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": resample_curve_1})

    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)

    float_curve_1 = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": spline_parameter_1.outputs["Factor"]}
    )
    node_utils.assign_curve(
        float_curve_1.mapping.curves[0],
        [
            (0.0000, 0.1500),
            (0.0500, 0.1700),
            (0.1500, 0.2000),
            (0.5500, 0.3800),
            (0.8000, 0.3500),
            (0.9568, 0.2200),
            (1.0000, 0.0000),
        ],
    )

    set_curve_radius_1 = nw.new_node(
        Nodes.SetCurveRadius, input_kwargs={"Curve": reroute, "Radius": float_curve_1}
    )

    curve_circle_3 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": integer}
    )

    curve_to_mesh_3 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius_1,
            "Profile Curve": curve_circle_3.outputs["Curve"],
        },
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": curve_to_mesh_3,
            "Material": group_input.outputs["LampshadeMaterial"],
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, set_material_1]}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Translation": (0.0000, 0.0000, 0.3000),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_reversiable_bulb_003_gp", singleton=False, type="GeometryNodeTree"
)
def nodegroup_reversiable_bulb_003_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Scale", 0.3000),
            ("NodeSocketBool", "Reverse", False),
            ("NodeSocketMaterial", "BlackMaterial", None),
            ("NodeSocketMaterial", "LampshadeMaterial", None),
            ("NodeSocketMaterial", "MetalMaterial", None),
        ],
    )

    bulb_003_gp = nw.new_node(
        nodegroup_bulb_003_gp().name,
        input_kwargs={
            "LampshadeMaterial": group_input.outputs["LampshadeMaterial"],
            "MetalMaterial": group_input.outputs["MetalMaterial"],
        },
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Scale"],
            "Y": group_input.outputs["Scale"],
            "Z": group_input.outputs["Scale"],
        },
    )

    transform_geometry = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": bulb_003_gp, "Scale": combine_xyz}
    )

    geometry_to_instance = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": transform_geometry}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Reverse"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: 3.1415},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply})

    rotate_instances = nw.new_node(
        Nodes.RotateInstances,
        input_kwargs={"Instances": geometry_to_instance, "Rotation": combine_xyz_1},
    )

    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_2, 1: 2.0000, 2: -1.0000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: -0.0150, 1: multiply_add},
        attrs={"operation": "MULTIPLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": rotate_instances, "RackSupport": multiply_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_bulb_rack_003_gp", singleton=False, type="GeometryNodeTree"
)
def nodegroup_bulb_rack_003_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    curve_line = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (-1.0000, 0.0000, 0.0000),
            "End": (1.0000, 0.0000, 0.0000),
        },
    )

    geometry_to_instance = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": curve_line}
    )

    amount = nw.new_node(
        Nodes.GroupInput,
        label="amount",
        expose_input=[
            ("NodeSocketFloat", "Thickness", 0.0200),
            ("NodeSocketInt", "Amount", 3),
            ("NodeSocketFloat", "InnerRadius", 1.0000),
            ("NodeSocketFloat", "OuterRadius", 1.0000),
            ("NodeSocketFloat", "InnerHeight", 0.0000),
            ("NodeSocketFloat", "OuterHeight", 0.0000),
            ("NodeSocketBool", "ShadeTop", False),
            ("NodeSocketFloat", "Sides", 0.0000),
        ],
    )

    less_than = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: amount.outputs["Sides"], 3: 10},
        attrs={"operation": "LESS_THAN", "data_type": "INT"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": amount.outputs["Amount"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": amount.outputs["Sides"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": less_than, "False": reroute_3, "True": reroute_1},
        attrs={"input_type": "INT"},
    )

    duplicate_elements = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={"Geometry": geometry_to_instance, "Amount": switch},
        attrs={"domain": "INSTANCE"},
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": duplicate_elements.outputs["Geometry"]}
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": reroute_13}
    )

    endpoint_selection = nw.new_node(
        Nodes.EndpointSelection, input_kwargs={"Start Size": 0}
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={
            "Resolution": amount.outputs["Sides"],
            "Radius": amount.outputs["OuterRadius"],
        },
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": curve_circle.outputs["Curve"]}
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": amount.outputs["OuterHeight"]}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_4})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_9, "Translation": combine_xyz},
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: 1.0000, 1: switch}, attrs={"operation": "DIVIDE"}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: divide},
        attrs={"operation": "MULTIPLY"},
    )

    sample_curve = nw.new_node(
        Nodes.SampleCurve,
        input_kwargs={"Curves": reroute_12, "Factor": multiply},
        attrs={"use_all_curves": True},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": realize_instances,
            "Selection": endpoint_selection,
            "Position": sample_curve.outputs["Position"],
        },
    )

    endpoint_selection_1 = nw.new_node(
        Nodes.EndpointSelection, input_kwargs={"End Size": 0}
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: amount.outputs["Thickness"], 2: amount.outputs["InnerRadius"]},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    curve_circle_1 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 8, "Radius": multiply_add}
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": amount.outputs["InnerHeight"]}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_8})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle_1.outputs["Curve"],
            "Translation": combine_xyz_1,
        },
    )

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_1}
    )

    sample_curve_1 = nw.new_node(
        Nodes.SampleCurve,
        input_kwargs={"Curves": reroute_10, "Factor": multiply},
        attrs={"use_all_curves": True},
    )

    reroute_18 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": sample_curve_1.outputs["Position"]}
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position,
            "Selection": endpoint_selection_1,
            "Position": reroute_18,
        },
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [set_position_1, reroute_15, reroute_16]},
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": amount.outputs["Thickness"]}
    )

    curve_circle_2 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 8, "Radius": reroute_7}
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": join_geometry,
            "Profile Curve": curve_circle_2.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": curve_to_mesh}
    )

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": amount.outputs["ShadeTop"]}
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    multiply_add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: amount.outputs["OuterRadius"], 1: 1.0000, 2: 0.0000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    multiply_add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: amount.outputs["Thickness"], 1: 1.0000, 2: 0.0010},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": reroute_1,
            "Radius": multiply_add_1,
            "Depth": multiply_add_2,
        },
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cylinder.outputs["Mesh"], "Translation": combine_xyz},
    )

    switch_1 = nw.new_node(
        Nodes.Switch, input_kwargs={"Switch": reroute_6, "True": transform_geometry_2}
    )

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_1})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_1, "LampShadeTop": reroute_17},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_015_gp",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_015_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": group_input.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_016_gp",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_016_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": group_input.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_014_gp",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_014_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": group_input.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_012_gp",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_012_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": group_input.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_013_gp",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_013_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": group_input.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_011_gp",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_011_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": group_input.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_010_gp",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_010_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": group_input.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_string_nodes_v2_gp", singleton=False, type="GeometryNodeTree"
)
def nodegroup_string_nodes_v2_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    reroute_3 = nw.new_node(Nodes.Reroute)

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Radius", 0.0000),
            ("NodeSocketFloat", "RadialLength", 0.0000),
            ("NodeSocketFloat", "Length", 0.0000),
            ("NodeSocketFloat", "Depth", 0.0000),
            ("NodeSocketInt", "Resolution", 0),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": multiply,
            "Y": group_input.outputs["RadialLength"],
            "Z": group_input.outputs["Depth"],
        },
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_3})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["RadialLength"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_1})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Translation": combine_xyz_4},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [reroute_3, transform_geometry_1]}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["RadialLength"], 1: 1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_2})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Translation": combine_xyz_5},
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0150

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 2: value},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add})

    multiply_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: value, 1: 2.0000}, attrs={"operation": "MULTIPLY"}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Length"], 1: multiply_3},
        attrs={"operation": "DIVIDE"},
    )

    float_to_integer = nw.new_node(Nodes.FloatToInt, input_kwargs={"Float": divide})

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: float_to_integer, 1: multiply_3},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 2: multiply_4},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add_1})

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz, "End": combine_xyz_1}
    )

    curve_to_points = nw.new_node(
        Nodes.CurveToPoints,
        input_kwargs={"Curve": curve_line, "Count": float_to_integer},
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Resolution"]}
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_6, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    uv_sphere = nw.new_node(
        Nodes.MeshUVSphere,
        input_kwargs={"Segments": reroute_6, "Rings": divide_1, "Radius": value},
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": curve_to_points.outputs["Points"],
            "Instance": uv_sphere.outputs["Mesh"],
        },
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points}
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_3, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Length"], 1: -1.0000, 2: multiply_5},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add_2})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": realize_instances, "Translation": combine_xyz_2},
    )

    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": group_input.outputs["Resolution"],
            "Radius": group_input.outputs["Radius"],
            "Depth": group_input.outputs["Depth"],
        },
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cylinder.outputs["Mesh"]}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [reroute_4, transform_geometry_1]}
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_1, "Translation": combine_xyz_5},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Depth"]}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute, 1: multiply_3},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": transform_geometry_2,
            "Child": reroute_2,
            "Parent": transform_geometry_3,
            "Max String Dist": reroute_5,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_009_gp",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_009_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": group_input.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_gp", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    sqrt = nw.new_node(Nodes.Math, attrs={"operation": "SQRT"})

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 4, "Side Segments": 2, "Radius": sqrt},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Rotation": (0.0000, 0.0000, 0.7854),
        },
    )

    index = nw.new_node(Nodes.Index)

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: 4, 3: index},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": transform_geometry_2,
            "Selection": equal,
            "Offset": (0.0000, -0.4500, 0.0000),
        },
    )

    index_1 = nw.new_node(Nodes.Index)

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: 5, 3: index_1},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position,
            "Selection": equal_1,
            "Offset": (0.0000, -0.4500, 0.0000),
        },
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": set_position_1,
            "Translation": (0.0000, 0.5000, 0.5000),
            "Rotation": (1.5708, 1.5708, -1.5708),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_4},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_008_gp",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_008_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": group_input.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_007_gp",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_007_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": group_input.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_006_gp",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_006_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": group_input.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_005_gp",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_005_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": group_input.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_004_gp",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_004_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": group_input.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_003_gp",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_003_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": group_input.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_lamp_head_final_gp", singleton=False, type="GeometryNodeTree"
)
def nodegroup_lamp_head_final_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "ShadeHeight", 0.0000),
            ("NodeSocketFloat", "TopRadius", 0.3000),
            ("NodeSocketFloat", "BotRadius", 0.5000),
            ("NodeSocketBool", "ReverseBulb", True),
            ("NodeSocketFloat", "RackThickness", 0.0050),
            ("NodeSocketFloat", "RackHeight", 0.5000),
            ("NodeSocketMaterial", "BlackMaterial", None),
            ("NodeSocketMaterial", "LampshadeMaterial", None),
            ("NodeSocketMaterial", "MetalMaterial", None),
            ("NodeSocketMaterial", "LampShadeInteriorMaterial", None),
            ("NodeSocketBool", "ShadeTop", False),
            ("NodeSocketBool", "ShadeCurved", False),
            ("NodeSocketBool", "IncludeLightbulb", False),
            ("NodeSocketBool", "HeadSidways", False),
            ("NodeSocketGeometry", "string", None),
            ("NodeSocketInt", "Sides", 100),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["HeadSidways"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["RackThickness"]}
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["TopRadius"], 1: 0.8000},
        attrs={"operation": "MULTIPLY"},
    )

    maximum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: 0.0600},
        attrs={"operation": "MAXIMUM"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: maximum, 1: 0.1500},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["TopRadius"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_17 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BlackMaterial"]}
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_17})

    reroute_19 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["LampShadeInteriorMaterial"]},
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["MetalMaterial"]}
    )

    reversiable_bulb_003_gp = nw.new_node(
        nodegroup_reversiable_bulb_003_gp().name,
        input_kwargs={
            "Scale": maximum,
            "BlackMaterial": reroute_18,
            "LampshadeMaterial": reroute_20,
            "MetalMaterial": reroute_8,
        },
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["RackHeight"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["ReverseBulb"], 1: 2.0000, 2: -1.0000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_add})

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_7, 1: reroute_24},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ShadeTop"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_15 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Sides"]}
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_15})

    bulb_rack_003_gp = nw.new_node(
        nodegroup_bulb_rack_003_gp().name,
        input_kwargs={
            "Thickness": reroute_14,
            "InnerRadius": multiply_1,
            "OuterRadius": reroute_3,
            "InnerHeight": reversiable_bulb_003_gp.outputs["RackSupport"],
            "OuterHeight": multiply_2,
            "ShadeTop": reroute_5,
            "Sides": reroute_16,
        },
    )

    reroute_21 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["LampshadeMaterial"]}
    )

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": bulb_rack_003_gp.outputs["LampShadeTop"],
            "Material": reroute_22,
        },
    )

    reroute_36 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_2})

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["IncludeLightbulb"]}
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

    greater_than = nw.new_node(Nodes.Compare, input_kwargs={0: reroute_29})

    reroute_31 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": reversiable_bulb_003_gp.outputs["Geometry"]},
    )

    reroute_32 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_31})

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_32})

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_26})

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": bulb_rack_003_gp.outputs["Geometry"],
            "Material": reroute_27,
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material_3, reroute_32]}
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": greater_than,
            "False": reroute_35,
            "True": join_geometry_1,
        },
    )

    switch_2 = nw.new_node(
        Nodes.Switch, input_kwargs={"Switch": reroute_12, "True": switch_1}
    )

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_2})

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ShadeCurved"]}
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_2})

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["ShadeHeight"],
            1: group_input.outputs["RackHeight"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract})

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_add, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_23, 1: multiply_3},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_4})

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz, "End": combine_xyz_1}
    )

    b_zier_segment = nw.new_node(
        Nodes.CurveBezierSegment,
        input_kwargs={
            "Start": combine_xyz,
            "Start Handle": (0.0000, 0.0000, 0.0000),
            "End": combine_xyz_1,
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_10,
            "False": curve_line,
            "True": b_zier_segment,
        },
    )

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": spline_parameter.outputs["Factor"],
            3: group_input.outputs["TopRadius"],
            4: group_input.outputs["BotRadius"],
        },
    )

    reroute_25 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": map_range.outputs["Result"]}
    )

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius, input_kwargs={"Curve": switch, "Radius": reroute_25}
    )

    reroute_30 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    curve_circle = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": reroute_30}
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius,
            "Profile Curve": curve_circle.outputs["Curve"],
        },
    )

    set_shade_smooth = nw.new_node(
        Nodes.SetShadeSmooth,
        input_kwargs={"Geometry": curve_to_mesh, "Shade Smooth": False},
    )

    flip_faces = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": set_shade_smooth})

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    set_material = nw.new_node(
        Nodes.SetMaterial, input_kwargs={"Geometry": flip_faces, "Material": reroute_28}
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": set_shade_smooth,
            "Offset Scale": 0.0050,
            "Individual": False,
        },
    )

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    reroute_34 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_33})

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": extrude_mesh.outputs["Mesh"], "Material": reroute_34},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, set_material_1]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry,
            "Translation": (0.0000, 0.0000, 0.0010),
        },
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [reroute_36, reroute_37, transform_geometry]},
    )

    reroute_38 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_2})

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": reroute_38}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_2,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": transform_geometry_1}
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_1,
            "False": join_geometry_3,
            "True": join_geometry_4,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": switch_3},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_002_gp",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_002_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": group_input.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_001_gp",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_001_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": group_input.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_gp",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_gp(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketString", "Label", ""),
        ],
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": group_input.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketBool", "SingleHinge", True),
            ("NodeSocketFloat", "FirstBarLength", 1.0000),
            ("NodeSocketFloat", "FirstBarExtension", 0.0000),
            ("NodeSocketBool", "DoubleHinge", False),
            ("NodeSocketFloat", "SecondBarLength", 1.0000),
            ("NodeSocketFloat", "SecondBarExtension", 0.0000),
            ("NodeSocketBool", "SecondBarSliding", False),
            ("NodeSocketFloat", "Radius", 0.0300),
            ("NodeSocketFloat", "Height", 1.0000),
            ("NodeSocketFloat", "BaseRadius", 0.2000),
            ("NodeSocketFloat", "BaseHeight", 0.0600),
            ("NodeSocketInt", "BaseSides", 40),
            ("NodeSocketFloat", "BaseButtonOffset", 0.1000),
            ("NodeSocketBool", "BaseRotate", False),
            ("NodeSocketBool", "ButtonOnLampShade", True),
            ("NodeSocketBool", "HeadSideways", False),
            ("NodeSocketFloat", "ShadeHeight", 0.5000),
            ("NodeSocketFloat", "RackHeight", 0.1000),
            ("NodeSocketFloat", "TopRadius", 0.3000),
            ("NodeSocketFloat", "BottomRadius", 0.3000),
            ("NodeSocketBool", "ShadeCurved", False),
            ("NodeSocketFloat", "RackThickness", 0.0050),
            ("NodeSocketBool", "ShadeTop", True),
            ("NodeSocketBool", "ReverseBulb", True),
            ("NodeSocketBool", "IncludeLightBulb", False),
            ("NodeSocketInt", "ShadeSides", 40),
            ("NodeSocketInt", "ButtonType", 1),
            ("NodeSocketBool", "IncludeButtonBase", False),
            ("NodeSocketFloat", "ButtonR1", 0.0500),
            ("NodeSocketFloat", "ButtonR2", 0.0300),
            ("NodeSocketFloat", "ButtonH1", 0.0300),
            ("NodeSocketFloat", "ButtonH2", 0.0800),
            ("NodeSocketMaterial", "BaseMaterial", None),
            ("NodeSocketMaterial", "StandMaterial", None),
            ("NodeSocketMaterial", "ShadeMaterial", None),
            ("NodeSocketMaterial", "ShadeInteriorMaterial", None),
            ("NodeSocketMaterial", "ButtonMaterial", None),
            ("NodeSocketMaterial", "LampRackMaterial", None),
            ("NodeSocketMaterial", "MetalMaterial", None),
        ],
    )

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["DoubleHinge"]}
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["SingleHinge"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_36 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BaseRotate"]}
    )

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_36})

    reroute_38 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["IncludeButtonBase"]}
    )

    reroute_39 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_38})

    reroute_131 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_39})

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": group_input.outputs["BaseSides"],
            "Radius": group_input.outputs["BaseRadius"],
            "Depth": group_input.outputs["BaseHeight"],
        },
    )

    add_jointed_geometry_metadata_gp = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_gp().name,
        input_kwargs={"Geometry": cylinder.outputs["Mesh"], "Label": "lamp_base"},
    )

    reroute_40 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BaseMaterial"]}
    )

    reroute_41 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_40})

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_gp,
            "Material": reroute_41,
        },
    )

    reroute_81 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material})

    integer = nw.new_node(Nodes.Integer)
    integer.integer = 32

    reroute_207 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": integer})

    reroute_208 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_207})

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": reroute_208,
            "Radius": group_input.outputs["Radius"],
            "Depth": group_input.outputs["Height"],
        },
    )

    reroute_45 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cylinder_1.outputs["Mesh"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_45, "Translation": combine_xyz},
    )

    add_jointed_geometry_metadata_001_gp = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_001_gp().name,
        input_kwargs={"Geometry": transform_geometry, "Label": "bar"},
    )

    reroute_73 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_001_gp}
    )

    reroute_42 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["StandMaterial"]}
    )

    reroute_43 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_42})

    reroute_69 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_43})

    reroute_87 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_69})

    reroute_100 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_87})

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": reroute_73, "Material": reroute_100},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [reroute_81, set_material_1]}
    )

    reroute_28 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ButtonOnLampShade"]}
    )

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_28})

    reroute_114 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_29})

    lamp_head_final_gp = nw.new_node(
        nodegroup_lamp_head_final_gp().name,
        input_kwargs={
            "ShadeHeight": group_input.outputs["ShadeHeight"],
            "TopRadius": group_input.outputs["TopRadius"],
            "BotRadius": group_input.outputs["BottomRadius"],
            "ReverseBulb": group_input.outputs["ReverseBulb"],
            "RackThickness": group_input.outputs["RackThickness"],
            "RackHeight": group_input.outputs["RackHeight"],
            "BlackMaterial": group_input.outputs["LampRackMaterial"],
            "LampshadeMaterial": group_input.outputs["ShadeMaterial"],
            "MetalMaterial": group_input.outputs["MetalMaterial"],
            "LampShadeInteriorMaterial": group_input.outputs["ShadeInteriorMaterial"],
            "ShadeTop": group_input.outputs["ShadeTop"],
            "ShadeCurved": group_input.outputs["ShadeCurved"],
            "IncludeLightbulb": group_input.outputs["IncludeLightBulb"],
            "HeadSidways": group_input.outputs["HeadSideways"],
            "Sides": group_input.outputs["ShadeSides"],
        },
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": lamp_head_final_gp}
    )

    reroute_56 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": realize_instances})

    add_jointed_geometry_metadata_002_gp = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_002_gp().name,
        input_kwargs={"Geometry": reroute_56, "Label": "head"},
    )

    reroute_105 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_002_gp}
    )

    reroute_106 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_105})

    reroute_116 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_106})

    reroute_112 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_002_gp}
    )

    reroute_30 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ButtonType"]}
    )

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_30})

    reroute_78 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_31})

    reroute_79 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_78})

    reroute_92 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_79})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_92},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    reroute_115 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": equal})

    reroute_109 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_92})

    reroute_110 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_109})

    reroute_119 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_110})

    reroute_120 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_119})

    reroute_122 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_120})

    reroute_123 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_122})

    reroute_126 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_123})

    greater_equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_126, 3: 2},
        attrs={"operation": "GREATER_EQUAL", "data_type": "INT"},
    )

    reroute_210 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": integer})

    cylinder_2 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": reroute_210,
            "Radius": group_input.outputs["ButtonR1"],
            "Depth": group_input.outputs["ButtonH1"],
        },
    )

    add_jointed_geometry_metadata_003_gp = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_003_gp().name,
        input_kwargs={"Geometry": cylinder_2.outputs["Mesh"], "Label": "button"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={1: group_input.outputs["ButtonH1"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_1})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_003_gp,
            "Translation": combine_xyz_1,
        },
    )

    reroute_61 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_1}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["HeadSideways"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_58 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ)

    reroute_21 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["RackHeight"]}
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_21, 1: -2.0000, 2: 0.0000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_add})

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_58,
            "False": combine_xyz_2,
            "True": combine_xyz_3,
        },
        attrs={"input_type": "VECTOR"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ)

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": 1.5700, "Z": 3.1400}
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_58,
            "False": combine_xyz_4,
            "True": combine_xyz_5,
        },
        attrs={"input_type": "VECTOR"},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_61,
            "Translation": switch,
            "Rotation": switch_1,
        },
    )

    reroute_95 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_2}
    )

    reroute_185 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_95})

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_185, "Label": "button_press_shade_base"},
    )

    reroute_211 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_210})

    cylinder_3 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": reroute_211,
            "Radius": group_input.outputs["ButtonR2"],
            "Depth": group_input.outputs["ButtonH1"],
        },
    )

    add_jointed_geometry_metadata_004_gp = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_004_gp().name,
        input_kwargs={"Geometry": cylinder_3.outputs["Mesh"], "Label": "button"},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ButtonH1"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_5})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_004_gp,
            "Translation": combine_xyz_6,
        },
    )

    reroute_70 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_3}
    )

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ)

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_70, "Rotation": combine_xyz_7},
    )

    reroute_59 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_59, 1: 0.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_2})

    reroute_85 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_1})

    reroute_86 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_85})

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_4,
            "Translation": combine_xyz_8,
            "Rotation": reroute_86,
        },
    )

    reroute_186 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_5}
    )

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_186, "Label": "button_prss_shade_button"},
    )

    reroute_74 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_58})

    reroute_75 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_74})

    reroute_97 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_75})

    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": -1.0000})

    combine_xyz_10 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": 1.0000})

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_97,
            "False": combine_xyz_9,
            "True": combine_xyz_10,
        },
        attrs={"input_type": "VECTOR"},
    )

    reroute_188 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_2})

    reroute_76 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_59})

    reroute_77 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_76})

    reroute_102 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_77})

    reroute_187 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_102})

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "button_press_shade",
            "Parent": add_jointed_geometry_metadata,
            "Child": add_jointed_geometry_metadata_1,
            "Axis": reroute_188,
            "Value": 0.0000,
            "Max": reroute_187,
        },
    )

    reroute_118 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": sliding_joint.outputs["Geometry"]}
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_120, 3: 3},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    cylinder_4 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": reroute_210,
            "Radius": group_input.outputs["ButtonR1"],
            "Depth": group_input.outputs["ButtonH1"],
        },
    )

    add_jointed_geometry_metadata_005_gp = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_005_gp().name,
        input_kwargs={"Geometry": cylinder_4.outputs["Mesh"], "Label": "button"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={1: group_input.outputs["ButtonH1"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_11 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_3})

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_005_gp,
            "Translation": combine_xyz_11,
        },
    )

    reroute_63 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_6}
    )

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_63,
            "Translation": switch,
            "Rotation": switch_1,
        },
    )

    reroute_90 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_7}
    )

    reroute_196 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_90})

    add_jointed_geometry_metadata_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_196, "Label": "twist_switch_base"},
    )

    cylinder_5 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": reroute_211,
            "Radius": group_input.outputs["ButtonR2"],
            "Depth": group_input.outputs["ButtonH2"],
        },
    )

    add_jointed_geometry_metadata_006_gp = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_006_gp().name,
        input_kwargs={"Geometry": cylinder_5.outputs["Mesh"], "Label": "button"},
    )

    reroute_62 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_006_gp}
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["ButtonR2"], 1: 0.4000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["ButtonR2"], 1: 1.6000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_12 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": multiply_4, "Y": multiply_5, "Z": multiply_4},
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_12})

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ButtonH2"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    multiply_add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_7, 2: 0.0000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_13 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add_1})

    transform_geometry_8 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Translation": combine_xyz_13},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [reroute_62, transform_geometry_8]},
    )

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["ButtonH1"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_7, 2: multiply_6},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_14 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add_2})

    reroute_71 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_14})

    transform_geometry_9 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_1, "Translation": reroute_71},
    )

    reroute_44 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_7})

    transform_geometry_10 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_9, "Rotation": reroute_44},
    )

    combine_xyz_15 = nw.new_node(Nodes.CombineXYZ)

    reroute_96 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_86})

    transform_geometry_11 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_10,
            "Translation": combine_xyz_15,
            "Rotation": reroute_96,
        },
    )

    reroute_197 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_11}
    )

    add_jointed_geometry_metadata_3 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_197, "Label": "twist_switch_switch"},
    )

    reroute_108 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_97})

    combine_xyz_16 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": 1.0000})

    combine_xyz_17 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": 1.0000})

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_108,
            "False": combine_xyz_16,
            "True": combine_xyz_17,
        },
        attrs={"input_type": "VECTOR"},
    )

    reroute_195 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_3})

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "twist_switch_shade",
            "Parent": add_jointed_geometry_metadata_2,
            "Child": add_jointed_geometry_metadata_3,
            "Axis": reroute_195,
        },
    )

    combine_xyz_18 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": group_input.outputs["ButtonH1"], "Y": 0.7000, "Z": 1.0000},
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_18})

    add_jointed_geometry_metadata_007_gp = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_007_gp().name,
        input_kwargs={"Geometry": cube_1.outputs["Mesh"], "Label": "button"},
    )

    combine_xyz_19 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["ButtonR1"],
            "Y": group_input.outputs["ButtonR1"],
            "Z": group_input.outputs["ButtonR1"],
        },
    )

    reroute_54 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_19})

    transform_geometry_12 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_007_gp,
            "Rotation": (1.5708, 0.0000, 1.5708),
            "Scale": reroute_54,
        },
    )

    transform_geometry_13 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_12,
            "Translation": switch,
            "Rotation": switch_1,
        },
    )

    reroute_191 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_13}
    )

    add_jointed_geometry_metadata_4 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_191, "Label": "flip_switch_base"},
    )

    node_group_gp = nw.new_node(nodegroup_node_group_gp().name)

    add_jointed_geometry_metadata_008_gp = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_008_gp().name,
        input_kwargs={"Geometry": node_group_gp, "Label": "button"},
    )

    transform_geometry_14 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_008_gp,
            "Translation": (0.0000, -0.5000, -0.5000),
        },
    )

    combine_xyz_21 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["ButtonR2"],
            "Y": group_input.outputs["ButtonR2"],
            "Z": group_input.outputs["ButtonR2"],
        },
    )

    reroute_55 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_21})

    transform_geometry_15 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_14,
            "Rotation": (1.5708, 0.0000, 1.5708),
            "Scale": reroute_55,
        },
    )

    transform_geometry_16 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_15,
            "Rotation": (0.0000, 3.1416, 0.0000),
        },
    )

    transform_geometry_17 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_16, "Rotation": switch_1},
    )

    reroute_192 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_17}
    )

    add_jointed_geometry_metadata_5 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_192, "Label": "flip_switch_switch"},
    )

    bounding_box_3 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": reroute_191}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_3.outputs["Max"],
            1: bounding_box_3.outputs["Min"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    multiply_7 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"], 1: (0.0000, 0.0000, 0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    vector_rotate = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={"Vector": multiply_7.outputs["Vector"], "Angle": -1.5708},
        attrs={"rotation_type": "Y_AXIS"},
    )

    switch_26 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["HeadSideways"],
            "False": multiply_7.outputs["Vector"],
            "True": vector_rotate,
        },
        attrs={"input_type": "VECTOR"},
    )

    combine_xyz_22 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": 1.0000})

    combine_xyz_23 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": 1.0000})

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_75,
            "False": combine_xyz_22,
            "True": combine_xyz_23,
        },
        attrs={"input_type": "VECTOR"},
    )

    reroute_193 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_4})

    hinge_joint_1 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "flip_switch",
            "Parent": add_jointed_geometry_metadata_4,
            "Child": add_jointed_geometry_metadata_5,
            "Position": switch_26,
            "Axis": reroute_193,
            "Value": 0.0000,
            "Min": -0.2500,
            "Max": 0.2500,
        },
    )

    reroute_194 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_1.outputs["Geometry"]}
    )

    reroute_107 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_194})

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_1,
            "False": hinge_joint.outputs["Geometry"],
            "True": reroute_107,
        },
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": greater_equal, "False": reroute_118, "True": switch_5},
    )

    reroute_32 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ButtonMaterial"]}
    )

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_32})

    reroute_67 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_33})

    set_material_2 = nw.new_node(
        Nodes.SetMaterial, input_kwargs={"Geometry": switch_6, "Material": reroute_67}
    )

    combine_xyz_24 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["RackHeight"]}
    )

    combine_xyz_25 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input.outputs["RackHeight"]}
    )

    switch_7 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_3,
            "False": combine_xyz_24,
            "True": combine_xyz_25,
        },
        attrs={"input_type": "VECTOR"},
    )

    reroute_65 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_7})

    reroute_66 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_65})

    reroute_101 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_66})

    transform_geometry_18 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": set_material_2, "Translation": reroute_101},
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0900

    string_nodes_v2_gp = nw.new_node(
        nodegroup_string_nodes_v2_gp().name,
        input_kwargs={
            "Radius": group_input.outputs["ButtonR1"],
            "RadialLength": value,
            "Length": group_input.outputs["ButtonH2"],
            "Depth": group_input.outputs["ButtonH1"],
            "Resolution": 12,
        },
    )

    add_jointed_geometry_metadata_6 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": string_nodes_v2_gp.outputs["Parent"],
            "Label": "string_base",
        },
    )

    add_jointed_geometry_metadata_7 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": string_nodes_v2_gp.outputs["Child"],
            "Label": "string",
        },
    )

    bounding_box_4 = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": string_nodes_v2_gp.outputs["Parent"]},
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_4.outputs["Max"],
            1: bounding_box_4.outputs["Min"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    multiply_8 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: subtract_1.outputs["Vector"], 1: (0.0000, -0.5000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_37 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": value})

    add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_8.outputs["Vector"], 1: combine_xyz_37},
    )

    sliding_joint_1 = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "lamp_string",
            "Parent": add_jointed_geometry_metadata_6,
            "Child": add_jointed_geometry_metadata_7,
            "Position": add.outputs["Vector"],
            "Axis": (0.0000, 0.0000, -1.0000),
            "Value": 0.0000,
            "Max": string_nodes_v2_gp.outputs["Max String Dist"],
        },
    )

    reroute_203 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": sliding_joint_1.outputs["Geometry"]}
    )

    add_jointed_geometry_metadata_009_gp = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_009_gp().name,
        input_kwargs={"Geometry": reroute_203, "Label": "button"},
    )

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_009_gp,
            "Material": reroute_33,
        },
    )

    reroute_80 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_3})

    switch_8 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_115,
            "False": transform_geometry_18,
            "True": reroute_80,
        },
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [reroute_112, switch_8]}
    )

    switch_9 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_114,
            "False": reroute_116,
            "True": join_geometry_2,
        },
    )

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

    combine_xyz_26 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_15})

    transform_geometry_19 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": switch_9, "Translation": combine_xyz_26},
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [join_geometry, transform_geometry_19]},
    )

    reroute_149 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_3})

    reroute_150 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_149})

    greater_equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_123, 3: 2},
        attrs={"operation": "GREATER_EQUAL", "data_type": "INT"},
    )

    reroute_19 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BaseButtonOffset"]}
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["ButtonR1"],
            1: group_input.outputs["Radius"],
        },
    )

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BaseRadius"]}
    )

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    multiply_add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: -2.0000, 1: add_1, 2: reroute_17},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_46 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add_1})

    multiply_add_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_20, 1: multiply_add_3, 2: reroute_46},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_18 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BaseHeight"]}
    )

    multiply_9 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_18}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_27 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_add_4, "Z": multiply_9}
    )

    transform_geometry_20 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_61,
            "Translation": combine_xyz_27,
            "Rotation": combine_xyz_7,
        },
    )

    add_jointed_geometry_metadata_8 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_20, "Label": "button_press_base"},
    )

    reroute_189 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_4}
    )

    add_jointed_geometry_metadata_9 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_189, "Label": "button_press_button"},
    )

    reroute_190 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_77})

    sliding_joint_2 = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "button_press",
            "Parent": add_jointed_geometry_metadata_8,
            "Child": add_jointed_geometry_metadata_9,
            "Axis": (0.0000, 0.0000, -1.0000),
            "Value": 0.0000,
            "Max": reroute_190,
        },
    )

    reroute_113 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": sliding_joint_2.outputs["Geometry"]}
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_110, 3: 3},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    transform_geometry_21 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_63,
            "Translation": combine_xyz_27,
            "Rotation": combine_xyz_7,
        },
    )

    reroute_91 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_21}
    )

    add_jointed_geometry_metadata_10 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_91, "Label": "twist_switch_base"},
    )

    reroute_183 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_10}
    )

    add_jointed_geometry_metadata_11 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_183, "Label": "twist_switch_switch"},
    )

    hinge_joint_2 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "twist_switch",
            "Parent": add_jointed_geometry_metadata_10,
            "Child": add_jointed_geometry_metadata_11,
            "Value": 0.0000,
        },
    )

    transform_geometry_22 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_12,
            "Translation": combine_xyz_27,
            "Rotation": combine_xyz_7,
        },
    )

    reroute_181 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_22}
    )

    add_jointed_geometry_metadata_12 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_181, "Label": "flip_switch_base"},
    )

    transform_geometry_23 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_16, "Rotation": combine_xyz_7},
    )

    reroute_180 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_23}
    )

    add_jointed_geometry_metadata_13 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_180, "Label": "flip_switch_switch"},
    )

    bounding_box_2 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": reroute_181}
    )

    subtract_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_2.outputs["Max"],
            1: bounding_box_2.outputs["Min"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    multiply_10 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: subtract_2.outputs["Vector"], 1: (0.0000, 0.0000, 0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    hinge_joint_3 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "flip_switch_base",
            "Parent": add_jointed_geometry_metadata_12,
            "Child": add_jointed_geometry_metadata_13,
            "Position": multiply_10.outputs["Vector"],
            "Axis": (1.0000, 0.0000, 0.0000),
            "Value": 0.0000,
            "Min": -0.2500,
            "Max": 0.2500,
        },
    )

    reroute_103 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_3.outputs["Geometry"]}
    )

    switch_10 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_2,
            "False": hinge_joint_2.outputs["Geometry"],
            "True": reroute_103,
        },
    )

    switch_11 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": greater_equal_1,
            "False": reroute_113,
            "True": switch_10,
        },
    )

    reroute_127 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_11})

    reroute_124 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_67})

    set_material_4 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": reroute_127, "Material": reroute_124},
    )

    reroute_136 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_4})

    reroute_137 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_136})

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [join_geometry_3, reroute_137]}
    )

    switch_12 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_131,
            "False": reroute_150,
            "True": join_geometry_4,
        },
    )

    reroute_47 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cylinder.outputs["Mesh"]}
    )

    reroute_48 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_47})

    add_jointed_geometry_metadata_010_gp = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_010_gp().name,
        input_kwargs={"Geometry": reroute_48, "Label": "lamp_base"},
    )

    reroute_68 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_41})

    set_material_5 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_010_gp,
            "Material": reroute_68,
        },
    )

    reroute_130 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_5})

    join_geometry_5 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material_4, set_material_5]}
    )

    switch_13 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_39,
            "False": reroute_130,
            "True": join_geometry_5,
        },
    )

    reroute_178 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_13})

    add_jointed_geometry_metadata_14 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_178, "Label": "lamp_base_short"},
    )

    reroute_132 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_1})

    reroute_133 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_132})

    join_geometry_6 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_19, reroute_133]},
    )

    reroute_177 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_6})

    add_jointed_geometry_metadata_15 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_177, "Label": "lamp_top_short"},
    )

    hinge_joint_4 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "hinge_rotate_z_short",
            "Parent": add_jointed_geometry_metadata_14,
            "Child": add_jointed_geometry_metadata_15,
            "Value": 0.0000,
        },
    )

    reroute_179 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_4.outputs["Geometry"]}
    )

    reroute_155 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_179})

    switch_14 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_37, "False": switch_12, "True": reroute_155},
    )

    reroute_164 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_14})

    reroute_145 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_37})

    reroute_142 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_131})

    reroute_209 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_208})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Radius"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    multiply_11 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 2.4000},
        attrs={"operation": "MULTIPLY"},
    )

    cylinder_6 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": reroute_209,
            "Radius": reroute_1,
            "Depth": multiply_11,
        },
    )

    transform_geometry_24 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_6.outputs["Mesh"],
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    add_jointed_geometry_metadata_011_gp = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_011_gp().name,
        input_kwargs={"Geometry": transform_geometry_24, "Label": "bar"},
    )

    set_material_6 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_011_gp,
            "Material": reroute_69,
        },
    )

    reroute_98 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_6})

    reroute_99 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_98})

    reroute_174 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_99})

    add_jointed_geometry_metadata_16 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_174, "Label": "hinge_connector_middle"},
    )

    add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["FirstBarLength"],
            1: group_input.outputs["FirstBarExtension"],
        },
    )

    cylinder_8 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": reroute_209, "Radius": reroute_1, "Depth": add_2},
    )

    reroute_64 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cylinder_8.outputs["Mesh"]}
    )

    reroute_24 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["FirstBarExtension"]}
    )

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_24})

    multiply_12 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["FirstBarLength"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_25, 1: -0.5000, 2: multiply_12},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_28 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add_5})

    transform_geometry_26 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_64, "Translation": combine_xyz_28},
    )

    add_jointed_geometry_metadata_013_gp = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_013_gp().name,
        input_kwargs={"Geometry": transform_geometry_26, "Label": "bar"},
    )

    set_material_8 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_013_gp,
            "Material": reroute_87,
        },
    )

    reroute_111 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_8})

    cylinder_7 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": reroute_209,
            "Radius": reroute_1,
            "Depth": multiply_11,
        },
    )

    transform_geometry_25 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_7.outputs["Mesh"],
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    add_jointed_geometry_metadata_012_gp = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_012_gp().name,
        input_kwargs={"Geometry": transform_geometry_25, "Label": "bar"},
    )

    set_material_7 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_012_gp,
            "Material": reroute_69,
        },
    )

    reroute_93 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_7})

    reroute_94 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_93})

    reroute_121 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_94})

    reroute_22 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["FirstBarLength"]}
    )

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    combine_xyz_29 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_23})

    reroute_117 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_29})

    transform_geometry_27 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": switch_9, "Translation": reroute_117}
    )

    join_geometry_7 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [reroute_111, reroute_121, transform_geometry_27]},
    )

    reroute_50 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_11})

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_50})

    combine_xyz_30 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": reroute_51})

    transform_geometry_28 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_7, "Translation": combine_xyz_30},
    )

    reroute_173 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_28}
    )

    add_jointed_geometry_metadata_17 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_173, "Label": "lamp_top_single_section"},
    )

    hinge_joint_5 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "hinge_top",
            "Parent": add_jointed_geometry_metadata_16,
            "Child": add_jointed_geometry_metadata_17,
            "Axis": (0.0000, 1.0000, 0.0000),
            "Value": 0.0000,
        },
    )

    reroute_176 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_5.outputs["Geometry"]}
    )

    reroute_128 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_26})

    reroute_129 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_128})

    transform_geometry_31 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_176, "Translation": reroute_129},
    )

    reroute_134 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry})

    reroute_135 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_134})

    join_geometry_10 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_31, reroute_135]},
    )

    reroute_161 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_10})

    reroute_153 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_137})

    reroute_154 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_153})

    join_geometry_11 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [join_geometry_10, reroute_154]}
    )

    switch_19 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_142,
            "False": reroute_161,
            "True": join_geometry_11,
        },
    )

    reroute_152 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_13})

    reroute_172 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_152})

    add_jointed_geometry_metadata_18 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_172, "Label": "lamp_base"},
    )

    reroute_139 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_133})

    reroute_140 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_139})

    join_geometry_12 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_31, reroute_140]},
    )

    reroute_170 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_12})

    add_jointed_geometry_metadata_19 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_170, "Label": "lamp_main"},
    )

    hinge_joint_6 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "hinge_rotate_z",
            "Parent": add_jointed_geometry_metadata_18,
            "Child": add_jointed_geometry_metadata_19,
            "Value": 0.0000,
        },
    )

    reroute_166 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_6.outputs["Geometry"]}
    )

    switch_20 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_145, "False": switch_19, "True": reroute_166},
    )

    switch_21 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_11, "False": reroute_164, "True": switch_20},
    )

    reroute_167 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_21})

    reroute_162 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_145})

    reroute_160 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_142})

    reroute_26 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["SecondBarSliding"]}
    )

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_26})

    multiply_13 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 2.4000},
        attrs={"operation": "MULTIPLY"},
    )

    cylinder_9 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": reroute_209,
            "Radius": reroute_1,
            "Depth": multiply_13,
        },
    )

    set_material_9 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": cylinder_9.outputs["Mesh"], "Material": reroute_43},
    )

    add_jointed_geometry_metadata_014_gp = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_014_gp().name,
        input_kwargs={"Geometry": set_material_9, "Label": "bar"},
    )

    reroute_82 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_014_gp}
    )

    transform_geometry_32 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_82, "Rotation": (1.5708, 0.0000, 0.0000)},
    )

    add_jointed_geometry_metadata_20 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": transform_geometry_32,
            "Label": "hinge_connector_two_sections",
        },
    )

    cylinder_11 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": reroute_209,
            "Radius": reroute_1,
            "Depth": multiply_13,
        },
    )

    set_material_11 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": cylinder_11.outputs["Mesh"], "Material": reroute_43},
    )

    add_jointed_geometry_metadata_016_gp = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_016_gp().name,
        input_kwargs={"Geometry": set_material_11, "Label": "bar"},
    )

    reroute_83 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_016_gp}
    )

    transform_geometry_35 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_83, "Rotation": (1.5708, 0.0000, 0.0000)},
    )

    reroute_146 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_35}
    )

    reroute_175 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_5.outputs["Geometry"]}
    )

    reroute_34 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["SecondBarLength"]}
    )

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_34})

    combine_xyz_32 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_35})

    transform_geometry_33 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_175, "Translation": combine_xyz_32},
    )

    add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["SecondBarLength"],
            1: group_input.outputs["SecondBarExtension"],
        },
    )

    cylinder_10 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": reroute_209, "Radius": reroute_1, "Depth": add_3},
    )

    set_material_10 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": cylinder_10.outputs["Mesh"], "Material": reroute_43},
    )

    add_jointed_geometry_metadata_015_gp = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_015_gp().name,
        input_kwargs={"Geometry": set_material_10, "Label": "bar"},
    )

    reroute_84 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_015_gp}
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["SecondBarExtension"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    multiply_14 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["SecondBarLength"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_9, 1: -0.5000, 2: multiply_14},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_33 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add_6})

    reroute_72 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_33})

    transform_geometry_34 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_84, "Translation": reroute_72},
    )

    reroute_144 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_34}
    )

    join_geometry_13 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [reroute_146, transform_geometry_33, reroute_144]},
    )

    reroute_52 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_13})

    reroute_53 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_52})

    multiply_15 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_53, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_34 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_15})

    transform_geometry_36 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_13, "Translation": combine_xyz_34},
    )

    add_jointed_geometry_metadata_21 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": transform_geometry_36,
            "Label": "lamp_top_two_sections",
        },
    )

    hinge_joint_7 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "hinge_bottom",
            "Parent": add_jointed_geometry_metadata_20,
            "Child": add_jointed_geometry_metadata_21,
            "Axis": (0.0000, 1.0000, 0.0000),
            "Value": 0.0000,
        },
    )

    reroute_156 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_32}
    )

    reroute_157 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_156})

    reroute_171 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_157})

    add_jointed_geometry_metadata_22 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_171, "Label": "hinge_connector"},
    )

    reroute_143 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_35}
    )

    reroute_199 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_143})

    add_jointed_geometry_metadata_23 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_199, "Label": "hinge_connector_sliding"},
    )

    reroute_147 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_34}
    )

    join_geometry_15 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [reroute_147, transform_geometry_33]},
    )

    reroute_198 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_15})

    add_jointed_geometry_metadata_24 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": reroute_198,
            "Label": "lamp_top_two_sections_sliding",
        },
    )

    reroute_138 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_35})

    multiply_add_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_138, 1: -1.0000, 2: 0.0500},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_200 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_add_7})

    reroute_60 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_60, 1: 0.0500},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_201 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract_3})

    sliding_joint_3 = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "lamp_sliding",
            "Parent": add_jointed_geometry_metadata_23,
            "Child": add_jointed_geometry_metadata_24,
            "Min": reroute_200,
            "Max": reroute_201,
        },
    )

    reroute_202 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": sliding_joint_3.outputs["Geometry"]}
    )

    reroute_148 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_34})

    transform_geometry_37 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_202, "Translation": reroute_148},
    )

    reroute_165 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_37}
    )

    add_jointed_geometry_metadata_25 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_165, "Label": "lamp_top"},
    )

    hinge_joint_8 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "hinge_bottom_with_sliding",
            "Parent": add_jointed_geometry_metadata_22,
            "Child": add_jointed_geometry_metadata_25,
            "Axis": (0.0000, 1.0000, 0.0000),
            "Value": 0.0000,
        },
    )

    switch_22 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_27,
            "False": hinge_joint_7.outputs["Geometry"],
            "True": hinge_joint_8.outputs["Geometry"],
        },
    )

    reroute_141 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_129})

    transform_geometry_38 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": switch_22, "Translation": reroute_141},
    )

    reroute_158 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_135})

    join_geometry_16 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_38, reroute_158]},
    )

    reroute_168 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_16})

    reroute_163 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_154})

    join_geometry_17 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [join_geometry_16, reroute_163]}
    )

    switch_23 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_160,
            "False": reroute_168,
            "True": join_geometry_17,
        },
    )

    reroute_151 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_13})

    add_jointed_geometry_metadata_26 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_151, "Label": "lamp_base_3_sections"},
    )

    reroute_159 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_140})

    join_geometry_18 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_38, reroute_159]},
    )

    reroute_184 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_18})

    add_jointed_geometry_metadata_27 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_184, "Label": "lamp_top_3_sections"},
    )

    hinge_joint_9 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "z_rotation_double_hinge",
            "Parent": add_jointed_geometry_metadata_26,
            "Child": add_jointed_geometry_metadata_27,
            "Value": 0.0000,
        },
    )

    reroute_169 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_9.outputs["Geometry"]}
    )

    switch_24 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_162, "False": switch_23, "True": reroute_169},
    )

    switch_25 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_13, "False": reroute_167, "True": switch_24},
    )

    reroute_205 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BaseHeight"]}
    )

    reroute_206 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_205})

    multiply_16 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_206}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_38 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_16})

    transform_geometry_40 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": switch_25, "Translation": combine_xyz_38},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_40},
        attrs={"is_active_output": True},
    )


def sample_lamp_parameters(lamp_type=None, materials={}):
    """
    Sample parameters for a procedural lamp.

    Args:
        lamp_type: Optional string to specify lamp type: 'standing', 'desk',
                  'single_bar', 'single_bar_modern', 'double_bar', or None for random.

    Returns:
        Dictionary of lamp parameters.
    """
    if lamp_type is None:
        lamp_type = np.random.choice(
            [
                "standing",
                "desk",
                "single_bar",
                "single_bar_modern",
                "double_bar",
                "double_bar_modern",
                "short_second_bar",
            ],
            p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.2],
        )

    # Initialize parameter dictionary with defaults
    params = {
        "SingleHinge": False,
        "DoubleHinge": False,
        "FirstBarLength": 0.0,
        "FirstBarExtension": 0.0,
        "SecondBarLength": 0.0,
        "SecondBarExtension": 0.0,
        "SecondBarSliding": False,
        "Radius": np.random.uniform(0.015, 0.03),  # Default bar thickness
        "Height": 0.0,
        "BaseRadius": np.random.uniform(0.1, 0.25),
        "BaseHeight": np.random.uniform(0.05, 0.07),
        "BaseButtonOffset": 0,
        "BaseRotate": np.random.choice([True, False]),
        "ButtonOnLampShade": False,
        "HeadSideways": False,
        "ShadeHeight": 0.0,
        "RackHeight": 0.0,
        "TopRadius": 0.0,
        "BottomRadius": 0.0,
        "ShadeCurved": np.random.choice([True, False], p=[0.2, 0.8]),
        "RackThickness": np.random.uniform(0.003, 0.01),
        "ShadeTop": False,
        "ReverseBulb": False,
        "IncludeLightBulb": True,
        "ButtonType": 0,
        "BaseSides": np.random.choice([40, 4, 6], p=[0.7, 0.2, 0.1]),
        "ShadeSides": np.random.choice([40, 4, 6], p=[0.7, 0.2, 0.1]),
    }
    params["BaseButtonOffset"] = np.random.uniform(0.05, params["BaseRadius"] - 0.05)
    params["BaseMaterial"] = materials["base"]
    params["StandMaterial"] = materials["stand"]
    params["ShadeMaterial"] = materials["shade_light"]
    params["ShadeInteriorMaterial"] = materials["shade_interior"]
    params["ButtonMaterial"] = materials["button"]
    params["LampRackMaterial"] = materials["lamp_rack"]
    params["MetalMaterial"] = materials["metal_mat"]

    def sample_button(button_type):
        params["BaseButtonOffset"] = np.random.uniform(0, 1)
        match button_type:
            case 0:
                params["ButtonR1"] = np.random.uniform(0.02, 0.03)
                params["ButtonH1"] = np.random.uniform(0.05, 0.08)
                params["ButtonH2"] = np.random.uniform(0.2, 0.3)
            case 1:
                params["ButtonR1"] = np.random.uniform(0.03, 0.06)
                params["ButtonR2"] = params["ButtonR1"] * np.random.uniform(0.6, 0.8)
                params["ButtonH1"] = params["ButtonR1"] * np.random.uniform(0.5, 1.0)
            case 2:
                params["ButtonR1"] = np.random.uniform(0.02, 0.04)
                params["ButtonR2"] = params["ButtonR1"] * np.random.uniform(0.6, 0.8)
                params["ButtonH1"] = params["ButtonR1"] * np.random.uniform(0.5, 1.0)
                params["ButtonH2"] = np.random.uniform(0.04, 0.1)
            case 3:
                params["ButtonR1"] = np.random.uniform(0.025, 0.04)
                params["ButtonR2"] = params["ButtonR1"] * np.random.uniform(0.6, 0.9)
                params["ButtonH1"] = 2

    # Set parameters based on lamp type
    if lamp_type == "standing" or lamp_type == "desk":
        if lamp_type == "standing":
            params["Height"] = np.random.uniform(1.4, 1.85)
        else:
            # desk lamp
            params["Height"] = np.random.uniform(0.4, 0.9)

        if np.random.rand() < 0.7:
            # Traditional shade
            params["TopRadius"] = np.random.uniform(0.1, 0.2)
            params["BottomRadius"] = params["TopRadius"] * np.random.uniform(1.5, 2.5)
        else:
            # Cylindrical shade (similar top and bottom radius)
            radius = np.random.uniform(0.15, 0.3)
            params["TopRadius"] = radius
            params["BottomRadius"] = radius
        params["RackHeight"] = -params["TopRadius"] - np.random.uniform(0.03, 0.3)
        params["ShadeHeight"] = params["RackHeight"] - np.random.uniform(0.0, 0.3)

        if lamp_type == "standing":
            params["ButtonOnLampShade"] = True
        else:
            params["ButtonOnLampShade"] = (
                False  # np.random.choice([True, False], p=[0.7, 0.3])
            )
        if params["ButtonOnLampShade"]:
            params["ButtonType"] = 0
            sample_button(params["ButtonType"])
        else:
            params["IncludeButtonBase"] = not params["ButtonOnLampShade"]
            params["ButtonType"] = np.random.choice([1, 2, 3])
            sample_button(params["ButtonType"])
        params["IncludeLightBulb"] = True

    elif lamp_type == "single_bar":
        params["SingleHinge"] = True
        if np.random.rand() < 0.7:
            # has extension
            params["FirstBarLength"] = np.random.uniform(0.5, 1.0)
            params["FirstBarExtension"] = params["FirstBarLength"] * np.random.uniform(
                0.2, 0.8
            )
        else:
            # no extension
            params["FirstBarLength"] = np.random.uniform(0.6, 1.0)
            params["FirstBarExtension"] = 0.0

        params["Height"] = np.random.uniform(0.6, 1.4)
        # Adjust base radius for stability based on height
        params["BaseRadius"] = np.random.uniform(0.1, 0.25)
        if params["Height"] > 1.0:
            params["BaseRadius"] = max(params["BaseRadius"], 0.15)

        # Add shade type variation
        if np.random.rand() < 0.8:
            # Traditional shade
            params["TopRadius"] = np.random.uniform(0.08, 0.15)
            params["BottomRadius"] = params["TopRadius"] * np.random.uniform(1.8, 3.0)
            params["RackHeight"] = params["TopRadius"] + np.random.uniform(0.05, 0.25)
        else:
            # Cylindrical shade
            radius = np.random.uniform(0.1, 0.18)
            params["TopRadius"] = radius
            params["BottomRadius"] = radius
            params["RackHeight"] = radius + np.random.uniform(0.05, 0.2)

        params["ShadeHeight"] = params["RackHeight"] + np.random.uniform(0.0, 0.3)
        params["HeadSideways"] = True
        params["ReverseBulb"] = True
        params["IncludeLightBulb"] = True

        params["ButtonOnLampShade"] = False
        params["IncludeButtonBase"] = True
        params["ButtonType"] = np.random.choice([1, 2, 3])
        sample_button(params["ButtonType"])

    elif lamp_type == "single_bar_modern":
        params["SingleHinge"] = True

        if np.random.rand() < 0.7:
            # has extension
            params["FirstBarLength"] = np.random.uniform(0.5, 1.0)
            params["FirstBarExtension"] = params["FirstBarLength"] * np.random.uniform(
                0.2, 0.8
            )
        else:
            # no extension
            params["FirstBarLength"] = np.random.uniform(0.6, 1.0)
            params["FirstBarExtension"] = 0.0

        params["Height"] = np.random.uniform(0.4, 1.2)
        # Adjust base for stability
        params["BaseRadius"] = np.random.uniform(0.12, 0.28)
        if params["FirstBarExtension"] > 0.8:
            params["BaseRadius"] = max(params["BaseRadius"], 0.18)

        radius = np.random.uniform(0.08, 0.18)
        params["TopRadius"] = radius
        params["BottomRadius"] = radius

        params["RackHeight"] = np.random.uniform(0.03, 0.08)
        params["ShadeHeight"] = params["RackHeight"] + np.random.uniform(0.25, 0.45)
        params["ButtonType"] = 1
        params["ShadeTop"] = True
        params["ReverseBulb"] = False
        params["HeadSideways"] = True
        params["ButtonOnLampShade"] = True
        params["IncludeLightBulb"] = False
        params["ShadeMaterial"] = materials["stand"]
        params["ButtonType"] = 3  # np.random.choice([1, 2, 3])
        sample_button(params["ButtonType"])

    elif lamp_type == "double_bar" or lamp_type == "double_bar_modern":
        params["SingleHinge"] = True
        params["DoubleHinge"] = True
        params["FirstBarLength"] = np.random.uniform(0.6, 1.0)
        params["FirstBarExtension"] = np.random.uniform(0.0, 0.7)
        params["SecondBarLength"] = np.random.uniform(0.4, 1.0)
        params["SecondBarExtension"] = np.random.uniform(0.0, 0.7)
        params["SecondBarSliding"] = np.random.choice([True, False], p=[0.7, 0.3])
        params["Height"] = np.random.uniform(0.6, 1.4)
        params["HeadSideways"] = True

        # Adjust base for stability
        params["BaseRadius"] = np.random.uniform(0.12, 0.28)
        if params["FirstBarExtension"] > 0.5 or params["SecondBarExtension"] > 0.5:
            params["BaseRadius"] = max(params["BaseRadius"], 0.18)

        if lamp_type == "double_bar_modern":
            radius = np.random.uniform(0.14, 0.18)
            params["TopRadius"] = radius
            params["BottomRadius"] = radius
            params["RackHeight"] = np.random.uniform(0.02, 0.08)
            params["ShadeHeight"] = np.random.uniform(0.2, 0.4)
            params["ButtonType"] = 8
            params["ShadeTop"] = True
            params["ReverseBulb"] = False
            params["IncludeLightBulb"] = False
            params["ButtonOnLampShade"] = True
            params["ShadeMaterial"] = materials["stand"]
            params["ButtonType"] = np.random.choice([1, 2, 3])
            sample_button(params["ButtonType"])
        else:
            # Traditional shade
            if np.random.rand() < 0.8:
                params["TopRadius"] = np.random.uniform(0.08, 0.15)
                params["BottomRadius"] = params["TopRadius"] * np.random.uniform(
                    1.8, 3.0
                )
            else:
                # Cylindrical shade
                radius = np.random.uniform(0.1, 0.18)
                params["TopRadius"] = radius
                params["BottomRadius"] = radius

            params["RackHeight"] = params["TopRadius"] + np.random.uniform(0.03, 0.2)
            params["ShadeHeight"] = params["RackHeight"] + np.random.uniform(0.0, 0.3)
            params["IncludeLightBulb"] = True
            params["ReverseBulb"] = True
            params["ButtonOnLampShade"] = False
            params["IncludeButtonBase"] = True
            params["ButtonType"] = np.random.choice([1, 2, 3])
            sample_button(params["ButtonType"])

    elif lamp_type == "short_second_bar":
        params["SingleHinge"] = True
        params["DoubleHinge"] = True
        params["FirstBarLength"] = np.random.uniform(0.15, 0.25)
        params["FirstBarExtension"] = 0
        params["SecondBarLength"] = np.random.uniform(0.7, 1.0)
        params["SecondBarExtension"] = np.random.uniform(0.5, 0.9)
        params["SecondBarSliding"] = True
        params["Height"] = np.random.uniform(0.7, 1.2)

        # Adjust base for stability with extended arm
        params["BaseRadius"] = np.random.uniform(0.15, 0.3)

        # Add shade type variation
        if np.random.rand() < 0.8:
            # Traditional shade
            params["TopRadius"] = np.random.uniform(0.15, 0.25)
            params["BottomRadius"] = params["TopRadius"] * np.random.uniform(1.8, 2.5)
        else:
            # More cylindrical shade
            radius = np.random.uniform(0.18, 0.25)
            params["TopRadius"] = radius
            params["BottomRadius"] = radius

        params["RackHeight"] = np.random.uniform(0.3, 0.5)
        params["ShadeHeight"] = params["RackHeight"] + np.random.uniform(0.0, 0.2)
        params["ReverseBulb"] = True
        params["HeadSideways"] = False
        params["IncludeLightBulb"] = True
        params["ButtonOnLampShade"] = True
        params["IncludeButtonBase"] = False
        params["ButtonType"] = 0
        sample_button(params["ButtonType"])

    if not params["IncludeLightBulb"]:
        params["RackThickness"] = 0.0
        params["IncludeLightBulb"] = True

    return params


def sample_white_interior():
    """Generate a white or near-white color for the lamp shade interior"""
    # Very high value (brightness), low saturation
    h = np.random.uniform(0, 1)  # Hue can be any value since saturation is low
    s = np.random.uniform(0, 0.1)  # Very low saturation to keep it close to white
    v = np.random.uniform(0.9, 1.0)  # High value for brightness

    return (h, s, v)


def sample_gold():
    """Generate a gold color variation"""
    # Gold colors are generally in yellow-orange hue range
    # 36/360 to 56/360 converted to 0-1 scale
    h = np.random.uniform(0.1, 0.155)  # Gold hue range
    s = np.random.uniform(0.65, 0.9)  # Moderate to high saturation
    v = np.random.uniform(0.75, 1.0)  # Bright

    return (h, s, v)


def sample_silver():
    """Generate a silver color variation"""
    # Silver colors are desaturated with high brightness
    h = np.random.uniform(0, 1)  # Hue doesn't matter much due to low saturation
    s = np.random.uniform(0, 0.1)  # Very low saturation
    v = np.random.uniform(0.75, 0.9)  # High but not maximum brightness

    return (h, s, v)


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


def shader_fine_knit_fabric_colored(color):
    def shader(nw: NodeWrangler):
        fabric_params = fabric.fine_knit_fabric.get_texture_params()
        fabric_params["_color"] = color[:3]  # np.ones(3) * np.random.uniform(0.8, 1.0)
        fabric_params["_map"] = "Object"
        return fabric.fine_knit_fabric.shader_material(nw, **fabric_params)

    return shader


class LampFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)
        self.type = None

    @classmethod
    @gin.configurable(module="LampFactory")
    def sample_joint_parameters(
        cls,
        hinge_rotate_z_short_stiffness_min: float = 0.0,
        hinge_rotate_z_short_stiffness_max: float = 0.0,
        hinge_rotate_z_short_damping_min: float = 2.0,
        hinge_rotate_z_short_damping_max: float = 5.0,
        flip_switch_stiffness_min: float = 0.0,
        flip_switch_stiffness_max: float = 0.0,
        flip_switch_damping_min: float = 2.0,
        flip_switch_damping_max: float = 5.0,
        lamp_string_stiffness_min: float = 3.0,
        lamp_string_stiffness_max: float = 5.0,
        lamp_string_damping_min: float = 1.0,
        lamp_string_damping_max: float = 3.0,
        hinge_rotate_z_stiffness_min: float = 0.0,
        hinge_rotate_z_stiffness_max: float = 0.0,
        hinge_rotate_z_damping_min: float = 2.0,
        hinge_rotate_z_damping_max: float = 5.0,
        hinge_top_stiffness_min: float = 0.0,
        hinge_top_stiffness_max: float = 0.0,
        hinge_top_damping_min: float = 10.0,
        hinge_top_damping_max: float = 20.0,
        z_rotation_double_hinge_stiffness_min: float = 0.0,
        z_rotation_double_hinge_stiffness_max: float = 0.0,
        z_rotation_double_hinge_damping_min: float = 10.0,
        z_rotation_double_hinge_damping_max: float = 20.0,
        hinge_bottom_stiffness_min: float = 0.0,
        hinge_bottom_stiffness_max: float = 0.0,
        hinge_bottom_damping_min: float = 10.0,
        hinge_bottom_damping_max: float = 20.0,
        lamp_sliding_stiffness_min: float = 0.0,
        lamp_sliding_stiffness_max: float = 0.0,
        lamp_sliding_damping_min: float = 10.0,
        lamp_sliding_damping_max: float = 20.0,
        twist_switch_stiffness_min: float = 0.0,
        twist_switch_stiffness_max: float = 0.0,
        twist_switch_damping_min: float = 10.0,
        twist_switch_damping_max: float = 20.0,
        hinge_bottom_with_sliding_stiffness_min: float = 0.0,
        hinge_bottom_with_sliding_stiffness_max: float = 0.0,
        hinge_bottom_with_sliding_damping_min: float = 10.0,
        hinge_bottom_with_sliding_damping_max: float = 20.0,
        twist_switch_shade_stiffness_min: float = 0.0,
        twist_switch_shade_stiffness_max: float = 0.0,
        twist_switch_shade_damping_min: float = 5.0,
        twist_switch_shade_damping_max: float = 10.0,
        button_press_shade_stiffness_min: float = 10.0,
        button_press_shade_stiffness_max: float = 20.0,
        button_press_shade_damping_min: float = 1.0,
        button_press_shade_damping_max: float = 2.0,
        flip_switch_base_stiffness_min: float = 0.0,
        flip_switch_base_stiffness_max: float = 0.0,
        flip_switch_base_damping_min: float = 2.0,
        flip_switch_base_damping_max: float = 5.0,
        button_press_stiffness_min: float = 10.0,
        button_press_stiffness_max: float = 20.0,
        button_press_damping_min: float = 1.0,
        button_press_damping_max: float = 2.0,
    ):
        return {
            "hinge_rotate_z_short": {
                "stiffness": np.random.uniform(
                    hinge_rotate_z_short_stiffness_min,
                    hinge_rotate_z_short_stiffness_max,
                ),
                "damping": np.random.uniform(
                    hinge_rotate_z_short_damping_min, hinge_rotate_z_short_damping_max
                ),
            },
            "flip_switch": {
                "stiffness": np.random.uniform(
                    flip_switch_stiffness_min, flip_switch_stiffness_max
                ),
                "damping": np.random.uniform(
                    flip_switch_damping_min, flip_switch_damping_max
                ),
            },
            "lamp_string": {
                "stiffness": np.random.uniform(
                    lamp_string_stiffness_min, lamp_string_stiffness_max
                ),
                "damping": np.random.uniform(
                    lamp_string_damping_min, lamp_string_damping_max
                ),
            },
            "hinge_rotate_z": {
                "stiffness": np.random.uniform(
                    hinge_rotate_z_stiffness_min, hinge_rotate_z_stiffness_max
                ),
                "damping": np.random.uniform(
                    hinge_rotate_z_damping_min, hinge_rotate_z_damping_max
                ),
            },
            "hinge_top": {
                "stiffness": np.random.uniform(
                    hinge_top_stiffness_min, hinge_top_stiffness_max
                ),
                "damping": np.random.uniform(
                    hinge_top_damping_min, hinge_top_damping_max
                ),
            },
            "z_rotation_double_hinge": {
                "stiffness": np.random.uniform(
                    z_rotation_double_hinge_stiffness_min,
                    z_rotation_double_hinge_stiffness_max,
                ),
                "damping": np.random.uniform(
                    z_rotation_double_hinge_damping_min,
                    z_rotation_double_hinge_damping_max,
                ),
            },
            "hinge_bottom": {
                "stiffness": np.random.uniform(
                    hinge_bottom_stiffness_min, hinge_bottom_stiffness_max
                ),
                "damping": np.random.uniform(
                    hinge_bottom_damping_min, hinge_bottom_damping_max
                ),
            },
            "lamp_sliding": {
                "stiffness": np.random.uniform(
                    lamp_sliding_stiffness_min, lamp_sliding_stiffness_max
                ),
                "damping": np.random.uniform(
                    lamp_sliding_damping_min, lamp_sliding_damping_max
                ),
            },
            "twist_switch": {
                "stiffness": np.random.uniform(
                    twist_switch_stiffness_min, twist_switch_stiffness_max
                ),
                "damping": np.random.uniform(
                    twist_switch_damping_min, twist_switch_damping_max
                ),
            },
            "hinge_bottom_with_sliding": {
                "stiffness": np.random.uniform(
                    hinge_bottom_with_sliding_stiffness_min,
                    hinge_bottom_with_sliding_stiffness_max,
                ),
                "damping": np.random.uniform(
                    hinge_bottom_with_sliding_damping_min,
                    hinge_bottom_with_sliding_damping_max,
                ),
            },
            "twist_switch_shade": {
                "stiffness": np.random.uniform(
                    twist_switch_shade_stiffness_min, twist_switch_shade_stiffness_max
                ),
                "damping": np.random.uniform(
                    twist_switch_shade_damping_min, twist_switch_shade_damping_max
                ),
            },
            "button_press_shade": {
                "stiffness": np.random.uniform(
                    button_press_shade_stiffness_min, button_press_shade_stiffness_max
                ),
                "damping": np.random.uniform(
                    button_press_shade_damping_min, button_press_shade_damping_max
                ),
            },
            "flip_switch_base": {
                "stiffness": np.random.uniform(
                    flip_switch_base_stiffness_min, flip_switch_base_stiffness_max
                ),
                "damping": np.random.uniform(
                    flip_switch_base_damping_min, flip_switch_base_damping_max
                ),
            },
            "button_press": {
                "stiffness": np.random.uniform(
                    button_press_stiffness_min, button_press_stiffness_max
                ),
                "damping": np.random.uniform(
                    button_press_damping_min, button_press_damping_max
                ),
            },
        }

    def sample_parameters(self):
        def sample_mat():
            gold = sample_gold()
            silver = sample_silver()

            shader = weighted_sample(
                [
                    (metal.MetalBasic, 0.7),
                    (plastic.Plastic, 0.2),
                    (plastic.BlackPlastic, 0.1),
                ]
            )()
            r = np.random.rand()
            if r < 0.3:
                return shader(color_hsv=gold)
            elif r < 0.6:
                return shader(color_hsv=silver)
            else:
                return shader()

        r = np.random.rand()
        if r < 1 / 3:
            # base, stand, button are same
            base = stand = button = sample_mat()
        elif r < 2 / 3:
            # base = stand, button is different
            base = stand = sample_mat()
            button = sample_mat()
        else:
            # base, stand, button are different
            base = sample_mat()
            stand = sample_mat()
            button = sample_mat()

        glasses = [
            (ceramic.Glass, 1.0),
            (ceramic.ColoredGlass, 0.5),
        ]

        shade_light = weighted_sample(material_assignments.lampshade)()()
        interior = metal.MetalBasic()(
            color_hsv=sample_white_interior()
        )  # TODO fix materials for lightbulb
        lamp_rack = sample_mat()
        metal_mat = weighted_sample(material_assignments.metal_neutral)()()

        materials = {
            "base": base,
            "stand": stand,
            "button": button,
            "lamp_rack": lamp_rack,
            "metal_mat": metal_mat,
            "shade_light": shade_light,
            "shade_interior": interior,
        }

        return sample_lamp_parameters(materials=materials, lamp_type=self.type)

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
