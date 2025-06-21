import functools

import numpy as np

from infinigen.assets.materials import fabrics, lamp_shaders, metal, plastic, ceramic
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.paths import blueprint_path_completion
from infinigen.assets.composition import material_assignments
from infinigen.core.util.random import weighted_sample

@node_utils.to_nodegroup("nodegroup_bulb_003", singleton=False, type="GeometryNodeTree")
def nodegroup_bulb_003(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    curve_line = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (0.0000, 0.0000, -0.2000),
            "End": (0.0000, 0.0000, 0.0000),
        },
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 100, "Radius": 0.1500}
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": curve_to_mesh, "Name": "joint14"},
        attrs={"data_type": "INT"},
    )

    spiral = nw.new_node(
        "GeometryNodeCurveSpiral",
        input_kwargs={
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
        Nodes.CurveCircle, input_kwargs={"Resolution": 100, "Radius": 0.0150}
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": transform_geometry,
            "Profile Curve": curve_circle_1.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": curve_to_mesh_1, "Name": "joint14", "Value": 1},
        attrs={"data_type": "INT"},
    )

    curve_line_1 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (0.0000, 0.0000, -0.2000),
            "End": (0.0000, 0.0000, -0.3000),
        },
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": curve_line_1, "Count": 100}
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
        Nodes.CurveCircle, input_kwargs={"Resolution": 100, "Radius": 0.1500}
    )

    curve_to_mesh_2 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius,
            "Profile Curve": curve_circle_2.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": curve_to_mesh_2, "Name": "joint14", "Value": 2},
        attrs={"data_type": "INT"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                store_named_attribute,
                store_named_attribute_1,
                store_named_attribute_2,
            ]
        },
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

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material, "Name": "joint13"},
        attrs={"data_type": "INT"},
    )

    curve_line_2 = nw.new_node(Nodes.CurveLine)

    resample_curve_1 = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": curve_line_2, "Count": 100}
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

    curve_circle_3 = nw.new_node(Nodes.CurveCircle, input_kwargs={"Resolution": 100})

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

    store_named_attribute_4 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material_1, "Name": "joint13", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_3, store_named_attribute_4]},
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
    "nodegroup_reversiable_bulb_003", singleton=False, type="GeometryNodeTree"
)
def nodegroup_reversiable_bulb_003(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

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

    bulb_003 = nw.new_node(
        nodegroup_bulb_003().name,
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
        Nodes.Transform, input_kwargs={"Geometry": bulb_003, "Scale": combine_xyz}
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
    "nodegroup_bulb_rack_003", singleton=False, type="GeometryNodeTree"
)
def nodegroup_bulb_rack_003(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

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
        attrs={"data_type": "INT", "operation": "LESS_THAN"},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": less_than,
            "False": amount.outputs["Amount"],
            "True": amount.outputs["Sides"],
        },
        attrs={"input_type": "INT"},
    )

    duplicate_elements = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={"Geometry": geometry_to_instance, "Amount": switch},
        attrs={"domain": "INSTANCE"},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": duplicate_elements.outputs["Geometry"]}
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": reroute_4}
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

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": amount.outputs["OuterHeight"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle.outputs["Curve"],
            "Translation": combine_xyz,
        },
    )

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
        input_kwargs={"Curves": transform_geometry, "Factor": multiply},
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
        Nodes.CurveCircle, input_kwargs={"Resolution": 100, "Radius": multiply_add}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": amount.outputs["InnerHeight"]}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_2})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle_1.outputs["Curve"],
            "Translation": combine_xyz_1,
        },
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply})

    sample_curve_1 = nw.new_node(
        Nodes.SampleCurve,
        input_kwargs={"Curves": transform_geometry_1, "Factor": reroute_6},
        attrs={"use_all_curves": True},
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": set_position,
            "Selection": endpoint_selection_1,
            "Position": sample_curve_1.outputs["Position"],
        },
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_position_1, "Name": "joint12"},
        attrs={"data_type": "INT"},
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry_1})

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_5, "Name": "joint12", "Value": 1},
        attrs={"data_type": "INT"},
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_3, "Name": "joint12", "Value": 2},
        attrs={"data_type": "INT"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                store_named_attribute_1,
                store_named_attribute_2,
                store_named_attribute_3,
            ]
        },
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": amount.outputs["Thickness"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    curve_circle_2 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 100, "Radius": reroute_1}
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": join_geometry,
            "Profile Curve": curve_circle_2.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    store_named_attribute_4 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": curve_to_mesh, "Name": "joint11"},
        attrs={"data_type": "INT"},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": store_named_attribute_4}
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": amount.outputs["Sides"]}
    )

    multiply_add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: amount.outputs["OuterRadius"], 1: 1.0000, 2: 0.0000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": amount.outputs["Thickness"]}
    )

    multiply_add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_7, 1: -1.0000, 2: 0.0000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": reroute_8,
            "Radius": multiply_add_1,
            "Depth": multiply_add_2,
        },
    )

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cylinder.outputs["Mesh"]}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_10, "Translation": combine_xyz},
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_2, "Name": "switch6", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": amount.outputs["ShadeTop"],
            "True": store_named_attribute,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_1, "LampShadeTop": switch_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_string_nodes_v2", singleton=False, type="GeometryNodeTree"
)
def nodegroup_string_nodes_v2(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    string = nw.new_node("FunctionNodeInputString", attrs={"string": "joint17"})

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Radius", 0.0000),
            ("NodeSocketFloat", "RadialLength", 0.0000),
            ("NodeSocketFloat", "Length", 0.0000),
            ("NodeSocketFloat", "Depth", 0.0000),
        ],
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["Radius"],
            "Depth": group_input.outputs["Depth"],
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": cylinder.outputs["Mesh"], "Name": "joint17"},
        attrs={"data_type": "INT"},
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0150

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 2: value},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add})

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: value, 1: 2.0000}, attrs={"operation": "MULTIPLY"}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Length"], 1: multiply},
        attrs={"operation": "DIVIDE"},
    )

    float_to_integer = nw.new_node(Nodes.FloatToInt, input_kwargs={"Float": divide})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: float_to_integer, 1: multiply},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 2: multiply_1},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add_1})

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_5, "End": combine_xyz_4}
    )

    curve_to_points = nw.new_node(
        Nodes.CurveToPoints,
        input_kwargs={"Curve": curve_line, "Count": float_to_integer},
    )

    uv_sphere = nw.new_node(Nodes.MeshUVSphere, input_kwargs={"Radius": value})

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

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Length"], 1: -1.0000, 2: multiply_2},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add_2})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": realize_instances, "Translation": combine_xyz_6},
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_4, "Name": "joint17", "Value": 1},
        attrs={"data_type": "INT"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Depth"]}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute, 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    sliding_joint_new = nw.new_node(
        nodegroup_sliding_joint_n_e_w().name,
        input_kwargs={
            "Joint ID (do not set)": string,
            "Parent": store_named_attribute,
            "Child": store_named_attribute_1,
            "Axis": (0.0000, 0.0000, -1.0000),
            "Max": subtract,
        },
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": sliding_joint_new.outputs["Geometry"],
            "Name": "joint16",
        },
        attrs={"data_type": "INT"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": multiply_3,
            "Y": group_input.outputs["RadialLength"],
            "Z": group_input.outputs["Depth"],
        },
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_1})

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["RadialLength"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_4})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Translation": combine_xyz_2},
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_1, "Name": "joint16", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_2, store_named_attribute_3]},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["RadialLength"], 1: 1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_5})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Translation": combine_xyz_3},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [sliding_joint_new.outputs["Parent"], transform_geometry_1]
        },
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_1, "Translation": combine_xyz_3},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_2, "Parent": transform_geometry_3},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    cube_1 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": (2.0000, 1.0000, 1.0000)}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Translation": (0.0000, 0.5000, 0.5000),
        },
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 6, "Radius": 6.0000, "Depth": 2.5200},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": (0.0000, -5.4400, 0.0000),
            "Rotation": (0.0000, 0.0000, 0.5250),
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": transform_geometry, "Mesh 2": transform_geometry_2},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": difference.outputs["Mesh"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_hinge_joint_n_e_w", singleton=False, type="GeometryNodeTree"
)
def nodegroup_hinge_joint_n_e_w(nw: NodeWrangler):
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
        attrs={"data_type": "INT", "operation": "EQUAL"},
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
        attrs={"data_type": "INT", "operation": "EQUAL"},
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
        attrs={"data_type": "INT", "operation": "EQUAL"},
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
        attrs={"data_type": "INT", "operation": "EQUAL"},
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
        attrs={"data_type": "INT", "operation": "EQUAL"},
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
    "nodegroup_sliding_joint_n_e_w", singleton=False, type="GeometryNodeTree"
)
def nodegroup_sliding_joint_n_e_w(nw: NodeWrangler):
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
        attrs={"data_type": "INT", "operation": "EQUAL"},
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
        attrs={"data_type": "INT", "operation": "EQUAL"},
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
        attrs={"data_type": "INT", "operation": "EQUAL"},
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
    "nodegroup_lamp_head_final", singleton=False, type="GeometryNodeTree"
)
def nodegroup_lamp_head_final(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

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

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ShadeCurved"]}
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["RackHeight"]}
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["ReverseBulb"], 1: 2.0000, 2: -1.0000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_add})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_11, 1: reroute_20},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["ShadeHeight"],
            1: group_input.outputs["RackHeight"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_add, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_19, 1: multiply_1},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_2})

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz, "End": combine_xyz_1}
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": curve_line, "Name": "switch5"},
        attrs={"data_type": "INT"},
    )

    b_zier_segment = nw.new_node(
        Nodes.CurveBezierSegment,
        input_kwargs={
            "Start": combine_xyz,
            "Start Handle": (0.0000, 0.0000, 0.0000),
            "End": combine_xyz_1,
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": b_zier_segment, "Name": "switch5", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_8,
            "False": store_named_attribute_1,
            "True": store_named_attribute,
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

    reroute_21 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": map_range.outputs["Result"]}
    )

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius, input_kwargs={"Curve": switch, "Radius": reroute_21}
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Sides"]}
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_17})

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    curve_circle = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": reroute_26}
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

    reroute_14 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["LampShadeInteriorMaterial"]},
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_15})

    set_material_5 = nw.new_node(
        Nodes.SetMaterial, input_kwargs={"Geometry": flip_faces, "Material": reroute_22}
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material_5, "Name": "joint9"},
        attrs={"data_type": "INT"},
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": set_shade_smooth,
            "Offset Scale": 0.0050,
            "Individual": False,
        },
    )

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["LampshadeMaterial"]}
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_27})

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": extrude_mesh.outputs["Mesh"], "Material": reroute_28},
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material_3, "Name": "joint9", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_2, store_named_attribute_3]},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry,
            "Translation": (0.0000, 0.0000, 0.0010),
        },
    )

    store_named_attribute_4 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry, "Name": "joint8"},
        attrs={"data_type": "INT"},
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["RackThickness"]}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["TopRadius"], 1: 0.8000},
        attrs={"operation": "MULTIPLY"},
    )

    maximum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_3, 1: 0.0600},
        attrs={"operation": "MAXIMUM"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: maximum, 1: 0.1500},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["TopRadius"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BlackMaterial"]}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["MetalMaterial"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reversiable_bulb_003 = nw.new_node(
        nodegroup_reversiable_bulb_003().name,
        input_kwargs={
            "Scale": maximum,
            "BlackMaterial": reroute_16,
            "LampshadeMaterial": reroute_15,
            "MetalMaterial": reroute_1,
        },
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ShadeTop"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    bulb_rack_003 = nw.new_node(
        nodegroup_bulb_rack_003().name,
        input_kwargs={
            "Thickness": reroute_6,
            "InnerRadius": multiply_4,
            "OuterRadius": reroute_5,
            "InnerHeight": reversiable_bulb_003.outputs["RackSupport"],
            "OuterHeight": multiply,
            "ShadeTop": reroute_3,
            "Sides": reroute_18,
        },
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": bulb_rack_003.outputs["LampShadeTop"],
            "Material": reroute_13,
        },
    )

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_2})

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_29, "Name": "joint8", "Value": 1},
        attrs={"data_type": "INT"},
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["IncludeLightbulb"]}
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    greater_than = nw.new_node(
        Nodes.Compare, input_kwargs={0: group_input.outputs["RackThickness"]}
    )

    reroute_25 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": reversiable_bulb_003.outputs["Geometry"]}
    )

    store_named_attribute_9 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_25, "Name": "switch8"},
        attrs={"data_type": "INT"},
    )

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": bulb_rack_003.outputs["Geometry"],
            "Material": reroute_24,
        },
    )

    store_named_attribute_6 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material, "Name": "joint10"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_7 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_25, "Name": "joint10", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_6, store_named_attribute_7]},
    )

    store_named_attribute_8 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_2, "Name": "switch8", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": greater_than,
            "False": store_named_attribute_9,
            "True": store_named_attribute_8,
        },
    )

    store_named_attribute_10 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_3, "Name": "switch7", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_10, "True": store_named_attribute_10},
    )

    reroute_30 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_1})

    store_named_attribute_11 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_30, "Name": "joint8", "Value": 2},
        attrs={"data_type": "INT"},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                store_named_attribute_4,
                store_named_attribute_5,
                store_named_attribute_11,
            ]
        },
    )

    store_named_attribute_12 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_1, "Name": "joint7"},
        attrs={"data_type": "INT"},
    )

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": store_named_attribute_12}
    )

    store_named_attribute_13 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_4, "Name": "switch4"},
        attrs={"data_type": "INT"},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    store_named_attribute_14 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_1, "Name": "joint15"},
        attrs={"data_type": "INT"},
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": store_named_attribute_14}
    )

    store_named_attribute_15 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_3, "Name": "switch4", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["HeadSidways"],
            "False": store_named_attribute_13,
            "True": store_named_attribute_15,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": switch_2},
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
            "Value": 7,
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
    # Code generated using version 2.6.5 of the node_transpiler

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

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BaseRotate"]}
    )

    reroute_64 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["IncludeButtonBase"]}
    )

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": group_input.outputs["BaseSides"],
            "Radius": group_input.outputs["BaseRadius"],
            "Depth": group_input.outputs["BaseHeight"],
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": cylinder_1.outputs["Mesh"], "Label": "lamp_base"},
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BaseMaterial"]}
    )

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata,
            "Material": reroute_17,
        },
    )

    store_named_attribute_56 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material_3, "Name": "joint31"},
        attrs={"data_type": "INT"},
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["Radius"],
            "Depth": group_input.outputs["Height"],
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_8,
        },
    )

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_1, "Label": "bar"},
    )

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["StandMaterial"]}
    )

    set_material_4 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_1,
            "Material": reroute_16,
        },
    )

    store_named_attribute_57 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material_4, "Name": "joint31", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_56, store_named_attribute_57]},
    )

    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_1})

    store_named_attribute_83 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_2, "Name": "joint38"},
        attrs={"data_type": "INT"},
    )

    reroute_20 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ButtonOnLampShade"]}
    )

    reroute_23 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ShadeHeight"]}
    )

    reroute_24 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["RackHeight"]}
    )

    reroute_25 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["HeadSideways"]}
    )

    lamp_head_final = nw.new_node(
        nodegroup_lamp_head_final().name,
        input_kwargs={
            "ShadeHeight": reroute_23,
            "TopRadius": group_input.outputs["TopRadius"],
            "BotRadius": group_input.outputs["BottomRadius"],
            "ReverseBulb": group_input.outputs["ReverseBulb"],
            "RackThickness": group_input.outputs["RackThickness"],
            "RackHeight": reroute_24,
            "BlackMaterial": group_input.outputs["LampRackMaterial"],
            "LampshadeMaterial": group_input.outputs["ShadeMaterial"],
            "MetalMaterial": group_input.outputs["MetalMaterial"],
            "LampShadeInteriorMaterial": group_input.outputs["ShadeInteriorMaterial"],
            "ShadeTop": group_input.outputs["ShadeTop"],
            "ShadeCurved": group_input.outputs["ShadeCurved"],
            "IncludeLightbulb": group_input.outputs["IncludeLightBulb"],
            "HeadSidways": reroute_25,
            "Sides": group_input.outputs["ShadeSides"],
        },
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": lamp_head_final}
    )

    add_jointed_geometry_metadata_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": realize_instances, "Label": "head"},
    )

    reroute_19 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_2}
    )

    store_named_attribute_19 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_19, "Name": "switch3"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_19, "Name": "joint6"},
        attrs={"data_type": "INT"},
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["ButtonType"]},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    reroute_54 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ButtonType"]}
    )

    greater_equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_54, 3: 2},
        attrs={"data_type": "INT", "operation": "GREATER_EQUAL"},
    )

    string_3 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint18"})

    reroute_29 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ButtonR1"]}
    )

    reroute_31 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ButtonH1"]}
    )

    cylinder_4 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Radius": reroute_29, "Depth": reroute_31},
    )

    add_jointed_geometry_metadata_3 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": cylinder_4.outputs["Mesh"], "Label": "button"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={1: reroute_31}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_19 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_1})

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_3,
            "Translation": combine_xyz_19,
        },
    )

    combine_xyz_36 = nw.new_node(Nodes.CombineXYZ)

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["RackHeight"], 1: -2.0000, 2: 0.0000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_35 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_add})

    switch_25 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["HeadSideways"],
            "False": combine_xyz_36,
            "True": combine_xyz_35,
        },
        attrs={"input_type": "VECTOR"},
    )

    combine_xyz_16 = nw.new_node(Nodes.CombineXYZ)

    combine_xyz_15 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": 1.5700, "Z": 3.1400}
    )

    switch_8 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_25,
            "False": combine_xyz_16,
            "True": combine_xyz_15,
        },
        attrs={"input_type": "VECTOR"},
    )

    transform_geometry_20 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_5,
            "Translation": switch_25,
            "Rotation": switch_8,
        },
    )

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_20, "Name": "joint18"},
        attrs={"data_type": "INT"},
    )

    reroute_30 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ButtonR2"]}
    )

    cylinder_3 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Radius": reroute_30, "Depth": reroute_31},
    )

    add_jointed_geometry_metadata_4 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": cylinder_3.outputs["Mesh"], "Label": "button"},
    )

    combine_xyz_20 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_31})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_4,
            "Translation": combine_xyz_20,
        },
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ)

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_1})

    reroute_42 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_4, "Rotation": reroute_42},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_31, 1: 0.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_21 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_2})

    transform_geometry_21 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_6,
            "Translation": combine_xyz_21,
            "Rotation": switch_8,
        },
    )

    store_named_attribute_4 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_21, "Name": "joint18", "Value": 1},
        attrs={"data_type": "INT"},
    )

    combine_xyz_18 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": -1.0000})

    combine_xyz_17 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": 1.0000})

    switch_10 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_25,
            "False": combine_xyz_18,
            "True": combine_xyz_17,
        },
        attrs={"input_type": "VECTOR"},
    )

    sliding_joint_new = nw.new_node(
        nodegroup_sliding_joint_n_e_w().name,
        input_kwargs={
            "Joint ID (do not set)": string_3,
            "Parent": store_named_attribute_5,
            "Child": store_named_attribute_4,
            "Axis": switch_10,
            "Max": reroute_31,
        },
    )

    store_named_attribute_6 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": sliding_joint_new.outputs["Geometry"],
            "Name": "switch10",
        },
        attrs={"data_type": "INT"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_54, 3: 3},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    string_4 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint19"})

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_29})

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_33})

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_31})

    reroute_39 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_35})

    cylinder_12 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Radius": reroute_37, "Depth": reroute_39},
    )

    add_jointed_geometry_metadata_5 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": cylinder_12.outputs["Mesh"], "Label": "button"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math, input_kwargs={1: reroute_39}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_22 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_3})

    transform_geometry_22 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_5,
            "Translation": combine_xyz_22,
        },
    )

    reroute_45 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_8})

    transform_geometry_26 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_22,
            "Translation": switch_25,
            "Rotation": reroute_45,
        },
    )

    store_named_attribute_7 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_26, "Name": "joint19"},
        attrs={"data_type": "INT"},
    )

    reroute_34 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_30})

    reroute_38 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_34})

    reroute_32 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ButtonH2"]}
    )

    reroute_36 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_32})

    reroute_40 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_36})

    cylinder_11 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Radius": reroute_38, "Depth": reroute_40},
    )

    add_jointed_geometry_metadata_6 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": cylinder_11.outputs["Mesh"], "Label": "button"},
    )

    store_named_attribute_8 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": add_jointed_geometry_metadata_6, "Name": "joint20"},
        attrs={"data_type": "INT"},
    )

    reroute_43 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_38})

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_43, 1: 0.4000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_43, 1: 1.6000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_25 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": multiply_4, "Y": multiply_5, "Z": multiply_4},
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_25})

    multiply_add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_40, 2: 0.0000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_26 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add_1})

    transform_geometry_28 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Translation": combine_xyz_26},
    )

    store_named_attribute_9 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_28, "Name": "joint20", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_13 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_8, store_named_attribute_9]},
    )

    multiply_6 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_39}, attrs={"operation": "MULTIPLY"}
    )

    multiply_add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_40, 2: multiply_6},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_23 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add_2})

    transform_geometry_24 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_13, "Translation": combine_xyz_23},
    )

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_42})

    transform_geometry_25 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_24, "Rotation": reroute_51},
    )

    combine_xyz_24 = nw.new_node(Nodes.CombineXYZ)

    transform_geometry_27 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_25,
            "Translation": combine_xyz_24,
            "Rotation": reroute_45,
        },
    )

    store_named_attribute_10 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_27, "Name": "joint19", "Value": 1},
        attrs={"data_type": "INT"},
    )

    reroute_46 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["HeadSideways"]}
    )

    combine_xyz_27 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": 1.0000})

    combine_xyz_28 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": 1.0000})

    switch_13 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_46,
            "False": combine_xyz_27,
            "True": combine_xyz_28,
        },
        attrs={"input_type": "VECTOR"},
    )

    hinge_joint_new = nw.new_node(
        nodegroup_hinge_joint_n_e_w().name,
        input_kwargs={
            "Joint ID (do not set)": string_4,
            "Parent": store_named_attribute_7,
            "Child": store_named_attribute_10,
            "Axis": switch_13,
        },
    )

    store_named_attribute_11 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": hinge_joint_new.outputs["Geometry"],
            "Name": "switch11",
        },
        attrs={"data_type": "INT"},
    )

    string_5 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint21"})

    reroute_49 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_35})

    combine_xyz_31 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_49, "Y": 0.7000, "Z": 1.0000}
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_31})

    add_jointed_geometry_metadata_7 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": cube_1.outputs["Mesh"], "Label": "button"},
    )

    reroute_47 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_33})

    combine_xyz_29 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": reroute_47, "Y": reroute_47, "Z": reroute_47},
    )

    transform_geometry_31 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_7,
            "Rotation": (1.5708, 0.0000, 1.5708),
            "Scale": combine_xyz_29,
        },
    )

    reroute_52 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_45})

    transform_geometry_35 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_31,
            "Translation": switch_25,
            "Rotation": reroute_52,
        },
    )

    store_named_attribute_12 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_35, "Name": "joint21"},
        attrs={"data_type": "INT"},
    )

    nodegroup = nw.new_node(nodegroup_node_group().name)

    add_jointed_geometry_metadata_8 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": nodegroup, "Label": "button"},
    )

    transform_geometry_29 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_8,
            "Translation": (0.0000, -0.5000, -0.5000),
        },
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_47, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_32 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_7})

    reroute_48 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_34})

    combine_xyz_30 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": reroute_48, "Y": reroute_48, "Z": reroute_48},
    )

    transform_geometry_30 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_29,
            "Translation": combine_xyz_32,
            "Rotation": (1.5708, 0.0000, 1.5708),
            "Scale": combine_xyz_30,
        },
    )

    transform_geometry_34 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_30,
            "Rotation": (0.0000, 3.1416, 0.0000),
        },
    )

    transform_geometry_36 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_34, "Rotation": reroute_52},
    )

    store_named_attribute_13 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_36, "Name": "joint21", "Value": 1},
        attrs={"data_type": "INT"},
    )

    reroute_53 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["HeadSideways"]}
    )

    combine_xyz_33 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": 1.0000})

    combine_xyz_34 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": 1.0000})

    switch_16 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_53,
            "False": combine_xyz_33,
            "True": combine_xyz_34,
        },
        attrs={"input_type": "VECTOR"},
    )

    hinge_joint_new_1 = nw.new_node(
        nodegroup_hinge_joint_n_e_w().name,
        input_kwargs={
            "Joint ID (do not set)": string_5,
            "Parent": store_named_attribute_12,
            "Child": store_named_attribute_13,
            "Axis": switch_16,
            "Min": -0.2500,
            "Max": 0.2500,
        },
    )

    store_named_attribute_14 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": hinge_joint_new_1.outputs["Geometry"],
            "Name": "switch11",
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    switch_15 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_1,
            "False": store_named_attribute_11,
            "True": store_named_attribute_14,
        },
    )

    store_named_attribute_15 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_15, "Name": "switch10", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch_11 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": greater_equal,
            "False": store_named_attribute_6,
            "True": store_named_attribute_15,
        },
    )

    reroute_26 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ButtonMaterial"]}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial, input_kwargs={"Geometry": switch_11, "Material": reroute_26}
    )

    combine_xyz_14 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_24})

    combine_xyz_13 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": reroute_24})

    switch_7 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_25,
            "False": combine_xyz_14,
            "True": combine_xyz_13,
        },
        attrs={"input_type": "VECTOR"},
    )

    transform_geometry_19 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": set_material, "Translation": switch_7},
    )

    store_named_attribute_16 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_19, "Name": "switch9"},
        attrs={"data_type": "INT"},
    )

    string_nodes_v2 = nw.new_node(
        nodegroup_string_nodes_v2().name,
        input_kwargs={
            "Radius": group_input.outputs["ButtonR1"],
            "RadialLength": 0.0900,
            "Length": group_input.outputs["ButtonH2"],
            "Depth": group_input.outputs["ButtonH1"],
        },
    )

    add_jointed_geometry_metadata_9 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": string_nodes_v2.outputs["Geometry"],
            "Label": "button",
        },
    )

    set_material_11 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_9,
            "Material": group_input.outputs["ButtonMaterial"],
        },
    )

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_11})

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_22, "Name": "switch9", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch_9 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal,
            "False": store_named_attribute_16,
            "True": store_named_attribute_3,
        },
    )

    store_named_attribute_17 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_9, "Name": "joint6", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_12 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_2, store_named_attribute_17]},
    )

    store_named_attribute_18 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_12, "Name": "switch3", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_20,
            "False": store_named_attribute_19,
            "True": store_named_attribute_18,
        },
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_6})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Height"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": reroute, "Translation": combine_xyz}
    )

    store_named_attribute_84 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry, "Name": "joint38", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_83, store_named_attribute_84]},
    )

    store_named_attribute_85 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_2, "Name": "switch20"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_86 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_2, "Name": "joint39"},
        attrs={"data_type": "INT"},
    )

    greater_equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_54, 3: 2},
        attrs={"data_type": "INT", "operation": "GREATER_EQUAL"},
    )

    string_8 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint26"})

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["ButtonR1"],
            1: group_input.outputs["Radius"],
        },
    )

    multiply_add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: -2.0000, 1: add, 2: group_input.outputs["BaseRadius"]},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    multiply_add_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["BaseButtonOffset"],
            1: multiply_add_3,
            2: add,
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    multiply_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["BaseHeight"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_add_4, "Z": multiply_8}
    )

    reroute_44 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_2})

    reroute_41 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_5,
            "Translation": reroute_44,
            "Rotation": reroute_41,
        },
    )

    store_named_attribute_40 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_3, "Name": "joint26"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_41 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_6, "Name": "joint26", "Value": 1},
        attrs={"data_type": "INT"},
    )

    sliding_joint_new_1 = nw.new_node(
        nodegroup_sliding_joint_n_e_w().name,
        input_kwargs={
            "Joint ID (do not set)": string_8,
            "Parent": store_named_attribute_40,
            "Child": store_named_attribute_41,
            "Axis": (0.0000, 0.0000, -1.0000),
            "Max": reroute_31,
        },
    )

    store_named_attribute_42 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": sliding_joint_new_1.outputs["Geometry"],
            "Name": "switch13",
        },
        attrs={"data_type": "INT"},
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_54, 3: 3},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    string_9 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint27"})

    transform_geometry_23 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_22,
            "Translation": reroute_44,
            "Rotation": reroute_41,
        },
    )

    store_named_attribute_43 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_23, "Name": "joint27"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_44 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_25, "Name": "joint27", "Value": 1},
        attrs={"data_type": "INT"},
    )

    hinge_joint_new_2 = nw.new_node(
        nodegroup_hinge_joint_n_e_w().name,
        input_kwargs={
            "Joint ID (do not set)": string_9,
            "Parent": store_named_attribute_43,
            "Child": store_named_attribute_44,
        },
    )

    store_named_attribute_45 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": hinge_joint_new_2.outputs["Geometry"],
            "Name": "switch14",
        },
        attrs={"data_type": "INT"},
    )

    string_10 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint28"})

    transform_geometry_33 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_31,
            "Translation": reroute_44,
            "Rotation": reroute_41,
        },
    )

    store_named_attribute_46 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_33, "Name": "joint28"},
        attrs={"data_type": "INT"},
    )

    transform_geometry_32 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_34, "Rotation": reroute_51},
    )

    store_named_attribute_47 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_32, "Name": "joint28", "Value": 1},
        attrs={"data_type": "INT"},
    )

    hinge_joint_new_3 = nw.new_node(
        nodegroup_hinge_joint_n_e_w().name,
        input_kwargs={
            "Joint ID (do not set)": string_10,
            "Parent": store_named_attribute_46,
            "Child": store_named_attribute_47,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Min": -0.2500,
            "Max": 0.2500,
        },
    )

    store_named_attribute_48 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": hinge_joint_new_3.outputs["Geometry"],
            "Name": "switch14",
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    switch_14 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_2,
            "False": store_named_attribute_45,
            "True": store_named_attribute_48,
        },
    )

    store_named_attribute_49 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_14, "Name": "switch13", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch_12 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": greater_equal_1,
            "False": store_named_attribute_42,
            "True": store_named_attribute_49,
        },
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial, input_kwargs={"Geometry": switch_12, "Material": reroute_26}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_1})

    store_named_attribute_87 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_1, "Name": "joint39", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_16 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_86, store_named_attribute_87]},
    )

    store_named_attribute_88 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_16, "Name": "switch20", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch_21 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_64,
            "False": store_named_attribute_85,
            "True": store_named_attribute_88,
        },
    )

    store_named_attribute_89 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_21, "Name": "switch19"},
        attrs={"data_type": "INT"},
    )

    string_12 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint36"})

    add_jointed_geometry_metadata_10 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": cylinder_1.outputs["Mesh"], "Label": "lamp_base"},
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_10,
            "Material": reroute_17,
        },
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_2})

    store_named_attribute_39 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_15, "Name": "switch12"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_50 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_1, "Name": "joint25"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_51 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_15, "Name": "joint25", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_8 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_50, store_named_attribute_51]},
    )

    store_named_attribute_52 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_8, "Name": "switch12", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch_20 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_64,
            "False": store_named_attribute_39,
            "True": store_named_attribute_52,
        },
    )

    store_named_attribute_81 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_20, "Name": "joint36"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_78 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry, "Name": "joint37"},
        attrs={"data_type": "INT"},
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_4})

    store_named_attribute_79 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_18, "Name": "joint37", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_9 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_78, store_named_attribute_79]},
    )

    store_named_attribute_80 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_9, "Name": "joint36", "Value": 1},
        attrs={"data_type": "INT"},
    )

    hinge_joint_new_4 = nw.new_node(
        nodegroup_hinge_joint_n_e_w().name,
        input_kwargs={
            "Joint ID (do not set)": string_12,
            "Parent": store_named_attribute_81,
            "Child": store_named_attribute_80,
        },
    )

    store_named_attribute_82 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": hinge_joint_new_4.outputs["Geometry"],
            "Name": "switch19",
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_4,
            "False": store_named_attribute_89,
            "True": store_named_attribute_82,
        },
    )

    store_named_attribute_90 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_3, "Name": "switch16"},
        attrs={"data_type": "INT"},
    )

    string_2 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint4"})

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Radius"]}
    )

    multiply_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_6, 1: 2.4000},
        attrs={"operation": "MULTIPLY"},
    )

    cylinder_7 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Radius": reroute_6, "Depth": multiply_9},
    )

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_7.outputs["Mesh"],
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    add_jointed_geometry_metadata_11 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_7, "Label": "bar"},
    )

    reroute_27 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["StandMaterial"]}
    )

    set_material_6 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_11,
            "Material": reroute_27,
        },
    )

    store_named_attribute_22 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material_6, "Name": "joint4"},
        attrs={"data_type": "INT"},
    )

    cylinder_5 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Radius": reroute_6, "Depth": multiply_9},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_5.outputs["Mesh"],
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    add_jointed_geometry_metadata_12 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_2, "Label": "bar"},
    )

    set_material_5 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_12,
            "Material": reroute_27,
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material_5, "Name": "joint5"},
        attrs={"data_type": "INT"},
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["FirstBarLength"]}
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["FirstBarExtension"]}
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_9, 1: reroute_8})

    cylinder_6 = nw.new_node(
        "GeometryNodeMeshCylinder", input_kwargs={"Radius": reroute_6, "Depth": add_1}
    )

    multiply_10 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_9}, attrs={"operation": "MULTIPLY"}
    )

    multiply_add_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_8, 1: -0.5000, 2: multiply_10},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add_5})

    transform_geometry_8 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_6.outputs["Mesh"],
            "Translation": combine_xyz_7,
        },
    )

    add_jointed_geometry_metadata_13 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_8, "Label": "bar"},
    )

    set_material_7 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_13,
            "Material": reroute_27,
        },
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_material_7, "Name": "joint5", "Value": 1},
        attrs={"data_type": "INT"},
    )

    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_9})

    transform_geometry_9 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute, "Translation": combine_xyz_9},
    )

    store_named_attribute_20 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_9, "Name": "joint5", "Value": 2},
        attrs={"data_type": "INT"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                store_named_attribute,
                store_named_attribute_1,
                store_named_attribute_20,
            ]
        },
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_9})

    transform_geometry_10 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Translation": combine_xyz_5},
    )

    store_named_attribute_21 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_10, "Name": "joint4", "Value": 1},
        attrs={"data_type": "INT"},
    )

    reroute_65 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    greater_equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_54, 3: 2},
        attrs={"data_type": "INT", "operation": "GREATER_EQUAL"},
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_54, 3: 3},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    switch_18 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_3,
            "False": transform_geometry_26,
            "True": transform_geometry_35,
        },
    )

    switch_17 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": greater_equal_2,
            "False": transform_geometry_20,
            "True": switch_18,
        },
    )

    transform_geometry_38 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": switch_17, "Translation": switch_7}
    )

    reroute_61 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_38}
    )

    reroute_62 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": string_nodes_v2.outputs["Parent"]}
    )

    switch_19 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": reroute_61, "True": reroute_62},
    )

    join_geometry_15 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [reroute_65, switch_19]}
    )

    switch_24 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["ButtonOnLampShade"],
            "False": reroute_65,
            "True": join_geometry_15,
        },
    )

    reroute_58 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_24})

    transform_geometry_37 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_58, "Translation": combine_xyz_9},
    )

    join_geometry_14 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [transform_geometry_37, set_material_7, set_material_5]
        },
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": join_geometry_14}
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": -0.5000, "Y": -0.5000, "Z": -0.5000}
    )

    multiply_11 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Min"], 1: combine_xyz_3},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_12 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: combine_xyz_3},
        attrs={"operation": "MULTIPLY"},
    )

    add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: multiply_11.outputs["Vector"],
            1: multiply_12.outputs["Vector"],
        },
    )

    hinge_joint_new_5 = nw.new_node(
        nodegroup_hinge_joint_n_e_w().name,
        input_kwargs={
            "Joint ID (do not set)": string_2,
            "Parent": store_named_attribute_22,
            "Child": store_named_attribute_21,
            "Position": add_2.outputs["Vector"],
            "Axis": (0.0000, 1.0000, 0.0000),
        },
    )

    transform_geometry_11 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": hinge_joint_new_5.outputs["Geometry"],
            "Translation": combine_xyz,
        },
    )

    store_named_attribute_70 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_11, "Name": "joint35"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_71 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_2, "Name": "joint35", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_70, store_named_attribute_71]},
    )

    store_named_attribute_75 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_3, "Name": "switch18"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_72 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_3, "Name": "joint34"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_73 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_1, "Name": "joint34", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_18 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_72, store_named_attribute_73]},
    )

    store_named_attribute_74 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_18, "Name": "switch18", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch_23 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_64,
            "False": store_named_attribute_75,
            "True": store_named_attribute_74,
        },
    )

    store_named_attribute_76 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_23, "Name": "switch17"},
        attrs={"data_type": "INT"},
    )

    string_11 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint32"})

    store_named_attribute_68 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_20, "Name": "joint32"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_65 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_11, "Name": "joint33"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_66 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_18, "Name": "joint33", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_11 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_65, store_named_attribute_66]},
    )

    store_named_attribute_67 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_11, "Name": "joint32", "Value": 1},
        attrs={"data_type": "INT"},
    )

    hinge_joint_new_6 = nw.new_node(
        nodegroup_hinge_joint_n_e_w().name,
        input_kwargs={
            "Joint ID (do not set)": string_11,
            "Parent": store_named_attribute_68,
            "Child": store_named_attribute_67,
        },
    )

    store_named_attribute_69 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": hinge_joint_new_6.outputs["Geometry"],
            "Name": "switch17",
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_4,
            "False": store_named_attribute_76,
            "True": store_named_attribute_69,
        },
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_5})

    store_named_attribute_77 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_3, "Name": "switch16", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["SingleHinge"],
            "False": store_named_attribute_90,
            "True": store_named_attribute_77,
        },
    )

    store_named_attribute_91 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_1, "Name": "switch0"},
        attrs={"data_type": "INT"},
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["SecondBarSliding"]}
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    string_1 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint2"})

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    multiply_13 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_7, 1: 2.4000},
        attrs={"operation": "MULTIPLY"},
    )

    cylinder_10 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Radius": reroute_7, "Depth": multiply_13},
    )

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_27})

    set_material_10 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": cylinder_10.outputs["Mesh"], "Material": reroute_28},
    )

    add_jointed_geometry_metadata_14 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_10, "Label": "bar"},
    )

    transform_geometry_13 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_14,
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_13}
    )

    store_named_attribute_27 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_12, "Name": "joint2"},
        attrs={"data_type": "INT"},
    )

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["SecondBarLength"]}
    )

    combine_xyz_10 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_10})

    transform_geometry_15 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": hinge_joint_new_5.outputs["Geometry"],
            "Translation": combine_xyz_10,
        },
    )

    store_named_attribute_23 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_15, "Name": "joint3"},
        attrs={"data_type": "INT"},
    )

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["SecondBarExtension"]}
    )

    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_10, 1: reroute_11})

    cylinder_9 = nw.new_node(
        "GeometryNodeMeshCylinder", input_kwargs={"Radius": reroute_7, "Depth": add_3}
    )

    set_material_8 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": cylinder_9.outputs["Mesh"], "Material": reroute_28},
    )

    add_jointed_geometry_metadata_15 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_8, "Label": "bar"},
    )

    multiply_14 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_10}, attrs={"operation": "MULTIPLY"}
    )

    multiply_add_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_11, 1: -0.5000, 2: multiply_14},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_12 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add_6})

    transform_geometry_14 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_15,
            "Translation": combine_xyz_12,
        },
    )

    store_named_attribute_24 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_14, "Name": "joint3", "Value": 1},
        attrs={"data_type": "INT"},
    )

    cylinder_8 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Radius": reroute_7, "Depth": multiply_13},
    )

    set_material_9 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": cylinder_8.outputs["Mesh"], "Material": reroute_28},
    )

    add_jointed_geometry_metadata_16 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_9, "Label": "bar"},
    )

    transform_geometry_12 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_16,
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    store_named_attribute_25 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_12, "Name": "joint3", "Value": 2},
        attrs={"data_type": "INT"},
    )

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                store_named_attribute_23,
                store_named_attribute_24,
                store_named_attribute_25,
            ]
        },
    )

    multiply_15 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_13, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_15})

    transform_geometry_16 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_4, "Translation": combine_xyz_6},
    )

    store_named_attribute_26 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_16, "Name": "joint2", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_6 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_14, transform_geometry_12]},
    )

    bounding_box_1 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": join_geometry_6}
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": -0.5000, "Y": -0.5000, "Z": -0.5000}
    )

    multiply_16 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box_1.outputs["Min"], 1: combine_xyz_4},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_17 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box_1.outputs["Max"], 1: combine_xyz_4},
        attrs={"operation": "MULTIPLY"},
    )

    add_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: multiply_16.outputs["Vector"],
            1: multiply_17.outputs["Vector"],
        },
    )

    multiply_18 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_7}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_11 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_18})

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_4.outputs["Vector"], 1: combine_xyz_11},
        attrs={"operation": "SUBTRACT"},
    )

    hinge_joint_new_7 = nw.new_node(
        nodegroup_hinge_joint_n_e_w().name,
        input_kwargs={
            "Joint ID (do not set)": string_1,
            "Parent": store_named_attribute_27,
            "Child": store_named_attribute_26,
            "Position": subtract.outputs["Vector"],
            "Axis": (0.0000, 1.0000, 0.0000),
        },
    )

    store_named_attribute_28 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": hinge_joint_new_7.outputs["Geometry"],
            "Name": "switch2",
        },
        attrs={"data_type": "INT"},
    )

    string_6 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint22"})

    store_named_attribute_34 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_13, "Name": "joint22"},
        attrs={"data_type": "INT"},
    )

    string_7 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint23"})

    store_named_attribute_32 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_12, "Name": "joint23"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_29 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_15, "Name": "joint24"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_30 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_14, "Name": "joint24", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_7 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_29, store_named_attribute_30]},
    )

    store_named_attribute_31 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_7, "Name": "joint23", "Value": 1},
        attrs={"data_type": "INT"},
    )

    multiply_add_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_10, 1: -1.0000, 2: 0.0500},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_11, 1: 0.0500},
        attrs={"operation": "SUBTRACT"},
    )

    sliding_joint_new_2 = nw.new_node(
        nodegroup_sliding_joint_n_e_w().name,
        input_kwargs={
            "Joint ID (do not set)": string_7,
            "Parent": store_named_attribute_32,
            "Child": store_named_attribute_31,
            "Min": multiply_add_7,
            "Max": subtract_1,
        },
    )

    transform_geometry_18 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": sliding_joint_new_2.outputs["Geometry"],
            "Translation": combine_xyz_6,
        },
    )

    store_named_attribute_33 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_18, "Name": "joint22", "Value": 1},
        attrs={"data_type": "INT"},
    )

    hinge_joint_new_8 = nw.new_node(
        nodegroup_hinge_joint_n_e_w().name,
        input_kwargs={
            "Joint ID (do not set)": string_6,
            "Parent": store_named_attribute_34,
            "Child": store_named_attribute_33,
            "Axis": (0.0000, 1.0000, 0.0000),
        },
    )

    store_named_attribute_35 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": hinge_joint_new_8.outputs["Geometry"],
            "Name": "switch2",
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_14,
            "False": store_named_attribute_28,
            "True": store_named_attribute_35,
        },
    )

    transform_geometry_17 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": switch_2, "Translation": combine_xyz}
    )

    store_named_attribute_55 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_17, "Name": "joint30"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_58 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_2, "Name": "joint30", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_5 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_55, store_named_attribute_58]},
    )

    store_named_attribute_62 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_5, "Name": "switch15"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_59 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_5, "Name": "joint29"},
        attrs={"data_type": "INT"},
    )

    reroute_63 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    store_named_attribute_60 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_63, "Name": "joint29", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_17 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_59, store_named_attribute_60]},
    )

    store_named_attribute_61 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_17, "Name": "switch15", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch_22 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_64,
            "False": store_named_attribute_62,
            "True": store_named_attribute_61,
        },
    )

    store_named_attribute_63 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_22, "Name": "switch1"},
        attrs={"data_type": "INT"},
    )

    string = nw.new_node("FunctionNodeInputString", attrs={"string": "joint0"})

    store_named_attribute_53 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_20, "Name": "joint0"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_36 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": transform_geometry_17, "Name": "joint1"},
        attrs={"data_type": "INT"},
    )

    store_named_attribute_37 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_18, "Name": "joint1", "Value": 1},
        attrs={"data_type": "INT"},
    )

    join_geometry_10 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute_36, store_named_attribute_37]},
    )

    store_named_attribute_38 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": join_geometry_10, "Name": "joint0", "Value": 1},
        attrs={"data_type": "INT"},
    )

    hinge_joint_new_9 = nw.new_node(
        nodegroup_hinge_joint_n_e_w().name,
        input_kwargs={
            "Joint ID (do not set)": string,
            "Parent": store_named_attribute_53,
            "Child": store_named_attribute_38,
        },
    )

    store_named_attribute_54 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": hinge_joint_new_9.outputs["Geometry"],
            "Name": "switch1",
            "Value": 1,
        },
        attrs={"data_type": "INT"},
    )

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_4,
            "False": store_named_attribute_63,
            "True": store_named_attribute_54,
        },
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_4})

    store_named_attribute_64 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": reroute_5, "Name": "switch0", "Value": 1},
        attrs={"data_type": "INT"},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["DoubleHinge"],
            "False": store_named_attribute_91,
            "True": store_named_attribute_64,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": switch},
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
        fabric_params = fabrics.fine_knit_fabric.get_texture_params()
        fabric_params["_color"] = color[:3]  # np.ones(3) * np.random.uniform(0.8, 1.0)
        fabric_params["_map"] = "Object"
        return fabrics.fine_knit_fabric.shader_material(nw, **fabric_params)

    return shader


class LampFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)
        self.sim_blueprint = blueprint_path_completion("lamp.json")
        self.type = None

    def sample_parameters(self):

        def sample_mat():
            gold = sample_gold()
            silver = sample_silver()
        
            shader = weighted_sample([
                (metal.MetalBasic, 0.7),
                (plastic.Plastic, 0.2),
                (plastic.BlackPlastic, 0.1),
            ])()
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
        interior = metal.MetalBasic()(color_hsv=sample_white_interior()) # TODO fix materials for lightbulb
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

    def create_asset(self, export=True, exporter="mjcf", asset_params=None, **kwargs):
        obj = butil.spawn_vert()
        butil.modify_mesh(
            obj,
            "NODES",
            apply=export,
            node_group=geometry_nodes(),
            ng_inputs=self.sample_parameters(),
        )

        return obj
