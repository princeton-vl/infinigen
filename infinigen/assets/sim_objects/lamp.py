import functools

import gin
import numpy as np
from numpy.random import uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.materials import ceramic, fabric, metal, plastic
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import weighted_sample


@node_utils.to_nodegroup("nodegroup_bulb_003", singleton=False, type="GeometryNodeTree")
def nodegroup_bulb_003(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

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
    "nodegroup_reversable_bulb", singleton=False, type="GeometryNodeTree"
)
def nodegroup_reversable_bulb(nw: NodeWrangler):
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
    "nodegroup_bulb_rack", singleton=False, type="GeometryNodeTree"
)
def nodegroup_bulb_rack(nw: NodeWrangler):
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

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": duplicate_elements.outputs["Geometry"]}
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": reroute}
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

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": amount.outputs["InnerHeight"]}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_1})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle_1.outputs["Curve"],
            "Translation": combine_xyz_1,
        },
    )

    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply})

    sample_curve_1 = nw.new_node(
        Nodes.SampleCurve,
        input_kwargs={"Curves": transform_geometry_1, "Factor": reroute_2},
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

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry_1})

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [set_position_1, reroute_3, reroute_4]},
    )

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": amount.outputs["Thickness"]}
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    curve_circle_2 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 100, "Radius": reroute_6}
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

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": amount.outputs["Sides"]}
    )

    multiply_add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: amount.outputs["OuterRadius"], 1: 1.0000, 2: 0.0000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": amount.outputs["Thickness"]}
    )

    multiply_add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_8, 1: -1.0000, 2: 0.0000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": reroute_7,
            "Radius": multiply_add_1,
            "Depth": multiply_add_2,
        },
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cylinder.outputs["Mesh"]}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_9, "Translation": combine_xyz},
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": amount.outputs["ShadeTop"],
            "True": transform_geometry_2,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_1, "LampShadeTop": switch_1},
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

    named_attribute = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    integer = nw.new_node(Nodes.Integer)
    integer.integer = 0

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": named_attribute.outputs["Exists"],
            "False": integer,
            "True": named_attribute.outputs["Attribute"],
        },
        attrs={"input_type": "INT"},
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Parent"],
            "Name": "part_id",
            "Value": switch,
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
            "Geometry": store_named_attribute,
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

    separate_geometry = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": store_named_attribute, "Selection": equal},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                separate_geometry.outputs["Selection"],
                separate_geometry.outputs["Inverted"],
            ]
        },
    )

    named_attribute_2 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    integer_1 = nw.new_node(Nodes.Integer)
    integer_1.integer = 1

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: named_attribute_2.outputs["Attribute"], 1: 1.0000}
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": named_attribute_2.outputs["Exists"],
            "False": integer_1,
            "True": add,
        },
        attrs={"input_type": "INT"},
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Child"],
            "Name": "part_id",
            "Value": switch_1,
        },
        attrs={"data_type": "INT"},
    )

    named_attribute_3 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    attribute_statistic_1 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": store_named_attribute_1,
            "Attribute": named_attribute_3.outputs["Attribute"],
        },
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_3.outputs["Attribute"],
            3: attribute_statistic_1.outputs["Min"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_1 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": store_named_attribute_1, "Selection": equal_1},
    )

    named_attribute_4 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "is_jointed"},
        attrs={"data_type": "BOOLEAN"},
    )

    attribute_statistic_2 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": separate_geometry_1.outputs["Selection"],
            "Attribute": named_attribute_4.outputs["Attribute"],
        },
    )

    greater_than = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: attribute_statistic_2.outputs["Sum"]},
        attrs={"data_type": "INT"},
    )

    combine_matrix = nw.new_node("FunctionNodeCombineMatrix")

    named_attribute_5 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": greater_than,
            "False": combine_matrix,
            "True": named_attribute_5.outputs["Attribute"],
        },
        attrs={"input_type": "MATRIX"},
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_1,
            "Name": "body_transform",
            "Value": switch_2,
        },
        attrs={"data_type": "FLOAT4X4"},
    )

    named_attribute_6 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "is_jointed"},
        attrs={"data_type": "BOOLEAN"},
    )

    attribute_statistic_3 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": separate_geometry_1.outputs["Selection"],
            "Attribute": named_attribute_6.outputs["Attribute"],
        },
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: attribute_statistic_3.outputs["Sum"]},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    position = nw.new_node(Nodes.InputPosition)

    position_1 = nw.new_node(Nodes.InputPosition)

    bounding_box = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry.outputs["Selection"]},
    )

    position_2 = nw.new_node(Nodes.InputPosition)

    attribute_statistic_4 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": position_2,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: position_1, 1: attribute_statistic_4.outputs["Mean"]},
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_2,
            "False": position,
            "True": add_1.outputs["Vector"],
        },
        attrs={"input_type": "VECTOR"},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": store_named_attribute_2, "Position": switch_3},
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": set_position, "Name": "is_jointed", "Value": True},
        attrs={"data_type": "BOOLEAN"},
    )

    position_3 = nw.new_node(Nodes.InputPosition)

    named_attribute_7 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    attribute_statistic_5 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": set_position,
            "Attribute": named_attribute_7.outputs["Attribute"],
        },
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_7.outputs["Attribute"],
            3: attribute_statistic_5.outputs["Min"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_2 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": set_position, "Selection": equal_3},
    )

    bounding_box_1 = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry_2.outputs["Selection"]},
    )

    position_4 = nw.new_node(Nodes.InputPosition)

    attribute_statistic_6 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box_1.outputs["Bounding Box"],
            "Attribute": position_4,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    named_attribute_8 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    transpose_matrix = nw.new_node(
        "FunctionNodeTransposeMatrix",
        input_kwargs={"Matrix": named_attribute_8.outputs["Attribute"]},
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
        input_kwargs={0: attribute_statistic_6.outputs["Mean"], 1: transform_point},
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

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": op_and,
            "False": clamp,
            "True": group_input.outputs["Value"],
        },
        attrs={"input_type": "FLOAT"},
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_4})

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

    named_attribute_9 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    separate_matrix = nw.new_node(
        "FunctionNodeSeparateMatrix",
        input_kwargs={"Matrix": named_attribute_9.outputs["Attribute"]},
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

    named_attribute_10 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    separate_matrix_1 = nw.new_node(
        "FunctionNodeSeparateMatrix",
        input_kwargs={"Matrix": named_attribute_10.outputs["Attribute"]},
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

    named_attribute_11 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    separate_matrix_2 = nw.new_node(
        "FunctionNodeSeparateMatrix",
        input_kwargs={"Matrix": named_attribute_11.outputs["Attribute"]},
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

    string = nw.new_node("FunctionNodeInputString", attrs={"string": "pos"})

    reroute_2 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Joint ID (do not set)"]},
    )

    join_strings = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string, reroute_2]},
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Position"]}
    )

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_4,
            "Name": join_strings,
            "Value": reroute_3,
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

    store_named_attribute_7 = nw.new_node(
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

    store_named_attribute_8 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_7,
            "Name": join_strings_3,
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
            "Translation": attribute_statistic_4.outputs["Mean"],
        },
    )

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Center of Parent"],
            "True": transform_geometry,
        },
    )

    store_named_attribute_9 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_5, "Name": "part_id", "Value": 999999999},
        attrs={"data_type": "INT"},
    )

    uv_sphere_1 = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Segments": 10, "Rings": 10, "Radius": 0.0500}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_position_1})

    named_attribute_12 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    attribute_statistic_7 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": reroute_4,
            "Attribute": named_attribute_12.outputs["Attribute"],
        },
    )

    equal_6 = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_12.outputs["Attribute"],
            3: attribute_statistic_7.outputs["Min"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_3 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": reroute_4, "Selection": equal_6},
    )

    bounding_box_2 = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry_3.outputs["Selection"]},
    )

    position_5 = nw.new_node(Nodes.InputPosition)

    attribute_statistic_8 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box_2.outputs["Bounding Box"],
            "Attribute": position_5,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere_1.outputs["Mesh"],
            "Translation": attribute_statistic_8.outputs["Mean"],
        },
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Center of Child"],
            "True": transform_geometry_1,
        },
    )

    store_named_attribute_10 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_6, "Name": "part_id", "Value": 999999999},
        attrs={"data_type": "INT"},
    )

    cone = nw.new_node(
        "GeometryNodeMeshCone", input_kwargs={"Radius Bottom": 0.0500, "Depth": 0.2000}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cone.outputs["Mesh"],
            "Translation": (0.0000, 0.0000, -0.0500),
        },
    )

    bounding_box_3 = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry_2.outputs["Selection"]},
    )

    position_6 = nw.new_node(Nodes.InputPosition)

    attribute_statistic_9 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box_3.outputs["Bounding Box"],
            "Attribute": position_6,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    add_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Position"],
            1: attribute_statistic_9.outputs["Mean"],
        },
    )

    attribute_statistic_10 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": separate_geometry_3.outputs["Selection"],
            "Attribute": transform_direction,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    normalize = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: attribute_statistic_10.outputs["Mean"]},
        attrs={"operation": "NORMALIZE"},
    )

    align_rotation_to_vector = nw.new_node(
        "FunctionNodeAlignRotationToVector",
        input_kwargs={"Vector": normalize.outputs["Vector"]},
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_2,
            "Translation": add_3.outputs["Vector"],
            "Rotation": align_rotation_to_vector,
        },
    )

    switch_7 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Joint"],
            "True": transform_geometry_3,
        },
    )

    store_named_attribute_11 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_7, "Name": "part_id", "Value": 999999999},
        attrs={"data_type": "INT"},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                join_geometry,
                store_named_attribute_8,
                store_named_attribute_9,
                store_named_attribute_10,
                store_named_attribute_11,
            ]
        },
    )

    store_named_attribute_12 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_4,
            "Name": join_strings,
            "Value": reroute_3,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute_13 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_12,
            "Name": join_strings_1,
            "Value": group_input.outputs["Axis"],
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute_14 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_13,
            "Name": join_strings_2,
            "Value": group_input.outputs["Min"],
        },
    )

    store_named_attribute_15 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_14,
            "Name": join_strings_3,
            "Value": group_input.outputs["Max"],
        },
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                store_named_attribute_15,
                store_named_attribute_9,
                store_named_attribute_10,
                store_named_attribute_11,
            ]
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Parent": join_geometry,
            "Child": join_geometry_2,
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

    named_attribute = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    integer = nw.new_node(Nodes.Integer)
    integer.integer = 0

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": named_attribute.outputs["Exists"],
            "False": integer,
            "True": named_attribute.outputs["Attribute"],
        },
        attrs={"input_type": "INT"},
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Parent"],
            "Name": "part_id",
            "Value": switch,
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
            "Geometry": store_named_attribute,
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

    separate_geometry = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": store_named_attribute, "Selection": equal},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                separate_geometry.outputs["Selection"],
                separate_geometry.outputs["Inverted"],
            ]
        },
    )

    cone = nw.new_node(
        "GeometryNodeMeshCone", input_kwargs={"Radius Bottom": 0.0500, "Depth": 0.2000}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cone.outputs["Mesh"],
            "Translation": (0.0000, 0.0000, -0.0500),
        },
    )

    named_attribute_2 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    integer_1 = nw.new_node(Nodes.Integer)
    integer_1.integer = 1

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: named_attribute_2.outputs["Attribute"], 1: 1.0000}
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": named_attribute_2.outputs["Exists"],
            "False": integer_1,
            "True": add,
        },
        attrs={"input_type": "INT"},
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": group_input.outputs["Child"],
            "Name": "part_id",
            "Value": switch_1,
        },
        attrs={"data_type": "INT"},
    )

    named_attribute_3 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "part_id"},
        attrs={"data_type": "INT"},
    )

    attribute_statistic_1 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": store_named_attribute_1,
            "Attribute": named_attribute_3.outputs["Attribute"],
        },
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            2: named_attribute_3.outputs["Attribute"],
            3: attribute_statistic_1.outputs["Min"],
        },
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    separate_geometry_1 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": store_named_attribute_1, "Selection": equal_1},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_geometry_1.outputs["Selection"]}
    )

    bounding_box = nw.new_node(Nodes.BoundingBox, input_kwargs={"Geometry": reroute})

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
        input_kwargs={
            0: group_input.outputs["Position"],
            1: attribute_statistic_2.outputs["Mean"],
        },
    )

    named_attribute_4 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    transpose_matrix = nw.new_node(
        "FunctionNodeTransposeMatrix",
        input_kwargs={"Matrix": named_attribute_4.outputs["Attribute"]},
    )

    transform_direction = nw.new_node(
        "FunctionNodeTransformDirection",
        input_kwargs={
            "Direction": group_input.outputs["Axis"],
            "Transform": transpose_matrix,
        },
    )

    attribute_statistic_3 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={"Geometry": reroute, "Attribute": transform_direction},
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    normalize = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: attribute_statistic_3.outputs["Mean"]},
        attrs={"operation": "NORMALIZE"},
    )

    align_rotation_to_vector = nw.new_node(
        "FunctionNodeAlignRotationToVector",
        input_kwargs={"Vector": normalize.outputs["Vector"]},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry,
            "Translation": add_1.outputs["Vector"],
            "Rotation": align_rotation_to_vector,
        },
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Joint"],
            "True": transform_geometry_1,
        },
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_2, "Name": "part_id", "Value": 999999999},
        attrs={"data_type": "INT"},
    )

    uv_sphere = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Segments": 10, "Rings": 10, "Radius": 0.0500}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere.outputs["Mesh"],
            "Translation": attribute_statistic_2.outputs["Mean"],
        },
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Center of Child"],
            "True": transform_geometry_2,
        },
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_3, "Name": "part_id", "Value": 999999999},
        attrs={"data_type": "INT"},
    )

    uv_sphere_1 = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Segments": 10, "Rings": 10, "Radius": 0.0500}
    )

    bounding_box_1 = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry.outputs["Selection"]},
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    attribute_statistic_4 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box_1.outputs["Bounding Box"],
            "Attribute": position_1,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere_1.outputs["Mesh"],
            "Translation": attribute_statistic_4.outputs["Mean"],
        },
    )

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Show Center of Parent"],
            "True": transform_geometry_3,
        },
    )

    store_named_attribute_4 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": switch_4, "Name": "part_id", "Value": 999999999},
        attrs={"data_type": "INT"},
    )

    named_attribute_5 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "is_jointed"},
        attrs={"data_type": "BOOLEAN"},
    )

    attribute_statistic_5 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": separate_geometry_1.outputs["Selection"],
            "Attribute": named_attribute_5.outputs["Attribute"],
        },
    )

    greater_than = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: attribute_statistic_5.outputs["Sum"]},
        attrs={"data_type": "INT"},
    )

    combine_matrix = nw.new_node("FunctionNodeCombineMatrix")

    named_attribute_6 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "body_transform"},
        attrs={"data_type": "FLOAT4X4"},
    )

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": greater_than,
            "False": combine_matrix,
            "True": named_attribute_6.outputs["Attribute"],
        },
        attrs={"input_type": "MATRIX"},
    )

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_1,
            "Name": "body_transform",
            "Value": switch_5,
        },
        attrs={"data_type": "FLOAT4X4"},
    )

    named_attribute_7 = nw.new_node(
        Nodes.NamedAttribute,
        input_kwargs={"Name": "is_jointed"},
        attrs={"data_type": "BOOLEAN"},
    )

    attribute_statistic_6 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": separate_geometry_1.outputs["Selection"],
            "Attribute": named_attribute_7.outputs["Attribute"],
        },
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: attribute_statistic_6.outputs["Sum"]},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    position_2 = nw.new_node(Nodes.InputPosition)

    position_3 = nw.new_node(Nodes.InputPosition)

    add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: position_3, 1: attribute_statistic_4.outputs["Mean"]},
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_2,
            "False": position_2,
            "True": add_2.outputs["Vector"],
        },
        attrs={"input_type": "VECTOR"},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": store_named_attribute_5, "Position": switch_6},
    )

    store_named_attribute_6 = nw.new_node(
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

    switch_7 = nw.new_node(
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
        input_kwargs={0: transform_direction, "Scale": switch_7},
        attrs={"operation": "SCALE"},
    )

    position_4 = nw.new_node(Nodes.InputPosition)

    add_3 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: scale.outputs["Vector"], 1: position_4}
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": store_named_attribute_6,
            "Position": add_3.outputs["Vector"],
        },
    )

    string = nw.new_node("FunctionNodeInputString", attrs={"string": "pos"})

    reroute_1 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Joint ID (do not set)"]},
    )

    join_strings = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string, reroute_1]},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Position"]}
    )

    store_named_attribute_7 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": set_position_1,
            "Name": join_strings,
            "Value": reroute_2,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    string_1 = nw.new_node("FunctionNodeInputString", attrs={"string": "axis"})

    join_strings_1 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_1, reroute_1]},
    )

    store_named_attribute_8 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_7,
            "Name": join_strings_1,
            "Value": group_input.outputs["Axis"],
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    string_2 = nw.new_node("FunctionNodeInputString", attrs={"string": "min"})

    join_strings_2 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_2, reroute_1]},
    )

    store_named_attribute_9 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_8,
            "Name": join_strings_2,
            "Value": group_input.outputs["Min"],
        },
    )

    string_3 = nw.new_node("FunctionNodeInputString", attrs={"string": "max"})

    join_strings_3 = nw.new_node(
        "GeometryNodeStringJoin",
        input_kwargs={"Delimiter": "_", "Strings": [string_3, reroute_1]},
    )

    store_named_attribute_10 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_9,
            "Name": join_strings_3,
            "Value": group_input.outputs["Max"],
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                join_geometry,
                store_named_attribute_2,
                store_named_attribute_3,
                store_named_attribute_4,
                store_named_attribute_10,
            ]
        },
    )

    store_named_attribute_11 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": set_position_1,
            "Name": join_strings,
            "Value": reroute_2,
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute_12 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_11,
            "Name": join_strings_1,
            "Value": group_input.outputs["Axis"],
        },
        attrs={"data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute_13 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_12,
            "Name": join_strings_2,
            "Value": group_input.outputs["Min"],
        },
    )

    store_named_attribute_14 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_13,
            "Name": join_strings_3,
            "Value": group_input.outputs["Max"],
        },
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                store_named_attribute_2,
                store_named_attribute_3,
                store_named_attribute_4,
                store_named_attribute_14,
            ]
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Parent": join_geometry,
            "Child": join_geometry_2,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata",
    singleton=False,
    type="GeometryNodeTree",
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
            "Value": 7,
        },
        attrs={"data_type": "INT"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": store_named_attribute},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_string_nodes_v2", singleton=False, type="GeometryNodeTree"
)
def nodegroup_string_nodes_v2(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

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

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cylinder.outputs["Mesh"]}
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0150

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": value})

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 2: reroute_7},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add})

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Depth"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: value, 1: 2.0000}, attrs={"operation": "MULTIPLY"}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Length"], 1: multiply},
        attrs={"operation": "DIVIDE"},
    )

    float_to_integer = nw.new_node(Nodes.FloatToInt, input_kwargs={"Float": divide})

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: float_to_integer, 1: reroute_9},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_10, 2: multiply_1},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add_1})

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": reroute_16, "End": combine_xyz_1}
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": float_to_integer})

    curve_to_points = nw.new_node(
        Nodes.CurveToPoints, input_kwargs={"Curve": curve_line, "Count": reroute_14}
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    uv_sphere = nw.new_node(Nodes.MeshUVSphere, input_kwargs={"Radius": reroute_13})

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

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Length"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_5, 1: -1.0000, 2: multiply_2},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add_2})

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_2})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": realize_instances, "Translation": reroute_15},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract})

    sliding_joint_001 = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint ID (do not set)": string,
            "Joint Label": "pullstring",
            "Parent": reroute_8,
            "Child": transform_geometry,
            "Axis": (0.0000, 0.0000, -1.0000),
            "Max": reroute_12,
        },
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["RadialLength"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_3, "Y": reroute_3, "Z": reroute_1}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_3})

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_3, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_4})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Translation": combine_xyz_4},
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_1}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [sliding_joint_001.outputs["Geometry"], reroute_17]},
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_11, 1: 1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_5})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Translation": combine_xyz_5},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [sliding_joint_001.outputs["Parent"], reroute_17]},
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_1, "Translation": combine_xyz_5},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_2, "Parent": transform_geometry_3},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_u_switch", singleton=False, type="GeometryNodeTree")
def nodegroup_u_switch(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (2.0000, 1.0000, 1.0000)})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Translation": (0.0000, 0.5000, 0.5000),
        },
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 6, "Radius": 6.0000, "Depth": 2.5200},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": (0.0000, -5.4400, 0.0000),
            "Rotation": (0.0000, 0.0000, 0.5250),
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": transform_geometry, "Mesh 2": transform_geometry_1},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": difference.outputs["Mesh"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_lamp_head_final", singleton=False, type="GeometryNodeTree"
)
def nodegroup_lamp_head_final(nw: NodeWrangler):
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

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ShadeCurved"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["RackHeight"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["ReverseBulb"], 1: 2.0000, 2: -1.0000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_add})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_7, 1: reroute_25},
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

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_add, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_24, 1: multiply_1},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_2})

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
            "Switch": reroute_11,
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

    reroute_26 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": map_range.outputs["Result"]}
    )

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius, input_kwargs={"Curve": switch, "Radius": reroute_26}
    )

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Sides"]}
    )

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_17})

    curve_circle = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": reroute_31}
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

    reroute_20 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["LampShadeInteriorMaterial"]},
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    set_material = nw.new_node(
        Nodes.SetMaterial, input_kwargs={"Geometry": flip_faces, "Material": reroute_29}
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": set_shade_smooth,
            "Offset Scale": 0.0050,
            "Individual": False,
        },
    )

    reroute_22 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["LampshadeMaterial"]}
    )

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    reroute_34 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

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

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["RackThickness"]}
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

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

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["TopRadius"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_18 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BlackMaterial"]}
    )

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["MetalMaterial"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    reversable_bulb = nw.new_node(
        nodegroup_reversable_bulb().name,
        input_kwargs={
            "Scale": maximum,
            "BlackMaterial": reroute_19,
            "LampshadeMaterial": reroute_21,
            "MetalMaterial": reroute_9,
        },
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ShadeTop"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    bulb_rack = nw.new_node(
        nodegroup_bulb_rack().name,
        input_kwargs={
            "Thickness": reroute_15,
            "InnerRadius": multiply_4,
            "OuterRadius": reroute_3,
            "InnerHeight": reversable_bulb.outputs["RackSupport"],
            "OuterHeight": multiply,
            "ShadeTop": reroute_5,
            "Sides": reroute_17,
        },
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": bulb_rack.outputs["LampShadeTop"],
            "Material": reroute_23,
        },
    )

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_2})

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["IncludeLightbulb"]}
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    reroute_30 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_15})

    greater_than = nw.new_node(Nodes.Compare, input_kwargs={0: reroute_30})

    reroute_32 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": reversable_bulb.outputs["Geometry"]}
    )

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_32})

    reroute_36 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_33})

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_27})

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": bulb_rack.outputs["Geometry"],
            "Material": reroute_28,
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material_3, reroute_33]}
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": greater_than,
            "False": reroute_36,
            "True": join_geometry_1,
        },
    )

    switch_2 = nw.new_node(
        Nodes.Switch, input_kwargs={"Switch": reroute_13, "True": switch_1}
    )

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_2})

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry, reroute_35, reroute_37]},
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

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": cylinder.outputs["Mesh"], "Label": "lamp_base"},
    )

    reroute_40 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BaseMaterial"]}
    )

    reroute_41 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_40})

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata,
            "Material": reroute_41,
        },
    )

    reroute_81 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material})

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
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

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry, "Label": "bar"},
    )

    reroute_73 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata}
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

    lamp_head_final = nw.new_node(
        nodegroup_lamp_head_final().name,
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
        Nodes.RealizeInstances, input_kwargs={"Geometry": lamp_head_final}
    )

    reroute_56 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": realize_instances})

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_56, "Label": "head"},
    )

    reroute_105 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata}
    )

    reroute_106 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_105})

    reroute_116 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_106})

    reroute_112 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata}
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

    string = nw.new_node("FunctionNodeInputString", attrs={"string": "joint18"})

    cylinder_2 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["ButtonR1"],
            "Depth": group_input.outputs["ButtonH1"],
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
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
            "Geometry": add_jointed_geometry_metadata,
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

    cylinder_3 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["ButtonR2"],
            "Depth": group_input.outputs["ButtonH1"],
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
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
            "Geometry": add_jointed_geometry_metadata,
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

    reroute_76 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_59})

    reroute_77 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_76})

    reroute_102 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_77})

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint ID (do not set)": string,
            "Joint Label": "button",
            "Parent Label": "button base",
            "Parent": reroute_95,
            "Child Label": "button",
            "Child": transform_geometry_5,
            "Axis": switch_2,
            "Max": reroute_102,
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

    string_1 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint19"})

    cylinder_4 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["ButtonR1"],
            "Depth": group_input.outputs["ButtonH1"],
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
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
            "Geometry": add_jointed_geometry_metadata,
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

    cylinder_5 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Radius": group_input.outputs["ButtonR2"],
            "Depth": group_input.outputs["ButtonH2"],
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": cylinder_5.outputs["Mesh"], "Label": "button"},
    )

    reroute_62 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata}
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

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint ID (do not set)": string_1,
            "Joint Label": "twist button",
            "Parent Label": "button base",
            "Parent": reroute_90,
            "Child Label": "button",
            "Child": transform_geometry_11,
            "Axis": switch_3,
            "Value": -0.5000,
        },
    )

    string_2 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint21"})

    combine_xyz_18 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": group_input.outputs["ButtonH1"], "Y": 0.7000, "Z": 1.0000},
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_18})

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
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
            "Geometry": add_jointed_geometry_metadata,
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

    u_switch = nw.new_node(nodegroup_u_switch().name)

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": u_switch, "Label": "button"},
    )

    transform_geometry_14 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata,
            "Translation": (0.0000, -0.5000, -0.5000),
        },
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["ButtonR1"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_20 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_7})

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
            "Translation": combine_xyz_20,
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

    hinge_joint_001 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint ID (do not set)": string_2,
            "Joint Label": "switch button",
            "Parent Label": "button base",
            "Parent": transform_geometry_13,
            "Child Label": "switch",
            "Child": transform_geometry_17,
            "Axis": switch_4,
            "Value": -0.9000,
            "Min": -0.2500,
            "Max": 0.2500,
        },
    )

    reroute_107 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_001.outputs["Geometry"]}
    )

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

    string_nodes_v2 = nw.new_node(
        nodegroup_string_nodes_v2().name,
        input_kwargs={
            "Radius": group_input.outputs["ButtonR1"],
            "RadialLength": 0.0900,
            "Length": group_input.outputs["ButtonH2"],
            "Depth": group_input.outputs["ButtonH1"],
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": string_nodes_v2.outputs["Geometry"],
            "Label": "button",
        },
    )

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata,
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

    string_3 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint26"})

    reroute_19 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BaseButtonOffset"]}
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    add = nw.new_node(
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
        input_kwargs={0: -2.0000, 1: add, 2: reroute_17},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_46 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add})

    multiply_add_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_20, 1: multiply_add_3, 2: reroute_46},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_18 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BaseHeight"]}
    )

    multiply_8 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_18}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_27 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_add_4, "Z": multiply_8}
    )

    transform_geometry_20 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_61,
            "Translation": combine_xyz_27,
            "Rotation": combine_xyz_7,
        },
    )

    sliding_joint_002 = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint ID (do not set)": string_3,
            "Joint Label": "button",
            "Parent Label": "button base",
            "Parent": transform_geometry_20,
            "Child Label": "button",
            "Child": transform_geometry_4,
            "Axis": (0.0000, 0.0000, -1.0000),
            "Value": -3.8000,
            "Max": reroute_77,
        },
    )

    reroute_113 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": sliding_joint_002.outputs["Geometry"]}
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_110, 3: 3},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    string_4 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint27"})

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

    hinge_joint_002 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint ID (do not set)": string_4,
            "Joint Label": "twist button",
            "Parent Label": "button base",
            "Parent": reroute_91,
            "Child Label": "twist",
            "Child": transform_geometry_10,
        },
    )

    string_5 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint28"})

    transform_geometry_22 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_12,
            "Translation": combine_xyz_27,
            "Rotation": combine_xyz_7,
        },
    )

    transform_geometry_23 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_16, "Rotation": combine_xyz_7},
    )

    hinge_joint_003 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint ID (do not set)": string_5,
            "Joint Label": "switch button",
            "Parent Label": "switch base",
            "Parent": transform_geometry_22,
            "Child Label": "switch",
            "Child": transform_geometry_23,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Value": -7.8000,
            "Min": -0.2500,
            "Max": 0.2500,
        },
    )

    reroute_103 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_003.outputs["Geometry"]}
    )

    switch_10 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_2,
            "False": hinge_joint_002.outputs["Geometry"],
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

    string_6 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint36"})

    reroute_47 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cylinder.outputs["Mesh"]}
    )

    reroute_48 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_47})

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_48, "Label": "lamp_base"},
    )

    reroute_68 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_41})

    set_material_5 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata,
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

    reroute_132 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_1})

    reroute_133 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_132})

    join_geometry_6 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_19, reroute_133]},
    )

    hinge_joint_004 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint ID (do not set)": string_6,
            "Joint Label": "lamp z rotation",
            "Parent Label": "lamp base",
            "Parent": switch_13,
            "Child Label": "lamp top",
            "Child": join_geometry_6,
        },
    )

    reroute_155 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_004.outputs["Geometry"]}
    )

    switch_14 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_37, "False": switch_12, "True": reroute_155},
    )

    reroute_164 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_14})

    reroute_145 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_37})

    reroute_142 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_131})

    string_7 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint4"})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Radius"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    multiply_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 2.4000},
        attrs={"operation": "MULTIPLY"},
    )

    cylinder_6 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Radius": reroute_1, "Depth": multiply_9},
    )

    transform_geometry_24 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_6.outputs["Mesh"],
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_24, "Label": "bar"},
    )

    set_material_6 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata,
            "Material": reroute_69,
        },
    )

    reroute_98 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_6})

    reroute_99 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_98})

    add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["FirstBarLength"],
            1: group_input.outputs["FirstBarExtension"],
        },
    )

    cylinder_8 = nw.new_node(
        "GeometryNodeMeshCylinder", input_kwargs={"Radius": reroute_1, "Depth": add_1}
    )

    reroute_64 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cylinder_8.outputs["Mesh"]}
    )

    reroute_24 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["FirstBarExtension"]}
    )

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_24})

    multiply_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["FirstBarLength"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_25, 1: -0.5000, 2: multiply_10},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_28 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add_5})

    transform_geometry_26 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_64, "Translation": combine_xyz_28},
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_26, "Label": "bar"},
    )

    set_material_8 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata,
            "Material": reroute_87,
        },
    )

    reroute_111 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_8})

    cylinder_7 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Radius": reroute_1, "Depth": multiply_9},
    )

    transform_geometry_25 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_7.outputs["Mesh"],
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_25, "Label": "bar"},
    )

    set_material_7 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata,
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

    reroute_50 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_9})

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_50})

    combine_xyz_30 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": reroute_51})

    transform_geometry_28 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_7, "Translation": combine_xyz_30},
    )

    greater_equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_79, 3: 2},
        attrs={"operation": "GREATER_EQUAL", "data_type": "INT"},
    )

    reroute_88 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_2}
    )

    reroute_89 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_88})

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_31, 3: 3},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    switch_15 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_3,
            "False": transform_geometry_7,
            "True": transform_geometry_13,
        },
    )

    switch_16 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": greater_equal_2,
            "False": reroute_89,
            "True": switch_15,
        },
    )

    transform_geometry_29 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": switch_16, "Translation": reroute_66}
    )

    reroute_49 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": string_nodes_v2.outputs["Parent"]}
    )

    switch_17 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal,
            "False": transform_geometry_29,
            "True": reroute_49,
        },
    )

    join_geometry_8 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [switch_17, add_jointed_geometry_metadata]},
    )

    switch_18 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_29,
            "False": reroute_106,
            "True": join_geometry_8,
        },
    )

    transform_geometry_30 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": switch_18, "Translation": combine_xyz_29},
    )

    reroute_104 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_8})

    join_geometry_9 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_30, reroute_94, reroute_104]},
    )

    reroute_125 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_9})

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": reroute_125}
    )

    combine_xyz_31 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": -0.5000, "Y": -0.5000, "Z": -0.5000}
    )

    multiply_11 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Min"], 1: combine_xyz_31},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_12 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: combine_xyz_31},
        attrs={"operation": "MULTIPLY"},
    )

    add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: multiply_11.outputs["Vector"],
            1: multiply_12.outputs["Vector"],
        },
    )

    hinge_joint_005 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint ID (do not set)": string_7,
            "Joint Label": "lamp second joint rotation",
            "Parent Label": "rotation axle",
            "Parent": reroute_99,
            "Child Label": "lamp top",
            "Child": transform_geometry_28,
            "Position": add_2.outputs["Vector"],
            "Axis": (0.0000, 1.0000, 0.0000),
            "Value": 5.7000,
        },
    )

    reroute_128 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_26})

    reroute_129 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_128})

    transform_geometry_31 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": hinge_joint_005.outputs["Geometry"],
            "Translation": reroute_129,
        },
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

    string_8 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint32"})

    reroute_152 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_13})

    reroute_139 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_133})

    reroute_140 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_139})

    join_geometry_12 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_31, reroute_140]},
    )

    hinge_joint_006 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint ID (do not set)": string_8,
            "Joint Label": "lamp z rotation",
            "Parent Label": "lamp base",
            "Parent": reroute_152,
            "Child Label": "lamp top",
            "Child": join_geometry_12,
            "Value": 3.4000,
        },
    )

    reroute_166 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_006.outputs["Geometry"]}
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

    string_9 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint2"})

    multiply_13 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 2.4000},
        attrs={"operation": "MULTIPLY"},
    )

    cylinder_9 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Radius": reroute_1, "Depth": multiply_13},
    )

    set_material_9 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": cylinder_9.outputs["Mesh"], "Material": reroute_43},
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_9, "Label": "bar"},
    )

    reroute_82 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata}
    )

    transform_geometry_32 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_82, "Rotation": (1.5708, 0.0000, 0.0000)},
    )

    cylinder_11 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Radius": reroute_1, "Depth": multiply_13},
    )

    set_material_11 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": cylinder_11.outputs["Mesh"], "Material": reroute_43},
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_11, "Label": "bar"},
    )

    reroute_83 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata}
    )

    transform_geometry_35 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_83, "Rotation": (1.5708, 0.0000, 0.0000)},
    )

    reroute_146 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_35}
    )

    reroute_34 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["SecondBarLength"]}
    )

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_34})

    combine_xyz_32 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_35})

    transform_geometry_33 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": hinge_joint_005.outputs["Geometry"],
            "Translation": combine_xyz_32,
        },
    )

    add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["SecondBarLength"],
            1: group_input.outputs["SecondBarExtension"],
        },
    )

    cylinder_10 = nw.new_node(
        "GeometryNodeMeshCylinder", input_kwargs={"Radius": reroute_1, "Depth": add_3}
    )

    set_material_10 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": cylinder_10.outputs["Mesh"], "Material": reroute_43},
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_10, "Label": "bar"},
    )

    reroute_84 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata}
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

    join_geometry_14 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_34, transform_geometry_35]},
    )

    bounding_box_1 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": join_geometry_14}
    )

    combine_xyz_35 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": -0.5000, "Y": -0.5000, "Z": -0.5000}
    )

    multiply_16 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box_1.outputs["Min"], 1: combine_xyz_35},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_17 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box_1.outputs["Max"], 1: combine_xyz_35},
        attrs={"operation": "MULTIPLY"},
    )

    add_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: multiply_16.outputs["Vector"],
            1: multiply_17.outputs["Vector"],
        },
    )

    reroute_57 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    multiply_18 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_57}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_36 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_18})

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_4.outputs["Vector"], 1: combine_xyz_36},
        attrs={"operation": "SUBTRACT"},
    )

    hinge_joint_007 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint ID (do not set)": string_9,
            "Joint Label": "lamp first joint rotation",
            "Parent Label": "rotation axle",
            "Parent": transform_geometry_32,
            "Child Label": "lamp top",
            "Child": transform_geometry_36,
            "Position": subtract.outputs["Vector"],
            "Axis": (0.0000, 1.0000, 0.0000),
            "Value": -0.3000,
        },
    )

    reroute_165 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_007.outputs["Geometry"]}
    )

    string_10 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint22"})

    reroute_156 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_32}
    )

    reroute_157 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_156})

    string_11 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint23"})

    reroute_143 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_35}
    )

    reroute_147 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_34}
    )

    join_geometry_15 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [reroute_147, transform_geometry_33]},
    )

    reroute_138 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_35})

    multiply_add_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_138, 1: -1.0000, 2: 0.0500},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_60 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_60, 1: 0.0500},
        attrs={"operation": "SUBTRACT"},
    )

    sliding_joint_003 = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint ID (do not set)": string_11,
            "Joint Label": "sliding bar",
            "Parent Label": "hinge",
            "Parent": reroute_143,
            "Child Label": "lamp top",
            "Child": join_geometry_15,
            "Min": multiply_add_7,
            "Max": subtract_1,
        },
    )

    reroute_148 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_34})

    transform_geometry_37 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": sliding_joint_003.outputs["Geometry"],
            "Translation": reroute_148,
        },
    )

    hinge_joint_008 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint ID (do not set)": string_10,
            "Joint Label": "lamp first joint rotation",
            "Parent Label": "rotation axle",
            "Parent": reroute_157,
            "Child Label": "lamp top",
            "Child": transform_geometry_37,
            "Axis": (0.0000, 1.0000, 0.0000),
            "Value": 4.9000,
        },
    )

    switch_22 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_27,
            "False": reroute_165,
            "True": hinge_joint_008.outputs["Geometry"],
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

    string_12 = nw.new_node("FunctionNodeInputString", attrs={"string": "joint0"})

    reroute_151 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_13})

    reroute_159 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_140})

    join_geometry_18 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_38, reroute_159]},
    )

    hinge_joint_009 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint ID (do not set)": string_12,
            "Joint Label": "lamp z rotation",
            "Parent Label": "lamp base",
            "Parent": reroute_151,
            "Child Label": "lamp top",
            "Child": join_geometry_18,
        },
    )

    reroute_169 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_009.outputs["Geometry"]}
    )

    switch_24 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_162, "False": switch_23, "True": reroute_169},
    )

    switch_25 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_13, "False": reroute_167, "True": switch_24},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": switch_25},
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
        lamp_first_joint_rotation_stiffness_min: float = 15000,
        lamp_first_joint_rotation_stiffness_max: float = 25000,
        lamp_first_joint_rotation_damping_min: float = 50000,
        lamp_first_joint_rotation_damping_max: float = 60000,
        lamp_z_rotation_stiffness_min: float = 0.0,
        lamp_z_rotation_stiffness_max: float = 0.0,
        lamp_z_rotation_damping_min: float = 10.0,
        lamp_z_rotation_damping_max: float = 20.0,
        twist_button_stiffness_min: float = 0.0,
        twist_button_stiffness_max: float = 0.0,
        twist_button_damping_min: float = 2.0,
        twist_button_damping_max: float = 5.0,
        pullstring_stiffness_min: float = 2.0,
        pullstring_stiffness_max: float = 5.0,
        pullstring_damping_min: float = 2.0,
        pullstring_damping_max: float = 5.0,
        switch_button_stiffness_min: float = 0.0,
        switch_button_stiffness_max: float = 0.0,
        switch_button_damping_min: float = 0.0,
        switch_button_damping_max: float = 0.0,
        lamp_second_joint_rotation_stiffness_min: float = 15000,
        lamp_second_joint_rotation_stiffness_max: float = 25000,
        lamp_second_joint_rotation_damping_min: float = 40000,
        lamp_second_joint_rotation_damping_max: float = 60000,
        sliding_bar_stiffness_min: float = 0.0,
        sliding_bar_stiffness_max: float = 0.0,
        sliding_bar_damping_min: float = 50000,
        sliding_bar_damping_max: float = 60000,
        button_stiffness_min: float = 20.0,
        button_stiffness_max: float = 30.0,
        button_damping_min: float = 5.0,
        button_damping_max: float = 10.0,
    ):
        return {
            "lamp first joint rotation": {
                "stiffness": uniform(
                    lamp_first_joint_rotation_stiffness_min,
                    lamp_first_joint_rotation_stiffness_max,
                ),
                "damping": uniform(
                    lamp_first_joint_rotation_damping_min,
                    lamp_first_joint_rotation_damping_max,
                ),
            },
            "lamp z rotation": {
                "stiffness": uniform(
                    lamp_z_rotation_stiffness_min, lamp_z_rotation_stiffness_max
                ),
                "damping": uniform(
                    lamp_z_rotation_damping_min, lamp_z_rotation_damping_max
                ),
            },
            "twist button": {
                "stiffness": uniform(
                    twist_button_stiffness_min, twist_button_stiffness_max
                ),
                "damping": uniform(twist_button_damping_min, twist_button_damping_max),
            },
            "pullstring": {
                "stiffness": uniform(
                    pullstring_stiffness_min, pullstring_stiffness_max
                ),
                "damping": uniform(pullstring_damping_min, pullstring_damping_max),
            },
            "switch button": {
                "stiffness": uniform(
                    switch_button_stiffness_min, switch_button_stiffness_max
                ),
                "damping": uniform(
                    switch_button_damping_min, switch_button_damping_max
                ),
            },
            "lamp second joint rotation": {
                "stiffness": uniform(
                    lamp_second_joint_rotation_stiffness_min,
                    lamp_second_joint_rotation_stiffness_max,
                ),
                "damping": uniform(
                    lamp_second_joint_rotation_damping_min,
                    lamp_second_joint_rotation_damping_max,
                ),
            },
            "sliding bar": {
                "stiffness": uniform(
                    sliding_bar_stiffness_min, sliding_bar_stiffness_max
                ),
                "damping": uniform(sliding_bar_damping_min, sliding_bar_damping_max),
                "friction": uniform(10000, 20000),
            },
            "button": {
                "stiffness": uniform(button_stiffness_min, button_stiffness_max),
                "damping": uniform(button_damping_min, button_damping_max),
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
