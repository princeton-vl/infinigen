import gin
from numpy.random import normal, randint, uniform

from infinigen.assets.utils.joints import (
    nodegroup_add_jointed_geometry_metadata,
    nodegroup_hinge_joint,
    nodegroup_sliding_joint,
)
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil


@node_utils.to_nodegroup(
    "nodegroup_distance_from_center_002_007", singleton=False, type="GeometryNodeTree"
)
def nodegroup_distance_from_center_002_007(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    named_attribute = nw.new_node(
        Nodes.NamedAttribute, input_kwargs={"Name": "part_id"}
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
        attrs={"data_type": "INT", "operation": "EQUAL"},
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

    attribute_statistic_1 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz.outputs["X"],
        },
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_1.outputs["Max"],
            1: attribute_statistic_1.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_1})

    attribute_statistic_2 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz_1.outputs["Y"],
        },
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_2.outputs["Max"],
            1: attribute_statistic_2.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    position_2 = nw.new_node(Nodes.InputPosition)

    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_2})

    attribute_statistic_3 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz_2.outputs["Z"],
        },
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_3.outputs["Max"],
            1: attribute_statistic_3.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": subtract_1, "Z": subtract_2}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Distance to AABB Center": combine_xyz},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_distance_from_center_002_006", singleton=False, type="GeometryNodeTree"
)
def nodegroup_distance_from_center_002_006(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    named_attribute = nw.new_node(
        Nodes.NamedAttribute, input_kwargs={"Name": "part_id"}
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
        attrs={"data_type": "INT", "operation": "EQUAL"},
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

    attribute_statistic_1 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz.outputs["X"],
        },
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_1.outputs["Max"],
            1: attribute_statistic_1.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_1})

    attribute_statistic_2 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz_1.outputs["Y"],
        },
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_2.outputs["Max"],
            1: attribute_statistic_2.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    position_2 = nw.new_node(Nodes.InputPosition)

    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_2})

    attribute_statistic_3 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz_2.outputs["Z"],
        },
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_3.outputs["Max"],
            1: attribute_statistic_3.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": subtract_1, "Z": subtract_2}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Distance to AABB Center": combine_xyz},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_distance_from_center_003_002", singleton=False, type="GeometryNodeTree"
)
def nodegroup_distance_from_center_003_002(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    named_attribute = nw.new_node(
        Nodes.NamedAttribute, input_kwargs={"Name": "part_id"}
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
        attrs={"data_type": "INT", "operation": "EQUAL"},
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

    attribute_statistic_1 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz.outputs["X"],
        },
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_1.outputs["Max"],
            1: attribute_statistic_1.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_1})

    attribute_statistic_2 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz_1.outputs["Y"],
        },
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_2.outputs["Max"],
            1: attribute_statistic_2.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    position_2 = nw.new_node(Nodes.InputPosition)

    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_2})

    attribute_statistic_3 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz_2.outputs["Z"],
        },
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_3.outputs["Max"],
            1: attribute_statistic_3.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": subtract_1, "Z": subtract_2}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Distance to AABB Center": combine_xyz},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_distance_from_center_002_004", singleton=False, type="GeometryNodeTree"
)
def nodegroup_distance_from_center_002_004(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    named_attribute = nw.new_node(
        Nodes.NamedAttribute, input_kwargs={"Name": "part_id"}
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
        attrs={"data_type": "INT", "operation": "EQUAL"},
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

    attribute_statistic_1 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz.outputs["X"],
        },
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_1.outputs["Max"],
            1: attribute_statistic_1.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_1})

    attribute_statistic_2 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz_1.outputs["Y"],
        },
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_2.outputs["Max"],
            1: attribute_statistic_2.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    position_2 = nw.new_node(Nodes.InputPosition)

    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_2})

    attribute_statistic_3 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz_2.outputs["Z"],
        },
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_3.outputs["Max"],
            1: attribute_statistic_3.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": subtract_1, "Z": subtract_2}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Distance to AABB Center": combine_xyz},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_distance_from_center_002_005", singleton=False, type="GeometryNodeTree"
)
def nodegroup_distance_from_center_002_005(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    named_attribute = nw.new_node(
        Nodes.NamedAttribute, input_kwargs={"Name": "part_id"}
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
        attrs={"data_type": "INT", "operation": "EQUAL"},
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

    attribute_statistic_1 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz.outputs["X"],
        },
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_1.outputs["Max"],
            1: attribute_statistic_1.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_1})

    attribute_statistic_2 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz_1.outputs["Y"],
        },
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_2.outputs["Max"],
            1: attribute_statistic_2.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    position_2 = nw.new_node(Nodes.InputPosition)

    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_2})

    attribute_statistic_3 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz_2.outputs["Z"],
        },
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_3.outputs["Max"],
            1: attribute_statistic_3.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": subtract_1, "Z": subtract_2}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Distance to AABB Center": combine_xyz},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_distance_from_center_007", singleton=False, type="GeometryNodeTree"
)
def nodegroup_distance_from_center_007(nw: NodeWrangler):
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
        attrs={"data_type": "INT", "operation": "EQUAL"},
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

    attribute_statistic_1 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz.outputs["X"],
        },
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_1.outputs["Max"],
            1: attribute_statistic_1.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_1})

    attribute_statistic_2 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz_1.outputs["Y"],
        },
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_2.outputs["Max"],
            1: attribute_statistic_2.outputs["Mean"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    position_2 = nw.new_node(Nodes.InputPosition)

    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_2})

    attribute_statistic_3 = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": bounding_box.outputs["Bounding Box"],
            "Attribute": separate_xyz_2.outputs["Z"],
        },
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: attribute_statistic_3.outputs["Max"],
            1: attribute_statistic_3.outputs["Mean"],
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


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "Samples", 0.0000),
            ("NodeSocketFloat", "Bottom Nozzle Length", 0.0000),
            ("NodeSocketFloat", "Bottom Nozzle Radius", 0.0000),
            ("NodeSocketFloat", "Nozzle Hinge", 0.0000),
            ("NodeSocketFloat", "Nozzle Slide", 0.0000),
            ("NodeSocketFloat", "Middle Nozzle Radius", 0.0030),
            ("NodeSocketFloat", "Middle Nozzle Length", 0.0500),
            ("NodeSocketFloat", "Top Nozzle Radius", 0.0060),
            ("NodeSocketFloat", "Top Nozzle Length", 0.0100),
            ("NodeSocketFloat", "Spout Length", 0.0400),
            ("NodeSocketFloat", "Spout Base Width", 0.0000),
            ("NodeSocketFloat", "Spout Mouth Width", 0.1000),
            ("NodeSocketFloat", "Spout Bend", 0.0000),
            ("NodeSocketFloat", "Spout Radius", 0.0030),
            ("NodeSocketFloat", "Cap Offset", -0.0150),
            ("NodeSocketMaterial", "Nozzle Material", None),
            ("NodeSocketMaterial", "Base Material", None),
            ("NodeSocketInt", "Cap Samples", 100),
            ("NodeSocketFloat", "Cap Thickness", 0.0050),
            ("NodeSocketInt", "Cylinder Vertices", 100),
            ("NodeSocketInt", "Spout Samples", 6),
            ("NodeSocketBool", "Top is cap", False),
        ],
    )

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_16})

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz})

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Samples"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    resample_curve = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": curve_line, "Count": reroute_5}
    )

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    float_curve = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": spline_parameter.outputs["Factor"]}
    )
    node_utils.assign_curve(float_curve.mapping.curves[0], [])


    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius,
        input_kwargs={"Curve": resample_curve, "Radius": float_curve},
    )

    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={"Radius": 0.1000})

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Base Material"]}
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": curve_to_mesh, "Material": reroute_10},
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material, "Label": "bottom base"},
    )

    reroute_19 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Cylinder Vertices"]}
    )

    reroute_18 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Bottom Nozzle Radius"]},
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Bottom Nozzle Length"]},
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": reroute_19,
            "Radius": reroute_18,
            "Depth": reroute_17,
        },
    )

    reroute_23 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cylinder.outputs["Mesh"]}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_16})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_23, "Translation": combine_xyz_1},
    )

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Nozzle Material"]}
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_15})

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry, "Material": reroute_27},
    )

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_1})

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_29, "Label": "dispenser neck"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [add_jointed_geometry_metadata, add_jointed_geometry_metadata_1]
        },
    )

    reroute_40 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry})

    reroute_41 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_40})

    reroute_45 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_41})

    reroute_46 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_45})

    reroute_53 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_46})

    add_jointed_geometry_metadata_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_53, "Label": "Base"},
    )

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Top is cap"]}
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    reroute_20 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Spout Length"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_20, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Spout Bend"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_1})

    quadratic_b_zier = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Resolution": 30,
            "Start": combine_xyz_4,
            "Middle": (0.0000, 0.0000, 0.0000),
            "End": combine_xyz_5,
        },
    )

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_4, 1: (-1.0000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": quadratic_b_zier,
            "Translation": multiply_1.outputs["Vector"],
        },
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Cap Samples"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    resample_curve_1 = nw.new_node(
        Nodes.ResampleCurve,
        input_kwargs={"Curve": transform_geometry_3, "Count": reroute_22},
    )

    curve_circle_1 = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={
            "Resolution": group_input.outputs["Spout Samples"],
            "Radius": group_input.outputs["Spout Radius"],
        },
    )

    reroute_21 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": curve_circle_1.outputs["Curve"]}
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={"Curve": resample_curve_1, "Profile Curve": reroute_21},
    )

    reroute_34 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": curve_to_mesh_1})

    cylinder_2 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": reroute_19,
            "Radius": group_input.outputs["Top Nozzle Radius"],
            "Depth": group_input.outputs["Top Nozzle Length"],
        },
    )

    reroute_25 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cylinder_2.outputs["Mesh"]}
    )

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": reroute_19,
            "Radius": group_input.outputs["Middle Nozzle Radius"],
            "Depth": group_input.outputs["Middle Nozzle Length"],
        },
    )

    reroute_24 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cylinder_1.outputs["Mesh"]}
    )

    distance_from_center_002_004 = nw.new_node(
        nodegroup_distance_from_center_002_004().name,
        input_kwargs={"Geometry": join_geometry},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": distance_from_center_002_004}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_2})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_24, "Translation": combine_xyz_2},
    )

    add_jointed_geometry_metadata_3 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_1, "Label": "Nozzle neck"},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [add_jointed_geometry_metadata_3, reroute_41]},
    )

    distance_from_center_002_005 = nw.new_node(
        nodegroup_distance_from_center_002_005().name,
        input_kwargs={"Geometry": join_geometry_1},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": distance_from_center_002_005}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Z"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_3})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_25, "Translation": combine_xyz_3},
    )

    add_jointed_geometry_metadata_4 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_2, "Label": "Nozzle top"},
    )

    reroute_47 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_1})

    reroute_48 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_47})

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [add_jointed_geometry_metadata_4, reroute_48]},
    )

    distance_from_center_007 = nw.new_node(
        nodegroup_distance_from_center_007().name,
        input_kwargs={"Geometry": join_geometry_2},
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": distance_from_center_007}
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: 1.9500},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_4})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_34, "Translation": combine_xyz_6},
    )

    add_jointed_geometry_metadata_5 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_4, "Label": "Spout"},
    )

    reroute_30 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_30})

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_31})

    grid = nw.new_node(
        Nodes.MeshGrid, input_kwargs={"Vertices X": 2, "Vertices Y": reroute_35}
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": group_input.outputs["Spout Base Width"]}
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": reroute_20, "Y": group_input.outputs["Spout Mouth Width"]},
    )

    quadratic_b_zier_1 = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Resolution": reroute_3,
            "Start": (0.0000, 0.0000, 0.0000),
            "Middle": combine_xyz_7,
            "End": combine_xyz_8,
        },
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Cap Offset"]}
    )

    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": reroute_8})

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": quadratic_b_zier_1, "Translation": combine_xyz_9},
    )

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_5,
            "Scale": (1.0000, -1.0000, 1.0000),
        },
    )

    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)

    float_curve_1 = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": spline_parameter_1.outputs["Factor"]}
    )
    node_utils.assign_curve(
        float_curve_1.mapping.curves[0],
        [(0.0000, 1.0000), (0.6545, 0.9900), (0.8618, 0.9400), (1.0000, 0.8300)],
    )

    map_range = nw.new_node(
        Nodes.MapRange, input_kwargs={"Value": float_curve_1, 3: reroute_1, 4: 0.0010}
    )

    reroute_26 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": map_range.outputs["Result"]}
    )

    combine_xyz_11 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_26})

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": transform_geometry_6, "Offset": combine_xyz_11},
    )

    curve_to_points_1 = nw.new_node(
        Nodes.CurveToPoints, input_kwargs={"Curve": set_position_1, "Count": reroute_31}
    )

    combine_xyz_10 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": map_range.outputs["Result"]}
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": transform_geometry_5, "Offset": combine_xyz_10},
    )

    curve_to_points = nw.new_node(
        Nodes.CurveToPoints, input_kwargs={"Curve": set_position, "Count": reroute_22}
    )

    reroute_32 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": curve_to_points.outputs["Points"]}
    )

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_32})

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [curve_to_points_1.outputs["Points"], reroute_33]},
    )

    position = nw.new_node(Nodes.InputPosition)

    index = nw.new_node(Nodes.Index)

    sample_index = nw.new_node(
        Nodes.SampleIndex,
        input_kwargs={"Geometry": join_geometry_3, "Value": position, "Index": index},
        attrs={"data_type": "FLOAT_VECTOR", "clamp": True},
    )

    set_position_2 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": grid.outputs["Mesh"], "Position": sample_index},
    )

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Cap Thickness"]}
    )

    combine_xyz_12 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_11})

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": set_position_2,
            "Offset": combine_xyz_12,
            "Individual": False,
        },
    )

    convex_hull = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": extrude_mesh.outputs["Mesh"]}
    )

    reroute_42 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": convex_hull})

    distance_from_center_003_002 = nw.new_node(
        nodegroup_distance_from_center_003_002().name,
        input_kwargs={"Geometry": join_geometry_2},
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": distance_from_center_003_002}
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["Z"], 1: 1.9500},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_13 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_5})

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_42, "Translation": combine_xyz_13},
    )

    add_jointed_geometry_metadata_6 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_7, "Label": "Cap"},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_13,
            "False": add_jointed_geometry_metadata_5,
            "True": add_jointed_geometry_metadata_6,
        },
    )

    reroute_43 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_3}
    )

    reroute_44 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_43})

    reroute_49 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_44})

    reroute_50 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_4}
    )

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_50})

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [switch, reroute_49, reroute_51]}
    )

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_15})

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": join_geometry_4, "Material": reroute_28},
    )

    add_jointed_geometry_metadata_7 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_2, "Label": "Nozzle"},
    )

    distance_from_center_002_006 = nw.new_node(
        nodegroup_distance_from_center_002_006().name,
        input_kwargs={"Geometry": reroute_46},
    )

    separate_xyz_4 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": distance_from_center_002_006}
    )

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_4.outputs["Z"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_14 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_6})

    reroute_36 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata}
    )

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_36})

    reroute_38 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_1}
    )

    reroute_39 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_38})

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 1": reroute_44,
            "Mesh 2": [add_jointed_geometry_metadata_4, reroute_37, reroute_39],
        },
    )

    distance_from_center_002_007 = nw.new_node(
        nodegroup_distance_from_center_002_007().name,
        input_kwargs={"Geometry": difference.outputs["Mesh"]},
    )

    separate_xyz_5 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": distance_from_center_002_007}
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_5.outputs["Z"], 1: -2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_52 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_7})

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "Nozzle Slide",
            "Parent": add_jointed_geometry_metadata_2,
            "Child": add_jointed_geometry_metadata_7,
            "Position": combine_xyz_14,
            "Min": reroute_52,
        },
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Nozzle Hinge"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "Nozzle Hinge",
            "Parent": sliding_joint.outputs["Parent"],
            "Child": sliding_joint.outputs["Child"],
            "Value": reroute_7,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": hinge_joint.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


class SoapDispenserFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    @gin.configurable(module="SoapDispenserFactory")
    def sample_joint_parameters(
        cls,
        Nozzle_Slide_stiffness_min: float = 50.0,
        Nozzle_Slide_stiffness_max: float = 60.0,
        Nozzle_Slide_damping_min: float = 0.0,
        Nozzle_Slide_damping_max: float = 0.0,
        Nozzle_Hinge_stiffness_min: float = 0.0,
        Nozzle_Hinge_stiffness_max: float = 0.0,
        Nozzle_Hinge_damping_min: float = 0.0001,
        Nozzle_Hinge_damping_max: float = 0.0001,
    ):
        return {
            "Nozzle Slide": {
                "stiffness": uniform(
                    Nozzle_Slide_stiffness_min, Nozzle_Slide_stiffness_max
                ),
                "damping": uniform(Nozzle_Slide_damping_min, Nozzle_Slide_damping_max),
                "friction": 20.0,
            },
            "Nozzle Hinge": {
                "stiffness": uniform(
                    Nozzle_Hinge_stiffness_min, Nozzle_Hinge_stiffness_max
                ),
                "damping": uniform(Nozzle_Hinge_damping_min, Nozzle_Hinge_damping_max),
                "friction": 0.0,
            },
        }

    def sample_parameters(self):
        from infinigen.assets.materials import ceramic, metal, plastic

        height = uniform(0.13, 0.21)
        bottom_nozzle_length = uniform(0.01, 0.03)
        bottom_nozzle_radius = uniform(0.007, 0.018)
        middle_nozzle_radius = uniform(0.002, 0.007)
        middle_nozzle_length = uniform(0.02, 0.06)
        top_nozzle_radius = uniform(0.002, 0.007)
        top_nozzle_length = uniform(0.01, 0.027)
        spout_length = uniform(0.035, 0.07)
        spout_base_width = uniform(0.01, 0.03)
        spout_mouth_width = uniform(0.002, 0.004)
        spout_bend = uniform(-0.03, 0)
        cap_offset = uniform(-0.015, -0.012)
        cap_thickness = uniform(0.0035, 0.008)
        samples = 45
        cylinder_vertices = 60
        cap_samples = 40
        is_cap = uniform(0, 1) > 0.5
        spout_radius = uniform(0.001, 0.003)
        spout_samples = 6

        # Sample materials according to specified combinations
        # 1) Ceramic body + Aluminum top
        # 2) Ceramic body + Brushed Metal top
        # 3) Translucent plastic body + Rough plastic top
        # 4) Translucent plastic body + Translucent plastic top
        # 5) Rough plastic body + Rough plastic top
        # 6) Rough plastic body + Translucent plastic top
        combo_idx = int(randint(0, 6))
        print(f"material combo_idx: {combo_idx}")
        if combo_idx == 0:
            base_material = ceramic.Ceramic()()
            nozzle_material = metal.Aluminum()()
        elif combo_idx == 1:
            base_material = ceramic.Ceramic()()
            nozzle_material = metal.BrushedMetal()()
        elif combo_idx == 2:
            base_material = plastic.PlasticTranslucent()()
            nozzle_material = plastic.PlasticRough()()
        elif combo_idx == 3:
            base_material = plastic.PlasticTranslucent()()
            nozzle_material = plastic.PlasticTranslucent()()
        elif combo_idx == 4:
            base_material = plastic.PlasticRough()()
            nozzle_material = plastic.PlasticRough()()
        else:
            base_material = plastic.PlasticRough()()
            nozzle_material = plastic.PlasticTranslucent()()

        return {
            "Height": height,
            "Bottom Nozzle Length": bottom_nozzle_length,
            "Bottom Nozzle Radius": bottom_nozzle_radius,
            "Middle Nozzle Radius": middle_nozzle_radius,
            "Middle Nozzle Length": middle_nozzle_length,
            "Top Nozzle Radius": top_nozzle_radius,
            "Top Nozzle Length": top_nozzle_length,
            "Spout Length": spout_length,
            "Spout Base Width": spout_base_width,
            "Spout Mouth Width": spout_mouth_width,
            "Spout Bend": spout_bend,
            "Spout Radius": spout_radius,
            "Cap Offset": cap_offset,
            "Cap Thickness": cap_thickness,
            "Samples": samples,
            "Cylinder Vertices": cylinder_vertices,
            "Cap Samples": cap_samples,
            "Base Material": base_material,
            "Nozzle Material": nozzle_material,
            "Top is cap": is_cap,
            "Spout Samples": spout_samples,
            "Nozzle Hinge": 0.0,
            "Nozzle Slide": 0.0,
        }

    def create_asset(self, asset_params=None, **kwargs):
        obj = butil.spawn_vert()

        # Create modifier and then adjust the FloatCurve mapping per instance using templates
        obj, mod = butil.modify_mesh(
            obj,
            "NODES",
            apply=False,
            return_mod=True,
            node_group=geometry_nodes(),
            ng_inputs=self.sample_parameters(),
        )

        # Templates: sample template → sample global shift (initial thickness) → sample per-point noise
        templates = [
            [
                (0.0618, 0.4200),
                (0.2945, 0.4600),
                (0.6436, 0.4300),
                (0.8582, 0.3250),
                (0.9964, 0.0650),
            ],
            [(0.0000, 0.3450), (0.5855, 0.3100), (0.9927, 0.1800)],
            [(0.0000, 0.3450), (0.5855, 0.3450), (0.9927, 0.3450)],
            [(0.0000, 0.2250), (0.1345, 0.3900), (0.9418, 0.3600), (1.0000, 0.2300)],
            [
                (0.0000, 0.5600),
                (0.3855, 0.3850),
                (0.6436, 0.3200),
                (0.8909, 0.3550),
                (0.9927, 0.1800),
            ],
            [
                (0.0509, 0.3650),
                (0.2364, 0.3050),
                (0.4909, 0.3150),
                (0.8036, 0.4300),
                (0.9855, 0.1850),
            ],
            [
                (0.0800, 0.4550),
                (0.3127, 0.4400),
                (0.5309, 0.4400),
                (0.8182, 0.3750),
                (0.9673, 0.2400),
            ],
            [
                (0.0727, 0.4000),
                (0.2945, 0.4750),
                (0.5418, 0.4950),
                (0.8582, 0.4100),
                (0.9673, 0.2400),
            ],
            [
                (0.0727, 0.4000),
                (0.2945, 0.4750),
                (0.5273, 0.3400),
                (0.7782, 0.4450),
                (0.9927, 0.2900),
            ],
            [
                (0.0255, 0.5600),
                (0.2945, 0.5150),
                (0.5745, 0.4750),
                (0.8436, 0.3900),
                (0.9855, 0.2250),
            ],
            [
                (0.0073, 0.4350),
                (0.2764, 0.4200),
                (0.5564, 0.3150),
                (0.8764, 0.3850),
                (0.9855, 0.2250),
            ],
        ]

        chosen = templates[int(randint(0, len(templates)))]
        noise_sigma = 0.03
        shift = normal(0.0, noise_sigma)  # initial thickness shift up/down
        # print(normal(0.0, noise_sigma), normal(0.0, noise_sigma), normal(0.0, noise_sigma))
        points = []
        for x, y in chosen:
            x_new = float(x + normal(0.0, noise_sigma))
            x_new = max(0.0, min(1.0, x_new))
            y_new = float(y + shift + normal(0.0, noise_sigma))
            points.append((float(x_new), y_new))
        points[-1] = (
            1.0,
            max(0.0, points[-1][1]),
        )  # ensure last point is at x=1.0 and y>=0

        # Prefer the FloatCurve that feeds SetCurveRadius; fallback to first FloatCurve
        try:
            float_curves = [
                n
                for n in mod.node_group.nodes
                if n.bl_idname in ("ShaderNodeFloatCurve", "GeometryNodeFloatCurve")
            ]
            target = None
            if float_curves:
                for n in float_curves:
                    for l in mod.node_group.links:
                        if l.from_node == n and (
                            "SetCurveRadius" in l.to_node.bl_idname
                        ):
                            target = n
                            break
                    if target is not None:
                        break
                if target is None:
                    target = float_curves[0]
                node_utils.assign_curve(target.mapping.curves[0], points)
        except Exception:
            pass

        return obj
