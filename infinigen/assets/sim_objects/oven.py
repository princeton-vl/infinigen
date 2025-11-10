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
    "nodegroup_rounded_quad_006", singleton=False, type="GeometryNodeTree"
)
def nodegroup_rounded_quad_006(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "Radius", 0.0000),
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

    map_range = nw.new_node(
        Nodes.MapRange, input_kwargs={"Value": group_input.outputs["Radius"], 4: 0.5000}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum, 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={"Curve": quadrilateral, "Count": 6, "Radius": multiply},
        attrs={"mode": "POLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Curve": fillet_curve},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_rounded_quad_004", singleton=False, type="GeometryNodeTree"
)
def nodegroup_rounded_quad_004(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "Radius", 0.0000),
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

    map_range = nw.new_node(
        Nodes.MapRange, input_kwargs={"Value": group_input.outputs["Radius"], 4: 0.5000}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum, 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={"Curve": quadrilateral, "Count": 6, "Radius": multiply},
        attrs={"mode": "POLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Curve": fillet_curve},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_handle_022", singleton=False, type="GeometryNodeTree"
)
def nodegroup_handle_022(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketBool", "Handle Type", False),
            ("NodeSocketFloat", "Handle Length", 0.5000),
            ("NodeSocketFloat", "Handle Protrusion", 0.0000),
            ("NodeSocketFloat", "Handle Width", 0.0300),
            ("NodeSocketFloat", "Handle Height", 0.0500),
            ("NodeSocketFloat", "Handle Radius", 0.0200),
        ],
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": divide})

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: (0.0000, -1.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Protrusion"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    quadratic_b_zier = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Resolution": 10,
            "Start": multiply.outputs["Vector"],
            "Middle": combine_xyz_1,
            "End": combine_xyz,
        },
    )

    rounded_quad_006 = nw.new_node(
        nodegroup_rounded_quad_006().name,
        input_kwargs={
            "Width": group_input.outputs["Handle Width"],
            "Height": group_input.outputs["Handle Height"],
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": quadratic_b_zier,
            "Profile Curve": rounded_quad_006,
            "Fill Caps": True,
        },
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 16,
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Length"],
        },
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input.outputs["Handle Protrusion"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_2,
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 16,
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Protrusion"],
        },
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Protrusion"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Radius"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: divide_2, 1: multiply_2})

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_1, "Y": add}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": combine_xyz_3,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_3, 1: (1.0000, -1.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": multiply_3.outputs["Vector"],
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [transform_geometry, transform_geometry_1, transform_geometry_2]
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Handle Type"],
            "False": curve_to_mesh,
            "True": join_geometry,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": switch},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_handle_021", singleton=False, type="GeometryNodeTree"
)
def nodegroup_handle_021(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketBool", "Handle Type", False),
            ("NodeSocketFloat", "Handle Length", 0.5000),
            ("NodeSocketFloat", "Handle Protrusion", 0.0000),
            ("NodeSocketFloat", "Handle Width", 0.0300),
            ("NodeSocketFloat", "Handle Height", 0.0500),
            ("NodeSocketFloat", "Handle Radius", 0.0200),
        ],
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": divide})

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: (0.0000, -1.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Protrusion"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    quadratic_b_zier = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Resolution": 10,
            "Start": multiply.outputs["Vector"],
            "Middle": combine_xyz_1,
            "End": combine_xyz,
        },
    )

    rounded_quad_004 = nw.new_node(
        nodegroup_rounded_quad_004().name,
        input_kwargs={
            "Width": group_input.outputs["Handle Width"],
            "Height": group_input.outputs["Handle Height"],
        },
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": quadratic_b_zier,
            "Profile Curve": rounded_quad_004,
            "Fill Caps": True,
        },
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 16,
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Length"],
        },
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input.outputs["Handle Protrusion"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_2,
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 16,
            "Radius": group_input.outputs["Handle Radius"],
            "Depth": group_input.outputs["Handle Protrusion"],
        },
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Protrusion"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Length"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Handle Radius"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: divide_2, 1: multiply_2})

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_1, "Y": add}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": combine_xyz_3,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_3, 1: (1.0000, -1.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": multiply_3.outputs["Vector"],
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [transform_geometry, transform_geometry_1, transform_geometry_2]
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Handle Type"],
            "False": curve_to_mesh,
            "True": join_geometry,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": switch},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_rounded_quad_005", singleton=False, type="GeometryNodeTree"
)
def nodegroup_rounded_quad_005(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "Radius", 0.0000),
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

    map_range = nw.new_node(
        Nodes.MapRange, input_kwargs={"Value": group_input.outputs["Radius"], 4: 0.5000}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum, 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={"Curve": quadrilateral, "Count": 6, "Radius": multiply},
        attrs={"mode": "POLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Curve": fillet_curve},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_oven_rack_003", singleton=False, type="GeometryNodeTree"
)
def nodegroup_oven_rack_003(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Radius", 0.0100),
            ("NodeSocketFloat", "Length", 0.5000),
            ("NodeSocketFloat", "Depth", 0.0000),
            ("NodeSocketInt", "Grate Count", 8),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Length"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply})

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": group_input.outputs["Radius"]}
    )

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: (0.0000, -1.0000, 0.0000), 2: combine_xyz_1},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_1})

    add = nw.new_node(Nodes.VectorMath, input_kwargs={0: combine_xyz, 1: combine_xyz_2})

    mesh_line = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={
            "Count": group_input.outputs["Grate Count"],
            "Start Location": multiply_add.outputs["Vector"],
            "Offset": add.outputs["Vector"],
        },
        attrs={"mode": "END_POINTS"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Radius"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 1: multiply_2},
        attrs={"operation": "SUBTRACT"},
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 12,
            "Radius": group_input.outputs["Radius"],
            "Depth": subtract,
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
            "Vertices": 10,
            "Radius": group_input.outputs["Radius"],
            "Depth": group_input.outputs["Length"],
        },
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_3, 1: group_input.outputs["Radius"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": subtract_1})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": combine_xyz_3,
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_4, 1: group_input.outputs["Radius"]}
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": add_1})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": combine_xyz_4,
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                realize_instances,
                transform_geometry_1,
                transform_geometry_2,
                transform_geometry_3,
            ]
        },
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": join_geometry}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_4},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_distance_from_center_005", singleton=False, type="GeometryNodeTree"
)
def nodegroup_distance_from_center_005(nw: NodeWrangler):
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


@node_utils.to_nodegroup(
    "nodegroup_drawer_004", singleton=False, type="GeometryNodeTree"
)
def nodegroup_drawer_004(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "Drawer Thickness", 0.0000),
            ("NodeSocketBool", "Handle Type", False),
            ("NodeSocketFloat", "Handle Length", 0.5000),
            ("NodeSocketFloat", "Handle Protrusion", 0.0000),
            ("NodeSocketFloat", "Handle Width", 0.0300),
            ("NodeSocketFloat", "Handle Height", 0.0500),
            ("NodeSocketFloat", "Handle Radius", 0.0200),
            ("NodeSocketMaterial", "Drawer Material", None),
            ("NodeSocketMaterial", "Drawer Handle Material", None),
        ],
    )

    cube = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": group_input.outputs["Size"]}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Drawer Thickness"],
            "Y": group_input.outputs["Drawer Thickness"],
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

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Translation": (0.0000, 0.0000, 0.0100),
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": cube.outputs["Mesh"], "Mesh 2": transform_geometry},
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: (0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": difference.outputs["Mesh"],
            "Translation": multiply.outputs["Vector"],
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_1,
            "Material": group_input.outputs["Drawer Material"],
        },
    )

    handle_022 = nw.new_node(
        nodegroup_handle_022().name,
        input_kwargs={
            "Handle Type": group_input.outputs["Handle Type"],
            "Handle Length": group_input.outputs["Handle Length"],
            "Handle Protrusion": group_input.outputs["Handle Protrusion"],
            "Handle Width": group_input.outputs["Handle Width"],
            "Handle Height": group_input.outputs["Handle Height"],
            "Handle Radius": group_input.outputs["Handle Radius"],
        },
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": handle_022,
            "Material": group_input.outputs["Drawer Handle Material"],
        },
    )

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: (1.0000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": set_material_1,
            "Translation": multiply_1.outputs["Vector"],
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [set_material, transform_geometry_2]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": join_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_distance_from_center_004", singleton=False, type="GeometryNodeTree"
)
def nodegroup_distance_from_center_004(nw: NodeWrangler):
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


@node_utils.to_nodegroup("nodegroup_door_003", singleton=False, type="GeometryNodeTree")
def nodegroup_door_003(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketBool", "Handle Type", False),
            ("NodeSocketFloat", "Handle Length", 0.5000),
            ("NodeSocketFloat", "Handle Protrusion", 0.0000),
            ("NodeSocketFloat", "Handle Width", 0.0300),
            ("NodeSocketFloat", "Handle Height", 0.0500),
            ("NodeSocketVector", "Handle Pos Offset", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "Handle Radius", 0.0200),
            ("NodeSocketFloat", "Window Width", 0.6000),
            ("NodeSocketFloat", "Window Height", 0.6000),
            ("NodeSocketFloat", "Window Radius", 0.0000),
            ("NodeSocketMaterial", "Door Handle Material", None),
            ("NodeSocketMaterial", "Oven Door Main Material", None),
            ("NodeSocketMaterial", "Door Window Border Material", None),
            ("NodeSocketMaterial", "Door Window Material", None),
        ],
    )

    rounded_quad_005 = nw.new_node(
        nodegroup_rounded_quad_005().name,
        input_kwargs={
            "Width": group_input.outputs["Window Height"],
            "Height": group_input.outputs["Window Width"],
            "Radius": group_input.outputs["Window Radius"],
        },
    )

    fill_curve = nw.new_node(
        Nodes.FillCurve,
        input_kwargs={"Curve": rounded_quad_005},
        attrs={"mode": "NGONS"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: separate_xyz.outputs["X"], 1: 0.0001}
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={"Mesh": fill_curve, "Offset Scale": add, "Individual": False},
    )

    flip_faces = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": fill_curve})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [extrude_mesh.outputs["Mesh"], flip_faces]},
    )

    convex_hull = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": join_geometry}
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: -2.0000}, attrs={"operation": "DIVIDE"}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": convex_hull,
            "Translation": combine_xyz,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry_1})

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_1,
            "Scale": (1.0001, 0.9000, 0.9000),
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": reroute_10, "Mesh 2": transform_geometry_2},
        attrs={"solver": "EXACT"},
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Window Border Material"]},
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": difference.outputs["Mesh"], "Material": reroute_6},
    )

    cube = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": group_input.outputs["Size"]}
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": cube.outputs["Mesh"]})

    difference_1 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": reroute_8, "Mesh 2": transform_geometry_1},
        attrs={"solver": "EXACT"},
    )

    reroute_5 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Oven Door Main Material"]},
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": difference_1.outputs["Mesh"], "Material": reroute_5},
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_1})

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    reroute_7 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Window Material"]},
    )

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_2, "Material": reroute_7},
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_3})

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [set_material_2, reroute_13, reroute_14]},
    )

    add_jointed_geometry_metadata_041 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": join_geometry_1, "Label": "door"},
    )

    handle_021 = nw.new_node(
        nodegroup_handle_021().name,
        input_kwargs={
            "Handle Type": group_input.outputs["Handle Type"],
            "Handle Length": group_input.outputs["Handle Length"],
            "Handle Protrusion": group_input.outputs["Handle Protrusion"],
            "Handle Width": group_input.outputs["Handle Width"],
            "Handle Height": group_input.outputs["Handle Height"],
            "Handle Radius": group_input.outputs["Handle Radius"],
        },
    )

    add_jointed_geometry_metadata_040 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": handle_021, "Label": "door handle"},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Handle Material"]},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_040,
            "Material": reroute_4,
        },
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": cube.outputs["Mesh"]}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Pos Offset"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box.outputs["Max"],
            1: (1.0000, 0.0000, 1.0000),
            2: reroute_3,
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": set_material,
            "Translation": multiply_add.outputs["Vector"],
        },
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [add_jointed_geometry_metadata_041, reroute_11]},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Size"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_1, 1: (0.5000, 0.0000, 0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_2,
            "Translation": multiply.outputs["Vector"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": transform_geometry_3},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_lower_drawer_base_003", singleton=False, type="GeometryNodeTree"
)
def nodegroup_lower_drawer_base_003(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Drawer Border Thickness", 0.0000),
            ("NodeSocketMaterial", "Lower Drawer Base Material", None),
        ],
    )

    cube = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": group_input.outputs["Size"]}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Drawer Border Thickness"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": divide,
            "Y": group_input.outputs["Drawer Border Thickness"],
            "Z": group_input.outputs["Drawer Border Thickness"],
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

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Drawer Border Thickness"], 1: 4.0000},
        attrs={"operation": "DIVIDE"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: divide_1, 1: 0.0010})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": add})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube_1.outputs["Mesh"], "Translation": combine_xyz_1},
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": cube.outputs["Mesh"], "Mesh 2": transform_geometry},
        attrs={"solver": "EXACT"},
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], 1: (0.0000, 0.0000, -0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": difference.outputs["Mesh"],
            "Translation": multiply.outputs["Vector"],
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_1,
            "Material": group_input.outputs["Lower Drawer Base Material"],
        },
    )

    add_jointed_geometry_metadata_039 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material, "Label": "drawer base"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": add_jointed_geometry_metadata_039},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_panel_box_003", singleton=False, type="GeometryNodeTree"
)
def nodegroup_panel_box_003(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Panel Width", 0.0000),
            ("NodeSocketFloat", "Panel Height", 0.0000),
            ("NodeSocketString", "Time", "12:38"),
            ("NodeSocketMaterial", "Panel Box Material", None),
            ("NodeSocketMaterial", "Panel Material", None),
            ("NodeSocketMaterial", "Text Material", None),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Size"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": reroute_11})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": 0.0010,
            "Y": group_input.outputs["Panel Width"],
            "Z": group_input.outputs["Panel Height"],
        },
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_1, 1: (0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Translation": multiply.outputs["Vector"],
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": cube.outputs["Mesh"], "Mesh 2": transform_geometry},
        attrs={"solver": "EXACT"},
    )

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Panel Box Material"]}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": difference.outputs["Mesh"], "Material": reroute_5},
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material})

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Panel Material"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry, "Material": reroute_7},
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_1})

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Time"]}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Panel Height"]},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Panel Width"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    string_to_curves = nw.new_node(
        "GeometryNodeStringToCurves",
        input_kwargs={
            "String": reroute_4,
            "Size": multiply_1,
            "Text Box Width": reroute_3,
        },
        attrs={"align_y": "MIDDLE", "overflow": "SCALE_TO_FIT", "align_x": "CENTER"},
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve,
        input_kwargs={
            "Curve": string_to_curves.outputs["Curve Instances"],
            "Count": 18,
        },
    )

    fill_curve = nw.new_node(
        Nodes.FillCurve, input_kwargs={"Curve": resample_curve}, attrs={"mode": "NGONS"}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": reroute_3})

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_1, 1: (-0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": fill_curve,
            "Translation": multiply_2.outputs["Vector"],
        },
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": transform_geometry_1,
            "Offset Scale": 0.0010,
            "Individual": False,
        },
    )

    separate_geometry = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={
            "Geometry": extrude_mesh.outputs["Mesh"],
            "Selection": extrude_mesh.outputs["Side"],
        },
        attrs={"domain": "FACE"},
    )

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_1}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [reroute_14, extrude_mesh.outputs["Mesh"]]},
    )

    flip_faces = nw.new_node(
        Nodes.FlipFaces,
        input_kwargs={"Mesh": join_geometry, "Selection": extrude_mesh.outputs["Top"]},
    )

    flip_faces_1 = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": flip_faces})

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [separate_geometry.outputs["Selection"], flip_faces_1]
        },
    )

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply.outputs["Vector"]}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_2,
            "Translation": reroute_12,
            "Rotation": (1.5708, 0.0000, 1.5708),
        },
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Text Material"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_2, "Material": reroute_9},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [reroute_16, reroute_15, set_material_2]},
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_13, 1: (0.0000, 0.0000, 0.5000)},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Translation": multiply_add.outputs["Vector"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": transform_geometry_3},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_oven_body_003", singleton=False, type="GeometryNodeTree"
)
def nodegroup_oven_body_003(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Major Border Thickness", 0.0000),
            ("NodeSocketFloat", "Minor Border Thickness", 0.5000),
            ("NodeSocketFloat", "Minor Border Depth", 0.0000),
            ("NodeSocketFloat", "Vent Cutter Length", 0.0000),
            ("NodeSocketFloat", "Vent Cutter Height", 0.0000),
            ("NodeSocketInt", "Vent Count", 7),
            ("NodeSocketBool", "Include Wire", False),
            ("NodeSocketFloat", "Wire Radius", 0.0200),
            ("NodeSocketFloat", "Grate Radius", 0.0100),
            ("NodeSocketInt", "Vertical Grate Count", 8),
            ("NodeSocketInt", "Rack Count", 10),
            ("NodeSocketFloat", "Rack Offset", 0.0000),
            ("NodeSocketFloat", "Rack Z Offset", 0.0000),
            ("NodeSocketMaterial", "Wire Material", None),
            ("NodeSocketMaterial", "Rack Material", None),
            ("NodeSocketMaterial", "Oven Body Material", None),
        ],
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Include Wire"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_19 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Size"]}
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    reroute = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Major Border Thickness"]},
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Major Border Thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_1, "Y": multiply, "Z": multiply}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_20, 1: combine_xyz},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: subtract.outputs["Vector"], 1: (0.0000, 1.0000, 1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_1.outputs["Vector"], 1: (0.0000, 0.0200, 0.0200)},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": add.outputs["Vector"]}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Width": separate_xyz.outputs["Z"],
            "Height": separate_xyz.outputs["Y"],
        },
    )

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": quadrilateral})

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Minor Border Thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply_2, "Z": multiply_2}
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_20, 1: combine_xyz_1},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Minor Border Depth"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": reroute_3})

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: subtract_1.outputs["Vector"],
            1: (0.0000, 1.0000, 1.0000),
            2: combine_xyz_2,
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    cube = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": multiply_add.outputs["Vector"]}
    )

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_25})

    multiply_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_2, 1: (-0.5000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: reroute_27,
            1: (0.5000, 0.0000, 0.0000),
            2: multiply_3.outputs["Vector"],
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Translation": multiply_add_1.outputs["Vector"],
        },
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": transform_geometry}
    )

    multiply_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Min"], 1: (1.0000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_31,
            "Translation": multiply_4.outputs["Vector"],
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Wire Radius"]}
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 8, "Radius": reroute_6}
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": transform_geometry_1,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    switch = nw.new_node(
        Nodes.Switch, input_kwargs={"Switch": reroute_5, "True": curve_to_mesh}
    )

    add_jointed_geometry_metadata_036 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": switch, "Label": "wire"},
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Wire Material"]}
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_036,
            "Material": reroute_14,
        },
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": reroute_25})

    reroute_28 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cube_1.outputs["Mesh"]}
    )

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_28})

    cube_2 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": subtract.outputs["Vector"]}
    )

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_23, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube_2.outputs["Mesh"], "Translation": combine_xyz_3},
    )

    reroute_30 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_2}
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": reroute_29, "Mesh 2": [transform_geometry, reroute_30]},
        attrs={"solver": "EXACT"},
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Oven Body Material"]}
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_17})

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": difference.outputs["Mesh"], "Material": reroute_18},
    )

    reroute_34 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_1})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, reroute_34]}
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Grate Radius"]}
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    bounding_box_1 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": cube_2.outputs["Mesh"]}
    )

    add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box_1.outputs["Max"], 1: (0.0000, 1.0000, 0.0000)},
    )

    add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box_1.outputs["Min"], 1: (0.0000, 1.0000, 0.0000)},
    )

    subtract_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_1.outputs["Vector"], 1: add_2.outputs["Vector"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_2.outputs["Vector"]}
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Major Border Thickness"],
            1: group_input.outputs["Minor Border Depth"],
        },
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: add_3},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": subtract_3})

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    reroute_9 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Vertical Grate Count"]},
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    oven_rack_003 = nw.new_node(
        nodegroup_oven_rack_003().name,
        input_kwargs={
            "Radius": reroute_8,
            "Length": separate_xyz_1.outputs["Y"],
            "Depth": reroute_22,
            "Grate Count": reroute_10,
        },
    )

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": oven_rack_003})

    bounding_box_2 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": cube_1.outputs["Mesh"]}
    )

    multiply_5 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box_2.outputs["Min"], 1: (1.0000, 0.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": multiply_5.outputs["Vector"]}
    )

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    add_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: separate_xyz_3.outputs["X"], 1: reroute_26}
    )

    bounding_box_3 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": transform_geometry}
    )

    separate_xyz_4 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box_3.outputs["Min"]}
    )

    add_5 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_4, 1: separate_xyz_4.outputs["X"]}
    )

    divide_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_5, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide_1})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.9000

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": value, "Y": 1.0000, "Z": 1.0000}
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_33,
            "Translation": combine_xyz_4,
            "Scale": combine_xyz_5,
        },
    )

    reroute_15 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Rack Material"]}
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_15})

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_3, "Material": reroute_16},
    )

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Rack Count"]}
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Rack Z Offset"]}
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Rack Offset"]}
    )

    mesh_line = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={
            "Count": reroute_12,
            "Start Location": combine_xyz_6,
            "Offset": combine_xyz_7,
        },
    )

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": mesh_line})

    reroute_32 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_32, 1: value},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_6})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Oven Base": join_geometry,
            "Oven Rack": set_material_2,
            "Points": reroute_24,
            "Max Oven": reroute_35,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_duplicate_joints_on_parent_003", singleton=False, type="GeometryNodeTree"
)
def nodegroup_duplicate_joints_on_parent_003(nw: NodeWrangler):
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
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Dimensions", (1.0000, 1.0000, 1.0000)),
            ("NodeSocketFloat", "Major Border Thickness", 0.0000),
            ("NodeSocketFloat", "Minor Border Thickness", 0.5000),
            ("NodeSocketFloat", "Minor Border Depth", 0.0000),
            ("NodeSocketBool", "Include Wire", False),
            ("NodeSocketFloat", "Wire Radius", 0.0200),
            ("NodeSocketFloat", "Grate Radius", 0.0100),
            ("NodeSocketInt", "Vertical Grate Count", 8),
            ("NodeSocketInt", "Rack Count", 10),
            ("NodeSocketFloat", "Rack Offset", 0.0000),
            ("NodeSocketFloat", "Rack Z Offset", 0.0000),
            ("NodeSocketFloat", "Door Thickness", 0.0000),
            ("NodeSocketBool", "Bar Handle", False),
            ("NodeSocketFloat", "Handle Length", 0.5000),
            ("NodeSocketFloat", "Handle Protrusion", 0.0000),
            ("NodeSocketFloat", "Handle Width", 0.0300),
            ("NodeSocketFloat", "Handle Height", 0.0500),
            ("NodeSocketVector", "Handle Pos Offset", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "Handle Radius", 0.0200),
            ("NodeSocketFloat", "Window Width", 0.6000),
            ("NodeSocketFloat", "Window Height", 0.6000),
            ("NodeSocketFloat", "Window Radius", 0.0000),
            ("NodeSocketBool", "Include Panel Box", False),
            ("NodeSocketFloat", "Panel Box Height", 0.0000),
            ("NodeSocketFloat", "Panel Width", 0.0000),
            ("NodeSocketFloat", "Panel Height", 0.0000),
            ("NodeSocketString", "Time", "12:38"),
            ("NodeSocketBool", "Include Lower Drawer", False),
            ("NodeSocketFloat", "Lower Drawer Height", 0.0000),
            ("NodeSocketFloat", "Drawer Thickness", 0.0000),
            ("NodeSocketBool", "Drawer Bar Handle", False),
            ("NodeSocketFloat", "Drawer Handle Length", 0.5000),
            ("NodeSocketFloat", "Drawer Handle Protrusion", 0.0000),
            ("NodeSocketFloat", "Drawer Handle Width", 0.0300),
            ("NodeSocketFloat", "Drawer Handle Height", 0.0500),
            ("NodeSocketFloat", "Drawer Handle Radius", 0.0200),
            ("NodeSocketFloat", "Drawer Border Thickness", 0.0000),
            ("NodeSocketMaterial", "Lower Drawer Base Material", None),
            ("NodeSocketMaterial", "Panel Box Material", None),
            ("NodeSocketMaterial", "Panel Material", None),
            ("NodeSocketMaterial", "Text Material", None),
            ("NodeSocketMaterial", "Wire Material", None),
            ("NodeSocketMaterial", "Rack Material", None),
            ("NodeSocketMaterial", "Oven Body Material", None),
            ("NodeSocketMaterial", "Door Handle Material", None),
            ("NodeSocketMaterial", "Oven Door Main Material", None),
            ("NodeSocketMaterial", "Door Window Border Material", None),
            ("NodeSocketMaterial", "Door Window Material", None),
            ("NodeSocketMaterial", "Drawer Material", None),
            ("NodeSocketMaterial", "Drawer Handle Material", None),
        ],
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Include Lower Drawer"]},
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    reroute_77 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    reroute_92 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_77})

    reroute_67 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Include Panel Box"]}
    )

    reroute_68 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_67})

    reroute_81 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_68})

    oven_body_003 = nw.new_node(
        nodegroup_oven_body_003().name,
        input_kwargs={
            "Size": group_input.outputs["Dimensions"],
            "Major Border Thickness": group_input.outputs["Major Border Thickness"],
            "Minor Border Thickness": group_input.outputs["Minor Border Thickness"],
            "Minor Border Depth": group_input.outputs["Minor Border Depth"],
            "Include Wire": group_input.outputs["Include Wire"],
            "Wire Radius": group_input.outputs["Wire Radius"],
            "Grate Radius": group_input.outputs["Grate Radius"],
            "Vertical Grate Count": group_input.outputs["Vertical Grate Count"],
            "Rack Count": group_input.outputs["Rack Count"],
            "Rack Offset": group_input.outputs["Rack Offset"],
            "Rack Z Offset": group_input.outputs["Rack Z Offset"],
            "Wire Material": group_input.outputs["Wire Material"],
            "Rack Material": group_input.outputs["Rack Material"],
            "Oven Body Material": group_input.outputs["Oven Body Material"],
        },
    )

    add_jointed_geometry_metadata_037 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": oven_body_003.outputs["Oven Base"],
            "Label": "oven base",
        },
    )

    add_jointed_geometry_metadata_038 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": oven_body_003.outputs["Oven Rack"],
            "Label": "oven rack",
        },
    )

    reroute_71 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": oven_body_003.outputs["Max Oven"]}
    )

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "rack_joint",
            "Parent": add_jointed_geometry_metadata_037,
            "Child": add_jointed_geometry_metadata_038,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Max": reroute_71,
        },
    )

    reroute_70 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": oven_body_003.outputs["Points"]}
    )

    duplicate_joints_on_parent_003 = nw.new_node(
        nodegroup_duplicate_joints_on_parent_003().name,
        input_kwargs={
            "Parent": sliding_joint.outputs["Parent"],
            "Child": sliding_joint.outputs["Child"],
            "Points": reroute_70,
        },
    )

    reroute_82 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": duplicate_joints_on_parent_003}
    )

    reroute_83 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_82})

    reroute_88 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_83})

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Dimensions"]}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Minor Border Thickness"]},
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: separate_xyz.outputs["X"], 1: reroute_3}
    )

    reroute_72 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["Y"]}
    )

    reroute_36 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Panel Box Height"]}
    )

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_36})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": reroute_72, "Z": reroute_37}
    )

    reroute_38 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Panel Width"]}
    )

    reroute_39 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_38})

    reroute_40 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Panel Height"]}
    )

    reroute_41 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_40})

    reroute_42 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Time"]}
    )

    reroute_43 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_42})

    reroute_44 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Panel Box Material"]}
    )

    reroute_45 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_44})

    reroute_46 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Panel Material"]}
    )

    reroute_47 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_46})

    reroute_48 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Text Material"]}
    )

    reroute_49 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_48})

    panel_box_003 = nw.new_node(
        nodegroup_panel_box_003().name,
        input_kwargs={
            "Size": combine_xyz,
            "Panel Width": reroute_39,
            "Panel Height": reroute_41,
            "Time": reroute_43,
            "Panel Box Material": reroute_45,
            "Panel Material": reroute_47,
            "Text Material": reroute_49,
        },
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Dimensions"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_74 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Minor Border Thickness"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide})

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_74, 1: (0.0000, 0.0000, 0.5000), 2: combine_xyz_1},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_84 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply_add.outputs["Vector"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": panel_box_003, "Translation": reroute_84},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_geometry, reroute_83]}
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_81, "False": reroute_88, "True": join_geometry},
    )

    reroute_91 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch})

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Lower Drawer Height"]}
    )

    multiply_add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_1, 1: (1.0000, 1.0000, 0.0000), 2: combine_xyz_2},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Drawer Border Thickness"]},
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_76 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    reroute_66 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Lower Drawer Base Material"]},
    )

    lower_drawer_base_003 = nw.new_node(
        nodegroup_lower_drawer_base_003().name,
        input_kwargs={
            "Size": multiply_add_1.outputs["Vector"],
            "Drawer Border Thickness": reroute_76,
            "Lower Drawer Base Material": reroute_66,
        },
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_74, 1: (0.0000, 0.0000, -0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": lower_drawer_base_003,
            "Translation": multiply.outputs["Vector"],
        },
    )

    reroute_87 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_1}
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [switch, reroute_87]}
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_77,
            "False": reroute_91,
            "True": join_geometry_1,
        },
    )

    reroute_93 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_1})

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_93, "Label": "oven_base"},
    )

    reroute_99 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata}
    )

    reroute_75 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Dimensions"]}
    )

    reroute_69 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz_1.outputs["Y"]}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Z"], 1: reroute_3},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_75, "Y": reroute_69, "Z": subtract}
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Bar Handle"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Length"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Protrusion"]}
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_12})

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Width"]}
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Height"]}
    )

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    reroute_18 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Pos Offset"]}
    )

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    reroute_20 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Handle Radius"]}
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    reroute_22 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Window Width"]}
    )

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    reroute_24 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Window Height"]}
    )

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_24})

    reroute_26 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Window Radius"]}
    )

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_26})

    reroute_28 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Handle Material"]},
    )

    reroute_29 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_28})

    reroute_30 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Oven Door Main Material"]},
    )

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_30})

    reroute_32 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Window Border Material"]},
    )

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_32})

    reroute_34 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Door Window Material"]},
    )

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_34})

    door_003 = nw.new_node(
        nodegroup_door_003().name,
        input_kwargs={
            "Size": combine_xyz_3,
            "Handle Type": reroute_9,
            "Handle Length": reroute_11,
            "Handle Protrusion": reroute_13,
            "Handle Width": reroute_15,
            "Handle Height": reroute_17,
            "Handle Pos Offset": reroute_19,
            "Handle Radius": reroute_21,
            "Window Width": reroute_23,
            "Window Height": reroute_25,
            "Window Radius": reroute_27,
            "Door Handle Material": reroute_29,
            "Oven Door Main Material": reroute_31,
            "Door Window Border Material": reroute_33,
            "Door Window Material": reroute_35,
        },
    )

    reroute_85 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": door_003})

    reroute_86 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_85})

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_86, "Label": "oven_door"},
    )

    distance_from_center_004 = nw.new_node(
        nodegroup_distance_from_center_004().name, input_kwargs={"Geometry": switch_1}
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0000

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.0000

    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = 0.0006

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Include Panel Box"],
            "False": value_1,
            "True": value_2,
        },
        attrs={"input_type": "FLOAT"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_3, 1: switch_2})

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1, 1: -1.0000}, attrs={"operation": "MULTIPLY"}
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_68, "False": value, "True": multiply_1},
        attrs={"input_type": "FLOAT"},
    )

    add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Lower Drawer Height"],
            1: group_input.outputs["Minor Border Thickness"],
        },
    )

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_7, "False": reroute_3, "True": add_2},
        attrs={"input_type": "FLOAT"},
    )

    reroute_79 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_4})

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": switch_3, "Z": reroute_79}
    )

    reroute_90 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_4})

    multiply_add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: distance_from_center_004,
            1: (1.0000, 0.0000, -1.0000),
            2: reroute_90,
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "door joint",
            "Parent": reroute_99,
            "Child": add_jointed_geometry_metadata_1,
            "Position": multiply_add_2.outputs["Vector"],
            "Axis": (0.0000, 1.0000, 0.0000),
            "Max": 1.5708,
        },
    )

    reroute_94 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint.outputs["Geometry"]}
    )

    add_jointed_geometry_metadata_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_94, "Label": "oven_with_door"},
    )

    reroute_95 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_2}
    )

    reroute_96 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_95})

    reroute_97 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_96})

    reroute_78 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": multiply_add_1.outputs["Vector"]}
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Drawer Border Thickness"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_1, "Y": reroute_5, "Z": reroute_5}
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_5, "Scale": -1.0000},
        attrs={"operation": "SCALE"},
    )

    add_3 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: reroute_78, 1: scale.outputs["Vector"]}
    )

    reroute_50 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Thickness"]}
    )

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_50})

    reroute_52 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Bar Handle"]}
    )

    reroute_53 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_52})

    reroute_54 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Drawer Handle Length"]},
    )

    reroute_55 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_54})

    reroute_56 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Drawer Handle Protrusion"]},
    )

    reroute_57 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Drawer Handle Width"]},
    )

    reroute_58 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_57})

    reroute_59 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Drawer Handle Height"]},
    )

    reroute_60 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Drawer Handle Radius"]},
    )

    reroute_61 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_60})

    reroute_62 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Drawer Material"]}
    )

    reroute_63 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_62})

    reroute_64 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Drawer Handle Material"]},
    )

    reroute_65 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_64})

    drawer_004 = nw.new_node(
        nodegroup_drawer_004().name,
        input_kwargs={
            "Size": add_3.outputs["Vector"],
            "Drawer Thickness": reroute_51,
            "Handle Type": reroute_53,
            "Handle Length": reroute_55,
            "Handle Protrusion": reroute_56,
            "Handle Width": reroute_58,
            "Handle Height": reroute_59,
            "Handle Radius": reroute_61,
            "Drawer Material": reroute_63,
            "Drawer Handle Material": reroute_65,
        },
    )

    reroute_89 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": drawer_004})

    add_jointed_geometry_metadata_3 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_89, "Label": "oven_drawer"},
    )

    distance_from_center_005 = nw.new_node(
        nodegroup_distance_from_center_005().name, input_kwargs={"Geometry": reroute_99}
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Drawer Border Thickness"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Lower Drawer Height"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_2, "Z": divide_3}
    )

    reroute_73 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_6})

    multiply_add_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: distance_from_center_005,
            1: (-1.0000, 0.0000, -1.0000),
            2: reroute_73,
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    reroute_80 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_74})

    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": reroute_80})

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "drawer joint",
            "Parent": reroute_95,
            "Child": add_jointed_geometry_metadata_3,
            "Position": multiply_add_3.outputs["Vector"],
            "Axis": (1.0000, 0.0000, 0.0000),
            "Max": separate_xyz_2.outputs["X"],
        },
    )

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_92,
            "False": reroute_97,
            "True": sliding_joint.outputs["Geometry"],
        },
    )

    reroute_98 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_5})

    bounding_box = nw.new_node(Nodes.BoundingBox, input_kwargs={"Geometry": switch_5})

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_1.outputs["Vector"]}
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": separate_xyz_3.outputs["Z"]}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_98, "Translation": combine_xyz_7},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": transform_geometry_2}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": realize_instances},
        attrs={"is_active_output": True},
    )


class OvenFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    @gin.configurable(module="OvenFactory")
    def sample_joint_parameters(
        cls,
        drawer_joint_stiffness_min: float = 0.0,
        drawer_joint_stiffness_max: float = 0.0,
        drawer_joint_damping_min: float = 80.0,
        drawer_joint_damping_max: float = 120.0,
        rack_joint_stiffness_min: float = 0.0,
        rack_joint_stiffness_max: float = 0.0,
        rack_joint_damping_min: float = 10.0,
        rack_joint_damping_max: float = 20.0,
        door_joint_stiffness_min: float = 115.0,
        door_joint_stiffness_max: float = 125.0,
        door_joint_damping_min: float = 50.0,
        door_joint_damping_max: float = 80.0,
    ):
        return {
            "drawer joint": {
                "stiffness": uniform(
                    drawer_joint_stiffness_min, drawer_joint_stiffness_max
                ),
                "damping": uniform(drawer_joint_damping_min, drawer_joint_damping_max),
            },
            "rack_joint": {
                "stiffness": uniform(
                    rack_joint_stiffness_min, rack_joint_stiffness_max
                ),
                "damping": uniform(rack_joint_damping_min, rack_joint_damping_max),
            },
            "door joint": {
                "stiffness": uniform(
                    door_joint_stiffness_min, door_joint_stiffness_max
                ),
                "damping": uniform(door_joint_damping_min, door_joint_damping_max),
                "friction": 100,
            },
        }

    def sample_time(self):
        hour = randint(1, 13)
        minute = randint(0, 60)
        return f"{hour:02d}:{minute:02d}"

    def sample_parameters(self):
        import numpy as np

        # add code here to randomly sample from parameters
        dimensions = (uniform(0.4, 0.8), uniform(0.6, 0.9), uniform(0.5, 0.7))

        major_border_thickness = uniform(0.04, 0.08)
        minor_border_thickness = uniform(0.005, major_border_thickness)
        minor_border_depth = uniform(0, 0.02)

        grate_radius = uniform(0.002, 0.005)

        inner_height = dimensions[2] - 2 * major_border_thickness
        condensed_inner_height = inner_height * 0.95
        if inner_height < 0.4:
            rack_count = randint(1, 3)
        elif inner_height < 0.6:
            rack_count = randint(2, 3)
        else:
            rack_count = randint(2, 4)

        # rack_offset = uniform(0.1, dimensions[2] / 2 * rack_count)
        rack_offset = condensed_inner_height / (rack_count + 1)

        # min_z_offset = dimensions[2] * -0.5 + major_border_thickness + grate_radius
        # rack_z_offset = uniform(min_z_offset, -0.2)

        rack_z_offset = -rack_offset * ((rack_count - 1) / 2)

        handle_length = uniform(min(dimensions[1], 0.2), dimensions[1])
        handle_protrusion = uniform(0.04, 0.08)
        handle_width = uniform(0.01, 0.03)
        handle_height = uniform(0.02, 0.06)
        handle_radius = handle_height / 2

        r = uniform()
        if r < 0.1:
            window_width = 0
        elif r < 0.7:
            window_width = uniform(dimensions[1] * 0.5, dimensions[1] * 0.8)
        else:
            window_width = dimensions[1] - 0.001

        window_height = uniform(dimensions[2] * 0.5, dimensions[2] * 0.8)

        panel_box_height = uniform(0.1, 0.2)
        panel_width = uniform(0.2, 0.4)
        panel_height = uniform(
            min(0.05, panel_box_height * 0.9), panel_box_height * 0.9
        )

        if uniform() < 0.2:
            y_pos_offset = uniform(
                -dimensions[1] / 2 + handle_length / 2,
                dimensions[1] / 2 - handle_length / 2,
            )
        else:
            y_pos_offset = 0
        handle_pos_offset = (0, y_pos_offset, -handle_height / 2 - uniform(0, 0.02))

        time = self.sample_time()

        # materials
        base_material = weighted_sample(material_assignments.kitchen_appliance_hard)()()
        glass_material = weighted_sample(material_assignments.appliance_front_glass)()()
        wire_material = weighted_sample(material_assignments.decorative_metal)()()
        handle_material = (
            weighted_sample(material_assignments.appliance_handle)()()
            if uniform() < 0.7
            else base_material
        )

        from infinigen.assets.materials import metal

        panel_material = metal.BlackGlass()()

        params = {
            "Dimensions": dimensions,
            "Major Border Thickness": major_border_thickness,
            "Minor Border Thickness": minor_border_thickness,
            "Minor Border Depth": minor_border_depth,
            "Include Wire": np.random.choice([True, False]),
            "Wire Radius": uniform(0.002, 0.01),
            "Grate Radius": grate_radius,
            "Vertical Grate Count": randint(10, 20),
            "Rack Count": rack_count,
            "Rack Offset": rack_offset,
            "Rack Z Offset": rack_z_offset,
            "Door Thickness": uniform(0.01, 0.05),
            "Bar Handle": np.random.choice([True, False]),
            "Handle Length": handle_length,
            "Handle Protrusion": handle_protrusion,
            "Handle Width": handle_width,
            "Handle Height": handle_height,
            "Handle Pos Offset": handle_pos_offset,
            "Handle Radius": handle_radius,
            "Window Width": window_width,
            "Window Height": window_height,
            "Window Radius": uniform() if uniform() < 0.5 else 0,
            "Include Panel Box": np.random.choice([True, False], p=[0.7, 0.3]),
            "Panel Box Height": panel_box_height,
            "Panel Width": panel_width,
            "Panel Height": panel_height,
            "Time": time,
            "Include Lower Drawer": np.random.choice([True, False]),
            "Lower Drawer Height": uniform(0.12, 0.2),
            "Drawer Thickness": uniform(0.005, 0.03),
            "Drawer Bar Handle": np.random.choice([True, False]),
            "Drawer Handle Length": handle_length,
            "Drawer Handle Protrusion": handle_protrusion,
            "Drawer Handle Width": handle_width,
            "Drawer Handle Height": handle_height,
            "Drawer Handle Radius": handle_radius,
            "Drawer Border Thickness": minor_border_thickness,
            "Lower Drawer Base Material": base_material,
            "Panel Box Material": base_material,
            "Panel Material": glass_material,
            "Wire Material": wire_material,
            "Rack Material": wire_material,
            "Oven Body Material": base_material,
            "Door Handle Material": handle_material,
            "Oven Door Main Material": base_material,
            "Door Window Border Material": wire_material
            if uniform() < 0.5
            else base_material,
            "Door Window Material": glass_material
            if uniform() < 0.5
            else panel_material,
            "Drawer Material": base_material,
            "Drawer Handle Material": handle_material,
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
