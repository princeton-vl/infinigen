# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Max Gonzalez Saez-Diez: Primary author
# - Abhishek Joshi: Updates for sim

import gin
from numpy.random import randint, uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.joints import nodegroup_hinge_joint, nodegroup_sliding_joint
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import weighted_sample


@node_utils.to_nodegroup(
    "nodegroup_node_group_009", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_009(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    bounding_box_2 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": group_input.outputs["Geometry"]}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_2.outputs["Max"],
            1: bounding_box_2.outputs["Min"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Z"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": bounding_box_2.outputs["Bounding Box"],
            "Translation": combine_xyz_2,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_3},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_010", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_010(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

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

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Width": separate_xyz.outputs["X"],
            "Height": separate_xyz.outputs["Y"],
        },
    )

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: separate_xyz.outputs["Y"]},
        attrs={"operation": "MINIMUM"},
    )

    map_range = nw.new_node(Nodes.MapRange, input_kwargs={"Value": 0.5000})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum, 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={"Curve": quadrilateral, "Count": 8, "Radius": multiply},
        attrs={"mode": "POLY"},
    )

    curve_to_mesh = nw.new_node(Nodes.CurveToMesh, input_kwargs={"Curve": fillet_curve})

    vector = nw.new_node(Nodes.Vector)
    vector.vector = (0.0000, 0.0000, 10.0000)

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={"Mesh": curve_to_mesh, "Offset": vector},
        attrs={"mode": "EDGES"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": extrude_mesh.outputs["Mesh"],
            "Translation": (0.0000, 0.0000, -5.0000),
            "Scale": (0.8000, 0.5000, 1.0000),
        },
    )

    convex_hull = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": transform_geometry}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Convex Hull": convex_hull},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_001", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_001(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketBool", "Version 4", False),
            ("NodeSocketGeometry", "Input", None),
        ],
    )

    position = nw.new_node(Nodes.InputPosition)

    attribute_statistic = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={"Geometry": group_input.outputs["Input"], "Attribute": position},
        attrs={"data_type": "FLOAT_VECTOR", "domain": "FACE"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": attribute_statistic.outputs["Max"]}
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": position})

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": reroute})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            0: separate_xyz.outputs["Z"],
            1: separate_xyz_1.outputs["Z"],
            "Epsilon": 0.0020,
        },
        attrs={"operation": "EQUAL"},
    )

    separate_geometry = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": group_input.outputs["Input"], "Selection": equal},
        attrs={"domain": "FACE"},
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": separate_geometry.outputs["Selection"]},
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Min"]}
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: separate_xyz_3.outputs["Z"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": subtract_1})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": separate_geometry.outputs["Selection"],
            "Translation": combine_xyz,
        },
    )

    convex_hull = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": transform_geometry}
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh, input_kwargs={"Mesh": convex_hull, "Offset Scale": 0.0100}
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["Version 4"],
            "False": extrude_mesh.outputs["Mesh"],
            "True": convex_hull,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Convex Hull": switch},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_008", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_008(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Mesh", None)]
    )

    separate_geometry = nw.new_node(
        Nodes.SeparateGeometry, input_kwargs={"Geometry": group_input.outputs["Mesh"]}
    )

    subdivision_surface = nw.new_node(
        Nodes.SubdivisionSurface,
        input_kwargs={
            "Mesh": separate_geometry.outputs["Selection"],
            "Level": 2,
            "Edge Crease": 0.1000,
        },
        attrs={"boundary_smooth": "PRESERVE_CORNERS"},
    )

    convex_hull = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": subdivision_surface}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Mesh"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    greater_than = nw.new_node(
        Nodes.Compare, input_kwargs={0: separate_xyz.outputs["Z"]}
    )

    delete_geometry = nw.new_node(
        Nodes.DeleteGeometry,
        input_kwargs={"Geometry": reroute_1, "Selection": greater_than},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [convex_hull, delete_geometry]}
    )

    convex_hull_1 = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": join_geometry}
    )

    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": convex_hull_1})

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": convex_hull_1, "Scale": (0.9300, 0.9300, 1.0000)},
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": reroute_3, "Mesh 2": transform_geometry},
        attrs={"solver": "EXACT"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": difference.outputs["Mesh"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_002", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_002(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Pedal Thickness", 0.0000),
            ("NodeSocketFloat", "Pedal Width", 0.0000),
            ("NodeSocketFloat", "Pedal Roundness", 0.0000),
            ("NodeSocketFloat", "Pedal Outward Stick", 0.0000),
            ("NodeSocketFloat", "Pedal Base Length", 0.0000),
        ],
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Pedal Thickness"]}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_2})

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz_1})

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius, input_kwargs={"Curve": curve_line, "Radius": 1.0000}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Pedal Outward Stick"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Pedal Width"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": multiply, "Height": reroute_1},
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": quadrilateral})

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Pedal Roundness"]}
    )

    maximum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_3, 1: 0.0100},
        attrs={"operation": "MAXIMUM"},
    )

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: reroute_1},
        attrs={"operation": "MINIMUM"},
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: minimum, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: maximum, 1: divide},
        attrs={"operation": "MULTIPLY"},
    )

    minimum_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: 1.0000},
        attrs={"operation": "MINIMUM"},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": reroute_7,
            "Count": 64,
            "Radius": minimum_1,
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius,
            "Profile Curve": fillet_curve,
            "Fill Caps": True,
        },
    )

    maximum_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Pedal Thickness"],
            1: group_input.outputs["Pedal Width"],
        },
        attrs={"operation": "MAXIMUM"},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Pedal Outward Stick"]},
    )

    maximum_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: maximum_1, 1: reroute_4},
        attrs={"operation": "MAXIMUM"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": maximum_2, "Y": maximum_2, "Z": maximum_2}
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_3})

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: maximum_2, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_2})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube_1.outputs["Mesh"], "Translation": combine_xyz_4},
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry})

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": curve_to_mesh, "Mesh 2": reroute_8},
        attrs={"solver": "EXACT"},
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Pedal Base Length"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Pedal Thickness"], 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_1, "Z": divide_2}
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_2})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": difference.outputs["Mesh"], "Translation": reroute_6},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Pedal Base Length"],
            "Y": group_input.outputs["Pedal Width"],
            "Z": group_input.outputs["Pedal Thickness"],
        },
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": cube.outputs["Mesh"]})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_geometry_1, reroute_5]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry})

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": join_geometry}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide_3})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_10, "Translation": combine_xyz_5},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": transform_geometry_2},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_007", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_007(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "PB Outer", None),
            ("NodeSocketGeometry", "Base", None),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Base"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    bouding_box_up = nw.new_node(
        nodegroup_node_group_009().name,
        input_kwargs={"Geometry": group_input.outputs["PB Outer"]},
        label="Bouding Box Up",
    )

    bounding_box_2 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": bouding_box_up}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": bounding_box_2.outputs["Bounding Box"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    cube = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": (100.0000, 100.0000, 0.0100)}
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box_2.outputs["Max"]}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": separate_xyz_1.outputs["Z"]}
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Translation": combine_xyz_2},
    )

    intersect = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 2": [transform_geometry_3, reroute_1]},
        attrs={"operation": "INTERSECT"},
    )

    bounding_box_3 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": intersect.outputs["Mesh"]}
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box_3.outputs["Max"]}
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

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["X"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide})

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: reroute_5},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": subtract_1})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_3, "Translation": combine_xyz_3},
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": reroute_6, "Mesh 2": transform_geometry_4},
        attrs={"solver": "EXACT"},
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_2})

    divide_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_4, 1: (1.0000, 1.0000, 2.0000)},
        attrs={"operation": "DIVIDE"},
    )

    add = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: combine_xyz_3, 1: divide_1.outputs["Vector"]}
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add.outputs["Vector"]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Base": difference.outputs["Mesh"], "Translation Box": reroute_7},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_004", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_004(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "PB Height", 0.0000),
            ("NodeSocketFloat", "PB Width", 0.0000),
            ("NodeSocketFloat", "PB Depth", 0.0000),
            ("NodeSocketFloat", "PB Thickness", 0.0000),
        ],
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["PB Depth"],
            "Y": group_input.outputs["PB Width"],
            "Z": group_input.outputs["PB Height"],
        },
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["PB Thickness"], 1: 1.0000}
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["PB Thickness"], 1: 1.0000}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 1.0000, "Y": add, "Z": add_1}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Scale": combine_xyz_1},
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": cube.outputs["Mesh"]})

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": transform_geometry, "Mesh 2": reroute_1},
        attrs={"solver": "EXACT"},
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": difference.outputs["Mesh"]}
    )

    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": cube.outputs["Mesh"]}
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Min"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 10.0000, 1: separate_xyz.outputs["Z"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_2,
            "Translation": combine_xyz_2,
            "Scale": (1.0000, 1.0000, 10.0000),
        },
    )

    difference_1 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": reroute_3, "Mesh 2": transform_geometry_1},
        attrs={"solver": "EXACT"},
    )

    bounding_box_1 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": reroute_2}
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box_1.outputs["Max"]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Mesh": difference_1.outputs["Mesh"],
            "Top Box Z": separate_xyz_1.outputs["Z"],
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_003", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_003(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketFloat", "Cap Height", 0.0000),
            ("NodeSocketMaterial", "Cap Material", None),
        ],
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Geometry"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    position = nw.new_node(Nodes.InputPosition)

    attribute_statistic = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Attribute": position,
        },
        attrs={"data_type": "FLOAT_VECTOR", "domain": "FACE"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": attribute_statistic.outputs["Max"]}
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": position})

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": reroute_6})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={
            0: separate_xyz.outputs["Z"],
            1: separate_xyz_1.outputs["Z"],
            "Epsilon": 0.0001,
        },
        attrs={"operation": "EQUAL"},
    )

    separate_geometry = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": reroute_7, "Selection": equal},
        attrs={"domain": "FACE"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Cap Height"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": separate_geometry.outputs["Selection"],
            "Offset Scale": reroute_1,
            "Individual": False,
        },
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_geometry.outputs["Selection"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [extrude_mesh.outputs["Mesh"], reroute_9]},
    )

    convex_hull = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": join_geometry}
    )

    bounding_box = nw.new_node(Nodes.BoundingBox, input_kwargs={"Geometry": reroute_5})

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": convex_hull, "Translation": combine_xyz},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Cap Material"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry, "Material": reroute_3},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": set_material},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_006", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_006(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketMaterial", "Round Rim Material", None),
            ("NodeSocketMaterial", "Lid Material", None),
            ("NodeSocketFloat", "Base Height", 0.5000),
            ("NodeSocketFloat", "Base Width", 0.5000),
            ("NodeSocketFloat", "Opening Size", 0.0000),
        ],
    )

    bounding_box_2 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": group_input.outputs["Geometry"]}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_2.outputs["Max"],
            1: bounding_box_2.outputs["Min"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_4 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_4.outputs["X"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    uv_sphere = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Rings": 32, "Radius": divide}
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz_4.outputs["Z"]}
    )

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_8})

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere.outputs["Mesh"],
            "Translation": combine_xyz_7,
        },
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_7}
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: divide, 1: 0.9900}, attrs={"operation": "MULTIPLY"}
    )

    uv_sphere_1 = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Segments": 16, "Radius": multiply}
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_7, 1: (0.0000, 0.0000, 0.0100)},
        attrs={"operation": "SUBTRACT"},
    )

    transform_geometry_9 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere_1.outputs["Mesh"],
            "Translation": subtract_1.outputs["Vector"],
            "Scale": (0.9600, 1.0000, 1.0000),
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": reroute_13, "Mesh 2": transform_geometry_9},
    )

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Base Width"]}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Opening Size"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply_1, 1: 2.0000})

    divide_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_5, 1: add}, attrs={"operation": "DIVIDE"}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": divide_1})

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder", input_kwargs={"Radius": reroute_9, "Depth": 3.4000}
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Base Height"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_1, 1: 1.4000},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_7, 1: multiply_2})

    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add_1})

    transform_geometry_8 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_8,
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_8}
    )

    difference_1 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": difference.outputs["Mesh"], "Mesh 2": reroute_14},
        attrs={"solver": "EXACT"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Round Rim Material"]}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": difference_1.outputs["Mesh"], "Material": reroute_2},
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Geometry"]}
    )

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    bounding_box = nw.new_node(Nodes.BoundingBox, input_kwargs={"Geometry": reroute_22})

    difference_2 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 1": reroute_16,
            "Mesh 2": bounding_box.outputs["Bounding Box"],
        },
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": difference_2.outputs["Mesh"]}
    )

    difference_3 = nw.new_node(Nodes.MeshBoolean, input_kwargs={"Mesh 1": reroute_17})

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [difference_3.outputs["Mesh"], reroute_1]},
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_2})

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    intersect = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 2": [difference.outputs["Mesh"], reroute_14]},
        attrs={"operation": "INTERSECT"},
    )

    reroute_15 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": intersect.outputs["Mesh"]}
    )

    bounding_box_3 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": join_geometry_2}
    )

    subtract_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_3.outputs["Max"],
            1: bounding_box_3.outputs["Min"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_5 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_2.outputs["Vector"]}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_5.outputs["Z"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_3})

    transform_geometry_10 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_15, "Translation": combine_xyz_9},
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_12, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_10 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide_2})

    multiply_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_10, 1: (0.0000, 0.0000, -1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_11 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_10,
            "Translation": multiply_4.outputs["Vector"],
            "Scale": (0.9000, 1.0000, 1.0000),
        },
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Lid Material"]}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_11, "Material": reroute_4},
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_10})

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": reroute_19,
            "Geometry 2": set_material_1,
            "Translation": reroute_21,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input_1 = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketFloat", "Base Height", 0.0000),
        ],
    )

    base_cut = nw.new_node(
        nodegroup_node_group_001().name,
        input_kwargs={"Input": group_input_1.outputs["Geometry"]},
        label="BaseCut",
    )

    beveled_cap = nw.new_node(
        nodegroup_node_group_008().name,
        input_kwargs={"Mesh": base_cut},
        label="Beveled Cap",
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["Geometry"]}
    )

    cutting_quad_cap = nw.new_node(
        nodegroup_node_group_010().name,
        input_kwargs={"Geometry": reroute},
        label="Cutting Quad Cap",
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": beveled_cap, "Mesh 2": cutting_quad_cap},
        attrs={"solver": "EXACT"},
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["Base Height"]}
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_1})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": difference.outputs["Mesh"],
            "Translation": combine_xyz_2,
        },
    )

    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry_3})

    intersect = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 2": [cutting_quad_cap, beveled_cap]},
        attrs={"operation": "INTERSECT"},
    )

    convex_hull = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": intersect.outputs["Mesh"]}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (4.0000, 4.0000, 4.0000)})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Translation": (0.0000, -2.0000, 0.0000),
            "Scale": (1.0000, 0.9970, 1.0000),
        },
    )

    intersect_1 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 2": [convex_hull, transform_geometry]},
        attrs={"solver": "EXACT", "operation": "INTERSECT"},
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": intersect_1.outputs["Mesh"]}
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": intersect_1.outputs["Mesh"]}
    )

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": bounding_box.outputs["Min"]}
    )

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": reroute_5})

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: separate_xyz_1.outputs["Z"], 1: divide}
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: -1.0000}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_3, "Translation": combine_xyz},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Translation": (0.0000, 2.0000, 0.0000),
            "Scale": (1.0000, 0.9970, 1.0000),
        },
    )

    intersect_2 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 2": [convex_hull, transform_geometry_1]},
        attrs={"solver": "EXACT", "operation": "INTERSECT"},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": intersect_2.outputs["Mesh"]}
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_4, "Translation": combine_xyz},
    )

    group_output_1 = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Mesh": reroute_2,
            "Left Flap": transform_geometry_2,
            "Right Flap": transform_geometry_4,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_011", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_011(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Base Geometry", None),
            ("NodeSocketGeometry", "Flap Container Geometry", None),
            ("NodeSocketGeometry", "Left Flap Geometry", None),
            ("NodeSocketGeometry", "Right Flap Geometry", None),
        ],
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                group_input.outputs["Flap Container Geometry"],
                group_input.outputs["Base Geometry"],
            ]
        },
    )

    bounding_box_2 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": join_geometry_2}
    )

    separate_xyz_4 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box_2.outputs["Max"]}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_4.outputs["Z"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide})

    bounding_box_3 = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": group_input.outputs["Left Flap Geometry"]},
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_3.outputs["Max"],
            1: bounding_box_3.outputs["Min"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_5 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": separate_xyz_5.outputs["Y"]}
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_8, 1: combine_xyz_9},
        attrs={"operation": "SUBTRACT"},
    )

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": group_input.outputs["Left Flap Geometry"],
            "Translation": combine_xyz_9,
        },
    )

    bounding_box_4 = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": group_input.outputs["Right Flap Geometry"]},
    )

    subtract_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_4.outputs["Max"],
            1: bounding_box_4.outputs["Min"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_6 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_2.outputs["Vector"]}
    )

    combine_xyz_10 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": separate_xyz_6.outputs["Y"]}
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_10, 1: (0.0000, -1.0000, 0.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_8 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": group_input.outputs["Right Flap Geometry"],
            "Translation": multiply.outputs["Vector"],
        },
    )

    add = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: combine_xyz_8, 1: combine_xyz_10}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Container Geometry": join_geometry_2,
            "Offset LF": subtract_1.outputs["Vector"],
            "LF Geometry": transform_geometry_7,
            "RF Geometry": transform_geometry_8,
            "Offset RF": add.outputs["Vector"],
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_013", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_013(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Base Height", 0.5000),
            ("NodeSocketGeometry", "Base Geometry", None),
            ("NodeSocketFloat", "Height Front Flap", 0.5000),
            ("NodeSocketGeometry", "Flap Cap Geometry", None),
        ],
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Base Geometry"]}
    )

    bounding_box = nw.new_node(Nodes.BoundingBox, input_kwargs={"Geometry": reroute_1})

    separate_xyz_4 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Max"]}
    )

    bounding_box_3 = nw.new_node(
        Nodes.BoundingBox,
        input_kwargs={"Geometry": group_input.outputs["Flap Cap Geometry"]},
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_3.outputs["Max"],
            1: bounding_box_3.outputs["Min"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_5 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_4.outputs["X"], 1: separate_xyz_5.outputs["X"]},
        attrs={"operation": "SUBTRACT"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Base Height"]}
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height Front Flap"]}
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_2, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: divide, 1: divide_1})

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add})

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract_1, "Z": reroute_4}
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Flap Cap Geometry"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_1, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": separate_xyz_5.outputs["X"], "Z": multiply}
    )

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_3, "Translation": combine_xyz_8},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Vector": combine_xyz_9, "Geometry": transform_geometry_7},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_005", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_005(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Base Width", 0.5000),
            ("NodeSocketFloat", "Base Depth", 0.5000),
            ("NodeSocketFloat", "Z", 0.0000),
            ("NodeSocketFloat", "Base Outward Tilt", 0.5000),
            ("NodeSocketFloat", "Base Roundness", 0.5000),
            ("NodeSocketFloat", "Wall Thickness", 0.0000),
            ("NodeSocketFloat", "Pedal Height", 0.0000),
        ],
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Z"]}
    )

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz})

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: spline_parameter.outputs["Factor"],
            1: group_input.outputs["Base Outward Tilt"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: 1.0000})

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius, input_kwargs={"Curve": curve_line, "Radius": add}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_curve_radius})

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Width": group_input.outputs["Base Width"],
            "Height": group_input.outputs["Base Depth"],
        },
    )

    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": quadrilateral})

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Base Width"],
            1: group_input.outputs["Base Depth"],
        },
        attrs={"operation": "MINIMUM"},
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: minimum, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Base Roundness"]}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide, 1: reroute},
        attrs={"operation": "MULTIPLY"},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={"Curve": reroute_2, "Count": 12, "Radius": multiply_1},
        attrs={"mode": "POLY"},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": reroute_4,
            "Profile Curve": fillet_curve,
            "Fill Caps": True,
        },
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": curve_to_mesh})

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Z"],
            1: group_input.outputs["Pedal Height"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_2, 1: 0.1000})

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add_1})

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_2})

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Wall Thickness"]}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: reroute_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": subtract, "Z": 1.0000}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_to_mesh,
            "Translation": reroute_3,
            "Scale": combine_xyz_1,
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": reroute_6, "Mesh 2": transform_geometry},
        attrs={"solver": "EXACT"},
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": difference.outputs["Mesh"], "Mesh for Cap": reroute_7},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_012", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_012(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input_1 = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketBool", "Version 4", False),
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketFloat", "Base Height", 0.0000),
        ],
    )

    base_cut = nw.new_node(
        nodegroup_node_group_001().name,
        input_kwargs={
            "Version 4": group_input_1.outputs["Version 4"],
            "Input": group_input_1.outputs["Geometry"],
        },
        label="BaseCut",
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_1.outputs["Base Height"], 1: 0.2400},
        attrs={"operation": "MULTIPLY"},
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh, input_kwargs={"Mesh": base_cut, "Offset Scale": multiply}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": extrude_mesh.outputs["Mesh"]}
    )

    beveled_cap = nw.new_node(
        nodegroup_node_group_008().name,
        input_kwargs={"Mesh": reroute_2},
        label="Beveled Cap",
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": extrude_mesh.outputs["Mesh"]}
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": bounding_box.outputs["Bounding Box"]}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    divide = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: subtract.outputs["Vector"],
            1: (2.0000, 1000000.0000, 1000000.0000),
        },
        attrs={"operation": "DIVIDE"},
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_4,
            "Translation": divide.outputs["Vector"],
            "Scale": (1.0000, 0.7000, 0.7000),
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": beveled_cap, "Mesh 2": transform_geometry},
    )

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": difference.outputs["Mesh"]}
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["Base Height"]}
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_1})

    bounding_box_2 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": difference.outputs["Mesh"]}
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_2.outputs["Max"],
            1: bounding_box_2.outputs["Min"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_1.outputs["Vector"]}
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide_1})

    add = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: combine_xyz_2, 1: combine_xyz_3}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_6, "Translation": add.outputs["Vector"]},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_1.outputs["Geometry"]}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_geometry_2, reroute]}
    )

    intersect = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 2": [transform_geometry, beveled_cap]},
        attrs={"operation": "INTERSECT"},
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": intersect.outputs["Mesh"]}
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    bounding_box_1 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": intersect.outputs["Mesh"]}
    )

    subtract_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_1.outputs["Max"],
            1: bounding_box_1.outputs["Min"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_2.outputs["Vector"]}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": separate_xyz_1.outputs["X"]}
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": bounding_box_1.outputs["Min"]}
    )

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": reroute_9})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": separate_xyz.outputs["X"]}
    )

    add_1 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: combine_xyz_1, 1: combine_xyz}
    )

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_1.outputs["Vector"], 1: (-1.0000, -1.0000, -1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_8,
            "Translation": multiply_1.outputs["Vector"],
            "Scale": (1.0000, 0.9700, 0.9700),
        },
    )

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_1}
    )

    bounding_box_3 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": transform_geometry}
    )

    subtract_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_3.outputs["Max"],
            1: bounding_box_3.outputs["Min"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_3.outputs["Vector"]}
    )

    group_output_1 = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Mesh": join_geometry,
            "Front Flap": reroute_10,
            "Height Front Flap": separate_xyz_3.outputs["Z"],
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_add_jointed_geometry_metadata_004",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_add_jointed_geometry_metadata_004(nw: NodeWrangler):
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
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketInt", "Version", 0),
            ("NodeSocketFloat", "Base Depth", 0.0000),
            ("NodeSocketFloat", "Base Width", 0.0000),
            ("NodeSocketFloat", "Base Height", 0.0000),
            ("NodeSocketFloat", "Base Roundness", 0.0000),
            ("NodeSocketFloat", "Base Thickness", 0.0000),
            ("NodeSocketFloat", "Base Outward Tilt", 0.0000),
            ("NodeSocketFloat", "Pedal Thickness", 0.0000),
            ("NodeSocketFloat", "Pedal Width", 0.0000),
            ("NodeSocketFloat", "Pedal Height", 0.0000),
            ("NodeSocketFloat", "Pedal Roundness", 0.0000),
            ("NodeSocketFloat", "Pedal Outward Length", 0.0000),
            ("NodeSocketFloat", "Pedal Box Thickness", 0.0000),
            ("NodeSocketFloat", "Flat Cap Height", 0.0000),
            ("NodeSocketFloat", "Round Opening Size", 0.0000),
            ("NodeSocketMaterial", "Base Material", None),
            ("NodeSocketMaterial", "Lid Material", None),
            ("NodeSocketMaterial", "Round Rim Material", None),
            ("NodeSocketMaterial", "Pedal Material", None),
        ],
    )

    reroute_21 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Version"]}
    )

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    reroute_40 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    reroute_41 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_40})

    reroute_53 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_41})

    reroute_54 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_53})

    reroute_103 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_54})

    reroute_104 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_103})

    reroute_105 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_104})

    reroute_108 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_105})

    less_than = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: reroute_108, 1: 4.5000},
        attrs={"operation": "LESS_THAN"},
    )

    not_equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_22, 3: 4},
        attrs={"data_type": "INT", "operation": "NOT_EQUAL"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Base Width"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_39 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_33})

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Version"], 3: 3},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Base Depth"]}
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": reroute_4, "True": reroute_1},
        attrs={"input_type": "FLOAT"},
    )

    reroute_34 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_2})

    reroute_35 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_34})

    reroute_23 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Base Height"]}
    )

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_23})

    reroute_5 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Base Outward Tilt"]}
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": equal})

    reroute_32 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_31})

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["Version"], 3: 5},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Base Roundness"]}
    )

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Base Roundness"], 1: 0.3000},
        attrs={"operation": "MINIMUM"},
    )

    switch_8 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal_1, "False": reroute_8, "True": minimum},
        attrs={"input_type": "FLOAT"},
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 1.0000

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_32, "False": switch_8, "True": value},
        attrs={"input_type": "FLOAT"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Base Thickness"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.0200

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": reroute_3, "True": value_1},
        attrs={"input_type": "FLOAT"},
    )

    reroute_36 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch_5})

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Pedal Height"]}
    )

    base = nw.new_node(
        nodegroup_node_group_005().name,
        input_kwargs={
            "Base Width": reroute_39,
            "Base Depth": reroute_35,
            "Z": reroute_24,
            "Base Outward Tilt": reroute_6,
            "Base Roundness": switch_3,
            "Wall Thickness": reroute_36,
            "Pedal Height": reroute_7,
        },
        label="Base",
    )

    reroute_42 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_24})

    flap_cap = nw.new_node(
        nodegroup_node_group_012().name,
        input_kwargs={
            "Version 4": not_equal,
            "Geometry": base.outputs["Mesh"],
            "Base Height": reroute_42,
        },
        label="Flap Cap",
    )

    reroute_25 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Base Material"]}
    )

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_25})

    reroute_37 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_26})

    reroute_38 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_37})

    reroute_57 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_38})

    set_material_6 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": flap_cap.outputs["Mesh"], "Material": reroute_57},
    )

    add_jointed_geometry_metadata_004 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_004().name,
        input_kwargs={"Geometry": set_material_6, "Label": "trash_front_flap_base"},
    )

    reroute_79 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_004}
    )

    reroute_55 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_42})

    reroute_56 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_55})

    front_flap_positioning = nw.new_node(
        nodegroup_node_group_013().name,
        input_kwargs={
            "Base Height": reroute_56,
            "Base Geometry": flap_cap.outputs["Mesh"],
            "Height Front Flap": flap_cap.outputs["Height Front Flap"],
            "Flap Cap Geometry": flap_cap.outputs["Front Flap"],
        },
        label="Front Flap Positioning",
    )

    reroute_27 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Lid Material"]}
    )

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_27})

    reroute_45 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_28})

    reroute_46 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_45})

    reroute_61 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_46})

    set_material_7 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": front_flap_positioning.outputs["Geometry"],
            "Material": reroute_61,
        },
    )

    add_jointed_geometry_metadata_004_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_004().name,
        input_kwargs={"Geometry": set_material_7, "Label": "trash_fron_flap"},
    )

    reroute_75 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": front_flap_positioning.outputs["Vector"]}
    )

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "flapping_cap_front",
            "Parent": reroute_79,
            "Child": add_jointed_geometry_metadata_004_1,
            "Position": reroute_75,
            "Axis": (0.0000, 1.0000, 0.0000),
            "Value": -0.1000,
            "Min": -0.1000,
            "Max": 0.8000,
        },
    )

    reroute_92 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint.outputs["Geometry"]}
    )

    reroute_106 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_92})

    less_than_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: reroute_105, 1: 3.5000},
        attrs={"operation": "LESS_THAN"},
    )

    reroute_59 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": base.outputs["Mesh"]}
    )

    reroute_60 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_59})

    reroute_73 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_60})

    reroute_74 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_73})

    flap_cap_double = nw.new_node(
        nodegroup_node_group().name,
        input_kwargs={"Geometry": base.outputs["Mesh"], "Base Height": reroute_42},
        label="Flap Cap Double ",
    )

    reroute_67 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": flap_cap_double.outputs["Mesh"]}
    )

    reroute_68 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_67})

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": flap_cap_double.outputs["Left Flap"],
            "Material": reroute_46,
        },
    )

    set_material_4 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": flap_cap_double.outputs["Right Flap"],
            "Material": reroute_46,
        },
    )

    offset_calculator = nw.new_node(
        nodegroup_node_group_011().name,
        input_kwargs={
            "Base Geometry": reroute_74,
            "Flap Container Geometry": reroute_68,
            "Left Flap Geometry": set_material_2,
            "Right Flap Geometry": set_material_4,
        },
        label="Offset Calculator",
    )

    reroute_69 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_57})

    reroute_70 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_69})

    set_material_5 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": offset_calculator.outputs["Container Geometry"],
            "Material": reroute_70,
        },
    )

    add_jointed_geometry_metadata_004_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_004().name,
        input_kwargs={
            "Geometry": set_material_5,
            "Label": "trash_base_double_flapping",
        },
    )

    reroute_88 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": offset_calculator.outputs["LF Geometry"]}
    )

    add_jointed_geometry_metadata_004_3 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_004().name,
        input_kwargs={"Geometry": reroute_88, "Label": "trash_left_flap"},
    )

    reroute_86 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": offset_calculator.outputs["Offset LF"]}
    )

    reroute_87 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_86})

    hinge_joint_1 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "flapping_cap_left",
            "Parent": add_jointed_geometry_metadata_004_2,
            "Child": add_jointed_geometry_metadata_004_3,
            "Position": reroute_87,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Min": -1.5708,
        },
    )

    add_jointed_geometry_metadata_004_4 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_004().name,
        input_kwargs={
            "Geometry": hinge_joint_1.outputs["Geometry"],
            "Label": "trash_double_flaps_left_attached",
        },
    )

    reroute_83 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": offset_calculator.outputs["RF Geometry"]}
    )

    add_jointed_geometry_metadata_004_5 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_004().name,
        input_kwargs={"Geometry": reroute_83, "Label": "trash_right_flap"},
    )

    reroute_84 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": offset_calculator.outputs["Offset RF"]}
    )

    reroute_85 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_84})

    hinge_joint_2 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "flapping_cap_right",
            "Parent": add_jointed_geometry_metadata_004_4,
            "Child": add_jointed_geometry_metadata_004_5,
            "Position": reroute_85,
            "Axis": (-1.0000, 0.0000, 0.0000),
            "Min": -1.5708,
        },
    )

    reroute_96 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_2.outputs["Geometry"]}
    )

    reroute_107 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_96})

    less_than_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: reroute_104, 1: 2.5000},
        attrs={"operation": "LESS_THAN"},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": base.outputs["Mesh"], "Material": reroute_38},
    )

    reroute_19 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Round Rim Material"]}
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    reroute_43 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    reroute_44 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_43})

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_39})

    reroute_52 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_51})

    reroute_17 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Round Opening Size"]}
    )

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_17})

    round_top = nw.new_node(
        nodegroup_node_group_006().name,
        input_kwargs={
            "Geometry": set_material,
            "Round Rim Material": reroute_44,
            "Lid Material": reroute_46,
            "Base Height": reroute_56,
            "Base Width": reroute_52,
            "Opening Size": reroute_18,
        },
        label="Round Top",
    )

    add_jointed_geometry_metadata_004_6 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_004().name,
        input_kwargs={
            "Geometry": round_top.outputs["Geometry"],
            "Label": "trash_base_spherical_top",
        },
    )

    add_jointed_geometry_metadata_004_7 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_004().name,
        input_kwargs={
            "Geometry": round_top.outputs["Geometry 2"],
            "Label": "trash_spherical_top",
        },
    )

    reroute_77 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": round_top.outputs["Translation"]}
    )

    reroute_78 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_77})

    hinge_joint_3 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "rotating_sphere_joint",
            "Parent": add_jointed_geometry_metadata_004_6,
            "Child": add_jointed_geometry_metadata_004_7,
            "Position": reroute_78,
            "Axis": (-1.0000, 0.0000, 0.0000),
            "Min": -1.0000,
            "Max": 1.0000,
        },
    )

    reroute_90 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_3.outputs["Geometry"]}
    )

    reroute_91 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_90})

    less_than_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: reroute_103, 1: 1.8000},
        attrs={"operation": "LESS_THAN"},
    )

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Flat Cap Height"]}
    )

    cap = nw.new_node(
        nodegroup_node_group_003().name,
        input_kwargs={
            "Geometry": base.outputs["Mesh for Cap"],
            "Cap Height": reroute_16,
            "Cap Material": group_input.outputs["Lid Material"],
        },
        label="Cap",
    )

    reroute_65 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": cap})

    reroute_66 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_65})

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 0.9500, 1: group_input.outputs["Base Thickness"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract, "Y": subtract, "Z": 1.0000}
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": cap, "Scale": combine_xyz_7}
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": reroute_66, "Mesh 2": transform_geometry_4},
        attrs={"solver": "EXACT"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_56, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide})

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": difference.outputs["Mesh"],
            "Translation": combine_xyz_5,
        },
    )

    reroute_63 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material})

    reroute_64 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_63})

    reroute_80 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_64})

    reroute_81 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_80})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_6, reroute_81]},
    )

    add_jointed_geometry_metadata_004_8 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_004().name,
        input_kwargs={"Geometry": join_geometry, "Label": "trash_base_open"},
    )

    reroute_72 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_4}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Base Height"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Flat Cap Height"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: divide_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": subtract_1})

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_72,
            "Translation": combine_xyz_3,
            "Scale": (0.9700, 0.9700, 1.0000),
        },
    )

    add_jointed_geometry_metadata_004_9 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_004().name,
        input_kwargs={
            "Geometry": transform_geometry_5,
            "Label": "trash_flat_rotating_top",
        },
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Base Height"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide_2})

    hinge_joint_4 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "flat_rotating_lid",
            "Parent": add_jointed_geometry_metadata_004_8,
            "Child": add_jointed_geometry_metadata_004_9,
            "Position": combine_xyz_6,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Min": -1.5708,
            "Max": 1.5708,
        },
    )

    reroute_95 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint_4.outputs["Geometry"]}
    )

    less_than_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: reroute_54, 1: 0.8000},
        attrs={"operation": "LESS_THAN"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Base Height"],
            1: group_input.outputs["Pedal Height"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    reroute_30 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_1})

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Pedal Width"]}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_13, 1: reroute_33},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: switch_2, 1: 0.3000},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_14 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Pedal Box Thickness"]},
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

    pedal_box = nw.new_node(
        nodegroup_node_group_004().name,
        input_kwargs={
            "PB Height": reroute_30,
            "PB Width": multiply_2,
            "PB Depth": multiply_3,
            "PB Thickness": reroute_15,
        },
        label="Pedal Box",
    )

    reroute_47 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": pedal_box.outputs["Mesh"]}
    )

    reroute_48 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_47})

    reroute_62 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_48})

    base_without_pedal_box = nw.new_node(
        nodegroup_node_group_007().name,
        input_kwargs={"PB Outer": reroute_48, "Base": set_material},
        label="Base without Pedal Box",
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_62,
            "Translation": base_without_pedal_box.outputs["Translation Box"],
        },
    )

    intersect = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 2": [transform_geometry_1, reroute_64]},
        attrs={"solver": "EXACT", "operation": "INTERSECT"},
    )

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": intersect.outputs["Mesh"],
            "Material": group_input.outputs["Pedal Material"],
        },
    )

    reroute_71 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": base_without_pedal_box.outputs["Base"]}
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material_3, reroute_71]}
    )

    reroute_93 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": join_geometry_1})

    reroute_94 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_93})

    reroute_99 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_94})

    add_jointed_geometry_metadata_004_10 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_004().name,
        input_kwargs={"Geometry": reroute_99, "Label": "trash_without_pedal"},
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Pedal Thickness"]}
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Pedal Roundness"]}
    )

    reroute_12 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Pedal Outward Length"]},
    )

    pedal = nw.new_node(
        nodegroup_node_group_002().name,
        input_kwargs={
            "Pedal Thickness": reroute_10,
            "Pedal Width": multiply_2,
            "Pedal Roundness": reroute_11,
            "Pedal Outward Stick": reroute_12,
            "Pedal Base Length": multiply_3,
        },
        label="Pedal",
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": pedal,
            "Material": group_input.outputs["Pedal Material"],
        },
    )

    reroute_58 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_1})

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ,
        input_kwargs={"Vector": base_without_pedal_box.outputs["Translation Box"]},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": separate_xyz.outputs["X"]}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_58, "Translation": combine_xyz},
    )

    reroute_82 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_2}
    )

    reroute_97 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_82})

    reroute_98 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_97})

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": join_geometry_1}
    )

    subtract_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_2.outputs["Vector"]}
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Z"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_4})

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": combine_xyz_1}
    )

    reroute_49 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": pedal_box.outputs["Top Box Z"]}
    )

    reroute_50 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_49})

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_50, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_5, 1: group_input.outputs["Pedal Thickness"]},
        attrs={"operation": "SUBTRACT"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: separate_xyz_3.outputs["Z"], 1: subtract_3}
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_98, "Translation": combine_xyz_2},
    )

    add_jointed_geometry_metadata_004_11 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_004().name,
        input_kwargs={"Geometry": transform_geometry, "Label": "pedal"},
    )

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_5, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_29 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Pedal Thickness"]}
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_6, 1: reroute_29})

    reroute_100 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add_1})

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "trash_pedal_joint",
            "Parent": add_jointed_geometry_metadata_004_10,
            "Child": add_jointed_geometry_metadata_004_11,
            "Min": reroute_100,
        },
    )

    add_jointed_geometry_metadata_004_12 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_004().name,
        input_kwargs={
            "Geometry": sliding_joint.outputs["Geometry"],
            "Label": "pedal_mechanism",
        },
    )

    reroute_76 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_66})

    bounding_box_1 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": reroute_94}
    )

    subtract_4 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_1.outputs["Max"],
            1: bounding_box_1.outputs["Min"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_4.outputs["Vector"]}
    )

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    divide_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": divide_3, "Z": divide_4}
    )

    multiply_7 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_4, 1: (-1.0000, 1.0000, -1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_76,
            "Translation": multiply_7.outputs["Vector"],
        },
    )

    add_jointed_geometry_metadata_004_13 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata_004().name,
        input_kwargs={"Geometry": transform_geometry_3, "Label": "rotating_top_lid"},
    )

    reroute_101 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_4})

    reroute_102 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_101})

    hinge_joint_5 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "rotating_neck_joint",
            "Parent": add_jointed_geometry_metadata_004_12,
            "Child": add_jointed_geometry_metadata_004_13,
            "Position": reroute_102,
            "Axis": (0.0000, -1.0000, 0.0000),
            "Max": 1.1000,
        },
    )

    reroute_109 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_81})

    reroute_110 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_109})

    s1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": less_than_4,
            "False": hinge_joint_5.outputs["Geometry"],
            "True": reroute_110,
        },
        label="S1",
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": less_than_3, "False": reroute_95, "True": s1},
    )

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": less_than_2, "False": reroute_91, "True": switch_1},
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": less_than_1, "False": reroute_107, "True": switch_4},
    )

    switch_7 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": less_than, "False": reroute_106, "True": switch_6},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": switch_7},
        attrs={"is_active_output": True},
    )


class TrashFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    @gin.configurable(module="TrashFactory")
    def sample_joint_parameters(
        cls,
        flapping_cap_front_stiffness_min: float = 0.5,
        flapping_cap_front_stiffness_max: float = 2.0,
        flapping_cap_front_damping_min: float = 0.5,
        flapping_cap_front_damping_max: float = 2.0,
        flapping_cap_left_stiffness_min: float = 0.1,
        flapping_cap_left_stiffness_max: float = 0.1,
        flapping_cap_left_damping_min: float = 0.001,
        flapping_cap_left_damping_max: float = 0.001,
        flapping_cap_right_stiffness_min: float = 0.1,
        flapping_cap_right_stiffness_max: float = 0.1,
        flapping_cap_right_damping_min: float = 0.001,
        flapping_cap_right_damping_max: float = 0.001,
        rotating_sphere_joint_stiffness_min: float = 0.5,
        rotating_sphere_joint_stiffness_max: float = 2.0,
        rotating_sphere_joint_damping_min: float = 0.5,
        rotating_sphere_joint_damping_max: float = 2.0,
        rotating_neck_joint_stiffness_min: float = 0.5,
        rotating_neck_joint_stiffness_max: float = 2.0,
        rotating_neck_joint_damping_min: float = 0.5,
        rotating_neck_joint_damping_max: float = 2.0,
        trash_pedal_joint_stiffness_min: float = 0.5,
        trash_pedal_joint_stiffness_max: float = 2.0,
        trash_pedal_joint_damping_min: float = 0.5,
        trash_pedal_joint_damping_max: float = 2.0,
        flat_rotating_lid_stiffness_min: float = 0.5,
        flat_rotating_lid_stiffness_max: float = 2.0,
        flat_rotating_lid_damping_min: float = 0.5,
        flat_rotating_lid_damping_max: float = 2.0,
    ):
        return {
            "flapping_cap_front": {
                "stiffness": uniform(
                    flapping_cap_front_stiffness_min, flapping_cap_front_stiffness_max
                ),
                "damping": uniform(
                    flapping_cap_front_damping_min, flapping_cap_front_damping_max
                ),
            },
            "flapping_cap_left": {
                "stiffness": uniform(
                    flapping_cap_left_stiffness_min, flapping_cap_left_stiffness_max
                ),
                "damping": uniform(
                    flapping_cap_left_damping_min, flapping_cap_left_damping_max
                ),
                "friction": 10,
            },
            "flapping_cap_right": {
                "stiffness": uniform(
                    flapping_cap_right_stiffness_min, flapping_cap_right_stiffness_max
                ),
                "damping": uniform(
                    flapping_cap_right_damping_min, flapping_cap_right_damping_max
                ),
                "friction": 10,
            },
            "rotating_sphere_joint": {
                "stiffness": uniform(
                    rotating_sphere_joint_stiffness_min,
                    rotating_sphere_joint_stiffness_max,
                ),
                "damping": uniform(
                    rotating_sphere_joint_damping_min, rotating_sphere_joint_damping_max
                ),
            },
            "rotating_neck_joint": {
                "stiffness": uniform(
                    rotating_neck_joint_stiffness_min, rotating_neck_joint_stiffness_max
                ),
                "damping": uniform(
                    rotating_neck_joint_damping_min, rotating_neck_joint_damping_max
                ),
            },
            "trash_pedal_joint": {
                "stiffness": uniform(
                    trash_pedal_joint_stiffness_min, trash_pedal_joint_stiffness_max
                ),
                "damping": uniform(
                    trash_pedal_joint_damping_min, trash_pedal_joint_damping_max
                ),
            },
            "flat_rotating_lid": {
                "stiffness": uniform(
                    flat_rotating_lid_stiffness_min, flat_rotating_lid_stiffness_max
                ),
                "damping": uniform(
                    flat_rotating_lid_damping_min, flat_rotating_lid_damping_max
                ),
            },
        }

    def sample_parameters(self):
        # add code here to randomly sample from parameters
        version = randint(0, 6)

        base_depth = uniform(0.2, 0.4)
        base_width = uniform(0.2, 0.4)
        base_height = min(
            max(base_depth, base_width) * uniform(1.3, 2.5), uniform(0.4, 0.6)
        )
        base_roundness = uniform(0.1, 1)
        base_thickness = uniform(0.03, 0.1)
        base_outward_tilt = uniform(0, 0.2)
        pedal_thickness = uniform(0.002, 0.006)
        pedal_width = uniform(0.08, 0.35)
        pedal_height = uniform(0.1, 0.25)
        pedal_roundness = uniform(0.01, 1)
        pedal_outward_length = uniform(0.02, 0.04)
        pedal_box_thickness = uniform(0.08, 0.3)
        flat_cap_height = uniform(0.005, 0.02)
        round_opening_size = uniform(0, 1)

        s = uniform()
        if s < 0.5:
            base_material = weighted_sample(material_assignments.plastics)()()
            lid_material = pedal_material = weighted_sample(
                material_assignments.plastics
            )()()
            round_rim_material = weighted_sample(material_assignments.plastics)()()
        else:
            base_material = lid_material = round_rim_material = weighted_sample(
                material_assignments.metal_plastic
            )()()
            pedal_material = weighted_sample(material_assignments.metal_plastic)()()

        return {
            "Version": version,
            "Base Depth": base_depth,
            "Base Width": base_width,
            "Base Height": base_height,
            "Base Roundness": base_roundness,
            "Base Thickness": base_thickness,
            "Base Outward Tilt": base_outward_tilt,
            "Pedal Thickness": pedal_thickness,
            "Pedal Width": pedal_width,
            "Pedal Height": pedal_height,
            "Pedal Roundness": pedal_roundness,
            "Pedal Outward Length": pedal_outward_length,
            "Pedal Box Thickness": pedal_box_thickness,
            "Flat Cap Height": flat_cap_height,
            "Round Opening Size": round_opening_size,
            "Base Material": base_material,
            "Lid Material": lid_material,
            "Round Rim Material": round_rim_material,
            "Pedal Material": pedal_material,
        }

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
