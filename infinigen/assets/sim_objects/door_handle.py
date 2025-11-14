# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Anna Calveri: Primary author
# - Max Gonzalez Saez-Diez: Updates for sim
# - Abhishek Joshi: Updates for sim

import gin
from numpy.random import uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.joints import (
    nodegroup_add_jointed_geometry_metadata,
    nodegroup_distance_from_center,
    nodegroup_hinge_joint,
    nodegroup_sliding_joint,
)
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import weighted_sample


@node_utils.to_nodegroup(
    "nodegroup_handle_lock_001", singleton=False, type="GeometryNodeTree"
)
def nodegroup_handle_lock_001(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Value", 0.5000),
            ("NodeSocketBool", "TurnLock", False),
            ("NodeSocketFloat", "ButtonDepth", 0.0070),
            ("NodeSocketFloat", "MiniLockDepth", 0.0020),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["TurnLock"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    stub_rad = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"]},
        label="StubRad",
        attrs={"operation": "MULTIPLY"},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["ButtonDepth"]}
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 64, "Radius": stub_rad, "Depth": reroute_2},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cylinder.outputs["Mesh"]}
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_4, "Rotation": (0.0000, 1.5708, 0.0000)},
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": transform_geometry_4})

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_7, "Label": "lock_base"},
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata}
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["MiniLockDepth"]}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_3})

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"], 1: 0.8000},
        attrs={"operation": "MULTIPLY"},
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": 0.0020, "Height": multiply},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={"Curve": quadrilateral, "Count": 32, "Limit Radius": True},
        attrs={"mode": "POLY"},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": fillet_curve,
            "Fill Caps": True,
        },
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_to_mesh,
            "Translation": (0.0020, 0.0000, 0.0000),
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    convex_hull = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": transform_geometry_3}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": convex_hull})

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_5})

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": transform_geometry_4}
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Max"]}
    )

    bounding_box_1 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": convex_hull}
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box_1.outputs["Min"]}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: separate_xyz_1.outputs["X"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": subtract})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_6, "Translation": combine_xyz_1},
    )

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry, "Label": "mini_turn_lock"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [add_jointed_geometry_metadata_1, add_jointed_geometry_metadata]
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_1, "False": reroute_8, "True": join_geometry},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Output": switch},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_handle_lever_001", singleton=False, type="GeometryNodeTree"
)
def nodegroup_handle_lever_001(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "StubHeight", 0.5000),
            ("NodeSocketFloat", "StubRadius", 0.5000),
            ("NodeSocketFloat", "StubDepth", 0.5000),
            ("NodeSocketFloat", "LeverLength", 2.0000),
            ("NodeSocketFloat", "LeverWidth", 0.0000),
        ],
    )

    outer_sleeve_height = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["StubHeight"], 1: 1.5000},
        label="OuterSleeveHeight",
        attrs={"operation": "MULTIPLY"},
    )

    quadrilateral_3 = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": outer_sleeve_height, "Height": outer_sleeve_height},
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": quadrilateral_3})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["StubDepth"]}
    )

    outer_sleeve_radius = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute, 1: 1.0000},
        label="OuterSleeveRadius",
        attrs={"operation": "MULTIPLY"},
    )

    fillet_curve_1 = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": quadrilateral_3,
            "Count": 32,
            "Radius": outer_sleeve_radius,
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    fillet_sleeve = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": True, "False": reroute_5, "True": fillet_curve_1},
        label="FilletSleeve",
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["StubRadius"]}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": reroute_4})

    curve_line_4 = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz})

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": fillet_sleeve,
            "Profile Curve": curve_line_4,
            "Fill Caps": True,
        },
    )

    convex_hull_2 = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": curve_to_mesh}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": convex_hull_2, "Rotation": (0.0000, -1.5708, 0.0000)},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["LeverWidth"]}
    )

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_2})

    curve_line_3 = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz_7})

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["LeverLength"]}
    )

    quadrilateral_2 = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": outer_sleeve_height, "Height": reroute_1},
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": quadrilateral_2})

    fillet_curve_3 = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": quadrilateral_2,
            "Count": 32,
            "Radius": outer_sleeve_radius,
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    fillet_lever = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": True, "False": reroute_6, "True": fillet_curve_3},
        label="FilletLever",
    )

    curve_to_mesh_3 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line_3,
            "Profile Curve": fillet_lever,
            "Fill Caps": True,
        },
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    lever_length_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["LeverLength"]},
        label="LeverLength/2",
        attrs={"operation": "MULTIPLY"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: lever_length_2, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    outer_sleeve_height_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: outer_sleeve_height},
        label="OuterSleeveHeight/2",
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: outer_sleeve_height_2})

    lever_translation = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": reroute_8, "Y": add},
        label="LeverTranslation",
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": lever_translation})

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    transform_geometry_8 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_to_mesh_3,
            "Translation": reroute_10,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    convex_hull = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": transform_geometry_8}
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_2, convex_hull]},
    )

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_2}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"FullHandle": join_geometry_3, "Output": reroute_11},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_handle_rose_001", singleton=False, type="GeometryNodeTree"
)
def nodegroup_handle_rose_001(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "RoseHeight", 0.0600),
            ("NodeSocketFloat", "RoseRadius", 10.8500),
            ("NodeSocketFloat", "RoseDepth", 0.2000),
        ],
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["RoseDepth"]}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_1})

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz})

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Width": group_input.outputs["RoseHeight"],
            "Height": group_input.outputs["RoseHeight"],
        },
    )

    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": quadrilateral})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["RoseRadius"]}
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": quadrilateral,
            "Count": 32,
            "Radius": reroute,
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    fillet_rose = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": True, "False": reroute_2, "True": fillet_curve},
        label="FilletRose",
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": fillet_rose,
            "Fill Caps": True,
        },
    )

    convex_hull = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": curve_to_mesh}
    )

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": convex_hull, "Rotation": (0.0000, 1.5708, 0.0000)},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": transform_geometry_7},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input_4 = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "RoseHeight", 0.0600),
            ("NodeSocketFloat", "RoseRadius", 1.0000),
            ("NodeSocketFloat", "RoseDepth", 0.0100),
            ("NodeSocketFloat", "StubHeight", 0.0120),
            ("NodeSocketFloat", "StubRadius", 0.0060),
            ("NodeSocketFloat", "StubDepth", 0.0300),
            ("NodeSocketFloat", "LeverLength", 0.1200),
            ("NodeSocketFloat", "LeverWidth", 0.0060),
            ("NodeSocketFloat", "ButtonDepth", 0.0070),
            ("NodeSocketFloat", "MiniLockDepth", 0.0020),
            ("NodeSocketBool", "TurnLock", True),
            ("NodeSocketBool", "HasLock", True),
            ("NodeSocketMaterial", "HandleMaterial", None),
            ("NodeSocketMaterial", "LockMaterial", None),
            ("NodeSocketMaterial", "RoseMaterial", None),
        ],
    )

    handlerose_001 = nw.new_node(
        nodegroup_handle_rose_001().name,
        input_kwargs={
            "RoseHeight": group_input_4.outputs["RoseHeight"],
            "RoseRadius": group_input_4.outputs["RoseRadius"],
            "RoseDepth": group_input_4.outputs["RoseDepth"],
        },
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": handlerose_001})

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": reroute_6,
            "Material": group_input_4.outputs["RoseMaterial"],
        },
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_2, "Label": "rose"},
    )

    handlelever_001 = nw.new_node(
        nodegroup_handle_lever_001().name,
        input_kwargs={
            "StubHeight": group_input_4.outputs["StubHeight"],
            "StubRadius": group_input_4.outputs["StubDepth"],
            "StubDepth": group_input_4.outputs["StubRadius"],
            "LeverLength": group_input_4.outputs["LeverLength"],
            "LeverWidth": group_input_4.outputs["LeverWidth"],
        },
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": handlelever_001.outputs["FullHandle"],
            "Material": group_input_4.outputs["HandleMaterial"],
        },
    )

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material_1, "Label": "lever"},
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_1}
    )

    distance_from_center = nw.new_node(
        nodegroup_distance_from_center().name, input_kwargs={"Geometry": handlerose_001}
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": distance_from_center}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": separate_xyz_1.outputs["X"]}
    )

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "lever_joint",
            "Parent": add_jointed_geometry_metadata,
            "Child": reroute_9,
            "Position": combine_xyz_2,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Min": -0.6000,
            "Max": 0.6000,
        },
    )

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": hinge_joint.outputs["Geometry"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_4.outputs["TurnLock"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    add_jointed_geometry_metadata_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_11, "Label": "lever_and_rose"},
    )

    handlelock_001 = nw.new_node(
        nodegroup_handle_lock_001().name,
        input_kwargs={
            "Value": group_input_4.outputs["StubHeight"],
            "TurnLock": group_input_4.outputs["TurnLock"],
            "ButtonDepth": group_input_4.outputs["ButtonDepth"],
            "MiniLockDepth": group_input_4.outputs["MiniLockDepth"],
        },
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": handlelock_001})

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_4.outputs["RoseDepth"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_4.outputs["ButtonDepth"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: divide, 1: divide_1})

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_4.outputs["LeverWidth"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: reroute_3})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_4.outputs["StubDepth"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: add_1, 1: reroute_1})

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": add_2})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_8, "Translation": combine_xyz},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry,
            "Material": group_input_4.outputs["LockMaterial"],
        },
    )

    add_jointed_geometry_metadata_3 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": set_material, "Label": "lock"},
    )

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "pushlock",
            "Parent": add_jointed_geometry_metadata_2,
            "Child": add_jointed_geometry_metadata_3,
            "Axis": (-1.0000, 0.0000, 0.0000),
            "Max": 0.0040,
        },
    )

    hinge_joint_1 = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "turnlock",
            "Parent": add_jointed_geometry_metadata_2,
            "Child": add_jointed_geometry_metadata_3,
            "Axis": (1.0000, 0.0000, 0.0000),
            "Value": 0.3000,
            "Min": -0.6000,
            "Max": 0.6000,
        },
    )

    has_turn_lock_joint = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_5,
            "False": sliding_joint.outputs["Geometry"],
            "True": hinge_joint_1.outputs["Geometry"],
        },
        label="hasTurnLockJoint",
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input_4.outputs["HasLock"],
            "False": reroute_12,
            "True": has_turn_lock_joint,
        },
    )

    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": switch})

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    bounding_box_1 = nw.new_node(Nodes.BoundingBox, input_kwargs={"Geometry": switch})

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box_1.outputs["Min"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_14, "Translation": combine_xyz_1},
    )

    group_output_1 = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_1},
        attrs={"is_active_output": True},
    )


class DoorHandleFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    @gin.configurable(module="DoorHandleFactory")
    def sample_joint_parameters(
        cls,
        lever_joint_stiffness_min: float = 0.6,
        lever_joint_stiffness_max: float = 1.3,
        lever_joint_damping_min: float = 0.2,
        lever_joint_damping_max: float = 0.6,
        pushlock_stiffness_min: float = 0.005,
        pushlock_stiffness_max: float = 0.01,
        pushlock_damping_min: float = 0.0001,
        pushlock_damping_max: float = 0.0001,
        turnlock_stiffness_min: float = 0.0,
        turnlock_stiffness_max: float = 0.0,
        turnlock_damping_min: float = 0.0,
        turnlock_damping_max: float = 0.0,
    ):
        return {
            "lever_joint": {
                "stiffness": uniform(
                    lever_joint_stiffness_min, lever_joint_stiffness_max
                ),
                "damping": uniform(lever_joint_damping_min, lever_joint_damping_max),
                "friction": 0.1,
            },
            "pushlock": {
                "stiffness": uniform(pushlock_stiffness_min, pushlock_stiffness_max),
                "damping": uniform(pushlock_damping_min, pushlock_damping_max),
                "friction": 0.1,
            },
            "turnlock": {
                "stiffness": uniform(turnlock_stiffness_min, turnlock_stiffness_max),
                "damping": uniform(turnlock_damping_min, turnlock_damping_max),
                "friction": 0.01,
            },
        }

    def sample_parameters(self):
        # add code here to randomly sample from parameters
        from numpy.random import choice

        rose_height = uniform(0.0500, 0.1000)
        rose_radius = uniform(0.0, rose_height / 2)
        rose_depth = uniform(0.0080, 0.0200)
        stub_height = uniform(0.008, 0.0160)
        stub_radius = uniform(0.000, 0.01)
        stub_depth = uniform(0.02, 0.05)
        lever_length = uniform(0.08, 0.1600)
        lever_width = uniform(0.0100, 0.04)

        button_depth = uniform(0.005, 0.01)
        minilock_depth = uniform(0.005, 0.01)

        turn_lock = choice([True, False], p=[0.5, 0.5])
        has_lock = choice([True, False], p=[0.5, 0.5])

        def pick_material():
            r = uniform(0, 1)
            if r < 0.5:
                return weighted_sample(material_assignments.metals)()()
            else:
                return weighted_sample(material_assignments.plastics)()()

        material_version = uniform(0, 1)
        if material_version < 0.5:
            material_handle = material_rose = material_lock = pick_material()
        elif material_version < 0.75:
            material_handle = material_lock = pick_material()
            material_rose = pick_material()
        else:
            material_rose = pick_material()
            material_handle = pick_material()
            material_lock = pick_material()

        return {
            "RoseHeight": rose_height,
            "RoseRadius": rose_radius,
            "RoseDepth": rose_depth,
            "StubHeight": stub_height,
            "StubRadius": stub_radius,
            "StubDepth": stub_depth,
            "LeverLength": lever_length,
            "LeverWidth": lever_width,
            "TurnLock": turn_lock,
            "HasLock": has_lock,
            "ButtonDepth": button_depth,
            "MiniLockDepth": minilock_depth,
            "HandleMaterial": material_handle,
            "LockMaterial": material_lock,
            "RoseMaterial": material_rose,
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
