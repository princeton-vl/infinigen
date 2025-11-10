import gin
from numpy.random import uniform

from infinigen.assets.utils.joints import (
    nodegroup_add_jointed_geometry_metadata,
    nodegroup_duplicate_joints_on_parent,
    nodegroup_hinge_joint,
    nodegroup_sliding_joint,
)
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.random import weighted_sample
@node_utils.to_nodegroup(
    "nodegroup_carriage_cylider", singleton=False, type="GeometryNodeTree"
)
def nodegroup_carriage_cylider(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler
    cylinder_2 = nw.new_node("GeometryNodeMeshCylinder", input_kwargs={"Vertices": 64})

    transform_geometry_17 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_2.outputs["Mesh"],
            "Translation": (0.2000, 0.0000, 0.0000),
            "Rotation": (1.5708, 0.0000, 0.0000),
            "Scale": (0.2000, 0.2000, 0.8000),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": transform_geometry_17},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_carriage_half_cylinder", singleton=False, type="GeometryNodeTree"
)
def nodegroup_carriage_half_cylinder(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    cylinder_2 = nw.new_node("GeometryNodeMeshCylinder")

    transform_geometry_17 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_2.outputs["Mesh"],
            "Translation": (0.5000, 0.0000, 0.0000),
            "Scale": (0.5000, 0.5000, 0.0800),
        },
    )

    position_3 = nw.new_node(Nodes.InputPosition)

    reroute_78 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": position_3})

    reroute_79 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_78})

    separate_xyz_5 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_3})

    map_range_7 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": separate_xyz_5.outputs["X"], 3: 1.0000, 4: 0.4000},
    )

    combine_xyz_15 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": 1.0000, "Y": 1.0000, "Z": map_range_7.outputs["Result"]},
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: reroute_79, 1: combine_xyz_15},
        attrs={"operation": "MULTIPLY"},
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": transform_geometry_17,
            "Position": multiply.outputs["Vector"],
        },
    )

    transform_geometry_18 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": set_position_1,
            "Translation": (-0.5100, 0.0000, 0.0000),
            "Scale": (1.5000, 1.5000, 2.0000),
        },
    )

    cube_3 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": (1.0000, 2.0000, 2.0000)}
    )

    transform_geometry_19 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_3.outputs["Mesh"],
            "Translation": (-0.5000, 0.0000, 0.0000),
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": transform_geometry_18, "Mesh 2": transform_geometry_19},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": difference.outputs["Mesh"]},
        attrs={"is_active_output": True},
    )

@node_utils.to_nodegroup(
    "nodegroup_carriage_eroded_sphere", singleton=False, type="GeometryNodeTree"
)
    # Code generated using version 2.7.1 of the node_transpiler
    uv_sphere = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Segments": 22, "Rings": 22}
    )

    transform_geometry_17 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere.outputs["Mesh"],
            "Translation": (0.1800, 0.0000, 0.0000),
            "Scale": (0.8000, 0.6000, 0.5000),
        },
    )

    cube_3 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": (1.0000, 2.0000, 2.0000)}
    )

    transform_geometry_18 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_3.outputs["Mesh"],
            "Translation": (-0.5000, 0.0000, 0.0000),
        },
    )

    transform_geometry_19 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere.outputs["Mesh"],
            "Translation": (0.9600, 0.0000, 1.0600),
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 1": transform_geometry_17,
            "Mesh 2": [transform_geometry_18, transform_geometry_19],
        },
    )

    transform_geometry_20 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": difference.outputs["Mesh"],
            "Scale": (1.0000, 1.2000, 0.6000),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_20},
        attrs={"is_active_output": True},
    )

@node_utils.to_nodegroup(
    "nodegroup_carriage_flat", singleton=False, type="GeometryNodeTree"
)
def nodegroup_carriage_flat(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    curve_line_2 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (0.0000, 0.0000, 0.0500),
            "End": (0.0000, 0.0000, -0.0500),
        },
    )

    quadrilateral_2 = nw.new_node("GeometryNodeCurvePrimitiveQuadrilateral")

    transform_geometry_17 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": quadrilateral_2,
            "Translation": (1.0000, 0.0000, 0.0000),
        },
    )

    resample_curve_6 = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": transform_geometry_17, "Count": 16}
    )

    fillet_curve_2 = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": resample_curve_6,
            "Count": 8,
            "Radius": 0.1000,
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    resample_curve_7 = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": fillet_curve_2, "Count": 32}
    )

    transform_geometry_18 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": resample_curve_7}
    )

    transform_geometry_19 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_18,
            "Scale": (0.7500, 0.7500, 1.0000),
        },
    )

    curve_to_mesh_2 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line_2,
            "Profile Curve": transform_geometry_19,
            "Fill Caps": True,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": curve_to_mesh_2},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_carriage_sphere", singleton=False, type="GeometryNodeTree"
)
def nodegroup_carriage_sphere(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler
    uv_sphere = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Segments": 20, "Rings": 20}
    )

    transform_geometry_17 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere.outputs["Mesh"],
            "Translation": (0.6000, 0.0000, 0.0000),
            "Scale": (0.6000, 0.6000, 0.6000),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_17},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler
    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketInt", "carriage_idx", 0)]
    )

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["carriage_idx"]},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["carriage_idx"], 3: 1},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["carriage_idx"], 3: 2},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["carriage_idx"], 3: 3},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    equal_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input.outputs["carriage_idx"], 3: 4},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "eq0": equal,
            "eq1": equal_1,
            "eq2": equal_2,
            "eq3": equal_3,
            "eq4": equal_4,
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input_7 = nw.new_node(
        Nodes.GroupInput,
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketInt", "carrage_idx", 0),
            ("NodeSocketInt", "base inclined", 0),
            ("NodeSocketFloat", "base inclination", 0.0000),
            ("NodeSocketInt", "material_type", 0),
            ("NodeSocketFloat", "material_reach", 0.0000),
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

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={3: group_input_7.outputs["material_type"]},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    cube_6 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": (4.1000, 4.8000, 4.7000)}
    )

    cube_5 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": (4.1000, 4.8000, 4.7000)}
    )

    reroute = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_7.outputs["base alternative style"]},
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_56 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_57 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_56})

    reroute_70 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_57})

    reroute_84 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_70})

    reroute_36 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_7.outputs["num_slots"]}
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_36}, attrs={"operation": "MULTIPLY"}
    )


    map_range = nw.new_node(
        Nodes.MapRange, input_kwargs={"Value": reroute_57, 3: add, 4: 1.6000}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    reroute_78 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz})

    reroute_72 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": map_range.outputs["Result"]}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": reroute_72})

    reroute_79 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_1})

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": reroute_78, "End": reroute_79}
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": curve_line, "Count": 64}
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    absolute = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"]},
        attrs={"operation": "ABSOLUTE"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: map_range.outputs["Result"]},
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

    reroute_83 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": float_curve})

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius,
        input_kwargs={"Curve": resample_curve, "Radius": reroute_83},
    )

    quadrilateral = nw.new_node("GeometryNodeCurvePrimitiveQuadrilateral")

    reroute_38 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": quadrilateral})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_38, "Translation": (1.0000, 0.0000, 0.0000)},
    )

    resample_curve_1 = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": transform_geometry, "Count": 16}
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": resample_curve_1,
            "Count": 16,
            "Radius": 0.1000,
            "Limit Radius": True,
        },
        attrs={"mode": "POLY"},
    )

    resample_curve_2 = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": fillet_curve, "Count": 32}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": resample_curve_2}
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_1})

    reroute_28 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_7.outputs["base side shape param"]},
    )

    power = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: reroute_28},
        attrs={"operation": "POWER"},
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_7.outputs["toaster length"]}
    )

    map_range_1 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": power, 3: 1.0000, 4: reroute_13},
        attrs={"clamp": False},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": 1.0000, "Y": map_range_1.outputs["Result"], "Z": 1.0000},
    )

    reroute_39 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": position_1})

    multiply_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_2, 1: reroute_39},
        attrs={"operation": "MULTIPLY"},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": transform_geometry_1,
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
        input_kwargs={0: reroute_36, 1: 0.4500},
        attrs={"operation": "MULTIPLY"},
    )

    map_range_2 = nw.new_node(
        Nodes.MapRange, input_kwargs={"Value": reroute_1, 3: 1.0000, 4: multiply_3}
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": map_range_2.outputs["Result"], "Y": 1.0000, "Z": 1.0000},
    )

    reroute_71 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_3})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_2, "Scale": reroute_71},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius,
            "Profile Curve": transform_geometry_3,
            "Fill Caps": True,
        },
    )

    reroute_90 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": curve_to_mesh})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": curve_to_mesh, "Rotation": (0.0000, 0.0000, 1.5708)},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_84,
            "False": reroute_90,
            "True": transform_geometry_4,
        },
    )

    bounding_box_2 = nw.new_node(Nodes.BoundingBox, input_kwargs={"Geometry": switch})

    reroute_92 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": bounding_box_2.outputs["Bounding Box"]}
    )

    reroute_93 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_92})

    reroute_43 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_7.outputs["base inclined"]}
    )

    equal_1 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: reroute_43},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    equal_2 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: group_input_7.outputs["base inclined"], 3: 1},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    position_3 = nw.new_node(Nodes.InputPosition)

    separate_xyz_5 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_3})

    reroute_49 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz_5.outputs["X"]}
    )

    reroute_52 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz_5.outputs["Y"]}
    )

    separate_xyz_6 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box_2.outputs["Min"]}
    )

    separate_xyz_7 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box_2.outputs["Max"]}
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: group_input_7.outputs["base inclination"]},
        attrs={"operation": "SUBTRACT"},
    )

    map_range_7 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": separate_xyz_5.outputs["Z"],
            1: separate_xyz_6.outputs["Z"],
            2: separate_xyz_7.outputs["Z"],
            3: 1.0000,
            4: subtract_1,
        },
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_52, 1: map_range_7.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_50 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz_5.outputs["Z"]}
    )

    reroute_51 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_50})

    combine_xyz_15 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": reroute_49, "Y": multiply_4, "Z": reroute_51},
    )

    position_4 = nw.new_node(Nodes.InputPosition)

    separate_xyz_8 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_4})

    reroute_53 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz_8.outputs["X"]}
    )

    separate_xyz_9 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box_2.outputs["Min"]}
    )

    separate_xyz_10 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box_2.outputs["Max"]}
    )

    map_range_8 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": separate_xyz_8.outputs["Z"],
            1: separate_xyz_9.outputs["Z"],
            2: separate_xyz_10.outputs["Z"],
            3: 1.0000,
            4: subtract_1,
        },
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_53, 1: map_range_8.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_54 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz_8.outputs["Y"]}
    )

    reroute_55 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz_8.outputs["Z"]}
    )

    combine_xyz_16 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": multiply_5, "Y": reroute_54, "Z": reroute_55},
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_2,
            "False": combine_xyz_15,
            "True": combine_xyz_16,
        },
        attrs={"input_type": "VECTOR"},
    )

    reroute_42 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": position_4})

    switch_7 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal_1, "False": switch_6, "True": reroute_42},
        attrs={"input_type": "VECTOR"},
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition, input_kwargs={"Geometry": reroute_93, "Position": switch_7}
    )

    subdivision_surface_1 = nw.new_node(
        Nodes.SubdivisionSurface,
        input_kwargs={"Mesh": set_position_1, "Level": 5, "Edge Crease": 0.5500},
        attrs={"boundary_smooth": "PRESERVE_CORNERS", "uv_smooth": "SMOOTH_ALL"},
    )

    convex_hull = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": subdivision_surface_1}
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": cube_5.outputs["Mesh"], "Mesh 2": convex_hull},
        attrs={"solver": "EXACT"},
    )

    difference_1 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 1": cube_6.outputs["Mesh"],
            "Mesh 2": difference.outputs["Mesh"],
        },
        attrs={"solver": "EXACT"},
    )

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_7.outputs["double slots"]}
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_7.outputs["num_slots"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})


    reroute_37 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_7.outputs["num_slots"]}
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_37, 2: -0.5000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: index, 1: multiply_add},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": subtract_2})

    points = nw.new_node(
        "GeometryNodePoints",
        input_kwargs={"Count": reroute_5, "Position": combine_xyz_4},
    )

    reroute_74 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": points})

    reroute_75 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_74})

    index_1 = nw.new_node(Nodes.Index)

    map_range_4 = nw.new_node(
        Nodes.MapRange, input_kwargs={"Value": index_1, 3: -0.2500, 4: 0.2500}
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": map_range_4.outputs["Result"]}
    )

    points_1 = nw.new_node(
        "GeometryNodePoints", input_kwargs={"Count": 2, "Position": combine_xyz_5}
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints, input_kwargs={"Points": points, "Instance": points_1}
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points}
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_12,
            "False": reroute_75,
            "True": realize_instances,
        },
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": switch_1, "Translation": (0.0000, 0.0000, 2.0000)},
    )

    reroute_7 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_7.outputs["slot width"]}
    )

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_7.outputs["slot length"]}
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_7.outputs["slot depth"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": reroute_8, "Y": reroute_10, "Z": multiply_6},
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_6})

    convex_hull_1 = nw.new_node(
        Nodes.ConvexHull, input_kwargs={"Geometry": cube.outputs["Mesh"]}
    )

    reroute_73 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": convex_hull_1})

    instance_on_points_1 = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={"Points": transform_geometry_5, "Instance": reroute_73},
    )

    reroute_89 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": instance_on_points_1}
    )

    difference_2 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": difference_1.outputs["Mesh"], "Mesh 2": reroute_89},
        attrs={"solver": "EXACT"},
    )

    map_range_5 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": group_input_7.outputs["base alternative style"],
            3: group_input_7.outputs["toaster length"],
            4: 1.5000,
        },
    )

    reroute_45 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": map_range_5.outputs["Result"]}
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": reroute_45, "Z": 1.0000}
    )

    reroute_67 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_7})

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": points,
            "Translation": reroute_67,
            "Scale": (1.0000, 1.0000, 0.0000),
        },
    )

    carriage_slit_width = nw.new_node(Nodes.Value, label="carriage slit width")

    carriage_slit_depth = nw.new_node(Nodes.Value, label="carriage slit depth")

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: carriage_slit_depth, 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    carriage_slit_height = nw.new_node(Nodes.Value, label="carriage slit height")

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": carriage_slit_width,
            "Y": multiply_7,
            "Z": carriage_slit_height,
        },
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_8})

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Translation": (0.0000, 0.0000, 0.2000),
        },
    )

    instance_on_points_2 = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={"Points": transform_geometry_6, "Instance": transform_geometry_7},
    )

    reroute_82 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": instance_on_points_2}
    )

    difference_3 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": difference_2.outputs["Mesh"], "Mesh 2": reroute_82},
        attrs={"solver": "EXACT"},
    )

    bounding_box_3 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": difference_3.outputs["Mesh"]}
    )

    equal_3 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: 1, 3: group_input_7.outputs["material_type"]},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 1.0000

    switch_9 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_3,
            "False": value,
            "True": group_input_7.outputs["material_reach"],
        },
        attrs={"input_type": "FLOAT"},
    )

    switch_10 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_3,
            "False": group_input_7.outputs["material_reach"],
            "True": value,
        },
        attrs={"input_type": "FLOAT"},
    )

    combine_xyz_18 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": switch_9, "Y": switch_10, "Z": 1.0010}
    )

    multiply_8 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_18, 1: (1.0010, 1.0010, 1.0010)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_17 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": bounding_box_3.outputs["Bounding Box"],
            "Translation": (0.0000, 0.0000, -0.0010),
            "Scale": multiply_8.outputs["Vector"],
        },
    )

    reroute_94 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": difference_3.outputs["Mesh"]}
    )

    reroute_95 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_94})

    intersect = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 2": [transform_geometry_17, reroute_95]},
        attrs={"operation": "INTERSECT"},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": intersect.outputs["Mesh"],
            "Material": group_input_7.outputs["body mat 2"],
        },
    )

    difference_4 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": reroute_95, "Mesh 2": transform_geometry_17},
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": difference_4.outputs["Mesh"],
            "Material": group_input_7.outputs["body mat 1"],
        },
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, set_material_1]}
    )

    reroute_96 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_95})

    set_material_5 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": reroute_96,
            "Material": group_input_7.outputs["body mat 1"],
        },
    )

    reroute_97 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_5})

    switch_11 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": equal, "False": join_geometry_2, "True": reroute_97},
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": switch_11, "Label": "toaster_body"},
    )

    reroute_100 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata}
    )

    reroute_101 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_100})

    reroute_35 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_7.outputs["carrage_idx"]}
    )

    nodegroup = nw.new_node(
        nodegroup_node_group().name, input_kwargs={"carriage_idx": reroute_35}
    )

    reroute_64 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": nodegroup.outputs["eq3"]}
    )

    reroute_65 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_64})

    reroute_63 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": nodegroup.outputs["eq2"]}
    )

    reroute_66 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": nodegroup.outputs["eq1"]}
    )

    reroute_68 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_66})

    reroute_69 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_68})

    carriage_sphere = nw.new_node(
        nodegroup_carriage_sphere().name, label="carriage_sphere"
    )

    carriage_flat = nw.new_node(nodegroup_carriage_flat().name, label="carriage_flat")

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": nodegroup.outputs["eq0"],
            "False": carriage_sphere,
            "True": carriage_flat,
        },
    )

    carriage_eroded_sphere = nw.new_node(
        nodegroup_carriage_eroded_sphere().name, label="carriage_eroded_sphere"
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_69,
            "False": switch_2,
            "True": carriage_eroded_sphere,
        },
    )

    carriage_half_cylinder = nw.new_node(
        nodegroup_carriage_half_cylinder().name, label="carriage_half_cylinder"
    )

    reroute_41 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": carriage_half_cylinder}
    )

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": reroute_63, "False": switch_3, "True": reroute_41},
    )

    carriage_cylider = nw.new_node(
        nodegroup_carriage_cylider().name, label="carriage_cylider"
    )

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": reroute_65,
            "False": switch_4,
            "True": carriage_cylider,
        },
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_7.outputs["carriage_dimensions"]},
    )

    transform_geometry_9 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": switch_5,
            "Translation": (2.2000, 0.0000, 0.0000),
            "Scale": reroute_6,
        },
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder", input_kwargs={"Vertices": 8, "Radius": 0.1000}
    )

    transform_geometry_8 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": (1.2000, 0.0000, 0.0000),
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_9, transform_geometry_8]},
    )

    reroute_44 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": map_range_5.outputs["Result"]}
    )

    subtract_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_44}, attrs={"operation": "SUBTRACT"}
    )

    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": subtract_3})

    transform_geometry_10 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry,
            "Translation": combine_xyz_9,
            "Rotation": (0.0000, 0.0000, 1.5708),
            "Scale": (0.3000, 0.3000, 0.3000),
        },
    )

    reroute_32 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_7.outputs["carriage mat"]}
    )

    reroute_33 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_32})

    reroute_80 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_33})

    reroute_81 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_80})

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_10, "Material": reroute_81},
    )

    reroute_91 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_2})

    bounding_box_1 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": add_jointed_geometry_metadata}
    )

    divide = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box_1.outputs["Max"], 1: (2.0000, 2.0000, 2.0000)},
        attrs={"operation": "DIVIDE"},
    )

    separate_xyz_4 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": divide.outputs["Vector"]}
    )

    multiply_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_4.outputs["Z"], 1: 0.6000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_14 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_9})

    transform_geometry_16 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_91, "Translation": combine_xyz_14},
    )

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_16, "Label": "slider"},
    )

    equal_4 = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: 2, 3: group_input_7.outputs["base inclined"]},
        attrs={"operation": "EQUAL", "data_type": "INT"},
    )

    vector = nw.new_node(Nodes.Vector)
    vector.vector = (0.0000, 0.0000, 1.0000)

    vector_1 = nw.new_node(Nodes.Vector)
    vector_1.vector = (0.0000, 0.0000, 1.0000)

    multiply_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_7.outputs["base inclination"], 1: -1.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_17 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_10})

    add_2 = nw.new_node(Nodes.VectorMath, input_kwargs={0: vector_1, 1: combine_xyz_17})

    switch_8 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": equal_4,
            "False": vector,
            "True": add_2.outputs["Vector"],
        },
        attrs={"input_type": "VECTOR"},
    )

    multiply_11 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: switch_8, 1: (-1.0000, -1.0000, -1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    sliding_joint = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "slider_joint",
            "Parent": reroute_101,
            "Child": add_jointed_geometry_metadata_1,
            "Axis": multiply_11.outputs["Vector"],
            "Max": 0.8000,
        },
    )

    reroute_85 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_75})

    reroute_86 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_85})

    duplicate_joints_on_parent = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Parent": sliding_joint.outputs["Parent"],
            "Child": sliding_joint.outputs["Child"],
            "Points": reroute_86,
        },
    )

    add_jointed_geometry_metadata_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": duplicate_joints_on_parent,
            "Label": "toaster_body_with_slider",
        },
    )

    cube_2 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": (0.2000, 1.0000, 0.5000)}
    )

    transform_geometry_11 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube_2.outputs["Mesh"],
            "Translation": (0.0000, 0.4800, 0.9500),
        },
    )

    cylinder_1 = nw.new_node("GeometryNodeMeshCylinder")

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_11, cylinder_1.outputs["Mesh"]]},
    )

    reroute_16 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_7.outputs["knob horizontal offset"]},
    )

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_16})

    reroute_14 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_7.outputs["knob vertical offset"]},
    )

    reroute_15 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_14})

    combine_xyz_10 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": reroute_17, "Y": reroute_45, "Z": reroute_15},
    )

    reroute_18 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_7.outputs["knob size"]}
    )

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    reroute_60 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    transform_geometry_12 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Translation": combine_xyz_10,
            "Rotation": (1.5708, 0.0000, 3.1416),
            "Scale": reroute_60,
        },
    )

    reroute_30 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_7.outputs["knob mat"]}
    )

    reroute_31 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_30})

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_12, "Material": reroute_31},
    )

    multiply_12 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_7.outputs["button horizontal offset"], 1: 10.0000},
        attrs={"operation": "MULTIPLY"},
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: multiply_12},
        attrs={"operation": "DIVIDE"},
    )

    multiply_13 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_1, 1: 0.0500},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_20 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_13})

    transform_geometry_22 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": set_material_3, "Translation": combine_xyz_20},
    )

    reroute_77 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_22}
    )

    add_jointed_geometry_metadata_3 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_77, "Label": "knob"},
    )

    reroute_98 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_3}
    )

    reroute_99 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_98})

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": add_jointed_geometry_metadata_3}
    )

    add_3 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: bounding_box.outputs["Min"]},
    )

    divide_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_3.outputs["Vector"], 1: (2.0000, 2.0000, 2.0000)},
        attrs={"operation": "DIVIDE"},
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": divide_2.outputs["Vector"]}
    )

    combine_xyz_13 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz_3.outputs["X"],
            "Z": separate_xyz_3.outputs["Z"],
        },
    )

    multiply_14 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_13, 1: (-1.0000, -1.0000, -1.0000)},
        attrs={"operation": "MULTIPLY"},
    )

    transform_geometry_15 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_99,
            "Translation": multiply_14.outputs["Vector"],
        },
    )

    transform_geometry_20 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": transform_geometry_15}
    )

    add_jointed_geometry_metadata_4 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_20, "Label": "knobs"},
    )

    reroute_102 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_13})

    reroute_103 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_102})

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "knob_joint",
            "Parent": add_jointed_geometry_metadata_2,
            "Child": add_jointed_geometry_metadata_4,
            "Position": reroute_103,
            "Axis": (0.0000, 1.0000, 0.0000),
            "Value": -1.8000,
            "Min": -2.5000,
            "Max": 2.5000,
        },
    )

    reroute_104 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_86})

    duplicate_joints_on_parent_1 = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Parent": hinge_joint.outputs["Parent"],
            "Child": hinge_joint.outputs["Child"],
            "Points": reroute_104,
        },
    )

    add_jointed_geometry_metadata_5 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": duplicate_joints_on_parent_1,
            "Label": "body_with_slider_knob",
        },
    )

    add_jointed_geometry_metadata_6 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": add_jointed_geometry_metadata_5,
            "Label": "toaster_slider_knob",
        },
    )

    cylinder_2 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 8,
            "Side Segments": 2,
            "Fill Segments": 2,
            "Depth": 4.5000,
        },
    )

    subdivision_surface = nw.new_node(
        Nodes.SubdivisionSurface, input_kwargs={"Mesh": cylinder_2.outputs["Mesh"]}
    )

    combine_xyz_11 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": reroute_45})

    reroute_20 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_7.outputs["button size"]}
    )

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_20})

    reroute_61 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    reroute_62 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_61})

    transform_geometry_14 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": subdivision_surface,
            "Translation": combine_xyz_11,
            "Rotation": (-1.5708, 0.0000, 0.0000),
            "Scale": reroute_62,
        },
    )

    reroute_29 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_7.outputs["button mat"]}
    )

    set_material_4 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": transform_geometry_14, "Material": reroute_29},
    )

    reroute_76 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_4})

    transform_geometry_21 = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": reroute_76}
    )

    add_jointed_geometry_metadata_7 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": transform_geometry_21, "Label": "button"},
    )

    reroute_87 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": add_jointed_geometry_metadata_7}
    )

    sliding_joint_1 = nw.new_node(
        nodegroup_sliding_joint().name,
        input_kwargs={
            "Joint Label": "button_joint",
            "Parent": add_jointed_geometry_metadata_6,
            "Child": reroute_87,
            "Axis": (0.0000, -1.0000, 0.0000),
            "Max": 0.1000,
        },
    )

    reroute_22 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input_7.outputs["num_buttons"]}
    )

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_22})

    reroute_25 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_7.outputs["button horizontal offset"]},
    )

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_25})

    index_2 = nw.new_node(Nodes.Index)

    reroute_40 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": index_2})

    reroute_48 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_40})

    reroute_46 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": map_range_5.outputs["Result"]}
    )

    subtract_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_46, 1: 1.0000},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_15 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input_7.outputs["base side shape param"], 1: -0.3500},
        attrs={"operation": "MULTIPLY"},
    )

    map_range_6 = nw.new_node(
        Nodes.MapRange, input_kwargs={"Value": reroute_1, 3: multiply_15, 4: 0.0000}
    )

    multiply_16 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_4, 1: map_range_6.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_17 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_48, 1: multiply_16},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_24 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_7.outputs["button vertical interval"]},
    )

    multiply_18 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_40, 1: reroute_24},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_27 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input_7.outputs["button vertical offset"]},
    )

    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_18, 1: reroute_27})

    combine_xyz_12 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_26, "Y": multiply_17, "Z": add_4}
    )

    points_2 = nw.new_node(
        "GeometryNodePoints",
        input_kwargs={"Count": reroute_23, "Position": combine_xyz_12},
    )

    instance_on_points_3 = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={"Points": points_2, "Instance": reroute_75},
    )

    realize_instances_1 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points_3}
    )

    reroute_88 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": realize_instances_1})

    duplicate_joints_on_parent_2 = nw.new_node(
        nodegroup_duplicate_joints_on_parent().name,
        input_kwargs={
            "Parent": sliding_joint_1.outputs["Parent"],
            "Child": sliding_joint_1.outputs["Child"],
            "Points": reroute_88,
        },
    )

    transform_geometry_18 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": duplicate_joints_on_parent_2,
            "Rotation": (0.0000, 0.0000, -1.5708),
        },
    )

    reroute_105 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_18}
    )

    bounding_box_4 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": transform_geometry_18}
    )

    subtract_5 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={1: bounding_box_4.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_11 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_5.outputs["Vector"]}
    )

    combine_xyz_19 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": separate_xyz_11.outputs["Z"]}
    )

    transform_geometry_19 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_105, "Translation": combine_xyz_19},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_19},
        attrs={"is_active_output": True},
    )


class ToasterFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)
    @classmethod
    @gin.configurable(module="ToasterFactory")
    def sample_joint_parameters(
        cls,
        slider_joint_stiffness_min: float = 800.0,
        slider_joint_stiffness_max: float = 1000.0,
        slider_joint_damping_min: float = 100.0,
        slider_joint_damping_max: float = 200.0,
        button_joint_stiffness_min: float = 500.0,
        button_joint_stiffness_max: float = 1000.0,
        button_joint_damping_min: float = 100.0,
        button_joint_damping_max: float = 200.0,
        knob_joint_stiffness_min: float = 0.0,
        knob_joint_stiffness_max: float = 0.0,
        knob_joint_damping_min: float = 100.0,
        knob_joint_damping_max: float = 200.0,
    ):
        return {
            "slider_joint": {
                "stiffness": uniform(
                    slider_joint_stiffness_min, slider_joint_stiffness_max
                ),
                "damping": uniform(slider_joint_damping_min, slider_joint_damping_max),
            },
            "button_joint": {
                "stiffness": uniform(
                    button_joint_stiffness_min, button_joint_stiffness_max
                ),
                "damping": uniform(button_joint_damping_min, button_joint_damping_max),
            },
            "knob_joint": {
                "stiffness": uniform(
                    knob_joint_stiffness_min, knob_joint_stiffness_max
                ),
                "damping": uniform(knob_joint_damping_min, knob_joint_damping_max),
            },
        }

        import numpy as np

        from infinigen.assets.composition.material_assignments import metal, plastic

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

        base_inclined = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])
        base_inclination = uniform(0.1, 0.2) if base_inclined else 0.0

        return {
            "base inclined": base_inclined,
            "base inclination": base_inclination,
            "material_type": np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3]),
            "material_reach": np.random.uniform(0.2, 0.8),
            "carrage_idx": np.random.choice([0, 1, 2, 3, 4]),
            "num_slots": np.random.choice([1, 2], p=[0.3, 0.7]),
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
            "knob size": uniform(0.13, 0.15),
            "button size": uniform(0.09, 0.11),
            "button width": uniform(2.0, 5.0),
            "num_buttons": np.random.choice([1, 2]),
            "button vertical interval": uniform(0.28, 0.4),
            "button horizontal offset": uniform(0.28, 0.35) * button_side,
            "button vertical offset": uniform(-0.65, -0.20),
            "base alternative style": np.random.choice([True, False]),
            "base side shape param": uniform(0.1, 0.2),
            "body mat 1": body_mat_1,
            "body mat 2": body_mat_2,
            "button mat": button_mat,
            "knob mat": knob_mat,
            "carriage mat": carriage_mat,
        }
    def create_asset(self, asset_params=None, **kwargs):
        obj = butil.spawn_vert()
            apply=False,
            node_group=geometry_nodes(),
            ng_inputs=self.sample_parameters(),
