import gin
from numpy.random import uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.joints import (
    nodegroup_add_jointed_geometry_metadata,
    nodegroup_hinge_joint,
)
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
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Body", None),
            ("NodeSocketFloat", "Radius", 1.0000),
            ("NodeSocketFloat", "Round Lid Factor", 0.0000),
            ("NodeSocketFloat", "Round Lid Compression", 0.0000),
            ("NodeSocketMaterial", "Material Mini Lid", None),
            ("NodeSocketMaterial", "Material Lid", None),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Round Lid Factor"],
            1: group_input.outputs["Radius"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: 0.1000},
        attrs={"operation": "MULTIPLY"},
    )

    uv_sphere_1 = nw.new_node(
        Nodes.MeshUVSphere,
        input_kwargs={"Segments": 6, "Rings": 6, "Radius": multiply_1},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Material Mini Lid"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": uv_sphere_1.outputs["Mesh"], "Material": reroute_5},
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material})

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply})

    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    uv_sphere = nw.new_node(
        Nodes.MeshUVSphere, input_kwargs={"Rings": 20, "Radius": reroute_8}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: group_input.outputs["Round Lid Compression"]},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: subtract},
        attrs={"operation": "MULTIPLY"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_2, 1: -1.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": uv_sphere.outputs["Mesh"],
            "Translation": combine_xyz_5,
        },
    )

    reroute_15 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_3}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Radius"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_1, 1: multiply},
        attrs={"operation": "DIVIDE"},
    )

    arccosine = nw.new_node(
        Nodes.Math, input_kwargs={0: divide_1}, attrs={"operation": "ARCCOSINE"}
    )

    sine = nw.new_node(
        Nodes.Math, input_kwargs={0: arccosine}, attrs={"operation": "SINE"}
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: sine, 1: reroute_10},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_3})

    reroute_2 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Round Lid Compression"]},
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 1.0000, "Y": 1.0000, "Z": reroute_3}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": reroute_15,
            "Translation": combine_xyz,
            "Scale": combine_xyz_4,
        },
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_9, 1: 4.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": multiply_4, "Y": multiply_4, "Z": multiply_4},
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_1})

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_4, 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide_2})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Translation": combine_xyz_2},
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": transform_geometry, "Mesh 2": transform_geometry_1},
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": difference.outputs["Mesh"]}
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Max"]}
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": separate_xyz.outputs["Z"]}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_14, "Translation": combine_xyz_3},
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Material Lid"]}
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": difference.outputs["Mesh"], "Material": reroute_6},
    )

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material_1})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_2, reroute_16]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Lid Height", 1.0000),
            ("NodeSocketFloat", "Radius", 1.0000),
            ("NodeSocketFloat", "Fraction Height Central Bar", 0.0000),
            ("NodeSocketMaterial", "Material Lid", None),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Radius"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Lid Height"],
            1: group_input.outputs["Fraction Height Central Bar"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_1, "Y": reroute_1, "Z": multiply}
    )

    divide = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz_4, 1: (6.0000, 6.0000, 1.0000)},
        attrs={"operation": "DIVIDE"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": divide.outputs["Vector"]}
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Lid Height"]}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: group_input.outputs["Fraction Height Central Bar"]},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_3, 1: subtract},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply_1})

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz_1.outputs["X"],
            "Y": separate_xyz_1.outputs["Y"],
            "Z": reroute_7,
        },
    )

    cube_2 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_8})

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cube_2.outputs["Mesh"]}
    )

    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_11})

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_6, 1: -2.0000},
        attrs={"operation": "DIVIDE"},
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Y"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: divide_1, 1: divide_2})

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 2.0000}, attrs={"operation": "MULTIPLY"}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": multiply})

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_5, 1: divide_3})

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add_1})

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply_2, "Z": reroute_10}
    )

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_12, "Translation": combine_xyz_9},
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": divide.outputs["Vector"]}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": reroute_8})

    divide_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Z"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide_4})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Translation": combine_xyz_5},
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_2}
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz_1.outputs["X"],
            "Y": reroute_6,
            "Z": separate_xyz_1.outputs["Y"],
        },
    )

    cube_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_6})

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": cube_1.outputs["Mesh"]}
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz_1.outputs["Z"]}
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": add, "Z": reroute_9}
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_14, "Translation": combine_xyz_7},
    )

    reroute_15 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_3}
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_4, reroute_13, reroute_15]},
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Material Lid"]}
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": join_geometry_2, "Material": reroute_2},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_material_2},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_010", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_010(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Depth", 2.0000),
            ("NodeSocketFloat", "Radius Bottom", 1.0000),
            ("NodeSocketMaterial", "Material Lid", None),
            ("NodeSocketFloat", "Lid Tilt", 0.0000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: 1.0000, 1: group_input.outputs["Lid Tilt"]}
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Radius Bottom"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: reroute_1}, attrs={"operation": "MULTIPLY"}
    )

    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Depth"]}
    )

    cone = nw.new_node(
        "GeometryNodeMeshCone",
        input_kwargs={
            "Radius Top": multiply,
            "Radius Bottom": reroute_4,
            "Depth": reroute_2,
        },
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Material Lid"]}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": cone.outputs["Mesh"], "Material": reroute_3},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_material},
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
            ("NodeSocketFloat", "Radius", 1.0000),
            ("NodeSocketFloat", "Foot Radius", 0.0000),
            ("NodeSocketInt", "Lid Version", 0),
            ("NodeSocketFloat", "Lid 0 Tilt", 0.0000),
            ("NodeSocketFloat", "Lid 1 Fraction Height (Handle)", 0.0000),
            ("NodeSocketFloat", "Lid  2 Factor (Sphere)", 0.0000),
            ("NodeSocketFloat", "Lid 2 Compression (Sphere)", 0.0000),
            ("NodeSocketMaterial", "Material", None),
            ("NodeSocketMaterial", "Material Mini Lid", None),
            ("NodeSocketFloat", "Depth", 2.0000),
            ("NodeSocketGeometry", "Body", None),
        ],
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Lid Version"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    greater_than = nw.new_node(Nodes.Compare, input_kwargs={0: reroute_1, 1: 1.5000})

    greater_than_1 = nw.new_node(
        Nodes.Compare, input_kwargs={0: group_input.outputs["Lid Version"], 1: 0.8000}
    )

    cone = nw.new_node(
        nodegroup_node_group_010().name,
        input_kwargs={
            "Depth": group_input.outputs["Depth"],
            "Radius Bottom": group_input.outputs["Radius"],
            "Material Lid": group_input.outputs["Material"],
            "Lid Tilt": group_input.outputs["Lid 0 Tilt"],
        },
        label="Cone",
    )

    handle = nw.new_node(
        nodegroup_node_group().name,
        input_kwargs={
            "Lid Height": group_input.outputs["Depth"],
            "Radius": group_input.outputs["Foot Radius"],
            "Fraction Height Central Bar": group_input.outputs[
                "Lid 1 Fraction Height (Handle)"
            ],
            "Material Lid": group_input.outputs["Material"],
        },
        label="Handle",
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": greater_than_1, "False": cone, "True": handle},
    )

    sphere = nw.new_node(
        nodegroup_node_group_009().name,
        input_kwargs={
            "Body": group_input.outputs["Body"],
            "Radius": group_input.outputs["Radius"],
            "Round Lid Factor": group_input.outputs["Lid  2 Factor (Sphere)"],
            "Round Lid Compression": group_input.outputs["Lid 2 Compression (Sphere)"],
            "Material Mini Lid": group_input.outputs["Material Mini Lid"],
            "Material Lid": group_input.outputs["Material"],
        },
        label="Sphere",
    )

    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": sphere})

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={"Switch": greater_than, "False": switch, "True": reroute_2},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": switch_1},
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
            ("NodeSocketFloat", "Radius", 1.0000),
            ("NodeSocketFloat", "Depth", 1.0000),
            ("NodeSocketFloat", "Base Depth", 0.5000),
            ("NodeSocketMaterial", "Material", None),
        ],
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Base Depth"]}
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_2})

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Base Depth"],
            1: group_input.outputs["Depth"],
        },
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add})

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_3, "End": combine_xyz_2}
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": curve_line, "Count": 32}
    )

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    float_curve = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": spline_parameter.outputs["Factor"]}
    )
    node_utils.assign_curve(
        float_curve.mapping.curves[0], [(0.0000, 1.0000), (1.0000, 1.0000)]
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Radius"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: float_curve, 1: reroute_1},
        attrs={"operation": "MULTIPLY"},
    )

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius, input_kwargs={"Curve": resample_curve, "Radius": multiply}
    )

    set_position = nw.new_node(
        Nodes.SetPosition, input_kwargs={"Geometry": set_curve_radius}
    )

    curve_circle = nw.new_node(Nodes.CurveCircle)

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_position,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Material"]}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": curve_to_mesh, "Material": reroute_3},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_material},
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
            ("NodeSocketMaterial", "Material", None),
            ("NodeSocketInt", "Foot Version", 0),
            ("NodeSocketFloat", "Foot Tilt", 0.0000),
            ("NodeSocketFloat", "Radius", 1.0000),
            ("NodeSocketFloat", "Depth", 2.0000),
        ],
    )

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Foot Version"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_7})

    less_than = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: reroute_10, 1: 0.5000},
        attrs={"operation": "LESS_THAN"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: group_input.outputs["Foot Tilt"]},
        attrs={"operation": "SUBTRACT"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Radius"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: reroute_1},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Depth"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    cone = nw.new_node(
        "GeometryNodeMeshCone",
        input_kwargs={
            "Radius Top": multiply,
            "Radius Bottom": reroute_9,
            "Depth": reroute_5,
        },
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

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 1: 2.0000},
        attrs={"operation": "DIVIDE"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": reroute_8, "Translation": combine_xyz},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": less_than,
            "False": cone.outputs["Mesh"],
            "True": transform_geometry,
        },
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Material"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    set_material = nw.new_node(
        Nodes.SetMaterial, input_kwargs={"Geometry": switch, "Material": reroute_3}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": set_material, "Scale": (0.7500, 0.7500, 1.0000)},
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": set_material, "Mesh 2": transform_geometry_1},
        attrs={"solver": "EXACT"},
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: reroute_7, 2: multiply, 3: reroute_9},
        attrs={"clamp_factor": False},
    )

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": mix.outputs["Result"]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": difference.outputs["Mesh"], "Radius": reroute_11},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_node_group_008", singleton=False, type="GeometryNodeTree"
)
def nodegroup_node_group_008(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketMaterial", "Material", None),
            ("NodeSocketFloat", "Body Wave Offset", 0.0000),
            ("NodeSocketFloat", "Radius", 1.0000),
            ("NodeSocketFloat", "Depth", 2.0000),
            ("NodeSocketFloat", "Base Height", 0.5000),
            ("NodeSocketFloat", "Knee Height", 0.5000),
            ("NodeSocketFloat", "Body Height", 0.5000),
            ("NodeSocketFloat", "Body Nr. Waves", 0.0000),
            ("NodeSocketFloat", "Body Wave Strength", 0.6000),
        ],
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Radius"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_17 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_3})

    reroute_18 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_17})

    reroute_28 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_18})

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Body Wave Offset"]}
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_9})

    cosine = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_10}, attrs={"operation": "COSINE"}
    )

    max_radius = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_3, 1: 0.0000}, label="MaxRadius"
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: group_input.outputs["Body Wave Strength"]},
        attrs={"operation": "SUBTRACT"},
    )

    min_radius = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_3, 1: subtract},
        label="MinRadius",
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: max_radius, 1: min_radius},
        attrs={"operation": "SUBTRACT"},
    )

    a = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: 2.0000},
        label="A",
        attrs={"operation": "DIVIDE"},
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: cosine, 1: a}, attrs={"operation": "MULTIPLY"}
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: max_radius, 1: min_radius})

    reroute_23 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add})

    c = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_23, 1: 2.0000},
        label="C",
        attrs={"operation": "DIVIDE"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: c})

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_18, 1: add_1},
        attrs={"operation": "SUBTRACT"},
    )

    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_28, 1: subtract_2})

    reroute_6 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Depth"]}
    )

    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_6})

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 35,
            "Side Segments": 16,
            "Radius": add_2,
            "Depth": reroute_7,
        },
    )

    add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Base Height"],
            1: group_input.outputs["Knee Height"],
        },
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Body Height"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: add_3, 1: multiply_1})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add_4})

    reroute_19 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": combine_xyz_1})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cylinder.outputs["Mesh"], "Translation": reroute_19},
    )

    reroute_29 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": transform_geometry_1}
    )

    reroute_26 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": a})

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: add_3},
        attrs={"operation": "SUBTRACT"},
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Body Nr. Waves"]}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_8, 1: 6.2832},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_3, 1: multiply_2},
        attrs={"operation": "MULTIPLY"},
    )

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Body Height"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_16 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_1})

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_3, 1: reroute_16},
        attrs={"operation": "DIVIDE"},
    )

    reroute_24 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_24})

    subtract_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide, 1: reroute_25},
        attrs={"operation": "SUBTRACT"},
    )

    cosine_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: subtract_4}, attrs={"operation": "COSINE"}
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_26, 1: cosine_1},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_27 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": c})

    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_4, 1: reroute_27})

    reroute_21 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": max_radius})

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_5, 1: reroute_22},
        attrs={"operation": "DIVIDE"},
    )

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["X"]}
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_1, 1: reroute_11},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["Y"]}
    )

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_1, 1: reroute_12},
        attrs={"operation": "MULTIPLY"},
    )

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["Z"]}
    )

    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_13})

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": multiply_5, "Y": multiply_6, "Z": reroute_14},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": reroute_29, "Position": combine_xyz},
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Material"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": set_position, "Material": reroute_5},
    )

    reroute_30 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": set_material})

    add_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Base Height"],
            1: group_input.outputs["Knee Height"],
        },
    )

    add_7 = nw.new_node(Nodes.Math, input_kwargs={0: add_6, 1: reroute_1})

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add_7})

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_1})

    not_equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={0: reroute_20, 1: separate_xyz_1.outputs["Z"]},
        attrs={"operation": "NOT_EQUAL"},
    )

    delete_geometry = nw.new_node(
        Nodes.DeleteGeometry,
        input_kwargs={"Geometry": set_position, "Selection": not_equal},
    )

    reroute_15 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz_1.outputs["X"]}
    )

    attribute_statistic = nw.new_node(
        Nodes.AttributeStatistic,
        input_kwargs={"Geometry": delete_geometry, "Attribute": reroute_15},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": reroute_30,
            "Radius": attribute_statistic.outputs["Max"],
        },
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("geometry_nodes", singleton=False, type="GeometryNodeTree")
def geometry_nodes(nw: NodeWrangler):
    # Code generated using version 2.7.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketInt", "Foot Version", 0),
            ("NodeSocketFloat", "Foot Height", 0.0000),
            ("NodeSocketFloat", "Foot Radius", 0.0000),
            ("NodeSocketFloat", "Foot Tilt", 0.0000),
            ("NodeSocketFloat", "Knee Height", 0.0000),
            ("NodeSocketFloat", "Body Height", 0.0000),
            ("NodeSocketFloat", "Body Nr. Waves", 0.0000),
            ("NodeSocketFloat", "Body Wave Strength", 0.0000),
            ("NodeSocketFloat", "Body Wave Offset", 0.0000),
            ("NodeSocketInt", "Lid Version", 0),
            ("NodeSocketFloat", "Lid Height", 0.0000),
            ("NodeSocketFloat", "Lid Version 0 Tilt", 0.0000),
            ("NodeSocketFloat", "Lid Version 1 Fraction (Handle)", 0.0000),
            ("NodeSocketFloat", "Lid Version 2 Factor (Sphere)", 0.0000),
            ("NodeSocketFloat", "Lid Version 2 Compression (Sphere)", 0.0000),
            ("NodeSocketMaterial", "Base Material", None),
            ("NodeSocketMaterial", "Knee Material", None),
            ("NodeSocketMaterial", "Chamber Material", None),
            ("NodeSocketMaterial", "Cap Material", None),
            ("NodeSocketMaterial", "Dot Material", None),
        ],
    )

    reroute_23 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Chamber Material"]}
    )

    reroute_21 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Body Wave Offset"]}
    )

    reroute_22 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_21})

    foot = nw.new_node(
        nodegroup_node_group_005().name,
        input_kwargs={
            "Material": group_input.outputs["Base Material"],
            "Foot Version": group_input.outputs["Foot Version"],
            "Foot Tilt": group_input.outputs["Foot Tilt"],
            "Radius": group_input.outputs["Foot Radius"],
            "Depth": group_input.outputs["Foot Height"],
        },
        label="Foot",
    )

    reroute_4 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Body Height"]}
    )

    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_4})

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Foot Height"]}
    )

    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute})

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Knee Height"]}
    )

    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_2})

    reroute_18 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Body Nr. Waves"]}
    )

    reroute_19 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Body Wave Strength"]}
    )

    reroute_20 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_19})

    chamber = nw.new_node(
        nodegroup_node_group_008().name,
        input_kwargs={
            "Material": reroute_23,
            "Body Wave Offset": reroute_22,
            "Radius": foot.outputs["Radius"],
            "Depth": reroute_5,
            "Base Height": reroute_1,
            "Knee Height": reroute_3,
            "Body Height": reroute_5,
            "Body Nr. Waves": reroute_18,
            "Body Wave Strength": reroute_20,
        },
        label="Chamber",
    )

    add_jointed_geometry_metadata = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": chamber.outputs["Geometry"], "Label": "Chamber"},
    )

    reroute_24 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Knee Material"]}
    )

    reroute_25 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_24})

    knee = nw.new_node(
        nodegroup_node_group_006().name,
        input_kwargs={
            "Radius": foot.outputs["Radius"],
            "Depth": reroute_3,
            "Base Depth": reroute_1,
            "Material": reroute_25,
        },
        label="Knee",
    )

    add_jointed_geometry_metadata_1 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": knee, "Label": "Knee"},
    )

    reroute_26 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": foot.outputs["Geometry"]}
    )

    add_jointed_geometry_metadata_2 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": reroute_26, "Label": "Foot"},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                add_jointed_geometry_metadata,
                add_jointed_geometry_metadata_1,
                add_jointed_geometry_metadata_2,
            ]
        },
    )

    add_jointed_geometry_metadata_3 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": join_geometry, "Label": "Full Base"},
    )

    reroute_10 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Lid Version"]}
    )

    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_10})

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Lid Version 0 Tilt"]}
    )

    reroute_15 = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": group_input.outputs["Lid Version 2 Factor (Sphere)"]},
    )

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Cap Material"]}
    )

    reroute_17 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Dot Material"]}
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Lid Height"]}
    )

    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": reroute_8})

    lid = nw.new_node(
        nodegroup_node_group_011().name,
        input_kwargs={
            "Radius": chamber.outputs["Radius"],
            "Foot Radius": group_input.outputs["Foot Radius"],
            "Lid Version": reroute_11,
            "Lid 0 Tilt": reroute_12,
            "Lid 1 Fraction Height (Handle)": group_input.outputs[
                "Lid Version 1 Fraction (Handle)"
            ],
            "Lid  2 Factor (Sphere)": reroute_15,
            "Lid 2 Compression (Sphere)": group_input.outputs[
                "Lid Version 2 Compression (Sphere)"
            ],
            "Material": reroute_16,
            "Material Mini Lid": reroute_17,
            "Depth": reroute_9,
            "Body": chamber.outputs["Geometry"],
        },
        label="Lid",
    )

    add_jointed_geometry_metadata_4 = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={"Geometry": lid, "Label": "Lid"},
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Foot Height"],
            1: group_input.outputs["Knee Height"],
        },
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: add, 1: reroute_5})

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: add_1, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": divide})

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "Lid Joint",
            "Parent": add_jointed_geometry_metadata_3,
            "Child": add_jointed_geometry_metadata_4,
            "Position": combine_xyz,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": hinge_joint.outputs["Geometry"]},
        attrs={"is_active_output": True},
    )


class PepperGrinderFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)

    @classmethod
    @gin.configurable(module="PepperGrinderFactory")
    def sample_joint_parameters(
        cls,
        Lid_Joint_stiffness_min: float = 0.0,
        Lid_Joint_stiffness_max: float = 0.0,
        Lid_Joint_damping_min: float = 0.05,
        Lid_Joint_damping_max: float = 0.2,
    ):
        return {
            "Lid Joint": {
                "stiffness": uniform(Lid_Joint_stiffness_min, Lid_Joint_stiffness_max),
                "damping": uniform(Lid_Joint_damping_min, Lid_Joint_damping_max),
            },
        }

    def sample_parameters(self):
        # add code here to randomly sample from parameters
        foot_height = uniform(0.02, 0.04)  # 2cm - 4cm
        foot_radius = uniform(0.025, 0.05)  # 2cm - 4cm
        foot_version = 0  # only straight foot, no inclination
        foot_tilt = 0.0  # no tilt
        knee_height = uniform(0.01, 0.04)  # 1cm - 4cm
        body_height = uniform(0.09, 0.16)  # 12cm - 16cm
        body_wave_strength = weighted_sample(
            [(uniform(0, 0.05), 0.1), (uniform(0.25, 0.45), 0.9)]
        )  # waviness of body (innnermost point is body_wave_strength * radius wide)
        body_nr_waves = uniform(0.8, 1.0)  # number of waves in body
        body_wave_offset = uniform(
            -1.4, 1.4
        )  # phase offset of wave (-3.142 means "start slim", 0 is "start thick")
        lid_height = uniform(0.015, 0.03)  # size of lid
        lid_version = weighted_sample([(0, 0.1), (1, 0.1), (2, 0.8)])
        lid_version_0_tilt = 0  # if version 0 select, amount of outward tilt of lid. Top will be 1+lid_version_0_tilt % wider. kept at 0 for now.
        lid_version_1_fraction = uniform(
            0.3, 0.6
        )  # fraction of handle that is "grabbed" vs. "sticks out"
        lid_version_2_factor_sphere = uniform(
            1.2, 1.45
        )  # sphere thickness if version is 2
        lid_version_2_compression_sphere = uniform(0.65, 1.0)

        def pick_material():
            r = uniform(0, 1)
            if r < 0.3:
                return material_assignments.kitchen_appliance_hard
            elif r < 0.6:
                return material_assignments.kitchen_appliance_hard
            else:
                return material_assignments.metal_plastic

        material_group1 = pick_material()
        material_group2 = pick_material()
        base_material = weighted_sample(material_group1)()()
        s = uniform()
        if s < 0.2:
            knee_material = chamber_material = cap_material = dot_material = (
                base_material
            )
        elif s < 0.7:
            knee_material = chamber_material = base_material
            cap_material = dot_material = weighted_sample(material_group2)()()
        else:
            knee_material = base_material
            chamber_material = cap_material = dot_material = weighted_sample(
                material_group2
            )()()

        return {
            "Foot Height": foot_height,
            "Foot Radius": foot_radius,
            "Foot Version": foot_version,
            "Foot Tilt": foot_tilt,
            "Knee Height": knee_height,
            "Body Height": body_height,
            "Body Nr. Waves": body_nr_waves,
            "Body Wave Strength": body_wave_strength,
            "Body Wave Offset": body_wave_offset,
            "Lid Height": lid_height,
            "Lid Version": lid_version,
            "Lid Version 0 Tilt": lid_version_0_tilt,
            "Lid Version 1 Fraction (Handle)": lid_version_1_fraction,
            "Lid Version 2 Factor (Sphere)": lid_version_2_factor_sphere,
            "Lid Version 2 Compression (Sphere)": lid_version_2_compression_sphere,
            "Base Material": base_material,
            "Knee Material": knee_material if uniform() < 0.1 else base_material,
            "Chamber Material": chamber_material if uniform() < 0.1 else base_material,
            "Cap Material": cap_material,
            "Dot Material": dot_material,
        }

    def create_asset(self, asset_params=None, **kwargs):
        obj = butil.spawn_vert()

        params = self.sample_parameters()
        all_nodes = geometry_nodes()
        nr_knees = int(float(params["Knee Height"]) / 0.02)
        if nr_knees == 0:
            control_points = [(0.000, 1.000), (1.000, 1.000)]
        else:
            placement_1th_knee = 1 / (nr_knees + 1)
            depth_center = uniform(0.7, 0.9)
            depth_sides = 1 - uniform(0, 0.05)
            dist_sides = uniform(0, 0.05)
            control_points = [
                (0.000, 1.000),
                (dist_sides, depth_center),
                (2 * dist_sides, depth_sides),
            ]

            for i in range(1, nr_knees + 2):
                control_points.append((i * placement_1th_knee, depth_center))
                control_points.append(
                    (i * placement_1th_knee - dist_sides, depth_sides)
                )
                control_points.append(
                    (i * placement_1th_knee + dist_sides, depth_sides)
                )

            control_points.extend([(1.0 - dist_sides, depth_sides), (1.000, 1.000)])

            for n in all_nodes.nodes:
                if n.name == "Knee":
                    for nn in n.node_tree.nodes:
                        if nn.bl_idname in (
                            "GeometryNodeFloatCurve",
                            "ShaderNodeFloatCurve",
                        ):
                            curve = nn.mapping.curves[0]

                            for p in control_points:
                                curve.points.new(p[0], p[1])

        # Create the modifier and then randomize the FloatCurve mapping per instance
        obj = butil.modify_mesh(
            obj, "NODES", apply=False, node_group=all_nodes, ng_inputs=params
        )

        return obj
