# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hongyu Wen


import numpy as np
from numpy.random import uniform as U

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.utils.misc import generate_text
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.bevelling import add_bevel, complete_no_bevel, get_bevel_edges
from infinigen.core.util.blender import delete
from infinigen.core.util.math import FixedSeed


class MicrowaveFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, dimensions=[1.0, 1.0, 1.0]):
        super(MicrowaveFactory, self).__init__(factory_seed, coarse=coarse)

        self.dimensions = dimensions
        with FixedSeed(factory_seed):
            self.params = self.sample_parameters(dimensions)
            self.material_params, self.scratch, self.edge_wear = (
                self.get_material_params()
            )
        self.params.update(self.material_params)

    def get_material_params(self):
        material_assignments = AssetList["MicrowaveFactory"]()
        params = {
            "Surface": material_assignments["surface"].assign_material(),
            "Back": material_assignments["back"].assign_material(),
            "BlackGlass": material_assignments["black_glass"].assign_material(),
            "Glass": material_assignments["glass"].assign_material(),
        }
        wrapped_params = {
            k: surface.shaderfunc_to_material(v) for k, v in params.items()
        }

        scratch_prob, edge_wear_prob = material_assignments["wear_tear_prob"]
        scratch, edge_wear = material_assignments["wear_tear"]

        is_scratch = np.random.uniform() < scratch_prob
        is_edge_wear = np.random.uniform() < edge_wear_prob
        if not is_scratch:
            scratch = None

        if not is_edge_wear:
            edge_wear = None

        return wrapped_params, scratch, edge_wear

    @staticmethod
    def sample_parameters(dimensions):
        depth = U(0.5, 0.7)
        width = U(0.6, 1.0)
        height = U(0.35, 0.45)
        panel_width = U(0.2, 0.4)
        margin_z = U(0.05, 0.1)
        door_thickness = U(0.02, 0.04)
        door_margin = U(0.03, 0.1)
        door_rotation = 0  # Set to 0 for now
        brand_name = generate_text()
        params = {
            "Depth": depth,
            "Width": width,
            "Height": height,
            "PanelWidth": panel_width,
            "MarginZ": margin_z,
            "DoorThickness": door_thickness,
            "DoorMargin": door_margin,
            "DoorRotation": door_rotation,
            "BrandName": brand_name,
        }
        return params

    def create_asset(self, **params):
        obj = butil.spawn_cube()
        butil.modify_mesh(
            obj,
            "NODES",
            node_group=nodegroup_microwave_geometry(preprocess=True),
            ng_inputs=self.params,
            apply=True,
        )
        bevel_edges = get_bevel_edges(obj)
        delete(obj)
        obj = butil.spawn_cube()
        butil.modify_mesh(
            obj,
            "NODES",
            node_group=nodegroup_microwave_geometry(),
            ng_inputs=self.params,
            apply=True,
        )
        obj = add_bevel(obj, bevel_edges)

        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)


@node_utils.to_nodegroup("nodegroup_plate", singleton=False, type="GeometryNodeTree")
def nodegroup_plate(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={"Resolution": 128})

    bezier_segment = nw.new_node(
        Nodes.CurveBezierSegment,
        input_kwargs={
            "Start Handle": (0.0000, 0.0000, 0.0000),
            "End": (1.0000, 0.0000, 0.4000),
        },
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": bezier_segment, "Rotation": (1.5708, 0.0000, 0.0000)},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_circle.outputs["Curve"],
            "Profile Curve": transform,
        },
    )

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[("NodeSocketVector", "Scale", (1.0000, 1.0000, 1.0000))],
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": curve_to_mesh, "Scale": group_input.outputs["Scale"]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": transform_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_text", singleton=False, type="GeometryNodeTree")
def nodegroup_text(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Translation", (1.5000, 0.0000, 0.0000)),
            ("NodeSocketString", "String", "BrandName"),
            ("NodeSocketFloat", "Size", 0.0500),
            ("NodeSocketFloat", "Offset Scale", 0.0020),
        ],
    )

    string_to_curves = nw.new_node(
        "GeometryNodeStringToCurves",
        input_kwargs={
            "String": group_input.outputs["String"],
            "Size": group_input.outputs["Size"],
        },
        attrs={"align_y": "BOTTOM_BASELINE", "align_x": "CENTER"},
    )

    fill_curve = nw.new_node(
        Nodes.FillCurve,
        input_kwargs={"Curve": string_to_curves.outputs["Curve Instances"]},
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": fill_curve,
            "Offset Scale": group_input.outputs["Offset Scale"],
        },
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": extrude_mesh.outputs["Mesh"],
            "Translation": group_input.outputs["Translation"],
            "Rotation": (1.5708, 0.0000, 1.5708),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_center", singleton=False, type="GeometryNodeTree")
def nodegroup_center(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketVector", "Vector", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "MarginX", 0.5000),
            ("NodeSocketFloat", "MarginY", 0.0000),
            ("NodeSocketFloat", "MarginZ", 0.0000),
        ],
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": group_input.outputs["Geometry"]}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Vector"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    greater_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: group_input.outputs["MarginX"]},
        attrs={"operation": "GREATER_THAN", "use_clamp": True},
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: group_input.outputs["Vector"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_1.outputs["Vector"]}
    )

    greater_than_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_1.outputs["X"],
            1: group_input.outputs["MarginX"],
        },
        attrs={"operation": "GREATER_THAN", "use_clamp": True},
    )

    op_and = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: greater_than, 1: greater_than_1}
    )

    greater_than_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: group_input.outputs["MarginY"]},
        attrs={"operation": "GREATER_THAN"},
    )

    greater_than_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_1.outputs["Y"],
            1: group_input.outputs["MarginY"],
        },
        attrs={"operation": "GREATER_THAN", "use_clamp": True},
    )

    op_and_1 = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: greater_than_2, 1: greater_than_3}
    )

    op_and_2 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: op_and, 1: op_and_1})

    greater_than_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: group_input.outputs["MarginZ"]},
        attrs={"operation": "GREATER_THAN", "use_clamp": True},
    )

    greater_than_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_1.outputs["Z"],
            1: group_input.outputs["MarginZ"],
        },
        attrs={"operation": "GREATER_THAN", "use_clamp": True},
    )

    op_and_3 = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: greater_than_4, 1: greater_than_5}
    )

    op_and_4 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: op_and_2, 1: op_and_3})

    op_not = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: op_and_4}, attrs={"operation": "NOT"}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"In": op_and_4, "Out": op_not},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_cube", singleton=False, type="GeometryNodeTree")
def nodegroup_cube(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Size", (0.1000, 10.0000, 4.0000)),
            ("NodeSocketVector", "Pos", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketInt", "Resolution", 10),
        ],
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": group_input.outputs["Size"],
            "Vertices X": group_input.outputs["Resolution"],
            "Vertices Y": group_input.outputs["Resolution"],
            "Vertices Z": group_input.outputs["Resolution"],
        },
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Name": "uv_map",
            3: cube.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": store_named_attribute_1, "Name": "uv_map"},
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Size"],
            1: (0.5000, 0.5000, 0.5000),
            2: group_input.outputs["Pos"],
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute,
            "Translation": multiply_add.outputs["Vector"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_microwave_geometry", singleton=False, type="GeometryNodeTree"
)
def nodegroup_microwave_geometry(nw: NodeWrangler, preprocess: bool = False):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Depth", 0.0000),
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "PanelWidth", 0.5000),
            ("NodeSocketFloat", "MarginZ", 0.0000),
            ("NodeSocketFloat", "DoorThickness", 0.0000),
            ("NodeSocketFloat", "DoorMargin", 0.0500),
            ("NodeSocketFloat", "DoorRotation", 0.0000),
            ("NodeSocketString", "BrandName", "BrandName"),
            ("NodeSocketMaterial", "Surface", None),
            ("NodeSocketMaterial", "Back", None),
            ("NodeSocketMaterial", "BlackGlass", None),
            ("NodeSocketMaterial", "Glass", None),
        ],
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Depth"],
            "Y": group_input.outputs["Width"],
            "Z": group_input.outputs["Height"],
        },
    )

    cube = nw.new_node(nodegroup_cube().name, input_kwargs={"Size": combine_xyz})

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Width"],
            1: group_input.outputs["PanelWidth"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Height"],
            1: group_input.outputs["MarginZ"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Depth"],
            "Y": subtract,
            "Z": subtract_1,
        },
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["MarginZ"], "Scale": 0.5000},
        attrs={"operation": "SCALE"},
    )

    cube_1 = nw.new_node(
        nodegroup_cube().name,
        input_kwargs={"Size": combine_xyz_1, "Pos": scale.outputs["Vector"]},
    )

    difference = nw.new_node(
        Nodes.MeshBoolean, input_kwargs={"Mesh 1": cube, "Mesh 2": cube_1}
    )

    cube_2 = nw.new_node(
        nodegroup_cube().name,
        input_kwargs={
            "Size": (0.0300, 0.0300, 0.0100),
            "Pos": (0.1000, 0.0000, 0.0500),
            "Resolution": 2,
        },
    )

    geometry_to_instance_1 = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": cube_2}
    )

    duplicate_elements = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={"Geometry": geometry_to_instance_1, "Amount": 10},
        attrs={"domain": "INSTANCE"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: 0.0400},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply})

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": duplicate_elements.outputs["Geometry"],
            "Offset": combine_xyz_7,
        },
    )

    duplicate_elements_1 = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={"Geometry": set_position_1, "Amount": 7},
        attrs={"domain": "INSTANCE"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements_1.outputs["Duplicate Index"], 1: 0.0200},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_1})

    set_position_2 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": duplicate_elements_1.outputs["Geometry"],
            "Offset": combine_xyz_8,
        },
    )

    difference_1 = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={
            "Mesh 1": difference.outputs["Mesh"],
            "Mesh 2": [duplicate_elements_1.outputs["Geometry"], set_position_2],
        },
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": difference_1.outputs["Mesh"],
            "Material": group_input.outputs["Back"],
        },
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["DoorThickness"],
            "Y": group_input.outputs["Width"],
            "Z": group_input.outputs["Height"],
        },
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input.outputs["Depth"]}
    )

    cube_3 = nw.new_node(
        nodegroup_cube().name,
        input_kwargs={"Size": combine_xyz_2, "Pos": combine_xyz_3, "Resolution": 10},
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Width"],
            1: group_input.outputs["PanelWidth"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["MarginZ"]},
        attrs={"operation": "MULTIPLY"},
    )

    add = nw.new_node(Nodes.Math, input_kwargs={0: subtract_2, 1: multiply_2})

    less_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: add},
        attrs={"operation": "LESS_THAN"},
    )

    separate_geometry = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": cube_3, "Selection": less_than},
        attrs={"domain": "FACE"},
    )

    convex_hull = nw.new_node(
        Nodes.ConvexHull,
        input_kwargs={"Geometry": separate_geometry.outputs["Selection"]},
    )

    subdivide_mesh = nw.new_node(
        Nodes.SubdivideMesh, input_kwargs={"Mesh": convex_hull, "Level": 0}
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    center = nw.new_node(
        nodegroup_center().name,
        input_kwargs={
            "Geometry": subdivide_mesh,
            "Vector": position_1,
            "MarginX": -1.0000,
            "MarginZ": group_input.outputs["DoorMargin"],
        },
    )

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": subdivide_mesh,
            "Selection": center.outputs["In"],
            "Material": group_input.outputs["BlackGlass"],
        },
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": set_material_3,
            "Selection": center.outputs["Out"],
            "Material": group_input.outputs["Surface"],
        },
    )

    add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Depth"],
            1: group_input.outputs["DoorThickness"],
        },
    )

    bounding_box_1 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": subdivide_mesh}
    )

    add_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_1.outputs["Min"],
            1: bounding_box_1.outputs["Max"],
        },
    )

    scale_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_2.outputs["Vector"], "Scale": 0.5000},
        attrs={"operation": "SCALE"},
    )

    separate_xyz_3 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": scale_1.outputs["Vector"]}
    )

    separate_xyz_4 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box_1.outputs["Min"]}
    )

    add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_4.outputs["Z"],
            1: group_input.outputs["DoorMargin"],
        },
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": add_1, "Y": separate_xyz_3.outputs["Y"], "Z": add_3},
    )

    text = nw.new_node(
        nodegroup_text().name,
        input_kwargs={
            "Translation": combine_xyz_5,
            "String": group_input.outputs["BrandName"],
            "Size": 0.0300,
            "Offset Scale": 0.0020,
        },
    )

    text = complete_no_bevel(nw, text, preprocess)

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material_2, text]}
    )

    geometry_to_instance = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": join_geometry_1}
    )

    z = nw.scalar_multiply(
        group_input.outputs["DoorRotation"], 1 if not preprocess else 0
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": z})

    rotate_instances = nw.new_node(
        Nodes.RotateInstances,
        input_kwargs={
            "Instances": geometry_to_instance,
            "Rotation": combine_xyz_6,
            "Pivot Point": combine_xyz_3,
        },
    )

    plate = nw.new_node(
        nodegroup_plate().name, input_kwargs={"Scale": (0.1000, 0.1000, 0.1000)}
    )

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: combine_xyz_1,
            1: (0.5000, 0.5000, 0.0000),
            2: scale.outputs["Vector"],
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": plate, "Offset": multiply_add.outputs["Vector"]},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": set_position,
            "Material": group_input.outputs["Glass"],
        },
    )

    convex_hull_1 = nw.new_node(
        Nodes.ConvexHull,
        input_kwargs={"Geometry": separate_geometry.outputs["Inverted"]},
    )

    subdivide_mesh_1 = nw.new_node(
        Nodes.SubdivideMesh, input_kwargs={"Mesh": convex_hull_1, "Level": 0}
    )

    position_2 = nw.new_node(Nodes.InputPosition)

    center_1 = nw.new_node(
        nodegroup_center().name,
        input_kwargs={
            "Geometry": subdivide_mesh_1,
            "Vector": position_2,
            "MarginX": -1.0000,
            "MarginY": 0.0010,
            "MarginZ": group_input.outputs["DoorMargin"],
        },
    )

    set_material_4 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": subdivide_mesh_1,
            "Selection": center_1.outputs["In"],
            "Material": group_input.outputs["BlackGlass"],
        },
    )

    set_material_5 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": set_material_4,
            "Selection": center_1.outputs["Out"],
            "Material": group_input.outputs["Surface"],
        },
    )

    add_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Depth"],
            1: group_input.outputs["DoorThickness"],
        },
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": subdivide_mesh_1}
    )

    add_5 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Min"], 1: bounding_box.outputs["Max"]},
    )

    scale_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_5.outputs["Vector"], "Scale": 0.5000},
        attrs={"operation": "SCALE"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": scale_2.outputs["Vector"]}
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Max"]}
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_2.outputs["Z"],
            1: group_input.outputs["DoorMargin"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: subtract_3, 1: -0.1000})

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": add_4, "Y": separate_xyz_1.outputs["Y"], "Z": add_6},
    )

    text_1 = nw.new_node(
        nodegroup_text().name,
        input_kwargs={
            "Translation": combine_xyz_4,
            "String": "12:01",
            "Offset Scale": 0.0050,
        },
    )

    text_1 = complete_no_bevel(nw, text_1, preprocess)

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                set_material_1,
                rotate_instances,
                set_material,
                set_material_5,
                text_1,
            ]
        },
    )
    geometry = nw.new_node(Nodes.RealizeInstances, [join_geometry])
    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": geometry},
        attrs={"is_active_output": True},
    )
