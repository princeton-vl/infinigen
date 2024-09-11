# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Hongyu Wen: primary author
# - Alexander Raistrick: add point light

import random

import bpy
import numpy as np
from numpy.random import uniform as U

from infinigen.assets.lighting.indoor_lights import PointLampFactory
from infinigen.assets.material_assignments import AssetList
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


class LampFactory(AssetFactory):
    def __init__(
        self,
        factory_seed,
        coarse=False,
        dimensions=[1.0, 1.0, 1.0],
        lamp_type="FloorLamp",
    ):
        super(LampFactory, self).__init__(factory_seed, coarse=coarse)

        self.bulb_fac = PointLampFactory(factory_seed)
        self.bulb_fac.params["Temperature"] = max(
            self.bulb_fac.params["Temperature"] * 0.6, 2500
        )
        self.bulb_fac.params["Wattage"] *= 0.5

        self.dimensions = dimensions
        self.lamp_type = lamp_type
        self.lamp_default_params = {
            "DeskLamp": {
                "StandRadius": 0.01,
                "StandHeight": 0.3,
                "BaseRadius": 0.07,
                "BaseHeight": 0.02,
                "ShadeHeight": 0.18,
                "HeadTopRadius": 0.08,
                "HeadBotRadius": 0.11,
                "ReverseLamp": True,
                "RackThickness": 0.002,
                "CurvePoint1": (0.0, 0.0, 0.0),
                "CurvePoint2": (0.0, 0.0, 0.2),
                "CurvePoint3": (0.0, 0.0, 0.3),
            },
            "FloorLamp1": {
                "StandRadius": 0.01,
                "StandHeight": 0.3,
                "BaseRadius": 0.1,
                "BaseHeight": 0.02,
                "ShadeHeight": 0.2,
                "HeadTopRadius": 0.1,
                "HeadBotRadius": 0.12,
                "ReverseLamp": False,
                "RackThickness": 0.002,
                "CurvePoint1": (0.0, 0.0, 1.0),
                "CurvePoint2": (0.05, 0.0, 1.2),
                "CurvePoint3": (0.2, 0.0, 1.0),
            },
            "FloorLamp2": {
                "StandRadius": 0.01,
                "StandHeight": 0.3,
                "BaseRadius": 0.1,
                "BaseHeight": 0.02,
                "ShadeHeight": 0.2,
                "HeadTopRadius": 0.1,
                "HeadBotRadius": 0.11,
                "ReverseLamp": True,
                "RackThickness": 0.002,
                "CurvePoint1": (0.0, 0.0, 1.0),
                "CurvePoint2": (0.0, 0.0, 1.1),
                "CurvePoint3": (0.0, 0.0, 1.2),
            },
        }
        with FixedSeed(factory_seed):
            self.params = self.sample_parameters(dimensions)
            self.material_params, self.scratch, self.edge_wear = (
                self.get_material_params()
            )

        self.params.update(self.material_params)

    def get_material_params(self):
        material_assignments = AssetList["LampFactory"]()
        black_material = material_assignments["black_material"].assign_material()
        white_material = material_assignments["metal"].assign_material()
        lampshade_material = material_assignments["lampshade"].assign_material()

        wrapped_params = {
            "BlackMaterial": surface.shaderfunc_to_material(black_material),
            "MetalMaterial": surface.shaderfunc_to_material(white_material),
            "LampshadeMaterial": surface.shaderfunc_to_material(lampshade_material),
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

    def sample_parameters(self, dimensions, use_default=False):
        if use_default:
            if self.lamp_type == "DeskLamp":
                return self.lamp_default_params["DeskLamp"]
            else:
                return random.choice(
                    [
                        self.lamp_default_params["FloorLamp1"],
                        self.lamp_default_params["FloorLamp2"],
                    ]
                )
        else:
            stand_radius = U(0.005, 0.015)
            base_radius = U(0.05, 0.15)
            base_height = U(0.01, 0.03)
            shade_height = U(0.18, 0.3)
            head_top_radius = U(0.07, 0.15)
            head_bot_radius = head_top_radius + U(0, 0.05)
            rack_thickness = U(0.001, 0.003)
            reverse_lamp = True

            if self.lamp_type == "DeskLamp":
                height = U(0.25, 0.4)
            else:
                height = U(1, 1.5)

            z1 = U(base_height, height)
            z2 = U(z1, height)
            z3 = height

            x1, x2, x3 = 0, 0, 0
            # if self.lamp_type == "FloorLamp" and U() < 0.5:
            #     x2 = U(0.03, 0.1)
            #     x3 = U(0.2, 0.4)
            #     z2, z3 = z3, z2
            #     reverse_lamp = False

            params = {
                "StandRadius": stand_radius,
                "BaseRadius": base_radius,
                "BaseHeight": base_height,
                "ShadeHeight": shade_height,
                "HeadTopRadius": head_top_radius,
                "HeadBotRadius": head_bot_radius,
                "ReverseLamp": reverse_lamp,
                "RackThickness": rack_thickness,
                "CurvePoint1": (x1, 0.0, z1),
                "CurvePoint2": (x2, 0.0, z2),
                "CurvePoint3": (x3, 0.0, z3),
            }
            return params

    def create_asset(self, i, **params):
        obj = butil.spawn_cube()
        butil.modify_mesh(
            obj,
            "NODES",
            node_group=nodegroup_lamp_geometry(),
            ng_inputs=self.params,
            apply=True,
        )

        if np.random.uniform() < 0.6:
            bulb = self.bulb_fac(i)
            butil.parent_to(bulb, obj, no_inverse=True, no_transform=True)
            bulb.location.z = obj.bound_box[-2][2] - self.params["ShadeHeight"] * 0.5

        with butil.SelectObjects(obj):
            bpy.ops.object.shade_flat()

        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)


class DeskLampFactory(LampFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse, lamp_type="DeskLamp")


class FloorLampFactory(LampFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(
            factory_seed,
            coarse,
            lamp_type=np.random.choice(["FloorLamp1", "FloorLamp2"]),
        )


@node_utils.to_nodegroup("nodegroup_bulb", singleton=False, type="GeometryNodeTree")
def nodegroup_bulb(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketMaterial", "LampshadeMaterial", None),
            ("NodeSocketMaterial", "MetalMaterial", None),
        ],
    )

    curve_line_1 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (0.0000, 0.0000, -0.2000),
            "End": (0.0000, 0.0000, 0.0000),
        },
    )

    curve_circle_1 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Radius": 0.1500, "Resolution": 100}
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line_1,
            "Profile Curve": curve_circle_1.outputs["Curve"],
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

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": spiral, "Translation": (0.0000, 0.0000, -0.2000)},
    )

    curve_circle_2 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Radius": 0.0150, "Resolution": 100}
    )

    curve_to_mesh_2 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": transform,
            "Profile Curve": curve_circle_2.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    curve_line_2 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={
            "Start": (0.0000, 0.0000, -0.2000),
            "End": (0.0000, 0.0000, -0.3000),
        },
    )

    resample_curve_1 = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": curve_line_2, "Count": 100}
    )

    spline_parameter_1 = nw.new_node(Nodes.SplineParameter)

    float_curve_1 = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": spline_parameter_1.outputs["Factor"]}
    )
    node_utils.assign_curve(
        float_curve_1.mapping.curves[0],
        [(0.0000, 1.0000), (0.4432, 0.5500), (1.0000, 0.2750)],
        handles=["AUTO", "VECTOR", "AUTO"],
    )

    set_curve_radius_1 = nw.new_node(
        Nodes.SetCurveRadius,
        input_kwargs={"Curve": resample_curve_1, "Radius": float_curve_1},
    )

    curve_circle_3 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Radius": 0.1500, "Resolution": 100}
    )

    curve_to_mesh_3 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius_1,
            "Profile Curve": curve_circle_3.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [curve_to_mesh_1, curve_to_mesh_2, curve_to_mesh_3]},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Material": group_input.outputs["MetalMaterial"],
        },
    )

    curve_line = nw.new_node(Nodes.CurveLine)

    resample_curve = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": curve_line, "Count": 100}
    )

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    float_curve = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": spline_parameter.outputs["Factor"]}
    )
    node_utils.assign_curve(
        float_curve.mapping.curves[0],
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

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius,
        input_kwargs={"Curve": resample_curve, "Radius": float_curve},
    )

    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={"Resolution": 100})

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius,
            "Profile Curve": curve_circle.outputs["Curve"],
        },
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": curve_to_mesh,
            "Material": group_input.outputs["LampshadeMaterial"],
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, set_material_1]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry,
            "Translation": (0.0000, 0.0000, 0.3000),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_bulb_rack", singleton=False, type="GeometryNodeTree"
)
def nodegroup_bulb_rack(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

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
        ],
    )

    curve_circle_2 = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={"Radius": amount.outputs["OuterRadius"], "Resolution": 100},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": amount.outputs["OuterHeight"]}
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle_2.outputs["Curve"],
            "Translation": combine_xyz,
        },
    )

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

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": amount.outputs["Amount"]}
    )

    duplicate_elements = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={"Geometry": geometry_to_instance, "Amount": reroute},
        attrs={"domain": "INSTANCE"},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances,
        input_kwargs={"Geometry": duplicate_elements.outputs["Geometry"]},
    )

    endpoint_selection = nw.new_node(
        Nodes.EndpointSelection, input_kwargs={"Start Size": 0}
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: 1.0000, 1: reroute}, attrs={"operation": "DIVIDE"}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: divide},
        attrs={"operation": "MULTIPLY"},
    )

    sample_curve = nw.new_node(
        Nodes.SampleCurve,
        input_kwargs={"Curves": transform, "Factor": multiply},
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

    curve_circle = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Radius": multiply_add, "Resolution": 100}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": amount.outputs["InnerHeight"]}
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle.outputs["Curve"],
            "Translation": combine_xyz_1,
        },
    )

    sample_curve_1 = nw.new_node(
        Nodes.SampleCurve,
        input_kwargs={"Curves": transform_1, "Factor": multiply},
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

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform, set_position_1, transform_1]},
    )

    curve_circle_1 = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={"Radius": amount.outputs["Thickness"], "Resolution": 100},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": join_geometry,
            "Profile Curve": curve_circle_1.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": curve_to_mesh},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_reversiable_bulb", singleton=False, type="GeometryNodeTree"
)
def nodegroup_reversiable_bulb(nw: NodeWrangler):
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

    bulb = nw.new_node(
        nodegroup_bulb().name,
        input_kwargs={
            "LampshadeMaterial": group_input.outputs["LampshadeMaterial"],
            "MetalMaterial": group_input.outputs["MetalMaterial"],
        },
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Scale"],
            "Y": group_input.outputs["Scale"],
            "Z": group_input.outputs["Scale"],
        },
    )

    transform = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": bulb, "Scale": combine_xyz_1}
    )

    geometry_to_instance = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": transform}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Reverse"], 1: 3.1415},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply})

    rotate_instances = nw.new_node(
        Nodes.RotateInstances,
        input_kwargs={"Instances": geometry_to_instance, "Rotation": combine_xyz_2},
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Reverse"], 1: 2.0000, 2: -1.0000},
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
    "nodegroup_lamp_head", singleton=False, type="GeometryNodeTree"
)
def nodegroup_lamp_head(nw: NodeWrangler):
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
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["TopRadius"], 1: 0.8000},
        attrs={"operation": "MULTIPLY"},
    )

    reversiable_bulb = nw.new_node(
        nodegroup_reversiable_bulb().name,
        input_kwargs={
            "Scale": multiply,
            "BlackMaterial": group_input.outputs["BlackMaterial"],
            "LampshadeMaterial": group_input.outputs["LampshadeMaterial"],
            "MetalMaterial": group_input.outputs["MetalMaterial"],
        },
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: 0.1500},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["ReverseBulb"], 1: 2.0000, 2: -1.0000},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["RackHeight"], 1: multiply_add},
        attrs={"operation": "MULTIPLY"},
    )

    bulb_rack = nw.new_node(
        nodegroup_bulb_rack().name,
        input_kwargs={
            "Thickness": group_input.outputs["RackThickness"],
            "InnerRadius": multiply_1,
            "OuterRadius": group_input.outputs["TopRadius"],
            "InnerHeight": reversiable_bulb.outputs["RackSupport"],
            "OuterHeight": multiply_2,
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": bulb_rack,
            "Material": group_input.outputs["BlackMaterial"],
        },
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_2})

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["ShadeHeight"],
            1: group_input.outputs["RackHeight"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_add, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: multiply_3},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_4})

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_1, "End": combine_xyz}
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

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius,
        input_kwargs={"Curve": curve_line, "Radius": map_range.outputs["Result"]},
    )

    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={"Resolution": 100})

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius,
            "Profile Curve": curve_circle.outputs["Curve"],
        },
    )

    flip_faces = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": curve_to_mesh})

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": curve_to_mesh,
            "Offset Scale": 0.0050,
            "Individual": False,
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [flip_faces, extrude_mesh.outputs["Mesh"]]},
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Material": group_input.outputs["LampshadeMaterial"],
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                reversiable_bulb.outputs["Geometry"],
                set_material,
                set_material_1,
            ]
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_lamp_geometry", singleton=False, type="GeometryNodeTree"
)
def nodegroup_lamp_geometry(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "StandRadius", 0.0200),
            ("NodeSocketFloat", "BaseRadius", 0.1000),
            ("NodeSocketFloat", "BaseHeight", 0.0200),
            ("NodeSocketFloat", "ShadeHeight", 0.0000),
            ("NodeSocketFloat", "HeadTopRadius", 0.3000),
            ("NodeSocketFloat", "HeadBotRadius", 0.5000),
            ("NodeSocketBool", "ReverseLamp", True),
            ("NodeSocketFloat", "RackThickness", 0.0050),
            ("NodeSocketVector", "CurvePoint1", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "CurvePoint2", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketVector", "CurvePoint3", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketMaterial", "BlackMaterial", None),
            ("NodeSocketMaterial", "LampshadeMaterial", None),
            ("NodeSocketMaterial", "MetalMaterial", None),
        ],
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["BaseHeight"]}
    )

    curve_line_1 = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz_1})

    curve_circle_1 = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={"Radius": group_input.outputs["BaseRadius"], "Resolution": 100},
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line_1,
            "Profile Curve": curve_circle_1.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["BaseHeight"]}
    )

    bezier_segment = nw.new_node(
        Nodes.CurveBezierSegment,
        input_kwargs={
            "Start": combine_xyz,
            "Start Handle": group_input.outputs["CurvePoint1"],
            "End Handle": group_input.outputs["CurvePoint2"],
            "End": group_input.outputs["CurvePoint3"],
            "Resolution": 100,
        },
    )

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz})

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [bezier_segment, curve_line]}
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={"Radius": group_input.outputs["StandRadius"], "Resolution": 100},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": join_geometry_2,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [curve_to_mesh_1, curve_to_mesh]}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": join_geometry,
            "Material": group_input.outputs["BlackMaterial"],
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["ShadeHeight"], 1: 0.4000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["ShadeHeight"], 1: 0.2000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: multiply,
            1: group_input.outputs["ReverseLamp"],
            2: multiply_1,
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    lamp_head = nw.new_node(
        nodegroup_lamp_head().name,
        input_kwargs={
            "ShadeHeight": group_input.outputs["ShadeHeight"],
            "TopRadius": group_input.outputs["HeadTopRadius"],
            "BotRadius": group_input.outputs["HeadBotRadius"],
            "ReverseBulb": group_input.outputs["ReverseLamp"],
            "RackThickness": group_input.outputs["RackThickness"],
            "RackHeight": multiply_add,
            "BlackMaterial": group_input.outputs["BlackMaterial"],
            "LampshadeMaterial": group_input.outputs["LampshadeMaterial"],
            "MetalMaterial": group_input.outputs["MetalMaterial"],
        },
    )

    sample_curve = nw.new_node(
        Nodes.SampleCurve,
        input_kwargs={"Curves": bezier_segment, "Factor": 1.0000},
        attrs={"use_all_curves": True},
    )

    align_euler_to_vector = nw.new_node(
        Nodes.AlignEulerToVector,
        input_kwargs={"Vector": sample_curve.outputs["Tangent"]},
        attrs={"axis": "Z"},
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": lamp_head,
            "Translation": sample_curve.outputs["Position"],
            "Rotation": align_euler_to_vector,
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, transform]}
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": join_geometry_1}
    )

    curve_line_2 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"End": (0.0000, 0.0000, 0.1000)}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_line_2,
            "Translation": sample_curve.outputs["Position"],
            "Rotation": align_euler_to_vector,
        },
    )

    sample_curve_1 = nw.new_node(
        Nodes.SampleCurve, input_kwargs={"Curves": transform_geometry, "Factor": 1.0000}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Bounding Box": bounding_box.outputs["Bounding Box"],
            "LightPosition": sample_curve_1.outputs["Position"],
        },
        attrs={"is_active_output": True},
    )
