# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Hongyu Wen: sink geometry
# - Meenal Parakh: material assignment
# - Stamatis Alexandropoulos: taps
# - Alexander Raistrick: placeholder, optimize detail, redo cutter


import bpy
import numpy as np
from numpy.random import uniform as U

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.utils import bbox_from_mesh
from infinigen.assets.utils.extract_nodegroup_parts import extract_nodegroup_geo
from infinigen.core import surface, tagging
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


class SinkFactory(AssetFactory):
    def __init__(
        self, factory_seed, coarse=False, dimensions=[1.0, 1.0, 1.0], upper_height=None
    ):
        super(SinkFactory, self).__init__(factory_seed, coarse=coarse)

        self.dimensions = dimensions
        self.factory_seed = factory_seed
        with FixedSeed(factory_seed):
            self.params = self.sample_parameters(dimensions, upper_height=upper_height)
            self.material_params, self.scratch, self.edge_wear = (
                self.get_material_params()
            )
        self.params.update(self.material_params)

        self.tap_factory = TapFactory(factory_seed)

    def get_material_params(self):
        material_assignments = AssetList["SinkFactory"]()
        params = {
            "Sink": material_assignments["sink"].assign_material(),
            "Tap": material_assignments["tap"].assign_material(),
        }
        wrapped_params = {
            k: surface.shaderfunc_to_material(v) for k, v in params.items()
        }

        scratch_prob, edge_wear_prob = material_assignments["wear_tear_prob"]
        scratch, edge_wear = material_assignments["wear_tear"]

        is_scratch = U() < scratch_prob
        is_edge_wear = U() < edge_wear_prob
        if not is_scratch:
            scratch = None

        if not is_edge_wear:
            edge_wear = None

        return wrapped_params, scratch, edge_wear

    @staticmethod
    def sample_parameters(dimensions, upper_height, use_default=False, open=False):
        width = U(0.4, 1.0)
        depth = U(0.4, 0.5)
        curvature = U(1.0, 1.0)
        if upper_height is None:
            upper_height = U(0.2, 0.4)
        lower_height = U(0.00, 0.01)
        hole_radius = U(0.02, 0.05)
        margin = U(0.02, 0.05)
        watertap_margin = U(0.1, 0.12)

        params = {
            "Width": width,
            "Depth": depth,
            "Curvature": curvature,
            "Upper Height": upper_height,
            "Lower Height": lower_height,
            "HoleRadius": hole_radius,
            "Margin": margin,
            "WaterTapMargin": watertap_margin,
            "ProtrudeAboveCounter": U(0.01, 0.025),
        }
        return params

    def _extract_geo_results(self):
        params = self.params.copy()
        params.pop("ProtrudeAboveCounter")

        with butil.TemporaryObject(butil.spawn_vert()) as temp:
            obj = extract_nodegroup_geo(
                temp, nodegroup_sink_geometry(), "Geometry", ng_params=params
            )
            cutter = extract_nodegroup_geo(
                temp, nodegroup_sink_geometry(), "Cutter", ng_params=params
            )

        return obj, cutter

    def create_placeholder(self, i, **kwargs) -> bpy.types.Object:
        obj, cutter = self._extract_geo_results()
        butil.delete(cutter)

        min_corner, max_corner = butil.bounds(obj)
        min_corner[-1] = max_corner[-1] - self.params["ProtrudeAboveCounter"]
        top_slice_placeholder = bbox_from_mesh.box_from_corners(min_corner, max_corner)

        butil.delete(obj)

        return top_slice_placeholder

    def create_asset(self, i, placeholder, state=None, **params):
        obj, cutter = self._extract_geo_results()
        tagging.tag_system.relabel_obj(obj)

        cutter.parent = obj
        cutter.name = repr(self) + f".spawn_placeholder({i}).cutter"
        cutter.hide_render = True

        tap_loc = (-self.params["Depth"] / 2, 0, self.params["Upper Height"])
        tap = self.tap_factory.spawn_asset(i, loc=tap_loc, rot=(0, 0, 0))
        tap.parent = obj

        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)


class TapFactory(AssetFactory):
    def __init__(self, factory_seed):
        super().__init__(factory_seed)
        with FixedSeed(factory_seed):
            self.params, self.scratch, self.edge_wear = self.get_material_params()

    @staticmethod
    def tap_parameters():
        params = {
            "base_width": U(0.570, 0.630),
            "tap_head": U(0.7, 1.1),
            "roation_z": U(5.5, 7.0),
            "tap_height": U(0.5, 1),
            "base_radius": U(0.0, 0.3),
            "Switch": True if U() > 0.5 else False,
            "Y": U(-0.5, -0.06),
            "hand_type": True if U() > 0.2 else False,
            "hands_length_x": U(0.750, 1.25),
            "hands_length_Y": U(0.950, 1.550),
            "one_side": True if U() > 0.5 else False,
            "different_type": True if U() > 0.8 else False,
        }
        return params

    def get_material_params(self):
        material_assignments = AssetList["TapFactory"]()
        tap_material = material_assignments["tap"].assign_material()

        wrapped_params = {"Tap": surface.shaderfunc_to_material(tap_material)}

        scratch_prob, edge_wear_prob = material_assignments["wear_tear_prob"]
        scratch, edge_wear = material_assignments["wear_tear"]

        is_scratch = U() < scratch_prob
        is_edge_wear = U() < edge_wear_prob
        if not is_scratch:
            scratch = None

        if not is_edge_wear:
            edge_wear = None

        return wrapped_params, scratch, edge_wear

    def create_asset(self, **_):
        obj = butil.spawn_cube()
        butil.modify_mesh(
            obj,
            "NODES",
            node_group=nodegroup_water_tap(),
            ng_inputs=self.params,
            apply=True,
        )
        obj.scale = (0.4,) * 3
        obj.rotation_euler.z += np.pi
        butil.apply_transform(obj)
        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)


@node_utils.to_nodegroup("nodegroup_handle", singleton=False, type="GeometryNodeTree")
def nodegroup_handle(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    bezier_segment = nw.new_node(
        Nodes.CurveBezierSegment,
        input_kwargs={
            "Start": (0.0000, 0.0000, 0.0000),
            "Start Handle": (0.0000, 0.0000, 0.7000),
            "End Handle": (0.2000, 0.0000, 0.7000),
            "End": (1.0000, 0.0000, 0.9000),
        },
    )

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    float_curve = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": spline_parameter.outputs["Factor"]}
    )
    node_utils.assign_curve(
        float_curve.mapping.curves[0], [(0.0000, 0.9750), (1.0000, 0.1625)]
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: float_curve, 1: 1.3000},
        attrs={"operation": "MULTIPLY"},
    )

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius, input_kwargs={"Curve": bezier_segment, "Radius": multiply}
    )

    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={"Radius": 0.2000})

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": separate_xyz.outputs["X"],
            1: 0.2000,
            3: 1.0000,
            4: 2.5000,
        },
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz.outputs["X"],
            "Y": multiply_1,
            "Z": separate_xyz.outputs["Z"],
        },
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": curve_to_mesh, "Position": combine_xyz},
    )

    subdivision_surface = nw.new_node(
        Nodes.SubdivisionSurface, input_kwargs={"Mesh": set_position, "Level": 2}
    )

    set_shade_smooth = nw.new_node(
        Nodes.SetShadeSmooth, input_kwargs={"Geometry": subdivision_surface}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_shade_smooth},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_water_tap", singleton=False, type="GeometryNodeTree"
)
def nodegroup_water_tap(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "base_width", U(0.2, 0.3)),
            ("NodeSocketFloat", "tap_head", U(0.7, 1.1)),
            ("NodeSocketFloat", "roation_z", U(5.5, 7.0)),
            ("NodeSocketFloat", "tap_height", U(0.5, 1)),
            ("NodeSocketFloat", "base_radius", U(0.0, 0.1)),
            ("NodeSocketBool", "Switch", True if U() > 0.5 else False),
            ("NodeSocketFloat", "Y", U(-0.5, -0.06)),
            ("NodeSocketBool", "hand_type", True if U() > 0.2 else False),
            ("NodeSocketFloat", "hands_length_x", U(0.750, 1.25)),
            ("NodeSocketFloat", "hands_length_Y", U(0.950, 1.550)),
            ("NodeSocketBool", "one_side", True if U() > 0.5 else False),
            ("NodeSocketBool", "different_type", True if U() > 0.8 else False),
            ("NodeSocketBool", "length_one_side", True if U() > 0.8 else False),
        ],
    )
    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketMaterial", "Tap", None)]
    )
    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={"Radius": 0.0500})

    fill_curve_1 = nw.new_node(
        Nodes.FillCurve, input_kwargs={"Curve": curve_circle.outputs["Curve"]}
    )

    extrude_mesh_1 = nw.new_node(
        Nodes.ExtrudeMesh, input_kwargs={"Mesh": fill_curve_1, "Offset Scale": 0.1500}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": 0.2000, "Height": 0.7000},
    )

    fillet_curve = nw.new_node(
        "GeometryNodeFilletCurve",
        input_kwargs={"Curve": quadrilateral, "Count": 19, "Radius": 0.1000},
        attrs={"mode": "POLY"},
    )

    fill_curve = nw.new_node(Nodes.FillCurve, input_kwargs={"Curve": fillet_curve})

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh, input_kwargs={"Mesh": fill_curve, "Offset Scale": 0.0500}
    )

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"End": (0.0000, 0.0000, 0.6000)}
    )

    curve_circle_1 = nw.new_node(Nodes.CurveCircle, input_kwargs={"Radius": 0.0300})

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": curve_circle_1.outputs["Curve"],
        },
    )

    curve_circle_2 = nw.new_node(Nodes.CurveCircle, input_kwargs={"Radius": 0.2000})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle_2.outputs["Curve"],
            "Translation": (0.0000, 0.2000, 0.0000),
        },
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry,
            "Rotation": (-1.5708, 1.5708, 0.0000),
            "Scale": (1.0000, 0.7000, 1.0000),
        },
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": 0.2000, "Y": group_input.outputs["Y"]}
    )

    bezier_segment = nw.new_node(
        Nodes.CurveBezierSegment,
        input_kwargs={
            "Resolution": 177,
            "Start": (0.0000, 0.0000, 0.0000),
            "Start Handle": (0.0000, 1.2000, 0.0000),
            "End Handle": combine_xyz_3,
            "End": (-0.0500, 0.1000, 0.0000),
        },
    )

    trim_curve = nw.new_node(
        Nodes.TrimCurve, input_kwargs={"Curve": bezier_segment, 3: 0.6625, 5: 3.0000}
    )

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": trim_curve,
            "Rotation": (1.5708, 0.0000, 2.5220),
            "Scale": (5.2000, 0.5000, 7.8000),
        },
    )

    curve_circle_3 = nw.new_node(Nodes.CurveCircle, input_kwargs={"Radius": 0.0300})

    curve_to_mesh_2 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": transform_geometry_6,
            "Profile Curve": curve_circle_3.outputs["Curve"],
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            0: group_input.outputs["Switch"],
            1: transform_geometry_1,
            2: curve_to_mesh_2,
        },
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": switch,
            "Profile Curve": curve_circle_1.outputs["Curve"],
        },
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    greater_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: -0.0100},
        attrs={"operation": "GREATER_THAN"},
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={0: group_input.outputs["Switch"], 1: greater_than, 2: 1.0000},
        attrs={"input_type": "FLOAT"},
    )

    separate_geometry = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={
            "Geometry": curve_to_mesh_1,
            "Selection": switch_1.outputs["Output"],
        },
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": 1.0000, "Y": 1.0000, "Z": group_input.outputs["tap_head"]},
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            0: group_input.outputs["Switch"],
            1: combine_xyz,
            2: (1.0000, 1.0000, 1.0000),
        },
        attrs={"input_type": "VECTOR"},
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": separate_geometry.outputs["Selection"],
            "Translation": (0.0000, 0.0000, 0.6000),
            "Scale": switch_2,
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [curve_to_mesh, transform_geometry_2]},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["roation_z"]}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": 1.0000, "Y": 1.0000, "Z": group_input.outputs["tap_height"]},
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry,
            "Rotation": combine_xyz_1,
            "Scale": combine_xyz_2,
        },
    )

    handle = nw.new_node(nodegroup_handle().name)

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": handle,
            "Translation": (0.0000, -0.2000, 0.0000),
            "Rotation": (0.0000, 0.0000, 3.6652),
            "Scale": (0.3000, 0.3000, 0.3000),
        },
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": handle,
            "Translation": (0.0000, 0.2000, 0.0000),
            "Rotation": (0.0000, 0.0000, 2.6180),
            "Scale": (0.3000, 0.3000, 0.3000),
        },
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_4, transform_geometry_3]},
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 41,
            "Side Segments": 39,
            "Radius": 0.0300,
            "Depth": 0.1000,
        },
    )

    transform_geometry_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": (0.0000, 0.0500, 0.1000),
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    switch_5 = nw.new_node(
        Nodes.Switch,
        input_kwargs={0: group_input.outputs["one_side"], 1: transform_geometry_7},
    )

    transform_geometry_8 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": (0.0000, -0.0500, 0.1000),
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [switch_5, transform_geometry_8]},
    )

    cylinder_1 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={
            "Vertices": 41,
            "Side Segments": 39,
            "Radius": 0.0050,
            "Depth": 0.1000,
        },
    )

    transform_geometry_9 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": (0.0000, 0.0800, 0.1500),
            "Scale": (1.0000, 1.0000, 1.1000),
        },
    )

    switch_4 = nw.new_node(
        Nodes.Switch,
        input_kwargs={0: group_input.outputs["one_side"], 1: transform_geometry_9},
    )

    transform_geometry_10 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_1.outputs["Mesh"],
            "Translation": (0.0000, -0.0800, 0.1500),
            "Rotation": (0.0000, 0.0000, 0.0855),
            "Scale": (1.0000, 1.0000, 1.1000),
        },
    )

    transform_geometry_17 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_10,
            "Translation": (0.0000, -0.0100, -0.0050),
            "Scale": (4.1000, 1.0000, 1.0000),
        },
    )

    switch_8 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            0: group_input.outputs["length_one_side"],
            1: transform_geometry_10,
            2: transform_geometry_17,
        },
    )

    switch_7 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            0: group_input.outputs["one_side"],
            1: transform_geometry_10,
            2: switch_8,
        },
    )

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [switch_4, switch_7]},
    )

    join_geometry_5 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [join_geometry_3, join_geometry_4]},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["hands_length_x"],
            "Y": group_input.outputs["hands_length_Y"],
            "Z": 1.0000,
        },
    )

    transform_geometry_11 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_5, "Scale": combine_xyz_4},
    )

    switch_3 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            0: group_input.outputs["hand_type"],
            1: join_geometry_2,
            2: transform_geometry_11,
        },
    )

    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={"Radius": 0.0500})

    fill_curve = nw.new_node(
        Nodes.FillCurve, input_kwargs={"Curve": curve_circle.outputs["Curve"]}
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh, input_kwargs={"Mesh": fill_curve, "Offset Scale": 0.1500}
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                transform_geometry_5,
                switch_3,
                extrude_mesh.outputs["Mesh"],
            ]
        },
    )

    bezier_segment_1 = nw.new_node(
        Nodes.CurveBezierSegment,
        input_kwargs={
            "Resolution": 54,
            "Start": (0.0000, 0.0000, 0.0000),
            "Start Handle": (0.0000, 0.0000, 0.7000),
            "End Handle": (0.2000, 0.0000, 0.7000),
            "End": (1.0000, 0.0000, 0.9000),
        },
    )

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    float_curve = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": spline_parameter.outputs["Factor"]}
    )
    node_utils.assign_curve(
        float_curve.mapping.curves[0],
        [(0.0000, 0.9750), (0.6295, 0.4125), (1.0000, 0.1625)],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: float_curve, 1: 1.3000},
        attrs={"operation": "MULTIPLY"},
    )

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius,
        input_kwargs={"Curve": bezier_segment_1, "Radius": multiply},
    )

    curve_circle_4 = nw.new_node(Nodes.CurveCircle, input_kwargs={"Radius": 0.1000})

    curve_to_mesh_3 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius,
            "Profile Curve": curve_circle_4.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_1})

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": separate_xyz_1.outputs["X"],
            1: 0.2000,
            3: 1.0000,
            4: 2.5000,
        },
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Y"], 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz_1.outputs["X"],
            "Y": multiply_1,
            "Z": separate_xyz_1.outputs["Z"],
        },
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": curve_to_mesh_3,
            "Position": combine_xyz_5,
            "Offset": (0.0000, 0.0000, 0.0000),
        },
    )

    subdivision_surface = nw.new_node(
        Nodes.SubdivisionSurface, input_kwargs={"Mesh": set_position, "Level": 1}
    )

    set_shade_smooth = nw.new_node(
        Nodes.SetShadeSmooth, input_kwargs={"Geometry": subdivision_surface}
    )

    transform_geometry_12 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": set_shade_smooth,
            "Translation": (0.0000, 0.0000, 0.1000),
            "Rotation": (0.0000, 0.0000, 0.6807),
            "Scale": (0.4000, 0.4000, 0.3000),
        },
    )

    curve_circle_5 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Resolution": 307, "Radius": 0.0550}
    )

    fill_curve_2 = nw.new_node(
        Nodes.FillCurve, input_kwargs={"Curve": curve_circle_5.outputs["Curve"]}
    )

    extrude_mesh_2 = nw.new_node(
        Nodes.ExtrudeMesh, input_kwargs={"Mesh": fill_curve_2, "Offset Scale": 0.1500}
    )

    cylinder_2 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 100, "Radius": 0.0100, "Depth": 0.7000},
    )

    set_position_1 = nw.new_node(
        Nodes.SetPosition, input_kwargs={"Geometry": cylinder_2.outputs["Mesh"]}
    )

    transform_geometry_13 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": set_position_1,
            "Translation": (0.3000, 0.0000, 0.2500),
            "Rotation": (0.0000, -2.0420, 0.0000),
            "Scale": (1.7000, 3.1000, 1.0000),
        },
    )

    cylinder_3 = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 318, "Radius": 0.0200, "Depth": 0.0300},
    )

    transform_geometry_14 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder_3.outputs["Mesh"],
            "Translation": (0.5950, 0.0000, 0.3800),
        },
    )

    join_geometry_7 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_geometry_13, transform_geometry_14]},
    )

    transform_geometry_15 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_7, "Scale": (0.9000, 1.0000, 1.0000)},
    )

    join_geometry_8 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                transform_geometry_12,
                extrude_mesh_2.outputs["Mesh"],
                transform_geometry_15,
            ]
        },
    )

    transform_geometry_16 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_8,
            "Rotation": (0.0000, 0.0000, 3.1416),
        },
    )

    switch_6 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            0: group_input.outputs["different_type"],
            1: join_geometry_1,
            2: transform_geometry_16,
        },
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": group_input.outputs["base_width"], "Height": 0.7000},
    )

    fillet_curve = nw.new_node(
        Nodes.FilletCurve,
        input_kwargs={
            "Curve": quadrilateral,
            "Count": 19,
            "Radius": group_input.outputs["base_radius"],
        },
        attrs={"mode": "POLY"},
    )

    fill_curve_1 = nw.new_node(Nodes.FillCurve, input_kwargs={"Curve": fillet_curve})

    extrude_mesh_1 = nw.new_node(
        Nodes.ExtrudeMesh, input_kwargs={"Mesh": fill_curve_1, "Offset Scale": 0.0500}
    )

    join_geometry_6 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [switch_6, extrude_mesh_1.outputs["Mesh"]]},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": join_geometry_6,
            "Material": group_input.outputs["Tap"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_material},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_sink_geometry", singleton=False, type="GeometryNodeTree"
)
def nodegroup_sink_geometry(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Width", 2.0000),
            ("NodeSocketFloat", "Depth", 2.0000),
            ("NodeSocketFloat", "Curvature", 0.9500),
            ("NodeSocketFloat", "Upper Height", 1.0000),
            ("NodeSocketFloat", "Lower Height", -0.0500),
            ("NodeSocketFloat", "HoleRadius", 0.1000),
            ("NodeSocketFloat", "Margin", 0.5000),
            ("NodeSocketFloat", "WaterTapMargin", 0.5000),
            ("NodeSocketMaterial", "Tap", None),
            ("NodeSocketMaterial", "Sink", None),
        ],
    )

    reroute_3 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Depth"]}
    )

    reroute_2 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": reroute_3, "Height": reroute_2},
    )

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_3, 1: reroute_2},
        attrs={"operation": "MINIMUM"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum, 1: 0.1000},
        attrs={"operation": "MULTIPLY"},
    )

    # inside of sink curve
    sink_interior_border = nw.new_node(
        "GeometryNodeFilletCurve",
        input_kwargs={"Curve": quadrilateral, "Count": 50, "Radius": multiply},
        attrs={"mode": "POLY"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Curvature"],
            "Y": group_input.outputs["Curvature"],
        },
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": sink_interior_border, "Scale": combine_xyz_1},
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Radius": group_input.outputs["HoleRadius"]}
    )

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_1, curve_circle.outputs["Curve"]]},
    )

    fill_curve_1 = nw.new_node(Nodes.FillCurve, input_kwargs={"Curve": join_geometry_4})

    # fill_curve_1 = tagging.tag_nodegroup(nw, fill_curve_1, t.Subpart.SupportSurface)

    reroute = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Lower Height"]}
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute})

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": fill_curve_1, "Translation": combine_xyz_2},
    )

    extrude_mesh_2 = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": transform_2,
            "Offset Scale": -0.0100,
            "Individual": False,
        },
    )

    transform_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_circle.outputs["Curve"],
            "Scale": (0.7000, 0.7000, 1.0000),
        },
    )

    join_geometry_6 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [curve_circle.outputs["Curve"], transform_5]},
    )

    fill_curve_4 = nw.new_node(Nodes.FillCurve, input_kwargs={"Curve": join_geometry_6})

    add = nw.new_node(Nodes.Math, input_kwargs={0: reroute, 1: -0.0100})

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add})

    transform_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": fill_curve_4, "Translation": combine_xyz_4},
    )

    extrude_mesh_4 = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": transform_6,
            "Offset Scale": group_input.outputs["Lower Height"],
            "Individual": False,
        },
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Lower Height"], 1: -0.0100}
    )

    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add_1})

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz_6})

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": curve_circle.outputs["Curve"],
        },
    )

    transform_7 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": curve_to_mesh, "Translation": combine_xyz_2},
    )

    join_geometry_5 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                extrude_mesh_2.outputs["Mesh"],
                transform_2,
                extrude_mesh_4.outputs["Mesh"],
                transform_7,
            ]
        },
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": sink_interior_border,
            "Scale": (0.9900, 0.9900, 1.0000),
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform, sink_interior_border]}
    )

    fill_curve = nw.new_node(Nodes.FillCurve, input_kwargs={"Curve": join_geometry})

    extrude_mesh_1 = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": fill_curve,
            "Offset Scale": group_input.outputs["Lower Height"],
        },
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    less_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: 0.0000},
        attrs={"operation": "LESS_THAN"},
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_1})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_1.outputs["X"],
            1: group_input.outputs["Curvature"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_1.outputs["Y"],
            1: group_input.outputs["Curvature"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": multiply_1,
            "Y": multiply_2,
            "Z": separate_xyz_1.outputs["Z"],
        },
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": extrude_mesh_1.outputs["Mesh"],
            "Selection": less_than,
            "Position": combine_xyz,
        },
    )

    add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Width"],
            1: group_input.outputs["Margin"],
        },
    )

    add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Depth"],
            1: group_input.outputs["Margin"],
        },
    )

    add_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_3, 1: group_input.outputs["WaterTapMargin"]}
    )

    quadrilateral_1 = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={"Width": add_4, "Height": add_2},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["WaterTapMargin"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_3})

    transform_8 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": quadrilateral_1, "Translation": combine_xyz_7},
    )

    fillet_curve_1 = nw.new_node(
        "GeometryNodeFilletCurve",
        input_kwargs={"Curve": transform_8, "Count": 10, "Radius": multiply},
        attrs={"mode": "POLY"},
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [sink_interior_border, fillet_curve_1]},
    )

    fill_curve_2 = nw.new_node(Nodes.FillCurve, input_kwargs={"Curve": join_geometry_2})

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Lower Height"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    add_5 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Upper Height"], 1: multiply_4}
    )

    extrude_mesh_3 = nw.new_node(
        Nodes.ExtrudeMesh, input_kwargs={"Mesh": fill_curve_2, "Offset Scale": add_5}
    )

    reroute_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Lower Height"]}
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_1})

    transform_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": extrude_mesh_3.outputs["Mesh"],
            "Translation": combine_xyz_3,
        },
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": transform_3}
    )

    # watertap = nw.new_node(nodegroup_water_tap().name, input_kwargs={'Tap': group_input.outputs['Tap']})

    add_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Depth"],
            1: group_input.outputs["WaterTapMargin"],
        },
    )

    multiply_5 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_6, 1: -0.5000}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": multiply_5, "Z": group_input.outputs["Upper Height"]},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                join_geometry_5,
                set_position,
                join_geometry_3,
            ]  # , transform_geometry]
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Material": group_input.outputs["Sink"],
        },
    )

    add_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["WaterTapMargin"],
            1: group_input.outputs["Margin"],
        },
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: add_7, 1: 2.5600}, attrs={"operation": "DIVIDE"}
    )

    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": divide})

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={"Geometry": set_material, "Offset": combine_xyz_8},
    )

    # region CREATE CUTTER (manually added by araistrick post-fact)

    sink_interior_border_simplified = nw.new_node(
        "GeometryNodeFilletCurve",
        input_kwargs={"Curve": quadrilateral, "Count": 3, "Radius": multiply},
        attrs={"mode": "POLY"},
    )

    scaled_sink_interior_border = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": sink_interior_border_simplified,
            "Scale": (1.01, 1.01, 1),  # scale it up just a little to avoid zclip
        },
    )

    fill_interior = nw.new_node(
        Nodes.FillCurve,
        input_kwargs={"Curve": scaled_sink_interior_border},
        attrs={"mode": "NGONS"},
    )

    extrude_amt = nw.scalar_add(
        group_input.outputs["Lower Height"], group_input.outputs["Upper Height"], 0.05
    )
    extrude = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={"Mesh": fill_interior, "Offset Scale": extrude_amt},
    )

    # same translation as set_position_1, to keep it in sync
    setpos_move_cutter = nw.new_node(
        Nodes.SetPosition, input_kwargs={"Geometry": extrude, "Offset": combine_xyz_8}
    )

    # endregion

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_position_1, "Cutter": setpos_move_cutter},
    )


def geometry_node_to_bbox(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler
    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": group_input.outputs["Geometry"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": bounding_box, "Scale": (0.100, 0.100, 0.1000)},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry},
        attrs={"is_active_output": True},
    )
