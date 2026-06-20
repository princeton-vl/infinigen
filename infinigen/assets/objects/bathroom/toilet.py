# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.decorate import (
    read_center,
    read_co,
    read_edge_center,
    read_edges,
    read_normal,
    select_edges,
    select_faces,
    select_vertices,
    subsurf,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.draw import align_bezier
from infinigen.assets.utils.object import join_objects, new_bbox, new_cube, new_cylinder
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed, normalize
from infinigen.core.util.random import log_uniform, weighted_sample


class ToiletParameters(AssetParameters):
    size: Annotated[float, Field(ge=0.4, le=0.5, json_schema_extra={"editable": True})]
    width_ratio: Annotated[float, Field(ge=0.7, le=0.8, json_schema_extra={"editable": True})]
    height_ratio: Annotated[float, Field(ge=0.8, le=0.9, json_schema_extra={"editable": True})]
    size_mid: Annotated[float, Field(ge=0.6, le=0.65, json_schema_extra={"editable": True})]
    curve_scale: tuple[float, float, float, float] = Field(
        json_schema_extra={"editable": False}
    )
    depth_ratio: Annotated[float, Field(ge=0.5, le=0.6, json_schema_extra={"editable": True})]
    tube_scale: Annotated[float, Field(ge=0.25, le=0.3, json_schema_extra={"editable": True})]
    thickness: Annotated[float, Field(ge=0.05, le=0.06, json_schema_extra={"editable": True})]
    extrude_height: Annotated[
        float, Field(ge=0.015, le=0.02, json_schema_extra={"editable": True})
    ]
    stand_depth_ratio: Annotated[
        float, Field(ge=0.85, le=0.95, json_schema_extra={"editable": True})
    ]
    stand_scale: Annotated[float, Field(ge=0.7, le=0.85, json_schema_extra={"editable": True})]
    bottom_offset: Annotated[float, Field(ge=0.5, le=1.5, json_schema_extra={"editable": True})]
    back_thickness_ratio: Annotated[
        float, Field(ge=0.0, le=0.8, json_schema_extra={"editable": True})
    ]
    back_size_ratio: Annotated[float, Field(ge=0.55, le=0.65, json_schema_extra={"editable": True})]
    back_scale: Annotated[float, Field(ge=0.8, le=1.0, json_schema_extra={"editable": True})]
    seat_thickness_ratio: Annotated[
        float, Field(ge=0.1, le=0.3, json_schema_extra={"editable": True})
    ]
    seat_size_ratio: Annotated[float, Field(ge=1.2, le=1.6, json_schema_extra={"editable": True})]
    has_seat_cut_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    tank_width_ratio: Annotated[float, Field(ge=1.0, le=1.2, json_schema_extra={"editable": True})]
    tank_height_ratio: Annotated[float, Field(ge=0.6, le=1.0, json_schema_extra={"editable": True})]
    tank_size_gap: Annotated[float, Field(ge=0.02, le=0.03, json_schema_extra={"editable": True})]
    tank_cap_height: Annotated[
        float, Field(ge=0.03, le=0.04, json_schema_extra={"editable": True})
    ]
    tank_cap_extrude_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    tank_cap_extrude_amount: Annotated[
        float, Field(ge=0.005, le=0.01, json_schema_extra={"editable": True})
    ] = 0.0075
    cover_rotation: Annotated[
        float, Field(ge=0.0, le=1.570796, json_schema_extra={"editable": True})
    ]
    hardware_type: Literal["button", "handle"] = Field(
        json_schema_extra={"editable": False}
    )
    hardware_cap: Annotated[
        float, Field(ge=0.01, le=0.015, json_schema_extra={"editable": True})
    ]
    hardware_radius: Annotated[
        float, Field(ge=0.015, le=0.02, json_schema_extra={"editable": True})
    ]
    hardware_length: Annotated[
        float, Field(ge=0.04, le=0.05, json_schema_extra={"editable": True})
    ]
    hardware_on_side_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    scratch_draw: Annotated[float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})]
    edge_wear_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    surface: Any = Field(json_schema_extra={"editable": False})
    hardware_surface: Any = Field(json_schema_extra={"editable": False})
    scratch: Any | None = Field(default=None, json_schema_extra={"editable": False})
    edge_wear: Any | None = Field(default=None, json_schema_extra={"editable": False})


class ToiletFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = ToiletParameters

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> ToiletParameters:
        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch_fn, edge_wear_fn = material_assignments.wear_tear
        scratch_draw = uniform()
        edge_wear_draw = uniform()
        tank_cap_extrude_draw = uniform()
        return ToiletParameters(
            seed=seed,
            size=uniform(0.4, 0.5),
            width_ratio=uniform(0.7, 0.8),
            height_ratio=uniform(0.8, 0.9),
            size_mid=uniform(0.6, 0.65),
            curve_scale=log_uniform(0.8, 1.2, 4),
            depth_ratio=uniform(0.5, 0.6),
            tube_scale=uniform(0.25, 0.3),
            thickness=uniform(0.05, 0.06),
            extrude_height=uniform(0.015, 0.02),
            stand_depth_ratio=uniform(0.85, 0.95),
            stand_scale=uniform(0.7, 0.85),
            bottom_offset=uniform(0.5, 1.5),
            back_thickness_ratio=uniform(0, 0.8),
            back_size_ratio=uniform(0.55, 0.65),
            back_scale=uniform(0.8, 1.0),
            seat_thickness_ratio=uniform(0.1, 0.3),
            seat_size_ratio=uniform(1.2, 1.6),
            has_seat_cut_draw=uniform(),
            tank_width_ratio=uniform(1.0, 1.2),
            tank_height_ratio=uniform(0.6, 1.0),
            tank_size_gap=uniform(0.02, 0.03),
            tank_cap_height=uniform(0.03, 0.04),
            tank_cap_extrude_draw=tank_cap_extrude_draw,
            tank_cap_extrude_amount=uniform(0.005, 0.01),
            cover_rotation=uniform(0, np.pi / 2),
            hardware_type=np.random.choice(["button", "handle"]),
            hardware_cap=uniform(0.01, 0.015),
            hardware_radius=uniform(0.015, 0.02),
            hardware_length=uniform(0.04, 0.05),
            hardware_on_side_draw=uniform(),
            scratch_draw=scratch_draw,
            edge_wear_draw=edge_wear_draw,
            surface=weighted_sample(material_assignments.ceramics)(),
            hardware_surface=weighted_sample(material_assignments.metal_neutral)(),
            scratch=None if scratch_draw > scratch_prob else scratch_fn(),
            edge_wear=None if edge_wear_draw > edge_wear_prob else edge_wear_fn(),
        )

    def apply_parameters(
        self, params: ToiletParameters, *, spawn_scope: bool = True
    ) -> None:
        self.size = params.size
        self.width = params.size * params.width_ratio
        self.height = params.size * params.height_ratio
        self.size_mid = params.size_mid
        self.curve_scale = params.curve_scale
        self.depth = params.size * params.depth_ratio
        self.tube_scale = params.tube_scale
        self.thickness = params.thickness
        self.extrude_height = params.extrude_height
        self.stand_depth = self.depth * params.stand_depth_ratio
        self.stand_scale = params.stand_scale
        self.bottom_offset = params.bottom_offset
        self.back_thickness = params.thickness * params.back_thickness_ratio
        self.back_size = params.size * params.back_size_ratio
        self.back_scale = params.back_scale
        self.seat_thickness = params.seat_thickness_ratio * params.thickness
        self.seat_size = params.thickness * params.seat_size_ratio
        self.has_seat_cut = params.has_seat_cut_draw < 0.1
        self.tank_width = self.width * params.tank_width_ratio
        self.tank_height = self.height * params.tank_height_ratio
        self.tank_size = self.back_size - self.seat_size - params.tank_size_gap
        self.tank_cap_height = params.tank_cap_height
        self.tank_cap_extrude = (
            0
            if params.tank_cap_extrude_draw < 0.5
            else params.tank_cap_extrude_amount
        )
        self.cover_rotation = -params.cover_rotation
        self.hardware_type = params.hardware_type
        self.hardware_cap = params.hardware_cap
        self.hardware_radius = params.hardware_radius
        self.hardware_length = params.hardware_length
        self.hardware_on_side = params.hardware_on_side_draw < 0.5
        self.surface = params.surface
        self.hardware_surface = params.hardware_surface
        self.scratch = params.scratch
        self.edge_wear = params.edge_wear
        self._use_fixed_spawn_draws = spawn_scope

    @property
    def mid_offset(self):
        return (1 - self.size_mid) * self.size

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(
            -self.mid_offset - self.back_size - self.tank_cap_extrude,
            self.size_mid * self.size + self.thickness + self.thickness,
            -self.width / 2 - self.thickness * 1.1,
            self.width / 2 + self.thickness * 1.1,
            -self.height,
            max(
                self.tank_height,
                -np.sin(self.cover_rotation)
                * (self.seat_size + self.size + self.thickness + self.thickness),
            ),
        )

    def create_asset(self, **params) -> bpy.types.Object:
        upper = self.build_curve()
        lower = deep_clone_obj(upper)
        lower.scale = [self.tube_scale] * 3
        lower.location = 0, self.tube_scale * self.mid_offset / 2, -self.depth
        butil.apply_transform(lower, True)
        bottom = deep_clone_obj(upper)
        bottom.scale = [self.stand_scale] * 3
        bottom.location = (
            0,
            self.tube_scale * (1 - self.size_mid) * self.size / 2 * self.bottom_offset,
            -self.height,
        )
        butil.apply_transform(bottom, True)

        obj = self.make_tube(lower, upper)
        seat, cover = self.make_seat(obj)
        stand = self.make_stand(obj, bottom)
        back = self.make_back(obj)
        tank = self.make_tank()
        butil.modify_mesh(obj, "BEVEL", segments=2)
        match self.hardware_type:
            case "button":
                hardware = self.add_button()
            case _:
                hardware = self.add_handle()
        write_attribute(hardware, 1, "hardware", "FACE")
        obj = join_objects([obj, seat, cover, stand, back, tank, hardware])
        obj.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(obj)
        return obj

    def build_curve(self):
        x_anchors = [0, self.width / 2, 0]
        y_anchors = [-self.size_mid * self.size, 0, self.mid_offset]
        axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([1, 0, 0])]
        obj = align_bezier([x_anchors, y_anchors, 0], axes, self.curve_scale)
        butil.modify_mesh(obj, "MIRROR", use_axis=(True, False, False))
        return obj

    def make_tube(self, lower, upper):
        obj = join_objects([upper, lower])
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.bridge_edge_loops(
                number_cuts=64,
                profile_shape_factor=uniform(0.1, 0.2),
                interpolation="SURFACE",
            )
        butil.modify_mesh(
            obj,
            "SOLIDIFY",
            thickness=self.thickness,
            offset=1,
            solidify_mode="NON_MANIFOLD",
            nonmanifold_boundary_mode="FLAT",
        )
        normal = read_normal(obj)
        select_faces(obj, normal[:, -1] > 0.9)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.extrude_region_move(
                TRANSFORM_OT_translate={
                    "value": (0, 0, self.thickness + self.extrude_height)
                }
            )
        x, y, z = read_co(obj).T
        write_co(obj, np.stack([x, y, np.clip(z, None, self.extrude_height)], -1))
        return obj

    def make_seat(self, obj):
        seat = self.make_plane(obj)
        cover = deep_clone_obj(seat)
        butil.modify_mesh(seat, "SOLIDIFY", thickness=self.extrude_height, offset=1)
        if self.has_seat_cut:
            cutter = new_cube()
            cutter.scale = [self.thickness] * 3
            cutter.location = 0, -self.thickness / 2 - self.size_mid * self.size, 0
            butil.apply_transform(cutter, True)
            butil.select_none()
            butil.modify_mesh(seat, "BOOLEAN", object=cutter, operation="DIFFERENCE")
            butil.delete(cutter)
        butil.modify_mesh(seat, "BEVEL", segments=2)

        x, y, _ = read_edge_center(cover).T
        i = np.argmin(np.abs(x) + np.abs(y))
        selection = np.full(len(x), False)
        selection[i] = True
        select_edges(cover, selection)
        with butil.ViewportMode(cover, "EDIT"):
            bpy.ops.mesh.loop_multi_select()
            bpy.ops.mesh.fill_grid()
        butil.modify_mesh(cover, "SOLIDIFY", thickness=self.extrude_height, offset=1)
        cover.location = [
            0,
            -self.mid_offset - self.seat_size + self.extrude_height / 2,
            -self.extrude_height / 2,
        ]
        butil.apply_transform(cover, True)
        cover.rotation_euler[0] = self.cover_rotation
        cover.location = [
            0,
            self.mid_offset + self.seat_size - self.extrude_height / 2,
            self.extrude_height * 1.5,
        ]
        butil.apply_transform(cover, True)
        butil.modify_mesh(cover, "BEVEL", segments=2)
        return seat, cover

    def make_plane(self, obj):
        select_faces(obj, lambda x, y, z: z > self.extrude_height * 2 / 3)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.duplicate_move()
            bpy.ops.mesh.separate(type="SELECTED")
        seat = next(o for o in bpy.context.selected_objects if o != obj)
        butil.select_none()
        select_vertices(seat, lambda x, y, z: y > self.mid_offset + self.seat_thickness)
        with butil.ViewportMode(seat, "EDIT"):
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={
                    "value": (0, self.seat_size + self.thickness * 2, 0)
                }
            )
        x, y, z = read_co(seat).T
        write_co(
            seat,
            np.stack([x, np.clip(y, None, self.mid_offset + self.seat_size), z], -1),
        )
        return seat

    def make_stand(self, obj, bottom):
        co = read_co(obj)[read_edges(obj).reshape(-1)].reshape(-1, 2, 3)
        horizontal = np.abs(normalize(co[:, 0] - co[:, 1])[:, -1]) < 0.1
        x, y, z = read_edge_center(obj).T
        under_depth = z < -self.stand_depth
        i = np.argmin(y - horizontal - under_depth)
        selection = np.full(len(co), False)
        selection[i] = True
        select_edges(obj, selection)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.loop_multi_select()
            bpy.ops.mesh.duplicate_move()
            bpy.ops.mesh.separate(type="SELECTED")
        stand = next(o for o in bpy.context.selected_objects if o != obj)
        stand = join_objects([stand, bottom])
        with butil.ViewportMode(stand, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.bridge_edge_loops(
                number_cuts=64,
                profile_shape_factor=uniform(0.0, 0.15),
            )
        return stand

    def make_back(self, obj):
        back = read_center(obj)[:, 1] > self.mid_offset - self.back_thickness
        back_facing = read_normal(obj)[:, 1] > 0.1
        butil.select_none()
        select_faces(obj, back & back_facing)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.duplicate_move()
            bpy.ops.mesh.separate(type="SELECTED")
        back = next(o for o in bpy.context.selected_objects if o != obj)
        butil.modify_mesh(back, "CORRECTIVE_SMOOTH")
        butil.select_none()
        with butil.ViewportMode(back, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={
                    "value": (0, self.back_size + self.thickness * 2, 0)
                }
            )
            bpy.ops.transform.resize(value=(self.back_scale, 1, 1))
            bpy.ops.mesh.edge_face_add()
        back.location[1] -= 0.01
        butil.apply_transform(back, True)
        x, y, z = read_co(back).T
        write_co(
            back,
            np.stack([x, np.clip(y, None, self.mid_offset + self.back_size), z], -1),
        )
        return back

    def make_tank(self):
        tank = new_cube()
        tank.scale = self.tank_width / 2, self.tank_size / 2, self.tank_height / 2
        tank.location = (
            0,
            self.mid_offset + self.back_size - self.tank_size / 2,
            self.tank_height / 2,
        )
        butil.apply_transform(tank, True)
        subsurf(tank, 2, True)
        butil.modify_mesh(tank, "BEVEL", segments=2)
        cap = new_cube()
        cap.scale = (
            self.tank_width / 2 + self.tank_cap_extrude,
            self.tank_size / 2 + self.tank_cap_extrude,
            self.tank_cap_height / 2,
        )
        cap.location = (
            0,
            self.mid_offset + self.back_size - self.tank_size / 2,
            self.tank_height,
        )
        butil.apply_transform(cap, True)
        butil.modify_mesh(
            cap, "BEVEL", width=uniform(0, self.extrude_height), segments=4
        )
        tank = join_objects([tank, cap])
        return tank

    def add_button(self):
        obj = new_cylinder()
        obj.scale = (
            self.hardware_radius,
            self.hardware_radius,
            self.tank_cap_height / 2 + 1e-3,
        )
        obj.location = (
            0,
            self.mid_offset + self.back_size - self.tank_size / 2,
            self.tank_height,
        )
        butil.apply_transform(obj, True)
        return obj

    def add_handle(self):
        obj = new_cylinder()
        obj.scale = self.hardware_radius, self.hardware_radius, self.hardware_cap
        obj.rotation_euler[0] = np.pi / 2
        butil.apply_transform(obj, True)
        lever = new_cylinder()
        lever.scale = (
            self.hardware_radius / 2,
            self.hardware_radius / 2,
            self.hardware_length,
        )
        lever.rotation_euler[1] = np.pi / 2
        lever.location = [
            -self.hardware_radius * uniform(0, 0.5),
            -self.hardware_cap,
            -self.hardware_radius * uniform(0, 0.5),
        ]
        butil.apply_transform(lever, True)
        obj = join_objects([obj, lever])
        if self.hardware_on_side:
            obj.location = [
                -self.tank_width / 2 + self.hardware_radius + uniform(0.01, 0.02),
                self.mid_offset + self.back_size - self.tank_size,
                self.tank_height - self.hardware_radius - uniform(0.02, 0.03),
            ]
        else:
            obj.location = [
                -self.tank_width / 2,
                self.mid_offset
                + self.back_size
                - self.tank_size
                + self.hardware_radius
                + uniform(0.01, 0.02),
                self.tank_height - self.hardware_radius - uniform(0.02, 0.03),
            ]
            obj.rotation_euler[-1] = -np.pi / 2
        butil.apply_transform(obj, True)
        butil.modify_mesh(obj, "BEVEL", width=uniform(0.005, 0.01), segments=2)
        return obj

    def finalize_assets(self, assets):
        self.surface.apply(assets, clear=True, metal_color="plain")
        self.hardware_surface.apply(assets, "hardware", metal_color="natural")
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
