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
    read_co,
    read_edge_center,
    read_edge_direction,
    remove_edges,
    remove_vertices,
    select_edges,
    solidify,
    subsurf,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.draw import align_bezier, bezier_curve
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import join_objects, new_bbox
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.surface import NoApply
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform, weighted_sample
from infinigen.core.util.random import random_general as rg


class ChairParameters(AssetParameters):
    width: Annotated[float, Field(ge=0.4, le=0.5, json_schema_extra={"editable": True})]
    size: Annotated[float, Field(ge=0.38, le=0.45, json_schema_extra={"editable": True})]
    thickness: Annotated[float, Field(ge=0.04, le=0.08, json_schema_extra={"editable": True})]
    bevel_width: Annotated[float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})]
    seat_back: Annotated[float, Field(ge=0.7, le=1.0, json_schema_extra={"editable": True})]
    seat_mid: Annotated[float, Field(ge=0.7, le=0.8, json_schema_extra={"editable": True})]
    seat_mid_x: Annotated[float, Field(ge=0.923382, le=1.0, json_schema_extra={"editable": True})]
    seat_mid_z: Annotated[float, Field(ge=0.0, le=0.5, json_schema_extra={"editable": True})]
    seat_front: Annotated[float, Field(ge=1.0, le=1.2, json_schema_extra={"editable": True})]
    is_seat_round_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    is_seat_subsurf_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    leg_thickness: Annotated[float, Field(ge=0.04, le=0.06, json_schema_extra={"editable": True})]
    limb_profile: Annotated[float, Field(ge=1.5, le=2.5, json_schema_extra={"editable": True})]
    leg_height: Annotated[float, Field(ge=0.45, le=0.5, json_schema_extra={"editable": True})]
    back_height: Annotated[float, Field(ge=0.4, le=0.5, json_schema_extra={"editable": True})]
    is_leg_round_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    leg_type: Literal["vertical", "straight", "up-curved", "down-curved"] = Field(
        json_schema_extra={"editable": False}
    )
    has_leg_x_bar_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    has_leg_y_bar_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    leg_offset_bar_low: Annotated[
        float, Field(ge=0.2, le=0.4, json_schema_extra={"editable": True})
    ]
    leg_offset_bar_high: Annotated[
        float, Field(ge=0.6, le=0.8, json_schema_extra={"editable": True})
    ]
    has_arm_draw: Annotated[float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})]
    arm_thickness: Annotated[float, Field(ge=0.04, le=0.06, json_schema_extra={"editable": True})]
    arm_height: Annotated[float, Field(ge=0.6, le=1.0, json_schema_extra={"editable": True})]
    arm_y: Annotated[float, Field(ge=0.8, le=1.0, json_schema_extra={"editable": True})]
    arm_z: Annotated[float, Field(ge=0.3, le=0.6, json_schema_extra={"editable": True})]
    arm_mid: tuple[float, float, float] = Field(json_schema_extra={"editable": False})
    arm_profile: tuple[float, float] = Field(json_schema_extra={"editable": False})
    back_thickness: Annotated[float, Field(ge=0.04, le=0.05, json_schema_extra={"editable": True})]
    back_type: Literal["whole", "partial", "horizontal-bar", "vertical-bar"] = Field(
        json_schema_extra={"editable": False}
    )
    back_vertical_cuts: Annotated[int, Field(ge=1, le=3, json_schema_extra={"editable": True})]
    back_partial_scale: Annotated[float, Field(ge=1.0, le=1.4, json_schema_extra={"editable": True})]
    panel_surface_same_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    scratch_draw: Annotated[float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})]
    edge_wear_draw: Annotated[float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})]
    leg_x_offset: float = Field(default=0.0, json_schema_extra={"editable": False})
    leg_y_offset: tuple[float, float] = Field(
        default=(0.0, 0.0), json_schema_extra={"editable": False}
    )
    back_x_offset: float = Field(default=0.0, json_schema_extra={"editable": False})
    back_y_offset: float = Field(default=0.0, json_schema_extra={"editable": False})
    back_profile: tuple[tuple[float, float], ...] = Field(
        default=((0.0, 1.0),), json_schema_extra={"editable": False}
    )
    smoothness: Annotated[float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})] = (
        0.0
    )
    profile_shape_factor: Annotated[
        float, Field(ge=0.0, le=0.4, json_schema_extra={"editable": True})
    ] = 0.0
    leg_x_bar_z_frac: float = Field(default=0.5, json_schema_extra={"editable": False})
    leg_y_bar_z_frac: float = Field(default=0.5, json_schema_extra={"editable": False})
    limb_surface: Any = Field(json_schema_extra={"editable": False})
    surface: Any = Field(json_schema_extra={"editable": False})
    panel_surface: Any = Field(json_schema_extra={"editable": False})
    scratch: Any | None = Field(default=None, json_schema_extra={"editable": False})
    edge_wear: Any | None = Field(default=None, json_schema_extra={"editable": False})


class ChairFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = ChairParameters

    back_types = (
        "weighted_choice",
        (1, "whole"),
        (1, "partial"),
        (1, "horizontal-bar"),
        (1, "vertical-bar"),
    )

    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> ChairParameters:
        width = uniform(0.4, 0.5)
        size = uniform(0.38, 0.45)
        thickness = uniform(0.04, 0.08)
        bevel_width = thickness * (0.1 if uniform() < 0.4 else 0.5)
        seat_back = uniform(0.7, 1.0) if uniform() < 0.75 else 1.0
        seat_mid = uniform(0.7, 0.8)
        seat_mid_x = uniform(seat_back + seat_mid * (1 - seat_back), 1)
        back_height = uniform(0.4, 0.5)
        arm_thickness = uniform(0.04, 0.06)
        limb_surface_gen_class = weighted_sample(material_assignments.furniture_leg)
        limb_surface_material_gen = limb_surface_gen_class()
        surface_gen_class = weighted_sample(material_assignments.furniture_hard_surface)
        surface_material_gen = surface_gen_class()
        surface_mat = surface_material_gen()
        panel_surface_same_draw = uniform()
        if panel_surface_same_draw < 0.3:
            panel_surface = surface_mat
        else:
            panel_surface = weighted_sample(material_assignments.furniture_hard_surface)()()
        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch_fn, edge_wear_fn = material_assignments.wear_tear
        scratch_draw = uniform()
        edge_wear_draw = uniform()
        return ChairParameters(
            seed=seed,
            width=width,
            size=size,
            thickness=thickness,
            bevel_width=bevel_width,
            seat_back=seat_back,
            seat_mid=seat_mid,
            seat_mid_x=seat_mid_x,
            seat_mid_z=uniform(0, 0.5),
            seat_front=uniform(1.0, 1.2),
            is_seat_round_draw=uniform(),
            is_seat_subsurf_draw=uniform(),
            leg_thickness=uniform(0.04, 0.06),
            limb_profile=uniform(1.5, 2.5),
            leg_height=uniform(0.45, 0.5),
            back_height=back_height,
            is_leg_round_draw=uniform(),
            leg_type=np.random.choice(
                ["vertical", "straight", "up-curved", "down-curved"]
            ),
            has_leg_x_bar_draw=uniform(),
            has_leg_y_bar_draw=uniform(),
            leg_offset_bar_low=uniform(0.2, 0.4),
            leg_offset_bar_high=uniform(0.6, 0.8),
            has_arm_draw=uniform(),
            arm_thickness=arm_thickness,
            arm_height=uniform(0.6, 1.0),
            arm_y=uniform(0.8, 1.0),
            arm_z=uniform(0.3, 0.6),
            arm_mid=(
                uniform(-0.03, 0.03),
                uniform(-0.03, 0.09),
                uniform(-0.09, 0.03),
            ),
            arm_profile=tuple(log_uniform(0.1, 3, 2)),
            back_thickness=uniform(0.04, 0.05),
            back_type=rg(self.back_types),
            back_vertical_cuts=int(np.random.randint(1, 4)),
            back_partial_scale=uniform(1, 1.4),
            panel_surface_same_draw=panel_surface_same_draw,
            scratch_draw=scratch_draw,
            edge_wear_draw=edge_wear_draw,
            limb_surface=limb_surface_material_gen(),
            surface=surface_mat,
            panel_surface=panel_surface,
            scratch=None if scratch_draw > scratch_prob else scratch_fn(),
            edge_wear=None if edge_wear_draw > edge_wear_prob else edge_wear_fn(),
        )

    def _sample_spawn_parameters(
        self, params: ChairParameters, seed: int, i: int
    ) -> ChairParameters:
        leg_offset_bar = (params.leg_offset_bar_low, params.leg_offset_bar_high)
        return params.model_copy(
            update={
                "smoothness": uniform(0, 1),
                "profile_shape_factor": uniform(0, 0.4),
                "leg_x_bar_z_frac": uniform(*leg_offset_bar),
                "leg_y_bar_z_frac": uniform(*leg_offset_bar),
            }
        )

    def sample_parameters(
        self, seed: int | None = None, *, i: int | None = None
    ) -> ChairParameters:
        params = super().sample_parameters(seed=seed, i=i)
        return self._materialize_post_init(params)

    def _materialize_post_init(self, params: ChairParameters) -> ChairParameters:
        self.apply_parameters(params, spawn_scope=False)
        with FixedSeed(params.seed):
            self.post_init()
        return params.model_copy(
            update={
                "leg_x_offset": self.leg_x_offset,
                "leg_y_offset": self.leg_y_offset,
                "back_x_offset": self.back_x_offset,
                "back_y_offset": self.back_y_offset,
                "back_profile": self.back_profile,
            }
        )

    def apply_parameters(
        self, params: ChairParameters, *, spawn_scope: bool = True
    ) -> None:
        self.width = params.width
        self.size = params.size
        self.thickness = params.thickness
        self.bevel_width = params.bevel_width
        self.seat_back = params.seat_back
        self.seat_mid = params.seat_mid
        self.seat_mid_x = params.seat_mid_x
        self.seat_mid_z = params.seat_mid_z
        self.seat_front = params.seat_front
        self.is_seat_round = params.is_seat_round_draw < 0.6
        self.is_seat_subsurf = params.is_seat_subsurf_draw < 0.5
        self.leg_thickness = params.leg_thickness
        self.limb_profile = params.limb_profile
        self.leg_height = params.leg_height
        self.back_height = params.back_height
        self.is_leg_round = params.is_leg_round_draw < 0.5
        self.leg_type = params.leg_type
        self.leg_x_offset = params.leg_x_offset
        self.leg_y_offset = params.leg_y_offset
        self.back_x_offset = params.back_x_offset
        self.back_y_offset = params.back_y_offset
        self.has_leg_x_bar = params.has_leg_x_bar_draw < 0.6
        self.has_leg_y_bar = params.has_leg_y_bar_draw < 0.6
        self.leg_offset_bar = (params.leg_offset_bar_low, params.leg_offset_bar_high)
        self.has_arm = params.has_arm_draw < 0.7
        self.arm_thickness = params.arm_thickness
        self.arm_height = params.arm_thickness * params.arm_height
        self.arm_y = params.arm_y * self.size
        self.arm_z = params.arm_z * self.back_height
        self.arm_mid = np.array(params.arm_mid)
        self.arm_profile = np.array(params.arm_profile)
        self.back_thickness = params.back_thickness
        self.back_type = params.back_type
        self.back_profile = list(params.back_profile)
        self.back_vertical_cuts = params.back_vertical_cuts
        self.back_partial_scale = params.back_partial_scale
        self.limb_surface = params.limb_surface
        self.surface = params.surface
        self.panel_surface = params.panel_surface
        self.scratch = params.scratch
        self.edge_wear = params.edge_wear
        self.clothes_scatter = NoApply()
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self.smoothness = params.smoothness
            self.profile_shape_factor = params.profile_shape_factor
            self.leg_x_bar_z_frac = params.leg_x_bar_z_frac
            self.leg_y_bar_z_frac = params.leg_y_bar_z_frac

    def generate(self, params: ChairParameters, i: int | None = None, **kwargs: Any):
        params = self._materialize_post_init(params)
        return super().generate(params, i=i, **kwargs)

    def post_init(self):
        with FixedSeed(self.factory_seed):
            if self.leg_type == "vertical":
                self.leg_x_offset = 0
                self.leg_y_offset = 0, 0
                self.back_x_offset = 0
                self.back_y_offset = 0
            else:
                self.leg_x_offset = self.width * uniform(0.05, 0.2)
                self.leg_y_offset = self.size * uniform(0.05, 0.2, 2)
                self.back_x_offset = self.width * uniform(-0.1, 0.15)
                self.back_y_offset = self.size * uniform(0.1, 0.25)

            match self.back_type:
                case "partial":
                    self.back_profile = ((uniform(0.4, 0.8), 1),)
                case "horizontal-bar":
                    n_cuts = np.random.randint(2, 4)
                    locs = uniform(1, 2, n_cuts).cumsum()
                    locs = locs / locs[-1]
                    ratio = uniform(0.5, 0.75)
                    locs = np.array(
                        [
                            (p + ratio * (l - p), l)
                            for p, l in zip([0, *locs[:-1]], locs)
                        ]
                    )
                    lowest = uniform(0, 0.4)
                    self.back_profile = locs * (1 - lowest) + lowest
                case "vertical-bar":
                    self.back_profile = ((uniform(0.8, 0.9), 1),)
                case _:
                    self.back_profile = [(0, 1)]

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        obj = new_bbox(
            -self.width / 2 - max(self.leg_x_offset, self.back_x_offset),
            self.width / 2 + max(self.leg_x_offset, self.back_x_offset),
            -self.size - self.leg_y_offset[1] - self.leg_thickness * 0.5,
            max(self.leg_y_offset[0], self.back_y_offset),
            -self.leg_height,
            self.back_height * 1.2,
        )
        obj.rotation_euler.z += np.pi / 2
        butil.apply_transform(obj)
        return obj

    def create_asset(self, **params) -> bpy.types.Object:
        obj = self.make_seat()
        legs = self.make_legs()
        backs = self.make_backs()

        parts = [obj] + legs + backs
        parts.extend(self.make_leg_decors(legs))
        if self.has_arm:
            parts.extend(self.make_arms(obj, backs))
        parts.extend(self.make_back_decors(backs))

        for obj in legs:
            self.solidify(obj, 2)
        for obj in backs:
            self.solidify(obj, 2, self.back_thickness)

        obj = join_objects(parts)
        obj.rotation_euler.z += np.pi / 2
        butil.apply_transform(obj)

        with FixedSeed(self.factory_seed):
            # TODO: wasteful to create unique materials for each individual asset
            # self.surface.apply(obj)

            # self.panel_surface.apply(obj, selection="panel")
            # self.limb_surface.apply(obj, selection="limb")
            surface.assign_material(obj, self.surface)
            surface.assign_material(obj, self.panel_surface, selection="panel")
            surface.assign_material(obj, self.limb_surface, selection="limb")

        return obj

    def finalize_assets(self, assets):
        pass
        # if self.scratch:
        #     self.scratch.apply(assets)
        # if self.edge_wear:
        #     self.edge_wear.apply(assets)

    def make_seat(self):
        x_anchors = (
            np.array(
                [
                    0,
                    0.1,
                    1,
                    self.seat_mid_x,
                    self.seat_back,
                    0,
                ]
            )
            * self.width
            / 2
        )
        y_anchors = (
            np.array([-self.seat_front, -self.seat_front, -1, -self.seat_mid, 0, 0])
            * self.size
        )
        z_anchors = np.array([0, 0, 0, self.seat_mid_z, 0, 0]) * self.thickness
        vector_locations = [4] if self.is_seat_round else [2, 4]
        obj = bezier_curve((x_anchors, y_anchors, z_anchors), vector_locations)
        butil.modify_mesh(obj, "MIRROR")
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.fill_grid(use_interp_simple=True)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=0)
        subsurf(obj, 1, not self.is_seat_subsurf)
        butil.modify_mesh(obj, "BEVEL", width=self.bevel_width, segments=8)
        return obj

    def make_legs(self):
        leg_starts = np.array(
            [[-self.seat_back, 0, 0], [-1, -1, 0], [1, -1, 0], [self.seat_back, 0, 0]]
        ) * np.array([[self.width / 2, self.size, 0]])
        leg_ends = leg_starts.copy()
        leg_ends[[0, 1], 0] -= self.leg_x_offset
        leg_ends[[2, 3], 0] += self.leg_x_offset
        leg_ends[[0, 3], 1] += self.leg_y_offset[0]
        leg_ends[[1, 2], 1] -= self.leg_y_offset[1]
        leg_ends[:, -1] = -self.leg_height
        return self.make_limb(leg_ends, leg_starts)

    def make_limb(self, leg_ends, leg_starts):
        limbs = []
        for leg_start, leg_end in zip(leg_starts, leg_ends):
            match self.leg_type:
                case "up-curved":
                    axes = [(0, 0, 1), None]
                    scale = [self.limb_profile, 1]
                case "down-curved":
                    axes = [None, (0, 0, 1)]
                    scale = [1, self.limb_profile]
                case _:
                    axes = None
                    scale = None
            limb = align_bezier(np.stack([leg_start, leg_end], -1), axes, scale)
            limb.location = (
                np.array(
                    [
                        1 if leg_start[0] < 0 else -1,
                        1 if leg_start[1] < -self.size / 2 else -1,
                        0,
                    ]
                )
                * self.leg_thickness
                / 2
            )
            butil.apply_transform(limb, True)
            limbs.append(limb)
        return limbs

    def make_backs(self):
        back_starts = (
            np.array([[-self.seat_back, 0, 0], [self.seat_back, 0, 0]]) * self.width / 2
        )
        back_ends = back_starts.copy()
        back_ends[:, 0] += np.array([self.back_x_offset, -self.back_x_offset])
        back_ends[:, 1] = self.back_y_offset
        back_ends[:, 2] = self.back_height
        return self.make_limb(back_starts, back_ends)

    def make_leg_decors(self, legs):
        decors = []
        if self.has_leg_x_bar:
            z_frac = (
                self.leg_x_bar_z_frac
                if self._use_fixed_spawn_draws
                else uniform(*self.leg_offset_bar)
            )
            z_height = -self.leg_height * z_frac
            locs = []
            for leg in legs:
                co = read_co(leg)
                locs.append(co[np.argmin(np.abs(co[:, -1] - z_height))])
            decors.append(
                self.solidify(bezier_curve(np.stack([locs[0], locs[3]], -1)), 0)
            )
            decors.append(
                self.solidify(bezier_curve(np.stack([locs[1], locs[2]], -1)), 0)
            )
        if self.has_leg_y_bar:
            z_frac = (
                self.leg_y_bar_z_frac
                if self._use_fixed_spawn_draws
                else uniform(*self.leg_offset_bar)
            )
            z_height = -self.leg_height * z_frac
            locs = []
            for leg in legs:
                co = read_co(leg)
                locs.append(co[np.argmin(np.abs(co[:, -1] - z_height))])
            decors.append(
                self.solidify(bezier_curve(np.stack([locs[0], locs[1]], -1)), 1)
            )
            decors.append(
                self.solidify(bezier_curve(np.stack([locs[2], locs[3]], -1)), 1)
            )
        for d in decors:
            write_attribute(d, 1, "limb", "FACE")
        return decors

    def make_back_decors(self, backs, finalize=True):
        obj = join_objects([deep_clone_obj(b) for b in backs])
        x, y, z = read_co(obj).T
        x += np.where(x > 0, self.back_thickness / 2, -self.back_thickness / 2)
        write_co(obj, np.stack([x, y, z], -1))
        smoothness = self.smoothness if self._use_fixed_spawn_draws else uniform(0, 1)
        profile_shape_factor = (
            self.profile_shape_factor
            if self._use_fixed_spawn_draws
            else uniform(0, 0.4)
        )
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            center = read_edge_center(obj)
            for z_min, z_max in self.back_profile:
                select_edges(
                    obj,
                    (z_min * self.back_height <= center[:, -1])
                    & (center[:, -1] <= z_max * self.back_height),
                )
                bpy.ops.mesh.bridge_edge_loops(
                    number_cuts=64,
                    interpolation="LINEAR",
                    smoothness=smoothness,
                    profile_shape_factor=profile_shape_factor,
                )
            bpy.ops.mesh.select_loose()
            bpy.ops.mesh.delete()
        butil.modify_mesh(
            obj,
            "SOLIDIFY",
            thickness=np.minimum(self.thickness, self.back_thickness),
            offset=0,
        )
        if finalize:
            butil.modify_mesh(obj, "BEVEL", width=self.bevel_width, segments=8)
        parts = [obj]
        if self.back_type == "vertical-bar":
            other = join_objects([deep_clone_obj(b) for b in backs])
            with butil.ViewportMode(other, "EDIT"):
                bpy.ops.mesh.select_mode(type="EDGE")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.bridge_edge_loops(
                    number_cuts=self.back_vertical_cuts,
                    interpolation="LINEAR",
                    smoothness=smoothness,
                    profile_shape_factor=profile_shape_factor,
                )
                bpy.ops.mesh.select_all(action="INVERT")
                bpy.ops.mesh.delete()
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.delete(type="ONLY_FACE")
            remove_edges(other, np.abs(read_edge_direction(other)[:, -1]) < 0.5)
            remove_vertices(other, lambda x, y, z: z < -self.thickness / 2)
            remove_vertices(
                other,
                lambda x, y, z: z
                > (self.back_profile[0][0] + self.back_profile[0][1])
                * self.back_height
                / 2,
            )
            parts.append(self.solidify(other, 2, self.back_thickness))
        elif self.back_type == "partial":
            co = read_co(obj)
            co[:, 1] *= self.back_partial_scale
            write_co(obj, co)
        for p in parts:
            write_attribute(p, 1, "panel", "FACE")
        return parts

    def make_arms(self, base, backs):
        co = read_co(base)
        end = co[np.argmin(co[:, 0] - (np.abs(co[:, 1] + self.arm_y) < 0.02))]
        end[0] += self.arm_thickness / 4
        end_ = end.copy()
        end_[0] = -end[0]
        arms = []
        co = read_co(backs[0])
        start = co[np.argmin(co[:, 0] - (np.abs(co[:, -1] - self.arm_z) < 0.02))]
        start[0] -= self.arm_thickness / 4
        start_ = start.copy()
        start_[0] = -start[0]
        for start, end in zip([start, start_], [end, end_]):
            mid = np.array(
                [
                    end[0] + self.arm_mid[0] * (-1 if end[0] > 0 else 1),
                    end[1] + self.arm_mid[1],
                    start[2] + self.arm_mid[2],
                ]
            )
            arm = align_bezier(
                np.stack([start, mid, end], -1),
                np.array(
                    [
                        [end[0] - start[0], end[1] - start[1], 0],
                        [0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                        [0, 0, 1],
                    ]
                ),
                [1, *self.arm_profile, 1],
            )
            if self.is_leg_round:
                surface.add_geomod(
                    arm,
                    geo_radius,
                    apply=True,
                    input_args=[self.arm_thickness / 2, 32],
                    input_kwargs={"to_align_tilt": False},
                )
            else:
                with butil.ViewportMode(arm, "EDIT"):
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.extrude_edges_move(
                        TRANSFORM_OT_translate={
                            "value": (
                                self.arm_thickness
                                if end[0] < 0
                                else -self.arm_thickness,
                                0,
                                0,
                            )
                        }
                    )
                butil.modify_mesh(arm, "SOLIDIFY", thickness=self.arm_height, offset=0)
            write_attribute(arm, 1, "limb", "FACE")
            arms.append(arm)
        return arms

    def solidify(self, obj, axis, thickness=None):
        if thickness is None:
            thickness = self.leg_thickness
        if self.is_leg_round:
            solidify(obj, axis, thickness)
            butil.modify_mesh(obj, "BEVEL", width=self.bevel_width, segments=8)
        else:
            surface.add_geomod(
                obj, geo_radius, apply=True, input_args=[thickness / 2, 32]
            )
        write_attribute(obj, 1, "limb", "FACE")
        return obj
