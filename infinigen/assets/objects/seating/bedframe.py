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
from infinigen.assets.objects.seating.chairs.chair import ChairFactory, ChairParameters
from infinigen.assets.objects.seating.mattress import make_coiled
from infinigen.assets.utils.decorate import (
    read_co,
    read_normal,
    remove_faces,
    select_faces,
    subdivide_edge_ring,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.object import join_objects, new_grid
from infinigen.assets.utils.shapes import dissolve_limited
from infinigen.core import surface
from infinigen.core.surface import NoApply
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform, weighted_sample
from infinigen.core.util.random import random_general as rg


class BedFrameParameters(ChairParameters):
    width: Annotated[float, Field(ge=1.4, le=2.4, json_schema_extra={"editable": True})]
    size: Annotated[float, Field(ge=2.0, le=2.4, json_schema_extra={"editable": True})]
    thickness: Annotated[float, Field(ge=0.05, le=0.12, json_schema_extra={"editable": True})]
    leg_thickness: Annotated[
        float, Field(ge=0.08, le=0.12, json_schema_extra={"editable": True})
    ]
    leg_height: Annotated[float, Field(ge=0.2, le=0.6, json_schema_extra={"editable": True})]
    back_height: Annotated[float, Field(ge=0.5, le=1.3, json_schema_extra={"editable": True})]
    has_all_legs_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    leg_decor_type: Literal["coiled", "pad", "plain", "legs", "none"] = Field(
        json_schema_extra={"editable": False}
    )
    leg_decor_wrapped_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    seat_subdivisions_x: Annotated[int, Field(ge=1, le=3, json_schema_extra={"editable": True})]
    seat_subdivisions_y: Annotated[
        float, Field(ge=4.0, le=10.0, json_schema_extra={"editable": True})
    ]
    dot_distance: Annotated[float, Field(ge=0.16, le=0.2, json_schema_extra={"editable": True})]
    dot_size: Annotated[float, Field(ge=0.005, le=0.02, json_schema_extra={"editable": True})]
    dot_depth: Annotated[float, Field(ge=0.04, le=0.08, json_schema_extra={"editable": True})]
    panel_distance: Annotated[float, Field(ge=0.3, le=0.5, json_schema_extra={"editable": True})]
    panel_margin: Annotated[float, Field(ge=0.01, le=0.02, json_schema_extra={"editable": True})]
    leg_trim_draw: Annotated[
        float, Field(ge=0.7, le=0.9, json_schema_extra={"editable": True})
    ] = 0.8
    divide_z_scale_draw: Annotated[
        float, Field(ge=0.5, le=1.0, json_schema_extra={"editable": True})
    ] = 0.75


class BedFrameFactory(ChairFactory):
    parameters_model: ClassVar[type[ChairParameters]] = BedFrameParameters
    scale = 1.0
    leg_decor_types = (
        "weighted_choice",
        (2, "coiled"),
        (2, "pad"),
        (1, "plain"),
        (2, "legs"),
    )
    back_types = (
        "weighted_choice",
        (3, "coiled"),
        (3, "pad"),
        (2, "whole"),
        (1, "horizontal-bar"),
        (1, "vertical-bar"),
    )

    def __init__(self, factory_seed, coarse=False):
        super(ChairFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> BedFrameParameters:
        width = log_uniform(1.4, 2.4)
        size = uniform(2, 2.4)
        thickness = uniform(0.05, 0.12)
        bevel_width = thickness * (0.1 if uniform() < 0.4 else 0.5)
        back_height = uniform(0.5, 1.3)
        arm_thickness = uniform(0.04, 0.06)
        limb_surface_gen_class = weighted_sample(material_assignments.furniture_leg)
        limb_surface_material_gen = limb_surface_gen_class()
        surface_gen_class = weighted_sample(material_assignments.bedframe)
        surface_material_gen = surface_gen_class()
        surface_mat = surface_material_gen()
        panel_surface_same_draw = uniform()
        panel_surface = (
            surface_mat
            if panel_surface_same_draw < 0.3
            else weighted_sample(material_assignments.furniture_hard_surface)()()
        )
        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch_fn, edge_wear_fn = material_assignments.wear_tear
        scratch_draw = uniform()
        edge_wear_draw = uniform()
        return BedFrameParameters(
            seed=seed,
            width=width,
            size=size,
            thickness=thickness,
            bevel_width=bevel_width,
            seat_back=1.0,
            seat_mid=uniform(0.7, 0.8),
            seat_mid_x=1.0,
            seat_mid_z=uniform(0, 0.5),
            seat_front=uniform(1.0, 1.2),
            is_seat_round_draw=uniform(),
            is_seat_subsurf_draw=uniform(),
            leg_thickness=uniform(0.08, 0.12),
            limb_profile=uniform(1.5, 2.5),
            leg_height=uniform(0.2, 0.6),
            back_height=back_height,
            is_leg_round_draw=uniform(),
            leg_type="vertical",
            has_leg_x_bar_draw=uniform(),
            has_leg_y_bar_draw=uniform(),
            leg_offset_bar_low=uniform(0.2, 0.4),
            leg_offset_bar_high=uniform(0.6, 0.8),
            has_arm_draw=1.0,
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
            has_all_legs_draw=uniform(),
            leg_decor_type=rg(self.leg_decor_types),
            leg_decor_wrapped_draw=uniform(),
            seat_subdivisions_x=int(np.random.randint(1, 4)),
            seat_subdivisions_y=int(log_uniform(4, 10)),
            dot_distance=log_uniform(0.16, 0.2),
            dot_size=uniform(0.005, 0.02),
            dot_depth=uniform(0.04, 0.08),
            panel_distance=uniform(0.3, 0.5),
            panel_margin=uniform(0.01, 0.02),
        )

    def _sample_spawn_parameters(
        self, params: BedFrameParameters, seed: int, i: int
    ) -> BedFrameParameters:
        params = super()._sample_spawn_parameters(params, seed, i)
        return params.model_copy(
            update={
                "leg_trim_draw": uniform(0.7, 0.9),
                "divide_z_scale_draw": uniform(0.5, 1.0),
            }
        )

    def apply_parameters(
        self, params: BedFrameParameters, *, spawn_scope: bool = True
    ) -> None:
        super().apply_parameters(params, spawn_scope=spawn_scope)
        self.has_all_legs = params.has_all_legs_draw < 0.2
        self.leg_decor_type = params.leg_decor_type
        self.leg_decor_wrapped = params.leg_decor_wrapped_draw < 0.5
        self.seat_subdivisions_x = params.seat_subdivisions_x
        self.seat_subdivisions_y = int(params.seat_subdivisions_y)
        self.dot_distance = params.dot_distance
        self.dot_size = params.dot_size
        self.dot_depth = params.dot_depth
        self.panel_distance = params.panel_distance
        self.panel_margin = params.panel_margin
        self.has_arm = False
        self.leg_type = "vertical"
        self.seat_back = 1
        self.clothes_scatter = NoApply()
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self.leg_trim_draw = params.leg_trim_draw
            self.divide_z_scale_draw = params.divide_z_scale_draw

    def make_seat(self):
        obj = new_grid(
            x_subdivisions=self.seat_subdivisions_x,
            y_subdivisions=self.seat_subdivisions_y,
        )
        obj.scale = (
            (self.width - self.leg_thickness) / 2,
            (self.size - self.leg_thickness) / 2,
            1,
        )
        butil.apply_transform(obj, True)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.delete(type="ONLY_FACE")
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={"value": (0, 0, self.thickness)}
            )
        butil.modify_mesh(
            obj,
            "SOLIDIFY",
            thickness=self.leg_thickness - 1e-3,
            offset=0,
            solidify_mode="NON_MANIFOLD",
        )
        obj.location = 0, -self.size / 2, -self.thickness / 2
        butil.apply_transform(obj, True)
        butil.modify_mesh(obj, "BEVEL", width=self.bevel_width, segments=8)
        return obj

    def make_legs(self):
        legs = super().make_legs()
        if self.has_all_legs:
            leg_starts = np.array(
                [[-1, -0.5, 0], [0, -1, 0], [0, 0, 0], [1, -0.5, 0]]
            ) * np.array([[self.width / 2, self.size, 0]])
            leg_ends = leg_starts.copy()
            leg_ends[0, 0] -= self.leg_x_offset
            leg_ends[3, 0] += self.leg_x_offset
            leg_ends[2, 1] += self.leg_y_offset[0]
            leg_ends[1, 1] -= self.leg_y_offset[1]
            leg_ends[:, -1] = -self.leg_height
            legs += self.make_limb(leg_ends, leg_starts)
        return legs

    def make_leg_decors(self, legs):
        if self.leg_decor_type == "none":
            return super().make_leg_decors(legs)
        obj = join_objects([deep_clone_obj(_) for _ in legs])
        x, y, z = read_co(obj).T
        leg_trim = (
            self.leg_trim_draw
            if self._use_fixed_spawn_draws
            else uniform(0.7, 0.9)
        )
        z = np.maximum(z, -self.leg_height * leg_trim)
        write_co(obj, np.stack([x, y, z], -1))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.convex_hull()
            bpy.ops.mesh.normals_make_consistent(inside=False)
        remove_faces(obj, np.abs(read_normal(obj)[:, -1]) > 0.5)
        if self.leg_decor_wrapped:
            x, y, z = read_co(obj).T
            x[x < 0] -= self.leg_thickness / 2 + 1e-3
            x[x > 0] += self.leg_thickness / 2 + 1e-3
            y[y < -self.size / 2] -= self.leg_thickness / 2 + 1e-3
            y[y > -self.size / 2] += self.leg_thickness / 2 + 1e-3
            write_co(obj, np.stack([x, y, z], -1))
        dissolve_limited(obj)
        match self.leg_decor_type:
            case "coiled":
                self.divide(obj, self.dot_distance)
                make_coiled(obj, self.dot_distance, self.dot_depth, self.dot_size)
            case "pad":
                self.divide(obj, self.panel_distance)
                with butil.ViewportMode(obj, "EDIT"):
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.inset(
                        thickness=self.panel_margin,
                        depth=self.panel_margin,
                        use_individual=True,
                    )
                butil.modify_mesh(obj, "BEVEL", segments=4)
        write_attribute(obj, 1, "panel", "FACE")
        return [obj]

    def divide(self, obj, distance):
        for i, size in enumerate(obj.dimensions):
            axis = np.zeros(3)
            axis[i] = 1
            z_scale = (
                self.divide_z_scale_draw
                if self._use_fixed_spawn_draws
                else uniform(0.5, 1.0)
            )
            distance = distance if i != 2 else distance * z_scale
            subdivide_edge_ring(obj, int(np.ceil(size / distance)), axis)

    def make_back_decors(self, backs, finalize=True):
        decors = super().make_back_decors(backs)
        match self.back_type:
            case "coiled":
                obj = self.make_back(backs)
                self.divide(obj, self.dot_distance)
                make_coiled(obj, self.dot_distance, self.dot_depth, self.dot_size)
                obj.scale = (1 - 1e-3,) * 3
                write_attribute(obj, 1, "panel", "FACE")
                with butil.ViewportMode(decors[0], "EDIT"):
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.bisect(
                        plane_co=(0, 0, self.back_height),
                        plane_no=(0, 0, 1),
                        clear_inner=True,
                    )
                return [obj] + decors
            case "pad":
                obj = self.make_back(backs)
                self.divide(obj, self.panel_distance)
                with butil.ViewportMode(obj, "EDIT"):
                    select_faces(obj, np.abs(read_normal(obj)[:, 1]) > 0.5)
                    bpy.ops.mesh.inset(
                        thickness=self.panel_margin,
                        depth=self.panel_margin,
                        use_individual=True,
                    )
                butil.modify_mesh(obj, "BEVEL", segments=4)
                write_attribute(obj, 1, "panel", "FACE")
                obj.scale = (1 - 1e-3,) * 3
                with butil.ViewportMode(decors[0], "EDIT"):
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.bisect(
                        plane_co=(0, 0, self.back_height),
                        plane_no=(0, 0, 1),
                        clear_inner=True,
                    )
                return [obj] + decors
            case _:
                return decors

    def make_back(self, backs):
        obj = join_objects([deep_clone_obj(b) for b in backs])
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.convex_hull()
        butil.modify_mesh(
            obj,
            "SOLIDIFY",
            thickness=np.minimum(self.thickness, self.leg_thickness),
            offset=0,
        )
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
        return obj
