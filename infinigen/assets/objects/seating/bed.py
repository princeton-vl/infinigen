# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from functools import cached_property
from typing import Annotated, ClassVar, Literal

import bpy
import numpy as np
import trimesh
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.seating import bedframe, mattress, pillow
from infinigen.assets.objects.seating.chairs.chair import ChairParameters
from infinigen.assets.scatters import clothes
from infinigen.assets.utils.decorate import decimate, read_co, subsurf
from infinigen.assets.utils.object import obj2trimesh
from infinigen.core import surface
from infinigen.core.placement.parameters import AssetParameters
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.random import log_uniform, weighted_sample
from infinigen.core.util.random import random_general as rg


class BedParameters(ChairParameters):
    width: Annotated[float, Field(ge=1.4, le=2.4, json_schema_extra={"editable": True})]
    size: Annotated[float, Field(ge=2.0, le=2.4, json_schema_extra={"editable": True})]
    thickness: Annotated[
        float, Field(ge=0.05, le=0.12, json_schema_extra={"editable": True})
    ]
    back_height: Annotated[
        float, Field(ge=0.5, le=1.3, json_schema_extra={"editable": True})
    ]
    leg_thickness: Annotated[
        float, Field(ge=0.08, le=0.12, json_schema_extra={"editable": True})
    ]
    leg_height: Annotated[float, Field(ge=0.2, le=0.6, json_schema_extra={"editable": True})]
    has_all_legs_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    leg_decor_type: Literal["coiled", "pad", "plain", "legs"] = Field(
        json_schema_extra={"editable": False}
    )
    leg_decor_wrapped_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    seat_subdivisions_x: Annotated[
        int, Field(ge=1, le=3, json_schema_extra={"editable": True})
    ]
    seat_subdivisions_y: Annotated[
        int, Field(ge=4, le=10, json_schema_extra={"editable": True})
    ]
    dot_distance: Annotated[
        float, Field(ge=0.16, le=0.2, json_schema_extra={"editable": True})
    ]
    dot_size: Annotated[
        float, Field(ge=0.005, le=0.02, json_schema_extra={"editable": True})
    ]
    dot_depth: Annotated[
        float, Field(ge=0.04, le=0.08, json_schema_extra={"editable": True})
    ]
    panel_distance: Annotated[
        float, Field(ge=0.3, le=0.5, json_schema_extra={"editable": True})
    ]
    panel_margin: Annotated[
        float, Field(ge=0.01, le=0.02, json_schema_extra={"editable": True})
    ]
    sheet_type: Literal["quilt", "comforter", "box_comforter", "none"] = Field(
        json_schema_extra={"editable": False}
    )
    sheet_folded_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    has_cover_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    mattress_type: Literal["coiled", "wrapped"] = Field(
        json_schema_extra={"editable": False}
    )
    mattress_width_scale: Annotated[
        float, Field(ge=0.88, le=0.96, json_schema_extra={"editable": True})
    ]
    mattress_size_scale: Annotated[
        float, Field(ge=0.88, le=0.96, json_schema_extra={"editable": True})
    ]
    quilt_width_scale: Annotated[
        float, Field(ge=1.4, le=1.6, json_schema_extra={"editable": True})
    ]
    quilt_size_scale: Annotated[
        float, Field(ge=0.9, le=1.1, json_schema_extra={"editable": True})
    ]
    comforter_width_scale: Annotated[
        float, Field(ge=1.4, le=1.8, json_schema_extra={"editable": True})
    ]
    comforter_size_scale: Annotated[
        float, Field(ge=0.9, le=1.2, json_schema_extra={"editable": True})
    ]
    box_comforter_width_scale: Annotated[
        float, Field(ge=1.4, le=1.8, json_schema_extra={"editable": True})
    ]
    box_comforter_size_scale: Annotated[
        float, Field(ge=0.9, le=1.2, json_schema_extra={"editable": True})
    ]
    cover_width_scale: Annotated[
        float, Field(ge=1.6, le=1.8, json_schema_extra={"editable": True})
    ]
    cover_size_scale: Annotated[
        float, Field(ge=0.3, le=0.4, json_schema_extra={"editable": True})
    ]
    back_type: Literal[
        "coiled", "pad", "whole", "partial", "horizontal-bar", "vertical-bar"
    ] = Field(json_schema_extra={"editable": False})
    n_pillows: Annotated[int, Field(ge=2, le=3, json_schema_extra={"editable": True})] = (
        2
    )
    n_towels: Annotated[int, Field(ge=1, le=1, json_schema_extra={"editable": True})] = 1
    pillow_scatter_x: tuple[float, ...] = Field(
        default=(), json_schema_extra={"editable": False}
    )
    pillow_scatter_y: tuple[float, ...] = Field(
        default=(), json_schema_extra={"editable": False}
    )
    towel_scatter_x: tuple[float, ...] = Field(
        default=(), json_schema_extra={"editable": False}
    )
    towel_scatter_y: tuple[float, ...] = Field(
        default=(), json_schema_extra={"editable": False}
    )
    sheet_pressure: float = Field(default=0.0, json_schema_extra={"editable": False})
    sheet_location_offset: Annotated[
        float, Field(ge=0.0, le=0.15, json_schema_extra={"editable": True})
    ] = 0.0
    cover_location_offset: Annotated[
        float, Field(ge=0.0, le=0.3, json_schema_extra={"editable": True})
    ] = 0.0
    pillow_rotations: tuple[float, ...] = Field(
        default=(), json_schema_extra={"editable": False}
    )


class BedFactory(bedframe.BedFrameFactory):
    parameters_model: ClassVar[type[AssetParameters]] = BedParameters

    mattress_types = "weighted_choice", (1, "coiled"), (3, "wrapped")
    sheet_types = (
        "weighted_choice",
        (4, "quilt"),
        (4, "comforter"),
        (4, "box_comforter"),
        (1, "none"),
    )

    def __init__(self, factory_seed, coarse=False):
        super(bedframe.BedFrameFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> BedParameters:
        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch_fn, edge_wear_fn = material_assignments.wear_tear
        scratch_draw = uniform()
        edge_wear_draw = uniform()
        sheet_type = rg(self.sheet_types)
        return BedParameters(
            seed=seed,
            width=log_uniform(1.4, 2.4),
            size=uniform(2, 2.4),
            thickness=uniform(0.05, 0.12),
            bevel_width=uniform(0.02, 0.05),
            seat_back=1.0,
            seat_mid=0.75,
            seat_mid_x=1.0,
            seat_mid_z=0.0,
            seat_front=1.0,
            is_seat_round_draw=1.0,
            is_seat_subsurf_draw=1.0,
            leg_thickness=uniform(0.08, 0.12),
            limb_profile=2.0,
            leg_height=uniform(0.2, 0.6),
            back_height=uniform(0.5, 1.3),
            is_leg_round_draw=1.0,
            leg_type="vertical",
            has_leg_x_bar_draw=1.0,
            has_leg_y_bar_draw=1.0,
            leg_offset_bar_low=0.2,
            leg_offset_bar_high=0.8,
            has_arm_draw=1.0,
            arm_thickness=0.05,
            arm_height=0.8,
            arm_y=0.9,
            arm_z=0.45,
            arm_mid=(0.0, 0.0, 0.0),
            arm_profile=(1.0, 1.0),
            back_thickness=0.045,
            back_type=rg(bedframe.BedFrameFactory.back_types),
            back_vertical_cuts=2,
            back_partial_scale=1.2,
            panel_surface_same_draw=0.0,
            scratch_draw=scratch_draw,
            edge_wear_draw=edge_wear_draw,
            limb_surface=weighted_sample(material_assignments.furniture_hard_surface)()(),
            surface=weighted_sample(material_assignments.bedframe)()(),
            panel_surface=weighted_sample(material_assignments.furniture_hard_surface)()(),
            scratch=None if scratch_draw > scratch_prob else scratch_fn(),
            edge_wear=None if edge_wear_draw > edge_wear_prob else edge_wear_fn(),
            has_all_legs_draw=uniform(),
            leg_decor_type=rg(bedframe.BedFrameFactory.leg_decor_types),
            leg_decor_wrapped_draw=uniform(),
            seat_subdivisions_x=int(np.random.randint(1, 4)),
            seat_subdivisions_y=int(log_uniform(4, 10)),
            dot_distance=log_uniform(0.16, 0.2),
            dot_size=uniform(0.005, 0.02),
            dot_depth=uniform(0.04, 0.08),
            panel_distance=uniform(0.3, 0.5),
            panel_margin=uniform(0.01, 0.02),
            sheet_type=sheet_type,
            sheet_folded_draw=uniform(),
            has_cover_draw=uniform(),
            mattress_type=rg(self.mattress_types),
            mattress_width_scale=uniform(0.88, 0.96),
            mattress_size_scale=uniform(0.88, 0.96),
            quilt_width_scale=uniform(1.4, 1.6),
            quilt_size_scale=uniform(0.9, 1.1),
            comforter_width_scale=uniform(1.4, 1.8),
            comforter_size_scale=uniform(0.9, 1.2),
            box_comforter_width_scale=uniform(1.4, 1.8),
            box_comforter_size_scale=uniform(0.9, 1.2),
            cover_width_scale=uniform(1.6, 1.8),
            cover_size_scale=uniform(0.3, 0.4),
        )

    def _sample_spawn_parameters(
        self, params: BedParameters, seed: int, i: int
    ) -> BedParameters:
        n_pillows = int(np.random.randint(2, 4))
        sheet_pressure = (
            0.0
            if params.sheet_type == "quilt"
            else uniform(1.0, 1.5)
            if params.sheet_type == "comforter"
            else log_uniform(8, 15)
        )
        return params.model_copy(
            update={
                "n_pillows": n_pillows,
                "n_towels": 1,
                "pillow_scatter_x": tuple(uniform(0.1, 0.4, 10)),
                "pillow_scatter_y": tuple(uniform(-0.3, 0.3, 10)),
                "towel_scatter_x": tuple(uniform(0.5, 0.8, 10)),
                "towel_scatter_y": tuple(uniform(-0.3, 0.3, 10)),
                "sheet_pressure": sheet_pressure,
                "sheet_location_offset": uniform(0, 0.15),
                "cover_location_offset": uniform(0, 0.3),
                "pillow_rotations": tuple(uniform(0, np.pi, n_pillows)),
            }
        )

    def apply_parameters(
        self, params: BedParameters, *, spawn_scope: bool = True
    ) -> None:
        super().apply_parameters(params, spawn_scope=False)
        self.has_all_legs = params.has_all_legs_draw < 0.2
        self.leg_decor_type = params.leg_decor_type
        self.leg_decor_wrapped = params.leg_decor_wrapped_draw < 0.5
        self.seat_subdivisions_x = params.seat_subdivisions_x
        self.seat_subdivisions_y = params.seat_subdivisions_y
        self.has_arm = False
        self.leg_type = "vertical"
        self.leg_x_offset = 0
        self.leg_y_offset = (0, 0)
        self.back_x_offset = 0
        self.back_y_offset = 0
        self.seat_back = 1
        self.clothes_scatter = surface.NoApply
        self.dot_distance = params.dot_distance
        self.dot_size = params.dot_size
        self.dot_depth = params.dot_depth
        self.panel_distance = params.panel_distance
        self.panel_margin = params.panel_margin
        self.sheet_type = params.sheet_type
        self.sheet_folded = params.sheet_folded_draw < 0.5
        self.has_cover = params.has_cover_draw < 0.5
        self.mattress_type = params.mattress_type
        self.mattress_width_scale = params.mattress_width_scale
        self.mattress_size_scale = params.mattress_size_scale
        self.quilt_width_scale = params.quilt_width_scale
        self.quilt_size_scale = params.quilt_size_scale
        self.comforter_width_scale = params.comforter_width_scale
        self.comforter_size_scale = params.comforter_size_scale
        self.box_comforter_width_scale = params.box_comforter_width_scale
        self.box_comforter_size_scale = params.box_comforter_size_scale
        self.cover_width_scale = params.cover_width_scale
        self.cover_size_scale = params.cover_size_scale
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self.n_pillows = params.n_pillows
            self.n_towels = params.n_towels
            self.pillow_scatter_x = params.pillow_scatter_x
            self.pillow_scatter_y = params.pillow_scatter_y
            self.towel_scatter_x = params.towel_scatter_x
            self.towel_scatter_y = params.towel_scatter_y
            self.sheet_pressure = params.sheet_pressure
            self.sheet_location_offset = params.sheet_location_offset
            self.cover_location_offset = params.cover_location_offset
            self.pillow_rotations = params.pillow_rotations

    @cached_property
    def mattress_factory(self):
        factory = mattress.MattressFactory(self.factory_seed, self.coarse)
        factory.type = self.mattress_type
        factory.width = self.width * self.mattress_width_scale
        factory.size = self.size * self.mattress_size_scale
        return factory

    @cached_property
    def quilt_factory(self):
        from infinigen.assets.objects.clothes.blanket import BlanketFactory

        factory = BlanketFactory(self.factory_seed, self.coarse)
        factory.width = self.mattress_factory.width * self.quilt_width_scale
        factory.size = self.mattress_factory.size * self.quilt_size_scale
        return factory

    @cached_property
    def comforter_factory(self):
        from infinigen.assets.objects.clothes.blanket import ComforterFactory

        factory = ComforterFactory(self.factory_seed, self.coarse)
        factory.width = self.mattress_factory.width * self.comforter_width_scale
        factory.size = self.mattress_factory.size * self.comforter_size_scale
        return factory

    @cached_property
    def box_comforter_factory(self):
        from infinigen.assets.objects.clothes.blanket import BoxComforterFactory

        factory = BoxComforterFactory(self.factory_seed, self.coarse)
        factory.width = self.mattress_factory.width * self.box_comforter_width_scale
        factory.size = self.mattress_factory.size * self.box_comforter_size_scale
        return factory

    @cached_property
    def cover_factory(self):
        from infinigen.assets.objects.clothes.blanket import BlanketFactory

        factory = BlanketFactory(self.factory_seed, self.coarse)
        factory.width = self.mattress_factory.width * self.cover_width_scale
        factory.size = self.mattress_factory.size * self.cover_size_scale
        return factory

    @cached_property
    def towel_factory(self):
        from infinigen.assets.objects.clothes import TowelFactory

        return TowelFactory(self.factory_seed)

    @cached_property
    def pillow_factory(self):
        return pillow.PillowFactory(self.factory_seed, self.coarse)

    def create_asset(self, i, **params) -> bpy.types.Object:
        frame = super().create_asset(i=i, **params)

        mattress_obj = self.make_mattress(i)
        sheet = self.make_sheet(i, mattress_obj, frame)
        cover = self.make_cover(i, sheet, mattress_obj)

        n_pillows = (
            self.n_pillows
            if self._use_fixed_spawn_draws
            else int(np.random.randint(2, 4))
        )
        if n_pillows > 0:
            pillow_obj = self.pillow_factory(i)
            pillows = [pillow_obj] + [
                deep_clone_obj(pillow_obj) for _ in range(n_pillows - 1)
            ]
        else:
            pillows = []
        self.pillow_factory.finalize_assets(pillows)
        if self._use_fixed_spawn_draws:
            points = np.stack(
                [
                    np.array(self.pillow_scatter_x) * self.size,
                    np.array(self.pillow_scatter_y) * self.width,
                    np.full(10, 1),
                ],
                -1,
            )
        else:
            points = np.stack(
                [
                    uniform(0.1, 0.4, 10) * self.size,
                    uniform(-0.3, 0.3, 10) * self.width,
                    np.full(10, 1),
                ],
                -1,
            )
        self.scatter(pillows, points, [sheet, mattress_obj])

        n_towels = self.n_towels if self._use_fixed_spawn_draws else np.random.randint(1, 2)
        if n_towels > 0:
            towel = self.towel_factory(i)
            towels = [towel] + [deep_clone_obj(towel) for _ in range(n_towels - 1)]
        else:
            towels = []
        self.towel_factory.finalize_assets(towels)
        if self._use_fixed_spawn_draws:
            points = np.stack(
                [
                    np.array(self.towel_scatter_x) * self.size,
                    np.array(self.towel_scatter_y) * self.width,
                    np.full(10, 1),
                ],
                -1,
            )
        else:
            points = np.stack(
                [
                    uniform(0.5, 0.8, 10) * self.size,
                    uniform(-0.3, 0.3, 10) * self.width,
                    np.full(10, 1),
                ],
                -1,
            )
        self.scatter(towels, points, [sheet, mattress_obj])

        for obj in [mattress_obj, sheet, cover] + pillows + towels:
            obj.parent = frame
        butil.select_none()
        return frame

    def make_mattress(self, i):
        mattress_obj = self.mattress_factory(i=i)
        mattress_obj.location = self.size / 2, 0, self.mattress_factory.thickness / 2
        mattress_obj.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(mattress_obj, True)
        self.mattress_factory.finalize_assets(mattress_obj)
        return mattress_obj

    def make_sheet(self, i, mattress_obj, obj):
        match self.sheet_type:
            case "quilt":
                factory = self.quilt_factory
                pressure = 0.0
            case "comforter":
                factory = self.comforter_factory
                pressure = (
                    self.sheet_pressure
                    if self._use_fixed_spawn_draws
                    else uniform(1.0, 1.5)
                )
            case _:
                factory = self.box_comforter_factory
                pressure = (
                    self.sheet_pressure
                    if self._use_fixed_spawn_draws
                    else log_uniform(8, 15)
                )
        sheet = factory(i)
        if self.sheet_folded:
            factory.fold(sheet)
        factory.finalize_assets(sheet)
        z_sheet = mattress_obj.location[-1] + np.max(read_co(mattress_obj)[:, -1])
        sheet_offset = (
            self.sheet_location_offset
            if self._use_fixed_spawn_draws
            else uniform(0, 0.15)
        )
        sheet.location = factory.size / 2 + sheet_offset, 0, z_sheet
        sheet.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(sheet, True)
        clothes.cloth_sim(
            sheet,
            [mattress_obj, obj],
            mass=0.05,
            tension_stiffness=2,
            distance_min=5e-3,
            use_pressure=True,
            uniform_pressure_force=pressure,
            use_self_collision=self.sheet_folded,
        )
        subsurf(sheet, 2)
        return sheet

    def make_cover(self, i, sheet, mattress_obj):
        cover = self.cover_factory(i)
        self.cover_factory.finalize_assets(cover)
        z_sheet = sheet.location[-1] + np.max(read_co(sheet)[:, -1])
        cover_offset = (
            self.cover_location_offset
            if self._use_fixed_spawn_draws
            else uniform(0, 0.3)
        )
        cover.location = self.size / 2 + cover_offset, 0, z_sheet
        cover.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(cover, True)
        clothes.cloth_sim(
            cover,
            [sheet, mattress_obj],
            80,
            mass=0.05,
            tension_stiffness=2,
            distance_min=5e-3,
        )
        subsurf(cover, 2)
        return cover

    def scatter(self, pillows, points, bases):
        direction = np.array([[0, 0, -1]])
        lengths = np.full(len(points), np.inf)
        for base in bases:
            lengths = np.minimum(
                lengths,
                trimesh.proximity.longest_ray(
                    obj2trimesh(base), points, np.repeat(direction, len(points), 0)
                ),
            )
        points += direction * lengths[:, np.newaxis]
        rotations = (
            list(self.pillow_rotations)
            if self._use_fixed_spawn_draws
            else [uniform(0, np.pi) for _ in pillows]
        )
        for asset, loc, rotation in zip(pillows, decimate(points, len(pillows)), rotations):
            asset.location = loc
            asset.location[-1] += 0.02 - np.min(read_co(asset)[:, -1])
            asset.rotation_euler[-1] = rotation
