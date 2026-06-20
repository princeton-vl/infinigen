# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Annotated, Any, ClassVar

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.objects.tableware.base import (
    TablewareFactory,
    apply_tableware_base,
    sample_tableware_base,
)
from infinigen.assets.utils.decorate import subsurf, write_co
from infinigen.assets.utils.object import join_objects, new_grid
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.random import log_uniform


class ChopsticksParameters(AssetParameters):
    y_length: Annotated[float, Field(ge=0.01, le=0.02, json_schema_extra={"editable": True})]
    y_shrink: Annotated[float, Field(ge=0.2, le=0.8, json_schema_extra={"editable": True})]
    is_square_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    has_guard_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    x_guard: Annotated[float, Field(ge=0.4, le=0.9, json_schema_extra={"editable": True})]
    lower_thresh: Annotated[float, Field(ge=0.5, le=0.8, json_schema_extra={"editable": True})]
    scale: Annotated[float, Field(ge=0.2, le=0.4, json_schema_extra={"editable": True})]
    scratch_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    thickness: float = Field(default=0.01, json_schema_extra={"editable": False})
    edge_wear_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    parallel_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ] = 0.0
    parallel_style_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ] = 0.0
    parallel_distance: Annotated[
        float, Field(ge=0.01, le=0.04, json_schema_extra={"editable": True})
    ] = 0.01
    parallel_rot_a: Annotated[
        float, Field(ge=0.0, le=0.392699, json_schema_extra={"editable": True})
    ] = 0.0
    parallel_rot_b: Annotated[
        float, Field(ge=0.0, le=0.392699, json_schema_extra={"editable": True})
    ] = 0.0
    crossed_loc_x: Annotated[
        float, Field(ge=-0.1, le=0.2, json_schema_extra={"editable": True})
    ] = 0.0
    crossed_loc_y: Annotated[
        float, Field(ge=-0.2, le=0.2, json_schema_extra={"editable": True})
    ] = 0.0
    crossed_rot: Annotated[
        float, Field(ge=0.392699, le=0.785398, json_schema_extra={"editable": True})
    ] = 0.392699
    surface: Any = Field(json_schema_extra={"editable": False})
    inside_surface: Any = Field(json_schema_extra={"editable": False})
    guard_surface: Any = Field(json_schema_extra={"editable": False})
    scratch: Any | None = Field(default=None, json_schema_extra={"editable": False})
    edge_wear: Any | None = Field(default=None, json_schema_extra={"editable": False})
    guard_depth: float = Field(default=0.0, json_schema_extra={"editable": False})
    metal_color: str = Field(default="bw+natural", json_schema_extra={"editable": False})


class ChopsticksFactory(ParameterizedAssetFactory, TablewareFactory):
    parameters_model: ClassVar[type[AssetParameters]] = ChopsticksParameters

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.pre_level = 2
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> ChopsticksParameters:
        base = sample_tableware_base(seed)
        return ChopsticksParameters(
            seed=seed,
            y_length=uniform(0.01, 0.02),
            y_shrink=log_uniform(0.2, 0.8),
            is_square_draw=uniform(0, 1),
            has_guard_draw=uniform(0, 1),
            x_guard=uniform(0.4, 0.9),
            lower_thresh=base["lower_thresh"],
            scale=log_uniform(0.2, 0.4),
            scratch_draw=base["scratch_draw"],
            edge_wear_draw=base["edge_wear_draw"],
            thickness=base["thickness"],
            surface=base["surface"],
            inside_surface=base["inside_surface"],
            guard_surface=base["guard_surface"],
            scratch=(
                None
                if base["scratch_draw"] > base["scratch_prob"]
                else base["scratch_fn"]()
            ),
            edge_wear=(
                None
                if base["edge_wear_draw"] > base["edge_wear_prob"]
                else base["edge_wear_fn"]()
            ),
            guard_depth=0.0,
            metal_color=base["metal_color"],
        )

    def _sample_spawn_parameters(
        self, params: ChopsticksParameters, seed: int, i: int
    ) -> ChopsticksParameters:
        parallel_draw = uniform(0, 1)
        if parallel_draw < 0.6:
            parallel_style_draw = uniform(0, 1)
            return params.model_copy(
                update={
                    "parallel_draw": parallel_draw,
                    "parallel_style_draw": parallel_style_draw,
                    "parallel_distance": log_uniform(params.y_length, 0.04),
                    "parallel_rot_a": uniform(0, np.pi / 8),
                    "parallel_rot_b": uniform(0, np.pi / 8),
                }
            )
        crossed_loc_y = uniform(-0.2, 0.2)
        sign = np.sign(crossed_loc_y)
        return params.model_copy(
            update={
                "parallel_draw": parallel_draw,
                "crossed_loc_x": uniform(-0.1, 0.2),
                "crossed_loc_y": crossed_loc_y,
                "crossed_rot": log_uniform(np.pi / 8, np.pi / 4) * sign,
            }
        )

    def apply_parameters(
        self, params: ChopsticksParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_tableware_base(self, params)
        self.y_length = params.y_length
        self.y_shrink = params.y_shrink
        self.is_square = params.is_square_draw < 0.5
        self.has_guard = params.has_guard_draw < 0.4
        self.x_guard = params.x_guard
        self.guard_depth = params.guard_depth
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._parallel_draw = params.parallel_draw
            self._parallel_style_draw = params.parallel_style_draw
            self._parallel_distance = params.parallel_distance
            self._parallel_rot_a = params.parallel_rot_a
            self._parallel_rot_b = params.parallel_rot_b
            self._crossed_loc_x = params.crossed_loc_x
            self._crossed_loc_y = params.crossed_loc_y
            self._crossed_rot = params.crossed_rot

    def create_asset(self, **params) -> bpy.types.Object:
        obj = self.make_single()
        if self._use_fixed_spawn_draws:
            is_parallel = self._parallel_draw < 0.6
        else:
            is_parallel = uniform(0, 1) < 0.6
        if is_parallel:
            obj = self.make_parallel(obj)
        else:
            obj = self.make_crossed(obj)
        return obj

    def make_parallel(self, obj):
        if self._use_fixed_spawn_draws:
            distance = self._parallel_distance
            style_draw = self._parallel_style_draw
            rot_a = self._parallel_rot_a
            rot_b = self._parallel_rot_b
        else:
            distance = log_uniform(self.y_length, 0.04)
            style_draw = uniform(0, 1)
            rot_a = uniform(0, np.pi / 8)
            rot_b = uniform(0, np.pi / 8)
        if style_draw < 0.5:
            other = deep_clone_obj(obj)
            obj.location[1] = distance
            obj.rotation_euler[-1] = rot_a
            other.location[1] = -distance
            other.rotation_euler[-1] = -rot_b
        else:
            obj.location[0] = -1
            butil.apply_transform(obj, loc=True)
            other = deep_clone_obj(obj)
            obj.location[1] = distance
            obj.rotation_euler[-1] = -rot_b
            other.location[1] = -distance
            other.rotation_euler[-1] = rot_a
        return join_objects([obj, other])

    def make_crossed(self, obj):
        other = deep_clone_obj(obj)
        if self._use_fixed_spawn_draws:
            other.location = (
                self._crossed_loc_x,
                self._crossed_loc_y,
                self.y_length,
            )
            other.rotation_euler[-1] = -self._crossed_rot
        else:
            other.location = uniform(-0.1, 0.2), uniform(-0.2, 0.2), self.y_length
            sign = np.sign(other.location[1])
            other.rotation_euler[-1] = -sign * log_uniform(np.pi / 8, np.pi / 4)
        return join_objects([obj, other])

    def make_single(self):
        n = int(1 / self.y_length)
        obj = new_grid(x_subdivisions=n - 1, y_subdivisions=1)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.y_length * 2)
        l = np.linspace(self.y_shrink, 1, n) * self.y_length
        x = np.concatenate([np.linspace(0, 1, n)] * 4)
        y = np.concatenate([-l, l, -l, l])
        z = np.concatenate([l, l, -l, -l])
        write_co(obj, np.stack([x, y, z], -1))
        subsurf(obj, 2, self.is_square)
        self.add_guard(obj, lambda nw, x: nw.compare("GREATER_THAN", x, self.x_guard))
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)
        return obj
