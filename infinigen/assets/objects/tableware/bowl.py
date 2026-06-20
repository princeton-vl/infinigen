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
from infinigen.assets.utils.decorate import set_shade_smooth, subsurf
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.object import new_bbox
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import log_uniform


class BowlParameters(AssetParameters):
    thickness_ratio: Annotated[
        float, Field(ge=0.01, le=0.03, json_schema_extra={"editable": True})
    ]
    lower_thresh: Annotated[float, Field(ge=0.5, le=0.8, json_schema_extra={"editable": True})]
    scale: Annotated[float, Field(ge=0.15, le=0.4, json_schema_extra={"editable": True})]
    x_bottom: Annotated[float, Field(ge=0.2, le=0.3, json_schema_extra={"editable": True})]
    x_mid: Annotated[float, Field(ge=0.8, le=0.95, json_schema_extra={"editable": True})]
    z_bottom: Annotated[float, Field(ge=0.02, le=0.05, json_schema_extra={"editable": True})]
    z_length: Annotated[float, Field(ge=0.4, le=0.8, json_schema_extra={"editable": True})]
    has_inside_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    scratch_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    edge_wear_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    bevel_segments: Annotated[int, Field(ge=2, le=4, json_schema_extra={"editable": True})] = (
        2
    )
    surface: Any = Field(json_schema_extra={"editable": False})
    inside_surface: Any = Field(json_schema_extra={"editable": False})
    guard_surface: Any = Field(json_schema_extra={"editable": False})
    scratch: Any | None = Field(default=None, json_schema_extra={"editable": False})
    edge_wear: Any | None = Field(default=None, json_schema_extra={"editable": False})
    has_guard: bool = Field(default=False, json_schema_extra={"editable": False})
    guard_depth: float = Field(default=0.01, json_schema_extra={"editable": False})
    metal_color: str = Field(default="bw+natural", json_schema_extra={"editable": False})


class BowlFactory(ParameterizedAssetFactory, TablewareFactory):
    allow_transparent = True
    parameters_model: ClassVar[type[AssetParameters]] = BowlParameters

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.x_end = 0.5
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> BowlParameters:
        base = sample_tableware_base(seed)
        scale = log_uniform(0.15, 0.4)
        return BowlParameters(
            seed=seed,
            thickness_ratio=uniform(0.01, 0.03),
            lower_thresh=base["lower_thresh"],
            scale=scale,
            x_bottom=uniform(0.2, 0.3),
            x_mid=uniform(0.8, 0.95),
            z_bottom=log_uniform(0.02, 0.05),
            z_length=log_uniform(0.4, 0.8),
            has_inside_draw=uniform(),
            scratch_draw=base["scratch_draw"],
            edge_wear_draw=base["edge_wear_draw"],
            surface=base["surface"],
            inside_surface=base["inside_surface"],
            guard_surface=base["guard_surface"],
            scratch=(
                None
                if base["scratch_draw"] > base["scratch_prob"]
                else base["scratch_fn"]()
            ),
            edge_wear=None,
            has_guard=False,
            guard_depth=base["guard_depth"],
            metal_color=base["metal_color"],
        )

    def _sample_spawn_parameters(
        self, params: BowlParameters, seed: int, i: int
    ) -> BowlParameters:
        return params.model_copy(update={"bevel_segments": int(np.random.randint(2, 5))})

    def apply_parameters(
        self, params: BowlParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_tableware_base(self, params)
        self.thickness = params.thickness_ratio * params.scale
        self.has_inside = params.has_inside_draw < 0.5
        self.x_bottom = params.x_bottom * self.x_end
        self.x_mid = params.x_mid * self.x_end
        self.z_bottom = params.z_bottom
        self.z_length = params.z_length
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._bevel_segments = params.bevel_segments

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        radius = self.x_end * self.scale
        return new_bbox(-radius, radius, -radius, radius, 0, self.z_length * self.scale)

    def create_asset(self, **params) -> bpy.types.Object:
        bevel_segments = (
            self._bevel_segments
            if self._use_fixed_spawn_draws
            else np.random.randint(2, 5)
        )
        x_anchors = (
            0,
            self.x_bottom,
            self.x_bottom + 1e-3,
            self.x_bottom,
            self.x_mid,
            self.x_end,
        )
        z_anchors = 0, 0, 0, self.z_bottom, self.z_length / 2, self.z_length
        anchors = np.array(x_anchors) * self.scale, 0, np.array(z_anchors) * self.scale
        obj = spin(anchors, [2, 3])
        self.solidify_with_inside(obj, self.thickness)
        butil.modify_mesh(obj, "BEVEL", width=self.thickness / 2, segments=bevel_segments)
        subsurf(obj, 1)
        set_shade_smooth(obj)
        return obj
