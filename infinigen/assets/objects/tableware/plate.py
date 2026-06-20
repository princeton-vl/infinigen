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
from infinigen.assets.utils.decorate import subsurf
from infinigen.assets.utils.draw import spin
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import log_uniform


class PlateParameters(AssetParameters):
    thickness_ratio: Annotated[
        float, Field(ge=0.01, le=0.03, json_schema_extra={"editable": True})
    ]
    lower_thresh: Annotated[float, Field(ge=0.5, le=0.8, json_schema_extra={"editable": True})]
    scale: Annotated[float, Field(ge=0.2, le=0.4, json_schema_extra={"editable": True})]
    x_mid: Annotated[float, Field(ge=0.3, le=1.0, json_schema_extra={"editable": True})]
    z_length: Annotated[float, Field(ge=0.05, le=0.2, json_schema_extra={"editable": True})]
    z_mid_ratio: Annotated[
        float, Field(ge=0.3, le=0.8, json_schema_extra={"editable": True})
    ]
    has_inside_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    scratch_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    edge_wear_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    surface: Any = Field(json_schema_extra={"editable": False})
    inside_surface: Any = Field(json_schema_extra={"editable": False})
    guard_surface: Any = Field(json_schema_extra={"editable": False})
    scratch: Any | None = Field(default=None, json_schema_extra={"editable": False})
    edge_wear: Any | None = Field(default=None, json_schema_extra={"editable": False})
    has_guard: bool = Field(default=False, json_schema_extra={"editable": False})
    guard_depth: float = Field(default=0.01, json_schema_extra={"editable": False})
    metal_color: str = Field(default="bw+natural", json_schema_extra={"editable": False})


class PlateFactory(ParameterizedAssetFactory, TablewareFactory):
    allow_transparent = True
    parameters_model: ClassVar[type[AssetParameters]] = PlateParameters

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.x_end = 0.5
        self.pre_level = 1
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> PlateParameters:
        base = sample_tableware_base(seed)
        scale = log_uniform(0.2, 0.4)
        z_length = log_uniform(0.05, 0.2)
        return PlateParameters(
            seed=seed,
            thickness_ratio=uniform(0.01, 0.03),
            lower_thresh=base["lower_thresh"],
            scale=scale,
            x_mid=uniform(0.3, 1.0) * self.x_end,
            z_length=z_length,
            z_mid_ratio=uniform(0.3, 0.8),
            has_inside_draw=uniform(),
            scratch_draw=base["scratch_draw"],
            edge_wear_draw=base["edge_wear_draw"],
            surface=base["surface"],
            inside_surface=base["inside_surface"],
            guard_surface=base["guard_surface"],
            scratch=None,
            edge_wear=None,
            has_guard=False,
            guard_depth=base["guard_depth"],
            metal_color=base["metal_color"],
        )

    def apply_parameters(
        self, params: PlateParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_tableware_base(self, params)
        self.thickness = params.thickness_ratio * params.scale
        self.has_inside = params.has_inside_draw < 0.2
        self.x_mid = params.x_mid
        self.z_length = params.z_length
        self.z_mid = params.z_mid_ratio * params.z_length
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, **params) -> bpy.types.Object:
        x_anchors = 0, self.x_mid, self.x_mid, self.x_end
        z_anchors = 0, 0, self.z_mid, self.z_length
        anchors = np.array(x_anchors) * self.scale, 0, np.array(z_anchors) * self.scale
        obj = spin(anchors, [1, 2])
        butil.modify_mesh(
            obj, "SUBSURF", render_levels=self.pre_level, levels=self.pre_level
        )
        self.solidify_with_inside(obj, self.thickness)
        subsurf(obj, 1)
        return obj
