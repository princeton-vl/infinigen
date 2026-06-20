# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Annotated, Any, ClassVar

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.tableware.base import (
    TablewareFactory,
    apply_tableware_base,
    sample_tableware_base,
)
from infinigen.assets.utils.draw import spin
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import log_uniform, weighted_sample


class WineglassParameters(AssetParameters):
    z_length: Annotated[float, Field(ge=0.6, le=2.0, json_schema_extra={"editable": True})]
    z_cup: Annotated[float, Field(ge=0.3, le=0.6, json_schema_extra={"editable": True})]
    z_mid: Annotated[float, Field(ge=0.3, le=0.5, json_schema_extra={"editable": True})]
    x_neck: Annotated[float, Field(ge=0.01, le=0.02, json_schema_extra={"editable": True})]
    x_top: Annotated[float, Field(ge=1.0, le=1.4, json_schema_extra={"editable": True})]
    x_mid: Annotated[float, Field(ge=0.9, le=1.2, json_schema_extra={"editable": True})]
    thickness: Annotated[float, Field(ge=0.01, le=0.03, json_schema_extra={"editable": True})]
    scale: Annotated[float, Field(ge=0.1, le=0.3, json_schema_extra={"editable": True})]
    lower_thresh: Annotated[float, Field(ge=0.5, le=0.8, json_schema_extra={"editable": True})]
    scratch_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    edge_wear_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    z_bottom: Annotated[
        float, Field(ge=0.01, le=0.05, json_schema_extra={"editable": True})
    ] = 0.03
    surface: Any = Field(json_schema_extra={"editable": False})
    inside_surface: Any = Field(json_schema_extra={"editable": False})
    guard_surface: Any = Field(json_schema_extra={"editable": False})
    scratch: Any | None = Field(default=None, json_schema_extra={"editable": False})
    edge_wear: Any | None = Field(default=None, json_schema_extra={"editable": False})
    has_guard: bool = Field(default=False, json_schema_extra={"editable": False})
    guard_depth: float = Field(default=0.01, json_schema_extra={"editable": False})
    metal_color: str = Field(default="bw+natural", json_schema_extra={"editable": False})


class WineglassFactory(ParameterizedAssetFactory, TablewareFactory):
    parameters_model: ClassVar[type[AssetParameters]] = WineglassParameters

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.x_end = 0.25
        self.has_guard = False
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> WineglassParameters:
        base = sample_tableware_base(seed)
        z_length = log_uniform(0.6, 2.0)
        return WineglassParameters(
            seed=seed,
            z_length=z_length,
            z_cup=uniform(0.3, 0.6),
            z_mid=uniform(0.3, 0.5),
            x_neck=log_uniform(0.01, 0.02),
            x_top=log_uniform(1, 1.4),
            x_mid=log_uniform(0.9, 1.2),
            thickness=uniform(0.01, 0.03),
            scale=log_uniform(0.1, 0.3),
            lower_thresh=base["lower_thresh"],
            scratch_draw=base["scratch_draw"],
            edge_wear_draw=base["edge_wear_draw"],
            surface=weighted_sample(material_assignments.glasses)()(),
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
            has_guard=False,
            guard_depth=base["guard_depth"],
            metal_color=base["metal_color"],
        )

    def _sample_spawn_parameters(
        self, params: WineglassParameters, seed: int, i: int
    ) -> WineglassParameters:
        return params.model_copy(update={"z_bottom": log_uniform(0.01, 0.05)})

    def apply_parameters(
        self, params: WineglassParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_tableware_base(self, params)
        self.z_length = params.z_length
        z_cup_abs = params.z_cup * params.z_length
        self.z_cup = z_cup_abs
        self.z_mid = z_cup_abs + params.z_mid * (params.z_length - z_cup_abs)
        self.x_neck = params.x_neck
        self.x_top = self.x_end * params.x_top
        self.x_mid = self.x_top * params.x_mid
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._z_bottom = params.z_bottom

    def create_asset(self, **params) -> bpy.types.Object:
        z_bottom = self.z_length * (
            self._z_bottom if self._use_fixed_spawn_draws else log_uniform(0.01, 0.05)
        )
        x_anchors = (
            self.x_end,
            self.x_end / 2,
            self.x_neck,
            self.x_neck,
            self.x_mid,
            self.x_top,
        )
        z_anchors = 0, z_bottom / 2, z_bottom, self.z_cup, self.z_mid, self.z_length
        anchors = x_anchors, np.zeros_like(x_anchors), z_anchors
        obj = spin(anchors, [0, 1, 2, 3])
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)

        with butil.SelectObjects(obj):
            bpy.ops.object.shade_smooth()

        return obj
