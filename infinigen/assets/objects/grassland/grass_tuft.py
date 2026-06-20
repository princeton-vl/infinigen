# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from __future__ import annotations

from typing import Annotated, Any, ClassVar

import bpy
import numpy as np
from numpy.random import normal, uniform
from pydantic import Field

from infinigen.assets.materials.plant import grass_blade
from infinigen.assets.utils.geometry.curve import Curve
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil


class GrassTuftParameters(AssetParameters):
    length_mean: Annotated[float, Field(ge=0.05, le=0.15, json_schema_extra={"editable": True})]
    length_std: Annotated[float, Field(ge=0.2, le=0.5, json_schema_extra={"editable": True})]
    curl_mean: Annotated[float, Field(ge=10.0, le=70.0, json_schema_extra={"editable": True})]
    curl_std: Annotated[float, Field(ge=0.0, le=0.6, json_schema_extra={"editable": True})]
    curl_power: Annotated[float, Field(ge=0.3, le=2.1, json_schema_extra={"editable": True})]
    blade_width_pct_mean: Annotated[
        float, Field(ge=0.01, le=0.03, json_schema_extra={"editable": True})
    ]
    blade_width_var: Annotated[float, Field(ge=0.0, le=0.05, json_schema_extra={"editable": True})]
    taper_var: Annotated[float, Field(ge=0.0, le=0.1, json_schema_extra={"editable": True})]
    base_spread: Annotated[
        float, Field(ge=0.0, le=0.037414, json_schema_extra={"editable": True})
    ]
    base_angle_var: Annotated[float, Field(ge=0.0, le=15.0, json_schema_extra={"editable": True})]
    n_blades: Annotated[int, Field(ge=30, le=59, json_schema_extra={"editable": True})] = 45
    taper_points: Any = Field(json_schema_extra={"editable": False})
    material_gen: Any = Field(json_schema_extra={"editable": False})


class GrassTuftFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = GrassTuftParameters
    n_seg = 4

    def __init__(self, seed):
        super().__init__(seed)
        self.init_legacy_parameters()

    def _sample_taper_points(self, taper_var: float) -> np.ndarray:
        taper_y = np.linspace(1, 0, self.n_seg) * normal(1, taper_var, self.n_seg)
        taper_x = np.linspace(0, 1, self.n_seg)
        return np.stack([taper_x, taper_y], axis=-1)

    def _sample_init_parameters(self, seed: int) -> GrassTuftParameters:
        length_mean = uniform(0.05, 0.15)
        taper_var = uniform(0, 0.1)
        return GrassTuftParameters(
            seed=seed,
            length_mean=length_mean,
            length_std=uniform(0.2, 0.5),
            curl_mean=uniform(10, 70),
            curl_std=float(np.clip(normal(0.3, 0.1), 0.01, 0.6)),
            curl_power=float(normal(1.2, 0.3)),
            blade_width_pct_mean=uniform(0.01, 0.03),
            blade_width_var=uniform(0, 0.05),
            taper_var=taper_var,
            base_spread=uniform(0, length_mean / 4),
            base_angle_var=uniform(0, 15),
            taper_points=self._sample_taper_points(taper_var),
            material_gen=grass_blade.GrassBlade(),
        )

    def _sample_spawn_parameters(
        self, params: GrassTuftParameters, seed: int, i: int
    ) -> GrassTuftParameters:
        return params.model_copy(update={"n_blades": int(np.random.randint(30, 60))})

    def apply_parameters(
        self, params: GrassTuftParameters, *, spawn_scope: bool = True
    ) -> None:
        self.length_mean = params.length_mean
        self.length_std = params.length_mean * params.length_std
        self.curl_mean = params.curl_mean
        self.curl_std = params.curl_mean * params.curl_std
        self.curl_power = params.curl_power
        self.blade_width_pct_mean = params.blade_width_pct_mean
        self.blade_width_var = params.blade_width_var
        self.taper_points = params.taper_points
        self.base_spread = params.base_spread
        self.base_angle_var = params.base_angle_var
        self.material_gen = params.material_gen
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self.n_blades = params.n_blades

    def create_asset(self, **params) -> bpy.types.Object:
        n_blades = (
            self.n_blades
            if self._use_fixed_spawn_draws
            else np.random.randint(30, 60)
        )

        blade_lengths = normal(self.length_mean, self.length_std, (n_blades, 1))
        seg_lens = blade_lengths / self.n_seg

        seg_curls = normal(self.curl_mean, self.curl_std, (n_blades, self.n_seg))
        seg_curls *= np.power(
            np.linspace(0, 1, self.n_seg).reshape(1, self.n_seg), self.curl_power
        )
        seg_curls = np.deg2rad(seg_curls)

        point_rads = np.arange(self.n_seg).reshape(1, self.n_seg) * seg_lens
        point_angles = np.cumsum(seg_curls, axis=-1)
        point_angles -= point_angles[:, [0]]

        points = np.empty((n_blades, self.n_seg, 2))
        points[..., 0] = np.cumsum(point_rads * np.cos(point_angles), axis=-1)
        points[..., 1] = np.cumsum(point_rads * np.sin(point_angles), axis=-1)

        taper = Curve(self.taper_points).to_curve_obj()

        widths = blade_lengths.reshape(-1) * normal(
            self.blade_width_pct_mean, self.blade_width_var, n_blades
        )
        objs = []
        for i in range(n_blades):
            obj = Curve(points[i], taper=taper).to_curve_obj(
                name=f"_blade_{i}", extrude=widths[i], resu=2
            )
            objs.append(obj)

        with butil.SelectObjects(objs):
            bpy.ops.object.convert(target="MESH")
        butil.delete(taper)

        base_angles = uniform(0, 2 * np.pi, n_blades)
        base_rads = uniform(0, self.base_spread, n_blades)
        facing_offsets = np.rad2deg(normal(0, self.base_angle_var, n_blades))
        for a, r, off, obj in zip(base_angles, base_rads, facing_offsets, objs):
            obj.location = (-r * np.cos(a), r * np.sin(a), -0.05 * self.length_mean)
            obj.rotation_euler = (np.pi / 2, -np.pi / 2, -a + off)

        with butil.SelectObjects(objs):
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        with butil.SelectObjects(objs):
            bpy.ops.object.join()
            bpy.ops.object.shade_flat()
            parent = objs[0]

        tag_object(parent, "grass_tuft")

        return parent

    def finalize_assets(self, assets):
        self.material_gen.apply(assets)
