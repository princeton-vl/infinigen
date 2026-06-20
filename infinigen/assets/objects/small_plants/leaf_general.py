# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han


from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
import numpy as np
from pydantic import Field

from infinigen.assets.objects.trees.utils import mesh
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil


class LeafParameters(AssetParameters):
    leaf_width: float = Field(default=0.5, json_schema_extra={"editable": False})
    alpha: float = Field(default=0.3, json_schema_extra={"editable": False})
    use_wave: bool = Field(default=True, json_schema_extra={"editable": False})
    x_offset: float = Field(default=0.0, json_schema_extra={"editable": False})
    flip_leaf: bool = Field(default=False, json_schema_extra={"editable": False})
    z_scaling: float = Field(default=0.0, json_schema_extra={"editable": False})
    width_rand: float = Field(default=0.33, json_schema_extra={"editable": False})
    width_noise: Annotated[
        float, Field(ge=-1.0, le=1.0, json_schema_extra={"editable": True})
    ] = 0.0
    wave_height: Annotated[
        float, Field(ge=-0.9, le=0.9, json_schema_extra={"editable": True})
    ] = 0.0
    wave_width: Annotated[
        float, Field(ge=0.55, le=0.95, json_schema_extra={"editable": True})
    ] = 0.75
    wave_speed: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ] = 0.5


class LeafFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = LeafParameters
    scale = 0.3

    def __init__(self, factory_seed, genome: dict | None = None, coarse=False):
        super(LeafFactory, self).__init__(factory_seed, coarse=coarse)
        self._genome_override = genome
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> LeafParameters:
        params = LeafParameters(seed=seed)
        if self._genome_override:
            updates = {
                k: v
                for k, v in self._genome_override.items()
                if k in LeafParameters.model_fields
            }
            params = params.model_copy(update=updates)
        return params

    def _sample_spawn_parameters(
        self, params: LeafParameters, seed: int, i: int
    ) -> LeafParameters:
        return params.model_copy(
            update={
                "width_noise": float(np.random.randn()),
                "wave_height": float(np.random.randn() * 0.3),
                "wave_width": float(0.75 + np.random.randn() * 0.1),
                "wave_speed": float(np.random.rand()),
            }
        )

    def apply_parameters(
        self, params: LeafParameters, *, spawn_scope: bool = True
    ) -> None:
        self.genome = {
            "leaf_width": params.leaf_width,
            "alpha": params.alpha,
            "use_wave": params.use_wave,
            "x_offset": params.x_offset,
            "flip_leaf": params.flip_leaf,
            "z_scaling": params.z_scaling,
            "width_rand": params.width_rand,
        }
        self.width_noise = params.width_noise
        self.wave_height = params.wave_height
        self.wave_width = params.wave_width
        self.wave_speed = params.wave_speed
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, **params) -> bpy.types.Object:
        bpy.ops.mesh.primitive_circle_add(
            enter_editmode=False, align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
        )
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.edge_face_add()

        obj = bpy.context.active_object
        n = len(obj.data.vertices) // 2

        mesh.select_vtx_by_idx(obj, [0, -1], deselect=True)
        bpy.ops.mesh.subdivide()

        a = np.linspace(0, np.pi, n)
        if self.genome["flip_leaf"]:
            a = a[::-1]
        width_noise = (
            self.width_noise if self._use_fixed_spawn_draws else float(np.random.randn())
        )
        x = (
            np.sin(a)
            * (self.genome["leaf_width"] + width_noise * self.genome["width_rand"])
            + self.genome["x_offset"]
        )
        y = -np.cos(0.9 * (a - self.genome["alpha"]))
        z = x**2 * self.genome["z_scaling"]

        full_coords = np.concatenate(
            [
                np.stack([x, y, z], 1),
                np.stack([-x[::-1], y[::-1], z], 1),
                np.array([[0, y[0], 0]]),
            ]
        ).flatten()
        bpy.ops.object.mode_set(mode="OBJECT")
        obj.data.vertices.foreach_set("co", full_coords)

        if self.genome["use_wave"]:
            wave_height = (
                self.wave_height
                if self._use_fixed_spawn_draws
                else float(np.random.randn() * 0.3)
            )
            wave_width = (
                self.wave_width
                if self._use_fixed_spawn_draws
                else float(0.75 + np.random.randn() * 0.1)
            )
            wave_speed = (
                self.wave_speed if self._use_fixed_spawn_draws else float(np.random.rand())
            )
            bpy.ops.object.modifier_add(type="WAVE")
            bpy.context.object.modifiers["Wave"].height = wave_height
            bpy.context.object.modifiers["Wave"].width = wave_width
            bpy.context.object.modifiers["Wave"].speed = wave_speed

        mesh.finalize_obj(obj)
        bpy.context.scene.cursor.location = obj.data.vertices[-1].co

        bpy.ops.object.origin_set(type="ORIGIN_CURSOR")

        obj.location = (0, 0, 0)
        obj.scale *= self.scale
        butil.apply_transform(obj)
        tag_object(obj, "leaf")
        return obj


if __name__ == "__main__":
    leaf = LeafFactory(factory_seed=0)
    leaf.create_asset()
