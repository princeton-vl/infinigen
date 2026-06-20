# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
import numpy as np
from numpy.random import uniform as U
from pydantic import Field

from infinigen.core.init import require_blender_addon
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil

require_blender_addon("extra_mesh_objects", fail="warn")


class BlenderRockParameters(AssetParameters):
    rock_seed: Annotated[int, Field(ge=0, le=99998, json_schema_extra={"editable": True})]
    zscale: Annotated[float, Field(ge=0.2, le=0.8, json_schema_extra={"editable": True})]
    zrand: Annotated[float, Field(ge=0.0, le=0.7, json_schema_extra={"editable": True})]
    deform: Annotated[float, Field(ge=2.0, le=10.0, json_schema_extra={"editable": True})]
    rough: Annotated[float, Field(ge=0.5, le=1.0, json_schema_extra={"editable": True})]


class BlenderRockFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = BlenderRockParameters

    def __init__(self, factory_seed, detail=1):
        super().__init__(factory_seed)
        self.detail = detail
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> BlenderRockParameters:
        return BlenderRockParameters(
            seed=seed,
            rock_seed=0,
            zscale=0.5,
            zrand=0.35,
            deform=6.0,
            rough=0.75,
        )

    def _sample_spawn_parameters(
        self, params: BlenderRockParameters, seed: int, i: int
    ) -> BlenderRockParameters:
        return params.model_copy(
            update={
                "rock_seed": int(np.random.randint(0, 99999)),
                "zscale": U(0.2, 0.8),
                "zrand": U(0, 0.7),
                "deform": U(2, 10),
                "rough": U(0.5, 1.0),
            }
        )

    def apply_parameters(
        self, params: BlenderRockParameters, *, spawn_scope: bool = True
    ) -> None:
        self.rock_seed = params.rock_seed
        self.zscale = params.zscale
        self.zrand = params.zrand
        self.deform = params.deform
        self.rough = params.rough
        self._use_fixed_spawn_draws = spawn_scope

    __repr__ = AssetFactory.__repr__

    def create_asset(self, **params):
        require_blender_addon("extra_mesh_objects")

        rock_seed = (
            self.rock_seed
            if self._use_fixed_spawn_draws
            else int(np.random.randint(0, 99999))
        )
        zscale = self.zscale if self._use_fixed_spawn_draws else U(0.2, 0.8)
        zrand = self.zrand if self._use_fixed_spawn_draws else U(0, 0.7)
        deform = self.deform if self._use_fixed_spawn_draws else U(2, 10)
        rough = self.rough if self._use_fixed_spawn_draws else U(0.5, 1.0)

        while True:
            try:
                kwargs = dict(
                    use_random_seed=False,
                    user_seed=rock_seed,
                    display_detail=self.detail,
                    detail=self.detail,
                    scale_Z=(zrand * zscale, zscale),
                    scale_fac=(1, 1, 1),
                    scale_X=(1.00, 1.01),
                    scale_Y=(1.00, 1.01),
                    deform=deform,
                    rough=rough,
                )
                bpy.ops.mesh.add_mesh_rock(**kwargs)
                break
            except IndexError:
                pass
            except RuntimeError:
                pass

        obj = bpy.context.active_object
        bpy.ops.object.shade_flat()

        butil.apply_modifiers(obj)

        tag_object(obj, "blender_rock")

        return obj
