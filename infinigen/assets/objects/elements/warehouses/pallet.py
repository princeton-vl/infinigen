# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Annotated, Any, ClassVar

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.materials.wood import wood
from infinigen.assets.utils.decorate import read_normal
from infinigen.assets.utils.object import join_objects, new_bbox, new_cube
from infinigen.core import surface
from infinigen.core import tags as t
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.surface import write_attr_data
from infinigen.core.tagging import PREFIX
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj


class PalletParameters(AssetParameters):
    depth: Annotated[float, Field(ge=1.2, le=1.4, json_schema_extra={"editable": True})]
    width: Annotated[float, Field(ge=1.2, le=1.4, json_schema_extra={"editable": True})]
    thickness: Annotated[
        float, Field(ge=0.01, le=0.015, json_schema_extra={"editable": True})
    ]
    tile_width: Annotated[
        float, Field(ge=0.06, le=0.1, json_schema_extra={"editable": True})
    ]
    tile_slackness: Annotated[
        float, Field(ge=1.5, le=2.0, json_schema_extra={"editable": True})
    ]
    height: Annotated[float, Field(ge=0.2, le=0.25, json_schema_extra={"editable": True})]
    surface: Any = Field(json_schema_extra={"editable": False})


class PalletFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = PalletParameters

    def __init__(self, factory_seed, coarse=False):
        super(PalletFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> PalletParameters:
        return PalletParameters(
            seed=seed,
            depth=uniform(1.2, 1.4),
            width=uniform(1.2, 1.4),
            thickness=uniform(0.01, 0.015),
            tile_width=uniform(0.06, 0.1),
            tile_slackness=uniform(1.5, 2),
            height=uniform(0.2, 0.25),
            surface=wood.Wood()(),
        )

    def apply_parameters(
        self, params: PalletParameters, *, spawn_scope: bool = True
    ) -> None:
        self.depth = params.depth
        self.width = params.width
        self.thickness = params.thickness
        self.tile_width = params.tile_width
        self.tile_slackness = params.tile_slackness
        self.height = params.height
        self.surface = params.surface
        self._use_fixed_spawn_draws = spawn_scope

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        bbox = new_bbox(0, self.width, 0, self.depth, 0, self.height)
        write_attr_data(
            bbox,
            f"{PREFIX}{t.Subpart.SupportSurface.value}",
            read_normal(bbox)[:, -1] > 0.5,
            "INT",
            "FACE",
        )
        return bbox

    def create_asset(self, **params) -> bpy.types.Object:
        vertical = self.make_vertical()
        vertical.location[-1] = self.thickness
        vertical_ = deep_clone_obj(vertical)
        vertical_.location[-1] = self.height - self.thickness
        horizontal = self.make_horizontal()
        horizontal_ = deep_clone_obj(horizontal)
        horizontal_.location[-1] = self.height - 2 * self.thickness
        support = self.make_support()
        support.location[-1] = 2 * self.thickness
        obj = join_objects([horizontal, horizontal_, vertical, vertical_, support])
        return obj

    def make_vertical(self):
        obj = new_cube()
        obj.location = 1, 1, 1
        butil.apply_transform(obj, True)
        obj.scale = self.tile_width / 2, self.depth / 2, self.thickness / 2
        butil.apply_transform(obj)
        count = (
            int(
                np.floor(
                    (self.width - self.tile_width)
                    / self.tile_width
                    / self.tile_slackness
                )
                / 2
            )
            * 2
        )
        butil.modify_mesh(
            obj,
            "ARRAY",
            use_relative_offset=False,
            use_constant_offset=True,
            constant_offset_displace=((self.width - self.tile_width) / count, 0, 0),
            count=count + 1,
        )
        return obj

    def make_horizontal(self):
        obj = new_cube()
        obj.location = 1, 1, 1
        butil.apply_transform(obj, True)
        obj.scale = self.width / 2, self.tile_width / 2, self.thickness / 2
        butil.apply_transform(obj)
        count = (
            int(
                np.floor(
                    (self.depth - self.tile_width)
                    / self.tile_width
                    / self.tile_slackness
                )
                / 2
            )
            * 2
        )
        butil.modify_mesh(
            obj,
            "ARRAY",
            use_relative_offset=False,
            use_constant_offset=True,
            constant_offset_displace=(0, (self.depth - self.tile_width) / count, 0),
            count=count + 1,
        )
        return obj

    def make_support(self):
        obj = new_cube()
        obj.location = 1, 1, 1
        butil.apply_transform(obj, True)
        obj.scale = (
            self.tile_width / 2,
            self.tile_width / 2,
            self.height / 2 - 2 * self.thickness,
        )
        butil.apply_transform(obj)
        butil.modify_mesh(
            obj,
            "ARRAY",
            use_relative_offset=False,
            use_constant_offset=True,
            constant_offset_displace=((self.width - self.tile_width) / 2, 0, 0),
            count=3,
        )
        butil.modify_mesh(
            obj,
            "ARRAY",
            use_relative_offset=False,
            use_constant_offset=True,
            constant_offset_displace=(0, (self.depth - self.tile_width) / 2, 0),
            count=3,
        )
        return obj

    def finalize_assets(self, assets):
        if isinstance(assets, bpy.types.Object):
            assets = [assets]
        for element in assets:
            surface.assign_material(element, self.surface)
