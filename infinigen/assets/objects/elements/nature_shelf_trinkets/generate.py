# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Stamatis Alexandropulos

from __future__ import annotations

from typing import Annotated, Any, ClassVar

import bpy
import mathutils
import numpy as np
import trimesh
from pydantic import Field

from infinigen.assets.objects import corals, creatures, mollusk, monocot, rocks
from infinigen.assets.utils import object as obj
from infinigen.assets.utils.object import join_objects
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil


class NatureShelfTrinketsParameters(AssetParameters):
    size: Annotated[float, Field(ge=0.1, le=0.15, json_schema_extra={"editable": True})]
    asset_index: Annotated[int, Field(ge=0, le=9999999, json_schema_extra={"editable": False})]
    base_factory: Any = Field(json_schema_extra={"editable": False})


class NatureShelfTrinketsFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = NatureShelfTrinketsParameters

    factories = [
        corals.CoralFactory,
        rocks.BlenderRockFactory,
        rocks.BoulderFactory,
        monocot.PineconeFactory,
        mollusk.MolluskFactory,
        mollusk.AugerFactory,
        mollusk.ClamFactory,
        mollusk.ConchFactory,
        mollusk.MusselFactory,
        mollusk.ScallopFactory,
        mollusk.VoluteFactory,
        creatures.CarnivoreFactory,
        creatures.HerbivoreFactory,
    ]
    probs = np.array([1, 1, 1, 1, 3, 2, 3, 2, 2, 2, 2, 5, 5])

    def __init__(self, factory_seed, coarse=False):
        super(NatureShelfTrinketsFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> NatureShelfTrinketsParameters:
        base_factory_fn = np.random.choice(
            self.factories, p=self.probs / self.probs.sum()
        )
        kwargs: dict[str, Any] = {}
        if base_factory_fn in [
            creatures.HerbivoreFactory,
            creatures.CarnivoreFactory,
        ]:
            kwargs["hair"] = False
        return NatureShelfTrinketsParameters(
            seed=seed,
            size=0.125,
            asset_index=0,
            base_factory=base_factory_fn(seed, **kwargs),
        )

    def _sample_spawn_parameters(
        self, params: NatureShelfTrinketsParameters, seed: int, i: int
    ) -> NatureShelfTrinketsParameters:
        return params.model_copy(
            update={
                "size": np.random.uniform(0.1, 0.15),
                "asset_index": np.random.randint(1e7),
            }
        )

    def apply_parameters(
        self, params: NatureShelfTrinketsParameters, *, spawn_scope: bool = True
    ) -> None:
        self.base_factory = params.base_factory
        self._asset_index = params.asset_index
        self._placeholder_size = params.size
        self._use_fixed_spawn_draws = spawn_scope

    def create_placeholder(self, **params) -> bpy.types.Object:
        size = (
            self._placeholder_size
            if self._use_fixed_spawn_draws
            else np.random.uniform(0.1, 0.15)
        )
        bpy.ops.mesh.primitive_cube_add(size=size, location=(0, 0, size / 2))
        placeholder = bpy.context.active_object
        return placeholder

    def create_asset(self, i, placeholder=None, **params):
        asset_index = (
            self._asset_index
            if self._use_fixed_spawn_draws
            else np.random.randint(1e7)
        )
        asset = self.base_factory.spawn_asset(
            asset_index, distance=200, adaptive_resolution=False
        )

        if list(asset.children):
            asset = join_objects(list(asset.children))

        butil.apply_transform(asset, loc=True)
        butil.apply_modifiers(asset)
        if isinstance(self.base_factory, creatures.HerbivoreFactory) or isinstance(
            self.base_factory, creatures.CarnivoreFactory
        ):
            pass
        else:
            if not isinstance(asset, trimesh.Trimesh):
                mesh = obj.obj2trimesh(asset)
            stable_poses, probs = trimesh.poses.compute_stable_poses(mesh)
            stable_pose = stable_poses[np.argmax(probs)]
            asset.rotation_euler = mathutils.Matrix(stable_pose[:3, :3]).to_euler()
        butil.apply_transform(asset, rot=True)
        dim = asset.dimensions
        bounding_box = placeholder.dimensions
        scale = min([bounding_box[i] / dim[i] for i in range(3)])
        asset.scale = [scale for i in range(3)]
        butil.apply_transform(asset, loc=True)
        bounds = butil.bounds(asset)
        cur_loc = asset.location
        new_location = [
            cur_loc[i] - (bounds[0][i] + bounds[1][i]) / 2 for i in range(3)
        ]
        new_location[2] = cur_loc[2] - (bounds[0][2] + bounding_box[2] / 2)
        asset.location = new_location
        butil.apply_transform(asset, loc=True)
        return asset
