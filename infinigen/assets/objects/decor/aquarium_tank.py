# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Annotated, Any, ClassVar

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
import infinigen.assets.materials.ceramic
import infinigen.assets.materials.fluid
from infinigen.assets.objects import (
    cactus,
    corals,
    mollusk,
    mushroom,
    rocks,
    underwater,
)
from infinigen.assets.utils.decorate import read_co, write_attribute
from infinigen.assets.utils.object import join_objects, new_bbox, new_cube, new_plane
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.random import log_uniform, weighted_sample


class AquariumTankParameters(AssetParameters):
    is_wet_draw: Annotated[float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})]
    width: Annotated[float, Field(ge=0.5, le=1.0, json_schema_extra={"editable": True})]
    depth: Annotated[float, Field(ge=0.5, le=0.8, json_schema_extra={"editable": True})]
    height: Annotated[float, Field(ge=0.5, le=1.0, json_schema_extra={"editable": True})]
    thickness: Annotated[
        float, Field(ge=0.01, le=0.02, json_schema_extra={"editable": True})
    ]
    belt_thickness: Annotated[
        float, Field(ge=0.02, le=0.05, json_schema_extra={"editable": True})
    ]
    scale: Annotated[float, Field(ge=0.7, le=0.9, json_schema_extra={"editable": True})] = (
        0.8
    )
    is_wet: bool = Field(default=False, json_schema_extra={"editable": False})
    base_factory: Any = Field(json_schema_extra={"editable": False})
    glass_surface: Any = Field(json_schema_extra={"editable": False})
    belt_surface: Any = Field(json_schema_extra={"editable": False})
    water_surface: Any = Field(json_schema_extra={"editable": False})


class AquariumTankFactory(ParameterizedAssetFactory, AssetFactory):
    dry_factories = [
        mushroom.MushroomFactory,
        cactus.CactusFactory,
        rocks.BoulderFactory,
    ]
    wet_factories = [
        mollusk.MolluskFactory,
        corals.CoralFactory,
        underwater.SeaweedFactory,
    ]
    parameters_model: ClassVar[type[AssetParameters]] = AquariumTankParameters

    def __init__(self, factory_seed, coarse=False):
        super(AquariumTankFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> AquariumTankParameters:
        is_wet_draw = uniform()
        is_wet = is_wet_draw < 0.5
        base_factory_fn = np.random.choice(
            self.wet_factories if is_wet else self.dry_factories
        )
        return AquariumTankParameters(
            seed=seed,
            is_wet_draw=is_wet_draw,
            is_wet=is_wet,
            base_factory=base_factory_fn(seed),
            width=log_uniform(0.5, 1),
            depth=log_uniform(0.5, 0.8),
            height=log_uniform(0.5, 1),
            thickness=uniform(0.01, 0.02),
            belt_thickness=log_uniform(0.02, 0.05),
            glass_surface=infinigen.assets.materials.ceramic.Glass(),
            belt_surface=weighted_sample(material_assignments.frame)(),
            water_surface=infinigen.assets.materials.fluid.Water(),
        )

    def _sample_spawn_parameters(
        self, params: AquariumTankParameters, seed: int, i: int
    ) -> AquariumTankParameters:
        return params.model_copy(update={"scale": uniform(0.7, 0.9)})

    def apply_parameters(
        self, params: AquariumTankParameters, *, spawn_scope: bool = True
    ) -> None:
        self.is_wet = params.is_wet
        self.base_factory = params.base_factory
        self.width = params.width
        self.depth = params.depth
        self.height = params.height
        self.thickness = params.thickness
        self.belt_thickness = params.belt_thickness
        self.glass_surface = params.glass_surface
        self.belt_surface = params.belt_surface
        self.water_surface = params.water_surface
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self.scale = params.scale

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(
            -self.thickness - self.depth,
            self.thickness,
            -self.thickness,
            self.width + self.thickness,
            0,
            self.height,
        )

    def create_asset(self, **params) -> bpy.types.Object:
        tank = new_cube(location=(1, 1, 1))
        butil.apply_transform(tank, loc=True)
        tank.scale = self.width / 2, self.depth / 2, self.height / 2
        butil.apply_transform(tank)
        butil.modify_mesh(tank, "SOLIDIFY", thickness=self.thickness)
        write_attribute(tank, 1, "glass", "FACE")
        parts = [tank]
        parts.extend(self.make_belts())
        base_obj = self.base_factory.create_asset(**params)
        co = read_co(base_obj)
        x_min, x_max = np.amin(co, 0), np.amax(co, 0)
        scale = (
            self.scale
            if self._use_fixed_spawn_draws
            else uniform(0.7, 0.9)
        ) / np.max(
            (x_max - x_min) / np.array([self.width, self.depth, self.height])
        )
        base_obj.location = -(x_min + x_max) * np.array(base_obj.scale) / 2
        base_obj.location[-1] = -(x_min * base_obj.scale)[-1]
        butil.apply_transform(base_obj, True)
        base_obj.location = self.width / 2, self.depth / 2, self.thickness
        base_obj.scale = [scale] * 3
        butil.apply_transform(base_obj)
        parts.append(base_obj)
        obj = join_objects(parts)
        obj.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(obj)
        return obj

    def make_belts(self):
        belt = new_plane()
        with butil.ViewportMode(belt, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.delete(type="ONLY_FACE")
        belt.location = self.width / 2, self.depth / 2, 0
        belt.scale = self.width / 2, self.depth / 2, 0
        butil.apply_transform(belt, loc=True)
        with butil.ViewportMode(belt, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={"value": (0, 0, self.belt_thickness)}
            )
        butil.modify_mesh(belt, "SOLIDIFY", thickness=self.thickness)
        write_attribute(belt, 1, "belt", "FACE")

        belt_ = deep_clone_obj(belt)
        belt_.location[-1] = self.height - self.belt_thickness
        butil.apply_transform(belt_, True)
        return [belt, belt_]

    def finalize_assets(self, assets):
        self.glass_surface.apply(assets, selection="glass")
        self.belt_surface.apply(assets, selection="belt")
