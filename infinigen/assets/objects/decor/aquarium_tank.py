# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
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
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class AquariumTankFactory(AssetFactory):
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

    def __init__(self, factory_seed, coarse=False):
        super(AquariumTankFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.is_wet = uniform() < 0.5
            base_factory_fn = np.random.choice(
                self.wet_factories if self.is_wet else self.dry_factories
            )
            self.base_factory = base_factory_fn(self.factory_seed)
            self.width = log_uniform(0.5, 1)
            self.depth = log_uniform(0.5, 0.8)
            self.height = log_uniform(0.5, 1)
            self.thickness = uniform(0.01, 0.02)
            self.belt_thickness = log_uniform(0.02, 0.05)

            materials = AssetList["AquariumTankFactory"]()
            self.glass_surface = materials["glass_surface"].assign_material()
            self.belt_surface = materials["belt_surface"].assign_material()
            self.water_surface = materials["water_surface"].assign_material()

            scratch_prob, edge_wear_prob = materials["wear_tear_prob"]
            self.scratch, self.edge_wear = materials["wear_tear"]
            is_scratch = uniform() < scratch_prob
            is_edge_wear = uniform() < edge_wear_prob
            if not is_scratch:
                self.scratch = None
            if not is_edge_wear:
                self.edge_wear = None

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
        scale = uniform(0.7, 0.9) / np.max(
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

        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
