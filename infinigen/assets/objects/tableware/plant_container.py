# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.objects.cactus import CactusFactory
from infinigen.assets.objects.monocot import MonocotFactory
from infinigen.assets.objects.mushroom import MushroomFactory
from infinigen.assets.objects.small_plants import (
    FernFactory,
    SnakePlantFactory,
    SpiderPlantFactory,
    SucculentFactory,
)
from infinigen.assets.objects.tableware.pot import PotFactory
from infinigen.assets.utils.decorate import (
    read_edge_center,
    read_edge_direction,
    remove_vertices,
    select_edges,
    subsurf,
)
from infinigen.assets.utils.object import join_objects, new_bbox, origin2lowest
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class PlantPotFactory(PotFactory):
    def __init__(self, factory_seed, coarse=False):
        super(PlantPotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.has_handle = self.has_bar = self.has_guard = False
            self.depth = log_uniform(0.5, 1.0)
            self.r_expand = uniform(1.1, 1.3)
            alpha = uniform(0.5, 0.8)
            self.r_mid = (self.r_expand - 1) * alpha + 1
            material_assignments = AssetList["PlantContainerFactory"]()
            self.surface = material_assignments["surface"].assign_material()
            self.scale = log_uniform(0.08, 0.12)


class PlantContainerFactory(AssetFactory):
    plant_factories = [
        CactusFactory,
        MushroomFactory,
        FernFactory,
        SucculentFactory,
        SpiderPlantFactory,
        SnakePlantFactory,
    ]

    def __init__(self, factory_seed, coarse=False):
        super(PlantContainerFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.base_factory = PlantPotFactory(self.factory_seed, coarse)
            self.dirt_ratio = uniform(0.7, 0.8)
            material_assignments = AssetList["PlantContainerFactory"]()
            self.dirt_surface = material_assignments["dirt_surface"].assign_material()
            fn = np.random.choice(self.plant_factories)
            self.plant_factory = fn(self.factory_seed)
            self.side_size = self.base_factory.scale * self.base_factory.r_expand
            self.top_size = uniform(0.4, 0.6)

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(
            -self.side_size,
            self.side_size,
            -self.side_size,
            self.side_size,
            -0.02,
            self.base_factory.depth * self.base_factory.scale + self.top_size,
        )

    def create_asset(self, i, **params) -> bpy.types.Object:
        obj = self.base_factory.create_asset(i=i, **params)
        horizontal = np.abs(read_edge_direction(obj)[:, -1]) < 0.1

        edge_center = read_edge_center(obj)
        z = edge_center[:, -1]
        dirt_z = self.dirt_ratio * self.base_factory.depth * self.base_factory.scale
        idx = np.argmin(np.abs(z - dirt_z) - horizontal * 10)
        radius = np.sqrt((edge_center[idx] ** 2)[:2].sum())

        selection = np.zeros_like(z).astype(bool)
        selection[idx] = True
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            select_edges(obj, selection)
            bpy.ops.mesh.loop_multi_select(ring=False)
            bpy.ops.mesh.duplicate_move()
            bpy.ops.mesh.separate(type="SELECTED")

        dirt_ = bpy.context.selected_objects[-1]
        butil.select_none()
        self.base_factory.finalize_assets(obj)
        with butil.ViewportMode(dirt_, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.fill_grid()
        subsurf(dirt_, 3)
        self.dirt_surface.apply(dirt_)
        butil.apply_modifiers(dirt_)

        remove_vertices(dirt_, lambda x, y, z: np.sqrt(x**2 + y**2) > radius * 0.92)
        dirt_.location[-1] -= 0.02

        plant = self.plant_factory.spawn_asset(i=i, loc=(0, 0, 0), rot=(0, 0, 0))
        origin2lowest(plant, approximate=True)
        self.plant_factory.finalize_assets(plant)

        scale = np.min(
            np.array([self.side_size, self.side_size, self.top_size])
            / np.max(np.abs(np.array(plant.bound_box)), 0)
        )
        plant.scale = [scale] * 3
        plant.location[-1] = dirt_z

        obj = join_objects([obj, plant, dirt_])
        return obj


class LargePlantContainerFactory(PlantContainerFactory):
    plant_factories = [MonocotFactory]

    def __init__(self, factory_seed, coarse=False):
        super(LargePlantContainerFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.base_factory.depth = log_uniform(1.0, 1.5)
            self.base_factory.scale = log_uniform(0.15, 0.25)
            self.side_size = (
                self.base_factory.scale * uniform(1.5, 2.0) * self.base_factory.r_expand
            )
            self.top_size = uniform(1, 1.5)
            # if WALL_HEIGHT - 2*WALL_THICKNESS < 3:
            #     self.top_size = uniform(1.5, WALL_HEIGHT - 2*WALL_THICKNESS)
            # else:
            #     self.top_size = uniform(1.5, 3)
            # print(f"{self.side_size=} {self.top_size=} {WALL_THICKNESS=} {WALL_HEIGHT=}")
