# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.cactus import CactusFactory
from infinigen.assets.monocot import MonocotFactory
from infinigen.assets.mushroom import MushroomFactory
from infinigen.assets.small_plants import FernFactory, SnakePlantFactory, SpiderPlantFactory, SucculentFactory
from infinigen.assets.tableware import PotFactory
from infinigen.assets.utils.decorate import (
    read_edge_center, read_edge_direction, remove_vertices,
    select_edges, subsurf,
)
from infinigen.assets.utils.object import center, join_objects, new_bbox, origin2lowest
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform
from infinigen.core.util import blender as butil


class PlantPotFactory(PotFactory):
    def __init__(self, factory_seed, coarse=False):
        super(PlantPotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.has_handle = self.has_bar = self.has_guard = False
            self.depth = log_uniform(.5, 1.)
            self.r_expand = uniform(1.1, 1.3)
            alpha = uniform(.5, .8)
            self.r_mid = (self.r_expand - 1) * alpha + 1
            self.scale = log_uniform(.08, .12)


class PlantContainerFactory(AssetFactory):
        SnakePlantFactory]

    def __init__(self, factory_seed, coarse=False):
        super(PlantContainerFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.base_factory = PlantPotFactory(self.factory_seed, coarse)
            self.dirt_ratio = uniform(.7, .8)
            fn = np.random.choice(self.plant_factories)
            self.plant_factory = fn(self.factory_seed)
            self.side_size = self.base_factory.scale * self.base_factory.r_expand
            self.top_size = uniform(.4, .6)

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
            -self.side_size,
            self.side_size,
            -self.side_size,
            self.side_size,
            -.02,

        horizontal = np.abs(read_edge_direction(obj)[:, -1]) < .1
        edge_center = read_edge_center(obj)
        z = edge_center[:, -1]
        dirt_z = self.dirt_ratio * self.base_factory.depth * self.base_factory.scale
        selection = np.zeros_like(z).astype(bool)
        with butil.ViewportMode(obj, 'EDIT'):
            bpy.ops.mesh.select_mode(type="EDGE")
            select_edges(obj, selection)
            bpy.ops.mesh.loop_multi_select(ring=False)
            bpy.ops.mesh.duplicate_move()
            bpy.ops.mesh.separate(type='SELECTED')
        dirt_ = bpy.context.selected_objects[-1]
        butil.select_none()
        self.base_factory.finalize_assets(obj)
        with butil.ViewportMode(dirt_, 'EDIT'):
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.fill_grid()
        subsurf(dirt_, 3)
        butil.apply_modifiers(dirt_)
        dirt_.location[-1] -= .02
        plant = self.plant_factory.spawn_asset(i=i, loc=(0, 0, 0), rot=(0, 0, 0))
        origin2lowest(plant, approximate=True)
        self.plant_factory.finalize_assets(plant)
        scale = np.min(
            np.array([self.side_size, self.side_size, self.top_size]) / np.max(
                np.abs(np.array(plant.bound_box)), 0
            )
        )
        plant.scale = [scale] * 3
        plant.location[-1] = dirt_z
        obj = join_objects([obj, plant, dirt_])
        return obj


class LargePlantContainerFactory(PlantContainerFactory):

    def __init__(self, factory_seed, coarse=False):
        super(LargePlantContainerFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.base_factory.depth = log_uniform(1., 1.5)
            self.base_factory.scale = log_uniform(.15, .25)
            self.side_size = self.base_factory.scale * uniform(1.5, 2.) * self.base_factory.r_expand
