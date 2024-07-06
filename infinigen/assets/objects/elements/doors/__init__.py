# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np

from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed

from .casing import DoorCasingFactory
from .lite import LiteDoorFactory
from .louver import LouverDoorFactory
from .panel import GlassPanelDoorFactory, PanelDoorFactory


def random_door_factory():
    door_factories = [
        PanelDoorFactory,
        GlassPanelDoorFactory,
        LouverDoorFactory,
        LiteDoorFactory,
    ]
    door_probs = np.array([4, 2, 3, 3])
    return np.random.choice(door_factories, p=door_probs / door_probs.sum())


class DoorFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, constants=None):
        super(DoorFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.base_factory = random_door_factory()(factory_seed, coarse, constants)

    def create_asset(self, **params) -> bpy.types.Object:
        return self.base_factory.create_asset(**params)

    def finalize_assets(self, assets):
        self.base_factory.finalize_assets(assets)
