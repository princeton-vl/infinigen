# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from statistics import mean
from typing import Any, ClassVar

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.fruits.general_fruit import FruitFactoryGeneralFruit
from infinigen.assets.objects.tableware.bowl import BowlFactory
from infinigen.assets.objects.tableware.pot import PotFactory
from infinigen.assets.utils.decorate import read_co, write_co
from infinigen.assets.utils.misc import make_normalized_factory, subclasses
from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.placement.instance_scatter import scatter_instances
from infinigen.core.placement.parameters import (
    LegacyBridgeParameters,
    ParameterizedAssetFactory,
    apply_bridge_parameters,
    legacy_init_to_parameters,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


class FruitCover:
    def __init__(self, factory_seed=0):
        with FixedSeed(factory_seed):
            fruit_factory_fns = list(
                subclasses(FruitFactoryGeneralFruit).difference(
                    [FruitFactoryGeneralFruit]
                )
            )
            fruit_factory_fn = make_normalized_factory(
                np.random.choice(fruit_factory_fns)
            )
            self.col = make_asset_collection(
                fruit_factory_fn(np.random.randint(1e5)), name="fruit", n=5
            )
            self.dimension = mean(mean(o.dimensions) for o in self.col.objects)
            self.shrink_rate = max(self.dimension, 2.0)

    def apply(self, obj, selection=None):
        for obj in obj if isinstance(obj, Iterable) else [obj]:
            scale = uniform(0.06, 0.08) / self.shrink_rate
            scattered = scatter_instances(
                base_obj=obj,
                collection=self.col,
                density=1e3,
                min_spacing=scale * self.dimension * uniform(0.5, 0.7),
                scale=scale,
                scale_rand=uniform(0.1, 0.3),
                selection=selection,
                ground_offset=self.dimension * 0.2 * scale,
                apply_geo=True,
                realize=True,
            )
            scattered.parent = obj


def _fruit_container_legacy_init(inst: Any, seed: int, coarse: bool) -> None:
    base_factory_fns = [BowlFactory, PotFactory]
    probs = np.array([1, 1])
    base_factory_fn = np.random.choice(base_factory_fns, p=probs / probs.sum())
    inst.base_factory = base_factory_fn(seed, coarse)
    inst.cover_seed = seed


class FruitContainerParameters(LegacyBridgeParameters):
    pass


class FruitContainerFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[LegacyBridgeParameters]] = FruitContainerParameters

    def __init__(self, factory_seed, coarse=False):
        super(FruitContainerFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> FruitContainerParameters:
        return legacy_init_to_parameters(
            FruitContainerParameters,
            FruitContainerFactory,
            seed,
            self.coarse,
            init_fn=_fruit_container_legacy_init,
        )

    def apply_parameters(
        self, params: FruitContainerParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    @cached_property
    def cover(self):
        return FruitCover(self.cover_seed)

    def create_placeholder(self, **params):
        box = self.base_factory.create_placeholder(**params)
        co = read_co(box)
        co[co[:, -1] > 0.02, -1] += 0.05
        co[co[:, -1] < 0.02, -1] -= 0.01
        write_co(box, co)
        butil.apply_transform(box)
        return box

    def create_asset(self, **params) -> bpy.types.Object:
        return self.base_factory.create_asset(**params)

    def finalize_assets(self, assets):
        self.base_factory.finalize_assets(assets)
        self.cover.apply(assets, selection="lower_inside")
