# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from __future__ import annotations

from typing import Any, ClassVar

import bpy
import numpy as np
from numpy.random import uniform

import infinigen.core.util.blender as butil
from infinigen.assets.objects.creatures.util.animation.driver_repeated import (
    repeated_driver,
)
from infinigen.assets.objects.monocot.growth import MonocotGrowthFactory
from infinigen.assets.utils.draw import bezier_curve, leaf
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.object import join_objects, origin2leftmost
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.detail import remesh_with_attrs
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    LegacyBridgeParameters,
    ParameterizedAssetFactory,
    apply_bridge_parameters,
    legacy_init_to_parameters,
)
from infinigen.core.util.random import log_uniform


def _monocot_base_legacy_init(inst: Any, seed: int, coarse: bool) -> None:
    base = MonocotGrowthFactory.__new__(MonocotGrowthFactory)
    AssetFactory.__init__(base, seed, coarse)
    MonocotGrowthFactory.__init__(base, seed, coarse)
    for key, value in vars(base).items():
        if key not in ("factory_seed", "coarse"):
            setattr(inst, key, value)


def _kelp_legacy_init(inst: Any, seed: int, coarse: bool) -> None:
    _monocot_base_legacy_init(inst, seed, coarse)
    inst.stem_offset = 10.0
    inst.angle = uniform(np.pi / 6, np.pi / 4)
    inst.z_drag = uniform(0.0, 0.2)
    inst.min_y_angle = uniform(0, np.pi * 0.1)
    inst.max_y_angle = inst.min_y_angle
    inst.bend_angle = uniform(0, np.pi / 6)
    inst.twist_angle = uniform(0, np.pi / 6)
    inst.count = 512
    inst.leaf_prob = uniform(0.6, 0.7)
    inst.align_angle = uniform(np.pi / 30, np.pi / 15)
    inst.radius = 0.02
    inst.align_factor = inst.make_align_factor()
    inst.align_direction = inst.make_align_direction()
    flow_angle = uniform(0, np.pi * 2)
    inst.align_direction = (
        np.cos(flow_angle),
        np.sin(flow_angle),
        uniform(-0.2, 0.2),
    )
    inst.anim_freq = 1 / log_uniform(100, 200)
    inst.anim_offset = uniform(0, 1)
    inst.anim_seed = np.random.randint(1e5)


class KelpMonocotParameters(LegacyBridgeParameters):
    pass


class KelpMonocotFactory(ParameterizedAssetFactory, MonocotGrowthFactory):
    parameters_model: ClassVar[type[AssetParameters]] = KelpMonocotParameters
    max_leaf_length = 1.2
    align_angle = uniform(np.pi / 24, np.pi / 12)

    def __init__(self, factory_seed, coarse=False):
        super(KelpMonocotFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> KelpMonocotParameters:
        return legacy_init_to_parameters(
            KelpMonocotParameters,
            KelpMonocotFactory,
            seed,
            self.coarse,
            init_fn=_kelp_legacy_init,
        )

    def apply_parameters(
        self, params: KelpMonocotParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    def make_align_factor(self):
        def align_factor(nw: NodeWrangler):
            rand = nw.uniform(0.7, 0.95)
            driver = rand.inputs[2].driver_add("default_value").driver
            driver.expression = repeated_driver(
                0.7, 0.85, self.anim_freq, self.anim_offset, self.anim_seed
            )
            return nw.scalar_multiply(nw.bernoulli(0.9), rand)

        return align_factor

    def make_align_direction(self):
        def align_direction(nw: NodeWrangler):
            direction = nw.combine(1, 0, 0)
            driver = direction.inputs[2].driver_add("default_value").driver
            driver.expression = repeated_driver(
                -0.5, -0.1, self.anim_freq, self.anim_offset, self.anim_seed
            )
            return direction

        return align_direction

    @staticmethod
    def build_base_hue():
        return uniform(0.05, 0.25)

    def build_instance(self, i, face_size):
        x_anchors = np.array([0, -0.02, -0.04])
        y_anchors = np.array([0, uniform(0.01, 0.02), 0])
        curves = []
        for angle in np.linspace(0, np.pi * 2, 6):
            anchors = [x_anchors, np.cos(angle) * y_anchors, np.sin(angle) * y_anchors]
            curves.append(bezier_curve(anchors))
        bud = butil.join_objects(curves)
        bud.location[0] += 0.02
        with butil.ViewportMode(bud, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.convex_hull()
        remesh_with_attrs(bud, face_size)

        x_anchors = 0, uniform(0.35, 0.65), uniform(0.8, 1.2)
        y_anchors = 0, uniform(0.06, 0.08), 0
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        obj = join_objects([obj, bud])
        self.decorate_leaf(
            obj,
            uniform(-2, 2),
            uniform(-np.pi / 4, np.pi / 4),
            uniform(-np.pi / 4, np.pi / 4),
        )
        origin2leftmost(obj)
        return obj

    def create_asset(self, **params):
        obj = self.create_raw(**params)
        self.decorate_monocot(obj)
        assign_material(obj, self.material)
        return obj
