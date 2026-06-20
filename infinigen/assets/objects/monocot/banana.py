# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from __future__ import annotations

from typing import Any, ClassVar

import bmesh
import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.monocot.growth import MonocotGrowthFactory
from infinigen.assets.utils.decorate import displace_vertices, read_co
from infinigen.assets.utils.draw import bezier_curve, leaf
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import join_objects, origin2lowest
from infinigen.assets.utils.shapes import point_normal_up
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    LegacyBridgeParameters,
    ParameterizedAssetFactory,
    apply_bridge_parameters,
    legacy_init_to_parameters,
)
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class BananaMonocotParameters(LegacyBridgeParameters):
    pass


def _banana_monocot_legacy_init(inst: Any, seed: int, coarse: bool) -> None:
    MonocotGrowthFactory.__init__(inst, seed, coarse)
    with FixedSeed(seed):
        inst.stem_offset = uniform(0.6, 1.0)
        inst.angle = uniform(np.pi / 4, np.pi / 3)
        inst.z_scale = uniform(1, 1.5)
        inst.z_drag = uniform(0.1, 0.2)
        inst.min_y_angle = uniform(np.pi * 0.05, np.pi * 0.1)
        inst.max_y_angle = uniform(np.pi * 0.25, np.pi * 0.45)
        inst.leaf_range = uniform(0.5, 0.7), 1
        inst.count = int(log_uniform(16, 24))
        inst.scale_curve = [(0, uniform(0.4, 1.0)), (1, uniform(0.6, 1.0))]
        inst.radius = uniform(0.04, 0.06)
        inst.bud_angle = uniform(np.pi / 8, np.pi / 6)
        inst.cut_angle = inst.bud_angle + uniform(np.pi / 20, np.pi / 12)
        inst.freq = log_uniform(100, 300)
        inst.n_cuts = np.random.randint(6, 10) if uniform(0, 1) < 0.8 else 0


class BananaMonocotFactory(ParameterizedAssetFactory, MonocotGrowthFactory):
    parameters_model: ClassVar[type[LegacyBridgeParameters]] = BananaMonocotParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> BananaMonocotParameters:
        return legacy_init_to_parameters(
            BananaMonocotParameters,
            BananaMonocotFactory,
            seed,
            self.coarse,
            init_fn=_banana_monocot_legacy_init,
        )

    def apply_parameters(
        self, params: BananaMonocotParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    @staticmethod
    def build_base_hue():
        return uniform(0.15, 0.35)

    def cut_leaf(self, obj):
        coords = read_co(obj)
        x, y, z = coords.T
        coords = coords[(np.abs(y) < 0.08) & (np.abs(y) > 0.01)]
        positive_coords = coords[coords.T[1] > 0]
        positive_coords = positive_coords[np.argsort(positive_coords[:, 0])]
        negative_coords = coords[coords.T[1] < 0]
        negative_coords = negative_coords[np.argsort(negative_coords[:, 0])]
        positive_coords = positive_coords[
            np.random.choice(len(positive_coords), self.n_cuts, replace=False)
        ]
        negative_coords = negative_coords[
            np.random.choice(len(negative_coords), self.n_cuts, replace=False)
        ]

        for (x1, y1, _), (x2, y2, _) in zip(
            np.concatenate([positive_coords[:-1], negative_coords[:-1]], 0),
            np.concatenate([positive_coords[1:], negative_coords[1:]], 0),
        ):
            coeff = 1 if y1 > 0 else -1
            ratio = uniform(-2.0, 0.4)
            exponent = uniform(1.2, 1.6)

            def cut(x, y, z):
                m1 = x1 * np.sin(self.cut_angle) - y1 * np.cos(self.cut_angle) * coeff
                m2 = x2 * np.sin(self.cut_angle) - y2 * np.cos(self.cut_angle) * coeff
                m = x * np.sin(self.cut_angle) - y * np.cos(self.cut_angle) * coeff
                dist = ((x - x1) * (y1 - y2) + (y - y1) * (x1 - x2)) / np.sqrt(
                    (x1 - x2) ** 2 + (y1 - y2) ** 2 + 0.1
                )
                return (
                    0,
                    0,
                    np.where(
                        (m1 < m) & (m < m2) & (dist * coeff < 0),
                        ratio * np.abs(dist) ** exponent,
                        0,
                    ),
                )

            displace_vertices(obj, cut)
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [e for e in bm.edges if e.calc_length() > 0.02]
            bmesh.ops.delete(bm, geom=geom, context="EDGES")
            bmesh.update_edit_mesh(obj.data)

    def build_leaf(self, face_size):
        x_anchors = 0, 0.2 * np.cos(self.bud_angle), uniform(0.8, 1.2), 2.0
        y_anchors = 0, 0.2 * np.sin(self.bud_angle), uniform(0.2, 0.25), 0
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        self.cut_leaf(obj)
        self.displace_veins(obj)
        self.decorate_leaf(obj)
        tag_object(obj, "banana")
        return obj

    def displace_veins(self, obj):
        vg = obj.vertex_groups.new(name="distance")
        x, y, z = read_co(obj).T
        branch = np.cos(
            (np.abs(y) * np.cos(self.cut_angle) - x * np.sin(self.cut_angle))
            * self.freq
        ) > uniform(0.85, 0.9, len(x))
        leaf = np.abs(y) < uniform(0.002, 0.008, len(x))
        weights = branch | leaf
        for i, l in enumerate(weights):
            vg.add([i], l, "REPLACE")
        butil.modify_mesh(
            obj,
            "DISPLACE",
            strength=-uniform(5e-3, 8e-3),
            mid_level=0,
            vertex_group="distance",
        )


class TaroMonocotParameters(LegacyBridgeParameters):
    pass


def _taro_monocot_legacy_init(
    inst: TaroMonocotFactory, seed: int, coarse: bool
) -> None:
    BananaMonocotFactory.__init__(inst, seed, coarse)
    with FixedSeed(seed):
        inst.stem_offset = uniform(0.05, 0.1)
        inst.radius = uniform(0.02, 0.04)
        inst.z_drag = uniform(0.2, 0.3)
        inst.bud_angle = uniform(np.pi * 0.6, np.pi * 0.7)
        inst.freq = log_uniform(10, 20)
        inst.count = int(log_uniform(12, 16))
        inst.n_cuts = np.random.randint(1, 2) if uniform(0, 1) < 0.5 else 0
        inst.min_y_angle = uniform(-np.pi * 0.25, -np.pi * 0.05)
        inst.max_y_angle = uniform(-np.pi * 0.05, 0)


class TaroMonocotFactory(BananaMonocotFactory):
    parameters_model: ClassVar[type[LegacyBridgeParameters]] = TaroMonocotParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> TaroMonocotParameters:
        return legacy_init_to_parameters(
            TaroMonocotParameters,
            TaroMonocotFactory,
            seed,
            self.coarse,
            init_fn=_taro_monocot_legacy_init,
        )

    def apply_parameters(
        self, params: TaroMonocotParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    def displace_veins(self, obj):
        point_normal_up(obj)
        vg = obj.vertex_groups.new(name="distance")
        x, y, z = read_co(obj).T
        branch = np.cos(
            uniform(0, np.pi * 2)
            + np.arctan2(
                y - np.where(y > 0, -1, 1) * uniform(0.1, 0.2), x - uniform(0.1, 0.4)
            )
            * self.freq
        ) > uniform(0.98, 0.99, len(x))
        leaf = np.abs(y) < uniform(0.002, 0.008, len(x))
        weights = branch | leaf
        for i, l in enumerate(weights):
            vg.add([i], l, "REPLACE")
        butil.modify_mesh(
            obj,
            "DISPLACE",
            strength=-uniform(5e-3, 8e-3),
            mid_level=0,
            vertex_group="distance",
        )

    def build_leaf(self, face_size):
        x_anchors = (
            0,
            0.2 * np.cos(self.bud_angle),
            uniform(0.4, 1.0),
            uniform(0.8, 1.0),
        )
        y_anchors = 0, 0.2 * np.sin(self.bud_angle), uniform(0.25, 0.3), 0
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        self.cut_leaf(obj)
        self.displace_veins(obj)
        self.decorate_leaf(obj, 2, leftmost=False)
        bezier = self.build_branch()
        obj = join_objects([obj, bezier])
        origin2lowest(obj)
        tag_object(obj, "taro")
        return obj

    def build_branch(self):
        offset = uniform(0.2, 0.3)
        length = uniform(1, 2)
        x_anchors = 0, -0.05, -offset - uniform(0.01, 0.02), -offset
        z_anchors = 0, 0, -length + 0.1, -length
        bezier = bezier_curve([x_anchors, 0, z_anchors])
        surface.add_geomod(
            bezier, geo_radius, apply=True, input_args=[uniform(0.02, 0.03), 32]
        )
        return bezier

    def build_instance(self, i, face_size):
        return self.build_leaf(face_size)
