# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from mathutils import Euler, kdtree
from numpy.random import uniform
from typing import Any, ClassVar

from infinigen.assets.utils.mesh import polygon_angles
from infinigen.assets.utils.object import join_objects
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    LegacyBridgeParameters,
    ParameterizedAssetFactory,
    apply_bridge_parameters,
    legacy_init_to_parameters,
)
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed

from .growth import MushroomGrowthFactory


def _mushroom_legacy_init(inst: Any, factory_seed: int, coarse: bool = False) -> None:
    AssetFactory.__init__(inst, factory_seed, coarse)
    with FixedSeed(factory_seed):
        inst.makers = [inst.directional_make, inst.cluster_make]
        inst.maker = np.random.choice(inst.makers)
        inst.lowered = uniform(0, 1) < 0.5
        inst.factory = MushroomGrowthFactory(factory_seed, coarse)
        inst.tolerant_length = uniform(0, 0.2)


class MushroomParameters(LegacyBridgeParameters):
    pass


class MushroomFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = MushroomParameters
    max_cluster = 10

    def __init__(self, factory_seed, coarse=False):
        super(MushroomFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> MushroomParameters:
        return legacy_init_to_parameters(
            MushroomParameters,
            MushroomFactory,
            seed,
            self.coarse,
            init_fn=_mushroom_legacy_init,
        )

    def apply_parameters(
        self, params: MushroomParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)

    def create_asset(self, i, face_size, **params):
        mushrooms, keypoints = self.build_mushrooms(i, face_size)
        locations, rotations, scales = self.maker(keypoints)
        for m, l, r, s in zip(mushrooms, locations, rotations, scales):
            m.location = l
            m.rotation_euler = r
            m.scale = s
            butil.apply_transform(m, loc=True)
        obj = join_objects(mushrooms)
        butil.modify_mesh(
            obj,
            "SIMPLE_DEFORM",
            deform_method="BEND",
            angle=uniform(-np.pi / 8, np.pi / 8),
            deform_axis=np.random.choice(["X", "Y"]),
        )
        tag_object(obj, "mushroom")
        return obj

    def build_mushrooms(self, i, face_size=0.01):
        n = np.random.randint(1, 6)
        mushrooms, keypoints = [], []
        for j in range(n):
            obj = self.factory.create_asset(
                i=j + i * self.max_cluster, face_size=face_size / 2
            )
            clone = deep_clone_obj(obj)
            butil.modify_mesh(clone, "REMESH", voxel_size=0.04)
            mushrooms.append(obj)
            k = np.array(
                [v.co for v in clone.data.vertices if v.co[-1] > self.tolerant_length]
            )
            if len(k) == 0:
                k = np.array([v.co for v in clone.data.vertices])
            if len(k) == 0:
                k = np.zeros((1, 3))
            keypoints.append(k)
            butil.delete(clone)
        return mushrooms, keypoints

    @property
    def radius(self):
        return self.factory.cap_factory.radius

    def find_closest(self, keypoints, rotations, start_locs, directions):
        vertices = [k.copy() for k in keypoints]
        locations, scales = [np.zeros(3)], []
        scales = np.tile(uniform(0.3, 1.2, len(keypoints))[:, np.newaxis], 3)
        for i in range(len(vertices)):
            vertices[i] = (
                np.array(Euler(rotations[i]).to_matrix())
                @ np.diag(scales[i])
                @ vertices[i].T
            ).T
        for i in range(1, len(vertices)):
            basis = np.concatenate(vertices[:i])
            kd = kdtree.KDTree(len(basis))
            for idx, v in enumerate(basis):
                kd.insert(v, idx)
            kd.balance()
            for d in np.linspace(0, 4, 20) * self.radius:
                offset = start_locs[i] + directions[i] * d
                if min(kd.find(v + offset)[-1] for v in vertices[i]) > 0.008:
                    break
            else:
                offset = start_locs[i] + directions[i] * 4 * self.radius
            vertices[i] += offset
            locations.append(offset)
        return locations, rotations, scales

    def cluster_make(self, keypoints):
        n = len(keypoints)
        angles = polygon_angles(n, np.pi / 10, np.pi * 2)
        rot_y = uniform(0, np.pi / 6, n) if self.lowered else np.zeros(n)
        rot_z = angles + uniform(-np.pi / 8, np.pi / 8, n)
        rotations = np.stack([np.zeros(n), rot_y, rot_z], -1)
        start_locs = np.zeros((n, 3))
        directions = np.stack([np.cos(angles), np.sin(angles), np.zeros(n)], -1)
        return self.find_closest(keypoints, rotations, start_locs, directions)

    def directional_make(self, keypoints):
        n = len(keypoints)
        rot_y = uniform(0, np.pi / 6, n) if self.lowered else np.zeros(n)
        rot_z = -np.pi / 2 + uniform(-np.pi / 8, np.pi / 8, n)
        rotations = np.stack([np.zeros(n), rot_y, rot_z], -1)
        start_locs = np.stack(
            [np.linspace(0, self.radius * n * 0.4, n), np.zeros(n), np.zeros(n)], -1
        )
        directions = np.tile([0, 1, 0], (n, 1))
        return self.find_closest(keypoints, rotations, start_locs, directions)
