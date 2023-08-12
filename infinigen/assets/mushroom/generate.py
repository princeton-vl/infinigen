# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


from copy import deepcopy

import numpy as np
from mathutils import Euler, kdtree
from numpy.random import uniform

from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from .growth import MushroomGrowthFactory
from infinigen.assets.utils.decorate import join_objects
from infinigen.assets.utils.mesh import polygon_angles
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from ..utils.misc import log_uniform
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class MushroomFactory(AssetFactory):
    max_cluster = 10

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.makers = [self.directional_make, self.cluster_make]
            self.maker = np.random.choice(self.makers)
            self.lowered = uniform(0, 1) < .5
            self.factory = MushroomGrowthFactory(factory_seed, coarse)
            self.tolerant_length = uniform(0, .2)

    def create_asset(self, i, face_size, **params):
        mushrooms, keypoints = self.build_mushrooms(i, face_size)
        locations, rotations, scales = self.maker(keypoints)
        for m, l, r, s in zip(mushrooms, locations, rotations, scales):
            m.location = l
            m.rotation_euler = r
            m.scale = s
            butil.apply_transform(m, loc=True)
        obj = join_objects(mushrooms)
        butil.modify_mesh(obj, 'SIMPLE_DEFORM', deform_method='BEND', angle=uniform(- np.pi / 8, np.pi / 8),
                          deform_axis=np.random.choice(['X', 'Y']))
        tag_object(obj, 'mushroom')
        return obj

    def build_mushrooms(self, i, face_size=.01):
        n = np.random.randint(1, 6)
        mushrooms, keypoints = [], []
        for j in range(n):
            obj = self.factory.create_asset(i=j + i * self.max_cluster, face_size=face_size / 2)
            clone = deep_clone_obj(obj)
            butil.modify_mesh(clone, 'REMESH', voxel_size=.04)
            mushrooms.append(obj)
            k = np.array([v.co for v in clone.data.vertices if v.co[-1] > self.tolerant_length])
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
        scales = np.tile(uniform(.3, 1.2, len(keypoints))[:, np.newaxis], 3)
        for i in range(len(vertices)):
            vertices[i] = (np.array(Euler(rotations[i]).to_matrix()) @ np.diag(scales[i]) @ vertices[i].T).T
        for i in range(1, len(vertices)):
            basis = np.concatenate(vertices[:i])
            kd = kdtree.KDTree(len(basis))
            for idx, v in enumerate(basis):
                kd.insert(v, idx)
            kd.balance()
            for d in np.linspace(0, 4, 20) * self.radius:
                offset = start_locs[i] + directions[i] * d
                if min(kd.find(v + offset)[-1] for v in vertices[i]) > .008:
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
        start_locs = np.stack([np.linspace(0, self.radius * n * .4, n), np.zeros(n), np.zeros(n)], -1)
        directions = np.tile([0, 1, 0], (n, 1))
        return self.find_closest(keypoints, rotations, start_locs, directions)
