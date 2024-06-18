# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Stamatis Alexandropulos

import colorsys

import bpy
import mathutils
import numpy as np
import trimesh
from numpy.random import uniform

from infinigen.assets.corals import CoralFactory
from infinigen.assets.creatures.beetle import AntSwarmFactory, BeetleFactory
from infinigen.assets.creatures.bird import BirdFactory, FlyingBirdFactory
from infinigen.assets.creatures.carnivore import CarnivoreFactory
from infinigen.assets.creatures.crustacean import CrabFactory, CrustaceanFactory, LobsterFactory, SpinyLobsterFactory
from infinigen.assets.creatures.herbivore import HerbivoreFactory
from infinigen.assets.creatures.insects.dragonfly import DragonflyFactory
from infinigen.assets.creatures.reptile import FrogFactory
from infinigen.assets.mollusk import (
    AugerFactory,
    ClamFactory,
    ConchFactory,
    MolluskFactory,
    MusselFactory,
    ScallopFactory,
    VoluteFactory,
)
from infinigen.assets.monocot import PineconeFactory
from infinigen.assets.rocks import BlenderRockFactory
from infinigen.assets.rocks.boulder import BoulderFactory
from infinigen.assets.utils import object as obj
from infinigen.assets.utils.decorate import remove_vertices
from infinigen.assets.utils.object import join_objects
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


class NatureShelfTrinketsFactory(AssetFactory):
    factories = [
        CoralFactory,
        BlenderRockFactory,
        BoulderFactory,
        PineconeFactory,
        MolluskFactory,
        AugerFactory,
        ClamFactory,
        ConchFactory,
        MusselFactory,
        ScallopFactory,
        VoluteFactory,
        CarnivoreFactory,
        HerbivoreFactory,
    ]
    probs = np.array([1, 1, 1, 1, 3, 2, 3, 2, 2, 2, 2, 5, 5])

    def __init__(self, factory_seed, coarse=False):
        super(NatureShelfTrinketsFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            base_factory_fn = np.random.choice(self.factories, p=self.probs / self.probs.sum())

            kwargs = {}
            if base_factory_fn in [HerbivoreFactory, CarnivoreFactory]:
                kwargs.update({"hair": False})

            self.base_factory = base_factory_fn(self.factory_seed, **kwargs)

    def create_placeholder(self, **params) -> bpy.types.Object:
        size = np.random.uniform(0.1, 0.15)
        bpy.ops.mesh.primitive_cube_add(size=size, location=(0, 0, size / 2))
        placeholder = bpy.context.active_object
        return placeholder

    def create_asset(self, i, placeholder=None, **params):
        asset = self.base_factory.spawn_asset(np.random.randint(1e7), distance=200, adaptive_resolution=False)

        if list(asset.children):
            asset = join_objects(list(asset.children))

        # butil.modify_mesh(asset, 'DECIMATE')
        butil.apply_transform(asset, loc=True)
        butil.apply_modifiers(asset)
        if isinstance(self.base_factory, HerbivoreFactory) or isinstance(self.base_factory, CarnivoreFactory):
            pass
        else:
            if not isinstance(asset, trimesh.Trimesh):
                mesh = obj.obj2trimesh(asset)
            stable_poses, probs = trimesh.poses.compute_stable_poses(mesh)
            stable_pose = stable_poses[np.argmax(probs)]
            asset.rotation_euler = mathutils.Matrix(stable_pose[:3, :3]).to_euler()
        butil.apply_transform(asset, rot=True)
        dim = asset.dimensions
        bounding_box = placeholder.dimensions
        scale = min([bounding_box[i] / dim[i] for i in range(3)])
        asset.scale = [scale for i in range(3)]
        # asset.dimensions = placeholder.dimensions
        butil.apply_transform(asset, loc=True)
        bounds = butil.bounds(asset)
        cur_loc = asset.location
        new_location = [cur_loc[i] - (bounds[0][i] + bounds[1][i]) / 2 for i in range(3)]
        new_location[2] = cur_loc[2] - (bounds[0][2] + bounding_box[2] / 2)
        asset.location = new_location
        butil.apply_transform(asset, loc=True)
        return asset
