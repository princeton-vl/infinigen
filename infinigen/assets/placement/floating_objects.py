# Copyright (C) 2024, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: David Yan

import logging

import bmesh
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import random_general as rg

logger = logging.getLogger(__name__)


def create_bvh_tree_from_object(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    bvh = BVHTree.FromBMesh(bm)
    bm.free()
    return bvh


def check_bvh_intersection(bvh1, bvh2):
    if isinstance(bvh2, list):
        return any([check_bvh_intersection(bvh1, bvh) for bvh in bvh2])
    else:
        return bvh1.overlap(bvh2)


def raycast_sample(min_dist, sensor_coords, pix_it, camera, bvhtree):
    cam_location = camera.matrix_world.to_translation()

    for _ in range(1500):
        x = pix_it[np.random.randint(0, pix_it.shape[0])][0]
        y = pix_it[np.random.randint(0, pix_it.shape[0])][1]

        direction = (sensor_coords[y, x] - camera.matrix_world.translation).normalized()

        location, normal, index, distance = bvhtree.ray_cast(cam_location, direction)

        if location:
            if distance <= min_dist:
                continue
            random_distance = np.random.uniform(min_dist, distance)
            sampled_point = cam_location + direction.normalized() * random_distance
            return sampled_point

    logger.info("Couldnt find far enough away pixel to raycast to")
    return None


def bbox_sample(bbox):
    raise NotImplementedError


class FloatingObjectPlacement:
    def __init__(
        self,
        generators: list[AssetFactory],
        camera: bpy.types.Object,
        background_objs: bpy.types.Object | list[bpy.types.Object],
        collision_objs: bpy.types.Object | list[bpy.types.Object],
        bbox=None,
    ):
        self.generators = generators
        self.camera = camera

        if not isinstance(background_objs, list):
            background_objs = [background_objs]
        self.background_objs = background_objs

        if not isinstance(collision_objs, list):
            collision_objs = [collision_objs]
        self.collision_objs = collision_objs

        self.bbox = bbox

    def place_objs(
        self,
        num_objs,
        min_dist=1,
        sample_retries=200,
        raycast=True,
        normalize=False,
        collision_placed=False,
        collision_existing=False,
    ):
        background_copied = butil.join_objects(
            [butil.copy(obj) for obj in self.background_objs]
        )
        room_bvh = create_bvh_tree_from_object(background_copied)
        butil.delete(background_copied)

        if len(self.collision_objs) > 0:
            objs_copied = butil.join_objects(
                [butil.copy(obj) for obj in self.collision_objs]
            )
            existing_obj_bvh = create_bvh_tree_from_object(objs_copied)
            butil.delete(objs_copied)
        else:
            existing_obj_bvh = None

        placed_obj_bvhs = []

        from infinigen.core.placement.camera import get_sensor_coords

        sensor_coords, pix_it = get_sensor_coords(self.camera, sparse=False)
        num_place = rg(num_objs)
        for i in range(num_place):
            fac = np.random.choice(self.generators)(np.random.randint(1e7))
            asset = fac.spawn_asset(0)
            fac.finalize_assets([asset])
            max_dim = max(asset.dimensions.x, asset.dimensions.y, asset.dimensions.z)

            if normalize:
                if max_dim != 0:
                    normalize_scale = 0.5 / max(
                        asset.dimensions.x, asset.dimensions.y, asset.dimensions.z
                    )
                else:
                    normalize_scale = 1

                asset.scale = (normalize_scale, normalize_scale, normalize_scale)

            for j in range(sample_retries):
                if raycast:
                    point = raycast_sample(
                        min_dist, sensor_coords, pix_it, self.camera, room_bvh
                    )
                else:
                    point = bbox_sample()

                if point is None:
                    continue

                asset.rotation_mode = "XYZ"
                asset.rotation_euler = np.random.uniform(-np.pi, np.pi, 3)
                asset.location = point

                bpy.context.view_layer.update()  # i can redo this later without view updates if necessary, but currently it doesn't incur significant overhead
                bvh = create_bvh_tree_from_object(asset)

                if (
                    check_bvh_intersection(bvh, room_bvh)
                    or (
                        not collision_existing
                        and existing_obj_bvh is not None
                        and check_bvh_intersection(bvh, existing_obj_bvh)
                    )
                    or (
                        not collision_placed
                        and check_bvh_intersection(bvh, placed_obj_bvhs)
                    )
                ):
                    logger.debug(f"Sample {j} of asset {i} rejected, resampling...")
                    if i == sample_retries - 1:
                        butil.delete(asset)
                else:
                    logger.info(
                        f"{self.__class__.__name__} placing object {i}/{num_place}, {asset.name=}"
                    )
                    placed_obj_bvhs.append(bvh)
                    break
