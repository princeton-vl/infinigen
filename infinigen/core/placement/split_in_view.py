# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import logging

import bpy
import numpy as np
from mathutils import Matrix
from mathutils.bvhtree import BVHTree
from tqdm import trange

from infinigen.core.placement.camera import get_sensor_coords
from infinigen.core.util import blender as butil
from infinigen.core.util import camera as cam_util
from infinigen.core.util.logging import Suppress
from infinigen.core.util.math import dehomogenize, homogenize

logger = logging.getLogger(__name__)


def raycast_visiblity_mask(
    obj: bpy.types.Object,
    cameras: list[bpy.types.Object],
    start=None,
    end=None,
    verbose=True,
):
    bvh = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())

    if start is None:
        start = bpy.context.scene.frame_start
    if end is None:
        end = bpy.context.scene.frame_end

    mask = np.zeros(len(obj.data.vertices), dtype=bool)
    rangeiter = trange if verbose else range
    for i in rangeiter(start, end + 1):
        bpy.context.scene.frame_set(i)
        invworld = obj.matrix_world.inverted()
        for cam in cameras:
            sensor_coords, pix_it = get_sensor_coords(cam)
            for x, y in pix_it:
                direction = (
                    sensor_coords[y, x] - cam.matrix_world.translation
                ).normalized()
                origin = cam.matrix_world.translation
                _, _, index, dist = bvh.ray_cast(
                    invworld @ origin, invworld.to_3x3() @ direction
                )
                if dist is None:
                    continue
                for vi in obj.data.polygons[index].vertices:
                    mask[vi] = True

    return mask


def select_vertmask(obj, mask):
    for i, v in enumerate(obj.data.vertices):
        v.select = mask[i]
    for f in obj.data.polygons:
        f.select = any(mask[vi] for vi in f.vertices)


def duplicate_mask(obj, mask, dilate=0, invert=False):
    butil.select_none()
    with butil.ViewportMode(obj, mode="EDIT"):
        bpy.ops.mesh.select_all(action="DESELECT")
    select_vertmask(obj, mask)
    with butil.ViewportMode(obj, mode="EDIT"):
        for _ in range(dilate):
            bpy.ops.mesh.select_more()
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type="FACE")
        if invert:
            bpy.ops.mesh.select_all(action="INVERT")
        with Suppress():
            bpy.ops.mesh.duplicate_move()
            try:
                bpy.ops.mesh.separate(type="SELECTED")
                return bpy.context.selected_objects[-1]
            except RuntimeError:
                return butil.spawn_point_cloud("duplicate_mask", [], [])


def compute_vis_dists(points: np.array, cam: bpy.types.Object):
    projmat, K, RT = map(np.array, cam_util.get_3x4_P_matrix_from_blender(cam))
    proj = points @ projmat.T
    uv, d = dehomogenize(proj), proj[:, -1]

    clamped_uv = np.clip(uv, [0, 0], butil.get_camera_res())
    clamped_d = np.maximum(d, 0)

    RT_4x4_inv = np.array(Matrix(RT).to_4x4().inverted())
    clipped_pos = (
        homogenize((homogenize(clamped_uv) * clamped_d[:, None]) @ np.linalg.inv(K).T)
        @ RT_4x4_inv.T
    )

    vis_dist = np.linalg.norm(points[:, :-1] - clipped_pos[:, :-1], axis=-1)

    return d, vis_dist


def compute_inview_distances(
    points: np.array,
    cameras: list[bpy.types.Object],
    dist_max,
    vis_margin,
    frame_start=None,
    frame_end=None,
    verbose=False,
):
    """
    Compute the minimum distance of each point to any of the cameras in the scene.

    Parameters:
    - points: an array of 3D points, in world space
    - cameras: a list of cameras in the scene
    - dist_max: the maximum distance to consider a point "in view"
    - vis_margin: how far outside the view frustum to consider a point "in view"

    Returns:
    - mask: boolean array of whether each point is within within vis_margin and dist_max of any frame of any camera
    - min_dists: the distance of each point the closest camera
    - min_vis_dists: the distance of each point to the nearest point in any camera's view frustum
    """

    assert len(points.shape) == 2 and points.shape[-1] == 3

    if frame_start is None:
        frame_start = bpy.context.scene.frame_start
    if frame_end is None:
        frame_end = bpy.context.scene.frame_end

    points = homogenize(points)

    mask = np.zeros(len(points), dtype=bool)
    min_dists = np.full(len(points), 1e7)
    min_vis_dists = np.full(len(points), 1e7)

    rangeiter = trange if verbose else range

    assert frame_start < frame_end + 1, (frame_start, frame_end)

    for frame in rangeiter(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        for cam in cameras:
            dists, vis_dists = compute_vis_dists(points, cam)
            mask |= (dists < dist_max) & (vis_dists < vis_margin)
            if mask.any():
                min_vis_dists[mask] = np.minimum(vis_dists[mask], min_vis_dists[mask])
                min_dists[mask] = np.minimum(dists[mask], min_dists[mask])

            logger.debug(f"Computed dists for {frame=} {cam.name} {mask.mean()=:.2f}")

    return mask, min_dists, min_vis_dists


def split_inview(
    obj: bpy.types.Object,
    cameras: list[bpy.types.Object],
    dist_max: float = 1e7,
    vis_margin: float = 0,
    raycast: bool = False,
    dilate: float = 0,
    outofview=True,
    verbose=False,
    hide_render=None,
    suffix=None,
    **kwargs,
):
    assert obj.type == "MESH"

    bpy.context.view_layer.update()
    verts = np.zeros((len(obj.data.vertices), 3))
    obj.data.vertices.foreach_get("co", verts.reshape(-1))
    verts = butil.apply_matrix_world(obj, verts)

    mask, dists, vis_dists = compute_inview_distances(
        verts,
        cameras,
        dist_max=dist_max,
        vis_margin=vis_margin,
        verbose=verbose,
        **kwargs,
    )

    logger.debug(f"split_inview {suffix=} {dist_max=} {vis_margin=} {mask.mean()=:.2f}")

    if raycast:
        mask *= raycast_visiblity_mask(obj, cameras)

    inview = duplicate_mask(obj, mask, dilate=dilate)

    if outofview:
        outview = duplicate_mask(obj, mask, dilate=dilate, invert=True)
    else:
        outview = butil.spawn_point_cloud("duplicate_mask", [], [])

    inview.name = obj.name + ".inview"
    outview.name = obj.name + ".outofview"

    if suffix is not None:
        inview.name += "_" + suffix
        outview.name += "_" + suffix

    if hide_render is not None:
        inview.hide_render = hide_render
        outview.hide_render = hide_render

    return inview, outview, dists, vis_dists
