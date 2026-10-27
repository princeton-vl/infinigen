# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma, Lahav Lipson: stationary camera selection in the original Infinigen camera placement (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/placement/camera.py)
# - Alexander Raistrick: proposal/validate refactor and camera animation in the original Infinigen camera placement (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/placement/camera.py)
# - Lingjie Mei: `get_camera_trajectory` in the original Infinigen camera placement (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/placement/camera.py)
# - David Yan, Jia Deng: optimized implementation

import inspect
import logging
import random
import typing
from copy import deepcopy

import bpy
import gin
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from tqdm import tqdm

from infinigen.core.placement import camera as _default
from infinigen.core.placement.camera import (
    camera_pose_proposal,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.organization import SelectionCriterions
from infinigen.core.util.random import random_general
from infinigen.perf.core.util import isolation as _isolation
from infinigen.terrain import Terrain

# these run in place of the default functions, so they log as the default module
logger = logging.getLogger(_default.__name__)

# Camera poses evaluated per batch by the batched search.
_SEARCH_BATCH_SIZE = 512


_sensor_grid_cache = {}


def _sensor_grid(key, build):
    """Pose-independent camera-space pixel grid for `key`, cached for the latest key."""
    grid = _sensor_grid_cache.get(key)
    if grid is None:
        grid = build()
        _sensor_grid_cache.clear()
        _sensor_grid_cache[key] = grid
    return grid


def _sensor_intrinsics(cam, H, W):
    """(u_0, v_0, s_u, s_v, f_in_m) of `cam` for an H x W image, computed as the default
    `get_sensor_coords` computes them."""
    camd = cam.data
    f_in_m = camd.lens / 1000
    scene = bpy.context.scene
    resolution_x_in_px = W
    resolution_y_in_px = H

    scale = scene.render.resolution_percentage / 100
    sensor_width_in_m = camd.sensor_width / 1000
    sensor_height_in_m = camd.sensor_height / 1000
    assert abs(sensor_width_in_m / sensor_height_in_m - W / H) < 1e-4, (
        sensor_width_in_m,
        sensor_height_in_m,
        W,
        H,
    )

    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if camd.sensor_fit == "VERTICAL":
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = (
            resolution_x_in_px * scale / sensor_width_in_m / pixel_aspect_ratio
        )  # pixels per milimeter
        s_v = resolution_y_in_px * scale / sensor_height_in_m

    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_m
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_m

    u_0 = resolution_x_in_px * scale / 2  # cx (in pixels) Usually is just W/2
    v_0 = resolution_y_in_px * scale / 2  # cx (in pixels) Usually is just H/2
    return u_0, v_0, s_u, s_v, f_in_m


def _sampled_sensor_coords(ii, W, intrinsics):
    """Camera-space coordinates of the sampled pixels `ii` (flat y * W + x) as float
    lists, equal to the grid entries `get_sensor_coords` builds for them."""
    u_0, v_0, s_u, s_v, f_in_m = intrinsics
    xx = (ii % W).astype(float)
    yy = (ii // W).astype(float)
    coords_x = (xx - u_0) / s_u
    coords_y = (yy - v_0 + 1) / s_v
    coords_z = np.full(coords_x.shape, -f_in_m)
    return np.stack((coords_x, coords_y, coords_z), axis=-1).tolist()


def get_sensor_coords(cam, H, W, sparse=False):
    u_0, v_0, s_u, s_v, f_in_m = _sensor_intrinsics(cam, H, W)

    def build_grid():
        xx, yy = np.meshgrid(np.arange(W).astype(float), np.arange(H).astype(float))
        coords_x = (xx - u_0) / s_u  # relative, in mm
        coords_y = (yy - v_0 + 1) / s_v  # relative, in mm
        coords_z = np.full(coords_x.shape, -f_in_m)
        relative_cam_coords = np.stack((coords_x, coords_y, coords_z), axis=-1)
        pixel_locs = np.stack(
            (np.meshgrid(np.arange(W), np.arange(H))), axis=-1
        ).reshape((W * H, 2))  # np.array(list(product(range(H), range(W))))
        return relative_cam_coords, pixel_locs

    relative_cam_coords, pixel_locs = _sensor_grid(
        (H, W, u_0, v_0, s_u, s_v, f_in_m), build_grid
    )

    if sparse:
        ii = np.random.choice(H * W, size=1000)
        pixel_locs = pixel_locs[ii]
        # only the sampled pixels are read (as coords[y, x]), so a dict keyed by
        # (y, x) stands in for the full object array; Vector() rounds Python and
        # numpy floats to the same float32
        matrix_world = cam.matrix_world
        cam_coords_vectors = {}
        rel = relative_cam_coords.reshape(H * W, 3)[ii].tolist()
        for (x, y), r in zip(pixel_locs.tolist(), rel):
            cam_coords_vectors[y, x] = matrix_world @ Vector(r)
        return cam_coords_vectors, pixel_locs

    cam_coords_vectors = np.empty((H, W), dtype=Vector)
    matrix_world = cam.matrix_world
    for x, y in pixel_locs:
        pixelVector = Vector(relative_cam_coords[y, x])
        cam_coords_vectors[y, x] = matrix_world @ pixelVector

    return cam_coords_vectors, pixel_locs


def terrain_camera_query(
    cam: bpy.types.Object,
    scene_bvh: BVHTree,
    terrain_tags_queries,
    vertexwise_min_dist,
    min_dist=0,
):
    dists = []
    sensor_coords, pix_it = _default.get_sensor_coords(cam, sparse=True)
    terrain_tags_queries_counts = {q: 0 for q in terrain_tags_queries}

    origin = cam.matrix_world.translation
    ray_cast = scene_bvh.ray_cast
    queries = list(terrain_tags_queries.items())
    for x, y in pix_it.tolist():
        direction = (sensor_coords[y, x] - origin).normalized()
        _, _, index, dist = ray_cast(origin, direction)
        if dist is None:
            continue
        dists.append(dist)
        if dist < min_dist or (
            vertexwise_min_dist is not None and dist < vertexwise_min_dist[index]
        ):
            logger.debug("Found dist=%s < min_dist=%s", dist, min_dist)
            dists = None  # means dist < min
            break
        for q, values in queries:
            terrain_tags_queries_counts[q] += values[index]

    n_pix = pix_it.shape[0]

    return dists, terrain_tags_queries_counts, n_pix


# The sparse sensor-pixel draw of the pose validator: H and W of
# get_sensor_coords, resolved once from its gin bindings.
_sparse_query = {"resolved": False, "H": None, "W": None, "first": True}
# (array, its list) of the last vertexwise minima seen
_vertexwise_list = [None, None]
# (camera name, x, y, z) -> terrain SDF (1-element array), filled per animation step
_sdf_prefetch = {}


def _sparse_pixels(cam):
    """Flat pixel indices of `get_sensor_coords(cam, sparse=True)`'s draw, consuming the
    random stream as it does. Only the first draw goes through it, which gin records in
    the operative config; None if one of its bindings is a gin reference."""
    st = _sparse_query
    if not st["resolved"]:
        sp = _resolved_call_params(_default.get_sensor_coords, {})
        st["resolved"] = True
        if sp is not None:
            st["H"], st["W"] = sp["H"], sp["W"]
    if st["H"] is None:
        return None
    if st["first"]:
        st["first"] = False
        _, pixel_locs = _default.get_sensor_coords(cam, sparse=True)
        return pixel_locs[:, 1] * st["W"] + pixel_locs[:, 0]
    return np.random.choice(st["H"] * st["W"], size=1000)


def _vertexwise_as_list(vertexwise_min_dist):
    """float64 vertexwise minima as a list of the same doubles, which the ray
    loop indexes faster; other inputs are returned unchanged."""
    if (
        isinstance(vertexwise_min_dist, np.ndarray)
        and vertexwise_min_dist.dtype == np.float64
    ):
        if _vertexwise_list[0] is not vertexwise_min_dist:
            _vertexwise_list[0] = vertexwise_min_dist
            _vertexwise_list[1] = vertexwise_min_dist.tolist()
        return _vertexwise_list[1]
    return vertexwise_min_dist


def _validator_query(
    cam, scene_bvh, terrain_tags_queries, vertexwise_min_dist, min_dist, coverage_range
):
    """`terrain_camera_query` for the pose validator, with the same random draw and ray
    casts but coordinates for the sampled pixels only. With `coverage_range` (min, max)
    it returns (None, zero sums, n) once the coverage test is sure to fail."""
    ii = _sparse_pixels(cam)
    if ii is None:
        return terrain_camera_query(
            cam, scene_bvh, terrain_tags_queries, vertexwise_min_dist, min_dist=min_dist
        )
    queries = list(terrain_tags_queries.items())
    rel = _sampled_sensor_coords(
        ii,
        _sparse_query["W"],
        _sensor_intrinsics(cam, _sparse_query["H"], _sparse_query["W"]),
    )
    min_coverage = max_coverage = None
    if coverage_range is not None:
        min_coverage, max_coverage = coverage_range[0], coverage_range[1]
    dists, hits = _ray_query(
        cam.matrix_world,
        rel,
        scene_bvh,
        _vertexwise_as_list(vertexwise_min_dist),
        min_dist,
        min_coverage,
        max_coverage,
    )
    if dists is None:
        return None, {q: 0 for q, _ in queries}, len(rel)
    return dists, _query_counts(queries, hits), len(rel)


def prefetch_terrain_sdf(terrain, cams, frames):
    """Evaluate the terrain SDF at the positions `cams` take on `frames` in one call and
    keep the values for `keep_cam_pose_proposal`, which evaluates one point at a time.
    Positions are read as the validator reads them, after `set_frame`, and each value is
    the one a single-point evaluation returns."""
    from infinigen.perf.core.placement import animation_policy as _policy

    if terrain is None or not cams:
        return
    keys = []
    pts = []
    for f in frames:
        _policy.set_frame(f)
        for cam in cams:
            p = cam.matrix_world.translation
            keys.append((cam.name, p.x, p.y, p.z))
            pts.append((p.x, p.y, p.z))
    sdf = terrain.compute_camera_space_sdf(np.array(pts, dtype=np.float64))
    _sdf_prefetch.clear()
    for i, k in enumerate(keys):
        _sdf_prefetch[k] = sdf[i : i + 1]


def _terrain_sdf_at(terrain, cam):
    """`terrain.compute_camera_space_sdf` at `cam`'s world position, taken from the
    prefetch when that position was prefetched, else computed as the default does."""
    p = cam.matrix_world.translation
    v = _sdf_prefetch.get((cam.name, p.x, p.y, p.z))
    if v is not None:
        return v
    return terrain.compute_camera_space_sdf(np.array(p).reshape((1, 3)))


def keep_cam_pose_proposal(
    cam: bpy.types.Object,
    terrain: Terrain,
    scene_bvh: BVHTree,
    placeholders_kd,
    camera_selection_answers,
    vertexwise_min_dist,
    camera_selection_ratio,
    min_placeholder_dist=0,
    min_terrain_distance=0,
    terrain_coverage_range=(0.5, 1),
):
    if terrain is not None:  # TODO refactor
        terrain_sdf = _terrain_sdf_at(terrain, cam)

    if not cam.type == "CAMERA":
        raise ValueError(f"{cam.name=} had {cam.type=}")

    bpy.context.view_layer.update()

    # Reject cameras too close to any placeholder vertex
    v, i, dist_to_placeholder = placeholders_kd.find(cam.matrix_world.translation)
    if dist_to_placeholder is not None and dist_to_placeholder < min_placeholder_dist:
        logger.debug(
            "keep_cam_pose_proposal rejects dist_to_placeholder=%s, %s",
            dist_to_placeholder,
            (v, i),
        )
        return None

    if terrain is not None and terrain_sdf <= 0:
        # rejected after the ray query below anyway: skip the query but keep its
        # sparse pixel draw so the random stream is unchanged
        _sparse_pixel_draw(cam)
        logger.debug("keep_cam_pose_proposal rejects terrain_sdf=%s early", terrain_sdf)
        return None
    dists, camera_selection_answers_counts, n_pix = _validator_query(
        cam,
        scene_bvh,
        camera_selection_answers,
        vertexwise_min_dist,
        min_terrain_distance,
        terrain_coverage_range,
    )

    if dists is None:
        logger.debug("keep_cam_pose_proposal rejects terrain dists")
        return None

    coverage = len(dists) / n_pix
    if terrain_coverage_range is not None and (
        coverage < terrain_coverage_range[0]
        or coverage > terrain_coverage_range[1]
        or coverage == 0
    ):
        logger.debug(
            f"keep_cam_pose_proposal rejects {coverage=} for {terrain_coverage_range=}"
        )
        return None

    if terrain is not None and terrain_sdf <= 0:
        logger.debug(
            f"keep_cam_pose_proposal rejects {terrain_sdf=} for {cam.matrix_world.translation=}"
        )
        return None

    if rparams := camera_selection_ratio:
        for q in rparams:
            if type(q) is tuple and q[0] == SelectionCriterions.CloseUp:
                closeup = len([d for d in dists if d < q[1]]) / n_pix
                if closeup < rparams[q][0] or closeup > rparams[q][1]:
                    logger.debug(f"keep_cam_pose_proposal rejects {closeup=} for {q=}")
                    return None
            else:
                minv, maxv = rparams[q][0], rparams[q][1]
                if q in camera_selection_answers_counts:
                    ratio = camera_selection_answers_counts[q] / n_pix
                    if ratio < minv or ratio > maxv:
                        logger.debug(
                            f"keep_cam_pose_proposal rejects {ratio=} for {q=}"
                        )
                        return None

    return np.std(dists) + 1.5 * np.min(dists)


def _ray_query(
    matrix_world,
    rel_rows,
    scene_bvh,
    vertexwise_min_dist,
    min_dist,
    min_coverage=None,
    max_coverage=None,
):
    """The ray loop of `terrain_camera_query` for one pose, with the same casts in the
    same order. Returns (hit distances, hit face indices), or (None, None) on a hit
    closer than `min_dist` or its face's vertexwise minimum, or as soon as the coverage
    test is sure to reject the pose for leaving `min_coverage` to `max_coverage`."""
    origin = matrix_world.translation
    ray_cast = scene_bvh.ray_cast
    dists = []
    hits = []
    n_pix = len(rel_rows)
    misses = 0
    for r in rel_rows:
        direction = (matrix_world @ Vector(r) - origin).normalized()
        _, _, index, dist = ray_cast(origin, direction)
        if dist is None:
            if min_coverage is not None:
                misses += 1
                if (n_pix - misses) / n_pix < min_coverage:
                    return None, None
            continue
        dists.append(dist)
        if dist < min_dist or (
            vertexwise_min_dist is not None and dist < vertexwise_min_dist[index]
        ):
            return None, None
        hits.append(index)
        if max_coverage is not None and len(hits) / n_pix > max_coverage:
            return None, None
    return dists, hits


def _query_counts(queries, hits):
    """`terrain_camera_query`'s per-query sums over the hit faces. For float64 and bool
    arrays a left-to-right `sum` of Python scalars makes the same additions as the
    default's loop from 0; other dtypes keep the loop."""
    counts = {}
    for q, values in queries:
        if values.dtype == np.float64 or values.dtype == np.bool_:
            counts[q] = sum(values[hits].tolist()) if hits else 0
        else:
            c = 0
            for index in hits:
                c += values[index]
            counts[q] = c
    return counts


def _score_pose(
    dists,
    counts,
    n_pix,
    camera_selection_ratio,
    terrain_coverage_range,
):
    """Acceptance tests of `keep_cam_pose_proposal` that follow the ray query."""
    if dists is None:
        logger.debug("keep_cam_pose_proposal rejects terrain dists")
        return None

    coverage = len(dists) / n_pix
    if terrain_coverage_range is not None and (
        coverage < terrain_coverage_range[0]
        or coverage > terrain_coverage_range[1]
        or coverage == 0
    ):
        logger.debug(
            "keep_cam_pose_proposal rejects coverage=%s for terrain_coverage_range=%s",
            coverage,
            terrain_coverage_range,
        )
        return None

    if rparams := camera_selection_ratio:
        for q in rparams:
            if type(q) is tuple and q[0] == SelectionCriterions.CloseUp:
                closeup = len([d for d in dists if d < q[1]]) / n_pix
                if closeup < rparams[q][0] or closeup > rparams[q][1]:
                    logger.debug(
                        "keep_cam_pose_proposal rejects closeup=%s for q=%s", closeup, q
                    )
                    return None
            else:
                minv, maxv = rparams[q][0], rparams[q][1]
                if q in counts:
                    ratio = counts[q] / n_pix
                    if ratio < minv or ratio > maxv:
                        logger.debug(
                            "keep_cam_pose_proposal rejects ratio=%s for q=%s", ratio, q
                        )
                        return None

    return np.std(dists) + 1.5 * np.min(dists)


def _has_gin_reference(value):
    """True if a binding value is or contains a reference to a configurable (`@fn` or
    `@fn()`), which gin re-resolves on every call, so such bindings cannot be resolved
    once. Macros (`%name`) are constants and do not count."""
    if isinstance(value, gin.config.ConfigurableReference):
        # a macro is a reference to gin's ``macro`` configurable
        # (selector ``gin.macro``, scoped as ``name/gin.macro``)
        name = str(getattr(value, "selector", "")).split("/")[-1].split(".")[-1]
        return name != "macro"
    if isinstance(value, dict):
        return any(_has_gin_reference(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_gin_reference(v) for v in value)
    return False


def _resolved_call_params(configurable_fn, explicit):
    """Keyword arguments `configurable_fn` would get when called with `explicit`: its
    gin bindings in the current scope (references resolved), overridden by `explicit`,
    plus signature defaults. None if a binding holds a gin reference."""
    unresolved = gin.get_bindings(configurable_fn, resolve_references=False)
    if any(_has_gin_reference(v) for k, v in unresolved.items() if k not in explicit):
        return None
    params = dict(gin.get_bindings(configurable_fn))
    params.update(explicit)
    sig = inspect.signature(configurable_fn.__wrapped__)
    for name, p in sig.parameters.items():
        if name not in params and p.default is not inspect.Parameter.empty:
            params[name] = p.default
    return params


def _search_sequential(
    camera_rig,
    its,
    propose,
    potential_views,
    n_min_candidates,
    pbar,
    terrain,
    scene_bvh,
    placeholders_kd,
    visualize,
    kwargs,
):
    """The default proposal loop over `its`; True once enough views were collected."""
    for it in its:
        props = propose()

        if props is None:
            logger.debug(f"{camera_pose_proposal.__name__} returned {props=} for {it=}")
            continue

        props.apply(camera_rig)

        all_scores = []
        for cam in camera_rig.children:
            score = _default.keep_cam_pose_proposal(
                cam,
                terrain,
                scene_bvh,
                placeholders_kd,
                **kwargs,
            )
            all_scores.append(score)

        if any(score is None for score in all_scores):
            criterion = None
        else:
            criterion = np.mean(all_scores)

        if visualize:
            criterion_str = f"{criterion:.2f}" if criterion is not None else "None"
            marker = butil.spawn_empty(f"attempt_{it}_{criterion_str}")
            marker.location = camera_rig.location
            marker.rotation_euler = camera_rig.rotation_euler

        if criterion is None:
            logger.debug("it=%s criterion=%s", it, criterion)
            continue

        # Compute focus distance
        destination = cam.matrix_world @ Vector((0.0, 0.0, -1.0))
        forward_dir = (destination - cam.location).normalized()
        *_, straight_ahead_dist = scene_bvh.ray_cast(cam.location, forward_dir)

        potential_views.append((criterion, deepcopy(props), straight_ahead_dist))
        pbar.update(1)

        if len(potential_views) >= n_min_candidates:
            return True
    return False


def _search_batched(
    camera_rig,
    max_tries,
    batch_size,
    propose,
    propose_fast,
    potential_views,
    n_min_candidates,
    pbar,
    terrain,
    scene_bvh,
    placeholders_kd,
    kwargs,
):
    """The proposal loop in batches, with the default loop's results; True once enough
    views were collected. Proposals and pixel draws are made one by one in the default
    order, one SDF call covers the batch, and decisions replay in proposal order. If it
    stops mid-batch, it rewinds and redoes the draws up to the last accepted proposal,
    so the search ends in the default loop's state."""
    children = list(camera_rig.children)
    for cam in children:
        if not cam.type == "CAMERA":
            raise ValueError(f"{cam.name=} had {cam.type=}")
    kp = _resolved_call_params(_default.keep_cam_pose_proposal, kwargs)
    sp = _resolved_call_params(_default.get_sensor_coords, {})
    if kp is None or sp is None:
        return _search_sequential(
            camera_rig,
            range(1, max_tries),
            propose,
            potential_views,
            n_min_candidates,
            pbar,
            terrain,
            scene_bvh,
            placeholders_kd,
            False,
            kwargs,
        )
    min_placeholder_dist = kp["min_placeholder_dist"]
    min_terrain_distance = kp["min_terrain_distance"]
    terrain_coverage_range = kp["terrain_coverage_range"]
    camera_selection_ratio = kp["camera_selection_ratio"]
    vertexwise_min_dist = kp["vertexwise_min_dist"]
    if (
        isinstance(vertexwise_min_dist, np.ndarray)
        and vertexwise_min_dist.dtype == np.float64
    ):
        # a list of the same doubles, which the ray loop indexes faster
        vertexwise_min_dist = vertexwise_min_dist.tolist()
    queries = list(kp["camera_selection_answers"].items())
    H, W = sp["H"], sp["W"]
    n_sample = 1000  # get_sensor_coords' sparse draw
    min_coverage = (
        terrain_coverage_range[0] if terrain_coverage_range is not None else None
    )

    vl = bpy.context.view_layer
    # of the sensor intrinsics only the focal length changes between
    # proposals; the rest is read once per camera
    static = {}

    def intrinsics(cam):
        st = static.get(cam.name)
        if st is None:
            u_0, v_0, s_u, s_v, _ = _sensor_intrinsics(cam, H, W)
            st = static[cam.name] = (u_0, v_0, s_u, s_v)
        return (*st, cam.data.lens / 1000)

    # keep_cam_pose_proposal reads the first camera's matrix_world for the SDF
    # test before it updates the view layer, so it sees the position the
    # previous evaluation left; later cameras read it after the update
    last_pos = [cam.matrix_world.translation.copy() for cam in children]
    first_draw = [True] * len(children)
    first_proposal = True

    def phase1(its):
        """Draw, apply and evaluate proposals `its` one by one, running each camera's
        placeholder test and then, if it passes, its sparse pixel draw, so the random
        stream is consumed as in the default loop."""
        nonlocal last_pos, first_proposal
        entries = []
        for i_it in its:
            if first_proposal:
                props = propose()  # through the gin wrapper once (operative config)
                first_proposal = False
            else:
                props = propose_fast()
            if props is None:
                logger.debug(
                    f"{camera_pose_proposal.__name__} returned {props=} for {i_it=}"
                )
                continue
            props.apply(camera_rig)
            vl.update()
            mats = [cam.matrix_world.copy() for cam in children]
            fresh = [m.translation.copy() for m in mats]
            e = {
                "it": i_it,
                "props": props,
                "mats": mats,
                "pts": [
                    last_pos[0] if j == 0 else fresh[j] for j in range(len(children))
                ],
                "ii": [],
                "intr": [],
            }
            for j, cam in enumerate(children):
                _, _, dist_to_placeholder = placeholders_kd.find(fresh[j])
                if (
                    dist_to_placeholder is not None
                    and dist_to_placeholder < min_placeholder_dist
                ):
                    logger.debug(
                        "keep_cam_pose_proposal rejects dist_to_placeholder=%s",
                        dist_to_placeholder,
                    )
                    e["ii"].append(None)
                    e["intr"].append(None)
                    continue
                if first_draw[j]:
                    # once through get_sensor_coords itself (operative config)
                    _, pixel_locs = _default.get_sensor_coords(cam, sparse=True)
                    ii = pixel_locs[:, 1] * W + pixel_locs[:, 0]
                    first_draw[j] = False
                else:
                    # the same numbers as get_sensor_coords' np.random.choice(H * W,
                    # size=1000): choice with replacement draws randint(0, H * W)
                    ii = np.random.randint(0, H * W, size=n_sample)
                e["ii"].append(ii)
                e["intr"].append(intrinsics(cam))
            last_pos = fresh
            entries.append(e)
        return entries

    it = 1
    while it < max_tries:
        its = range(it, min(it + batch_size, max_tries))
        it = its.stop
        np_state = np.random.get_state()
        py_state = random.getstate()
        rig_state = (
            camera_rig.location.copy(),
            camera_rig.rotation_euler.copy(),
            [cam.data.lens for cam in children],
        )
        last_pos_start = [p.copy() for p in last_pos]

        # phase 1: draw, apply, evaluate; placeholder test and pixel draw
        entries = phase1(its)

        # phase 2: one SDF evaluation for every camera of the batch
        if terrain is not None and entries:
            per_point = [np.array(p).reshape((1, 3)) for e in entries for p in e["pts"]]
            sdf_all = terrain.compute_camera_space_sdf(
                np.concatenate(per_point, axis=0)
            )
            k = 0
            for e in entries:
                e["sdf"] = sdf_all[k : k + len(children)]
                k += len(children)

        # phase 3: ray queries for the survivors, decisions in proposal order
        reached = None
        for local, e in enumerate(entries):
            all_scores = []
            for j, cam in enumerate(children):
                if e["ii"][j] is None:
                    all_scores.append(None)
                    continue
                if terrain is not None and e["sdf"][j] <= 0:
                    logger.debug(
                        "keep_cam_pose_proposal rejects terrain_sdf=%s early",
                        e["sdf"][j],
                    )
                    all_scores.append(None)
                    continue
                rel = _sampled_sensor_coords(e["ii"][j], W, e["intr"][j])
                dists, hits = _ray_query(
                    e["mats"][j],
                    rel,
                    scene_bvh,
                    vertexwise_min_dist,
                    min_terrain_distance,
                    min_coverage,
                )
                counts = _query_counts(queries, hits) if dists is not None else None
                all_scores.append(
                    _score_pose(
                        dists,
                        counts,
                        len(e["ii"][j]),
                        camera_selection_ratio,
                        terrain_coverage_range,
                    )
                )
            if any(score is None for score in all_scores):
                logger.debug("it=%s criterion=%s", e["it"], None)
                continue
            criterion = np.mean(all_scores)

            # Compute focus distance (from the last camera, as the default does)
            cam = children[-1]
            destination = e["mats"][-1] @ Vector((0.0, 0.0, -1.0))
            forward_dir = (destination - cam.location).normalized()
            *_, straight_ahead_dist = scene_bvh.ray_cast(cam.location, forward_dir)

            potential_views.append(
                (criterion, deepcopy(e["props"]), straight_ahead_dist)
            )
            pbar.update(1)
            if len(potential_views) >= n_min_candidates:
                reached = e["it"]
                break

        if reached is None:
            continue
        if reached == its[-1]:
            return True
        # the default loop stops at proposal `reached`: rewind to the batch start
        # and redo the draws up to it, so the random stream and the rig end in the
        # same state; the views already found are unchanged
        np.random.set_state(np_state)
        random.setstate(py_state)
        camera_rig.location, camera_rig.rotation_euler, lenses = rig_state
        for cam, lens in zip(children, lenses):
            cam.data.lens = lens
        vl.update()
        last_pos = last_pos_start
        phase1(range(its.start, reached + 1))
        return True
    return False


def compute_base_views(
    camera_rig: bpy.types.Object,
    n_views: int,
    terrain,
    scene_bvh: BVHTree,
    location_sample: typing.Callable,
    center_coordinate=None,
    radius=None,
    bbox=None,
    placeholders_kd=None,
    min_candidates_ratio=20,
    max_tries=30000,
    visualize=False,
    **kwargs,
):
    potential_views = []
    n_min_candidates = int(min_candidates_ratio * n_views)

    def propose():
        if center_coordinate:
            return _default.camera_pose_proposal(
                scene_bvh=scene_bvh,
                location_sample=location_sample,
                center_coordinate=center_coordinate,
                radius=random_general(radius),
                bbox=bbox,
            )
        return _default.camera_pose_proposal(
            scene_bvh=scene_bvh, location_sample=location_sample
        )

    # the same call without the gin wrapper, with its bindings resolved once
    # (they do not change during the search)
    proposal_fn = _default.camera_pose_proposal.__wrapped__
    proposal_params = _resolved_call_params(
        _default.camera_pose_proposal,
        dict(scene_bvh=scene_bvh, location_sample=location_sample),
    )
    if proposal_params is None:
        propose_fast = propose
    elif center_coordinate:
        proposal_params.update(center_coordinate=center_coordinate, bbox=bbox)

        def propose_fast():
            return proposal_fn(**{**proposal_params, "radius": random_general(radius)})

    else:

        def propose_fast():
            return proposal_fn(**proposal_params)

    # Every proposal needs a depsgraph update to refresh the cameras'
    # matrix_world, so the search runs in a scratch view layer holding the rig.
    rig_objects = [camera_rig, *camera_rig.children_recursive]
    with (
        _isolation.IsolatedScene(link=rig_objects),
        tqdm(total=n_min_candidates, desc="Searching for camera viewpoints") as pbar,
    ):
        if visualize:
            _search_sequential(
                camera_rig,
                range(1, max_tries),
                propose,
                potential_views,
                n_min_candidates,
                pbar,
                terrain,
                scene_bvh,
                placeholders_kd,
                visualize,
                kwargs,
            )
        else:
            _search_batched(
                camera_rig,
                max_tries,
                _SEARCH_BATCH_SIZE,
                propose,
                propose_fast,
                potential_views,
                n_min_candidates,
                pbar,
                terrain,
                scene_bvh,
                placeholders_kd,
                kwargs,
            )

    if len(potential_views) < n_views:
        if visualize:
            butil.save_blend("compute_base_views-failed.blend")
        raise ValueError(f"Could not find {n_views} camera views")

    views = sorted(potential_views, reverse=True)

    return views[:n_views]


def _sparse_pixel_draw(cam):
    """Advance the random stream as `get_sensor_coords(cam, sparse=True)` does, without
    building the coordinates unless its bindings cannot be resolved ahead of time."""
    sp = _resolved_call_params(_default.get_sensor_coords, {})
    if sp is None:
        _default.get_sensor_coords(cam, sparse=True)
    else:
        np.random.choice(sp["H"] * sp["W"], size=1000)
