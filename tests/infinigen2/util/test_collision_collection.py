# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import inspect

import numpy as np
import procfunc as pf

from infinigen.core.util import blender as butil
from infinigen2.scenes.collision_collection import (
    any_self_collision,
    box_intersection_test,
    collision_set,
    intersection_test,
    n_colliders,
    raycast,
)


def test_collision_add_and_query():
    objs = []
    col = collision_set(objs)

    assert n_colliders(col) == 0

    origin_probe = pf.ops.primitives.mesh_cube()
    origin_probe.item().location = (0, 0, 0)
    assert not intersection_test(col, origin_probe)

    cube = pf.ops.primitives.mesh_cube()
    cube.item().location = (10, 1, 1)
    objs.append(cube)
    col = collision_set(objs, existing=col)

    assert n_colliders(col) == 1
    assert not intersection_test(col, origin_probe)

    near_probe = pf.ops.primitives.mesh_cube()
    near_probe.item().location = (11, 1, 1)
    assert intersection_test(col, near_probe)


def test_collision_cube_and_sphere():
    cube = pf.ops.primitives.mesh_cube()
    cube.item().location = (0, 0, 0)
    sphere = pf.ops.primitives.mesh_uv_sphere()
    sphere.item().location = (10, 0, 0)
    col = collision_set([cube])
    assert n_colliders(col) == 1
    assert not intersection_test(col, sphere)
    near_sphere = pf.ops.primitives.mesh_uv_sphere()
    near_sphere.item().location = (1, 0, 0)
    assert intersection_test(col, near_sphere)


def test_collision_list_update_changes_queries():
    objs = []
    col = collision_set(objs)

    first = pf.ops.primitives.mesh_cube()
    first.item().location = (0, 0, 0)
    objs.append(first)
    col = collision_set(objs, existing=col)
    assert n_colliders(col) == 1

    second_probe = pf.ops.primitives.mesh_cube()
    second_probe.item().location = (5, 0, 0)
    assert not intersection_test(col, second_probe)

    second = pf.ops.primitives.mesh_cube()
    second.item().location = (5, 0, 0)
    objs.append(second)
    col = collision_set(objs, existing=col)
    assert n_colliders(col) == 2
    assert intersection_test(col, second_probe)


def test_collision_reuses_aliased_mesh_collider():
    base = pf.ops.primitives.mesh_cube()
    alias = pf.ops.object.alias(base)
    base.item().location = (0, 0, 0)
    alias.item().location = (6, 0, 0)

    col = collision_set([base, alias])

    # Two objects tracked, one shared mesh collider.
    assert n_colliders(col) == 2
    assert len(col.mesh_colliders) == 1

    probe_left = pf.ops.primitives.mesh_cube()
    probe_left.item().location = (0, 0, 0)
    probe_right = pf.ops.primitives.mesh_cube()
    probe_right.item().location = (6, 0, 0)
    assert intersection_test(col, probe_left)
    assert intersection_test(col, probe_right)


def test_collision_skips_non_mesh_objects_and_raycast_hits():
    cube = pf.ops.primitives.mesh_cube()
    cube.item().location = (0, 0, 0)
    light = pf.ops.primitives.point_lamp(energy=1000)
    light.item().location = (0, 0, 3)

    col = collision_set([cube, light])
    assert n_colliders(col) == 1
    assert not intersection_test(col, light)

    origins = np.array([[0.0, 0.0, 5.0]])
    directions = np.array([[0.0, 0.0, -1.0]])
    hit_points, _ray_idx, _tri_idx = raycast(col, origins, directions)
    assert hit_points.shape[0] >= 1


# --- any_self_collision tests ---


def test_any_self_collision_empty_and_single():
    col = collision_set([])
    assert not any_self_collision(col)

    cube = pf.ops.primitives.mesh_cube()
    col = collision_set([cube])
    assert not any_self_collision(col)


def test_any_self_collision_non_overlapping():
    a = pf.ops.primitives.mesh_cube()
    b = pf.ops.primitives.mesh_cube()
    a.item().location = (0, 0, 0)
    b.item().location = (10, 0, 0)
    col = collision_set([a, b])
    assert not any_self_collision(col)


def test_any_self_collision_overlapping():
    a = pf.ops.primitives.mesh_cube()
    b = pf.ops.primitives.mesh_cube()
    a.item().location = (0, 0, 0)
    b.item().location = (0.5, 0, 0)
    col = collision_set([a, b])
    assert any_self_collision(col)


def test_any_self_collision_updates_after_move():
    """Objects start clear; moving one into the other is detected without rebuilding."""
    a = pf.ops.primitives.mesh_cube()
    b = pf.ops.primitives.mesh_cube()
    a.item().location = (0, 0, 0)
    b.item().location = (10, 0, 0)
    col = collision_set([a, b])

    assert not any_self_collision(col)

    b.item().location = (0.5, 0, 0)
    assert any_self_collision(col)


def test_any_self_collision_clears_after_move():
    """Objects start overlapping; moving one away clears the collision."""
    a = pf.ops.primitives.mesh_cube()
    b = pf.ops.primitives.mesh_cube()
    a.item().location = (0, 0, 0)
    b.item().location = (0.5, 0, 0)
    col = collision_set([a, b])

    assert any_self_collision(col)

    b.item().location = (10, 0, 0)
    assert not any_self_collision(col)


# --- Moving object tests ---


def test_collision_detects_after_object_moves_into_range():
    """Object starts far away; after moving into probe position the set detects collision."""
    cube = pf.ops.primitives.mesh_cube()
    cube.item().location = (50, 0, 0)
    col = collision_set([cube])

    probe = pf.ops.primitives.mesh_cube()
    probe.item().location = (0, 0, 0)

    # No collision while cube is far away
    assert not intersection_test(col, probe)

    # Move cube on top of probe — do NOT rebuild col
    cube.item().location = (0, 0, 0)
    assert intersection_test(col, probe)


def test_collision_clears_after_object_moves_away():
    """Object starts at probe position; after moving away the collision clears."""
    cube = pf.ops.primitives.mesh_cube()
    cube.item().location = (0, 0, 0)
    col = collision_set([cube])

    probe = pf.ops.primitives.mesh_cube()
    probe.item().location = (0, 0, 0)

    # Starts colliding
    assert intersection_test(col, probe)

    # Move cube far away — do NOT rebuild col
    cube.item().location = (50, 0, 0)
    assert not intersection_test(col, probe)


def test_collision_box_intersection_detects_after_move():
    """box_intersection_test also auto-reflects object movement."""
    cube = pf.ops.primitives.mesh_cube()
    cube.item().location = (50, 0, 0)
    col = collision_set([cube])

    transform = np.eye(4)  # probe box at origin

    assert not box_intersection_test(col, transform, size=1.0)

    cube.item().location = (0, 0, 0)
    assert box_intersection_test(col, transform, size=1.0)


def test_collision_box_intersection_clears_after_move():
    """box_intersection_test clears when object moves out of probe box."""
    cube = pf.ops.primitives.mesh_cube()
    cube.item().location = (0, 0, 0)
    col = collision_set([cube])

    transform = np.eye(4)
    assert box_intersection_test(col, transform, size=1.0)

    cube.item().location = (50, 0, 0)
    assert not box_intersection_test(col, transform, size=1.0)


def test_collision_moving_one_of_multiple_objects():
    """Moving one object leaves the others unaffected in the set."""
    a = pf.ops.primitives.mesh_cube()
    b = pf.ops.primitives.mesh_cube()
    a.item().location = (0, 0, 0)
    b.item().location = (20, 0, 0)
    col = collision_set([a, b])

    probe_origin = pf.ops.primitives.mesh_cube()
    probe_origin.item().location = (0, 0, 0)
    probe_b = pf.ops.primitives.mesh_cube()
    probe_b.item().location = (20, 0, 0)

    assert intersection_test(col, probe_origin)
    assert intersection_test(col, probe_b)

    # Move only 'a' far away
    a.item().location = (50, 0, 0)

    assert not intersection_test(col, probe_origin)
    assert intersection_test(col, probe_b)  # b is still in place


def test_collision_object_moves_back_and_forth():
    """Object oscillating position is tracked correctly through multiple moves."""
    cube = pf.ops.primitives.mesh_cube()
    cube.item().location = (0, 0, 0)
    col = collision_set([cube])

    probe = pf.ops.primitives.mesh_cube()
    probe.item().location = (0, 0, 0)

    assert intersection_test(col, probe)

    cube.item().location = (50, 0, 0)
    assert not intersection_test(col, probe)

    cube.item().location = (0, 0, 0)
    assert intersection_test(col, probe)

    cube.item().location = (50, 0, 0)
    assert not intersection_test(col, probe)


def test_collision_raycast_detects_after_move():
    """raycast reflects object position after movement without rebuilding the set."""
    cube = pf.ops.primitives.mesh_cube()
    cube.item().location = (50, 0, 0)
    col = collision_set([cube])

    origins = np.array([[0.0, 0.0, 5.0]])
    directions = np.array([[0.0, 0.0, -1.0]])

    hit_points, _, _ = raycast(col, origins, directions)
    assert hit_points.shape[0] == 0  # cube is far away, no hit

    cube.item().location = (0, 0, 0)
    hit_points, _, _ = raycast(col, origins, directions)
    assert hit_points.shape[0] >= 1  # cube now under the ray


def test_collision_cube_passes_through(tmp_path, save_blend=True):
    """Animate a cube passing through a stationary cube and log collision per frame.

    The moving cube is placed in the collision set so that _sync_transforms is
    exercised on every frame. The stationary cube is the probe (always rebuilt
    fresh by intersection_test, so it does not test sync).

    Run with:
        uv run pytest tests/infinigen2/util/test_collision_collection.py::test_collision_cube_passes_through -v -s
    """
    moving = pf.ops.primitives.mesh_cube()
    moving.item().name = "moving"

    stationary = pf.ops.primitives.mesh_cube()
    stationary.item().name = "stationary"
    stationary.item().location = (0, 0, 0)

    # moving cube is in the set — _sync_transforms updates its FCL transform each frame
    col = collision_set([moving])

    n_frames = 40
    x_start, x_end = -5.0, 5.0

    print("\nframe | moving_x | colliding")
    print("------+----------+----------")

    results = []
    for frame in range(n_frames + 1):
        t = frame / n_frames
        x = x_start + t * (x_end - x_start)
        moving.item().location = (x, 0, 0)
        hit = intersection_test(col, stationary)
        print(f"{frame:5d} | {x:+.4f}  | {hit}")
        moving.item().keyframe_insert("location", frame=frame)
        results.append((x, hit))

    # both cubes have half-extent 1; overlap region is x in (-2, +2)
    for x, hit in results:
        if abs(x) < 2.0:
            assert hit, f"expected collision at x={x}"
        elif abs(x) > 2.0:
            assert not hit, f"expected no collision at x={x}"

    if save_blend:
        blend_path = tmp_path / f"{inspect.currentframe().f_code.co_name}.blend"
        butil.save_blend(blend_path)


def test_collision_detects_after_object_rotates():
    """Rotating an object correctly updates its collision geometry orientation.

    A unit cube (half-extent = 1) at the origin has corners at (±1, ±1, ±1).
    After 45° rotation around Z, corner (1, -1, 0) maps to (√2, 0, 0), so the
    cube now extends to √2 ≈ 1.414 along X.  A probe cube centred at (2.2, 0, 0)
    starts at x = 1.2 — just outside the unrotated cube (max x = 1.0, gap = 0.2)
    but inside the rotated cube (max x ≈ 1.414, overlap ≈ 0.21).

    This exercises the rotation component of _fcl_transform / setTransform, which
    the translation-only moving-object tests above do not cover.
    """
    import math

    cube = pf.ops.primitives.mesh_cube()
    cube.item().location = (0, 0, 0)
    cube.item().rotation_euler = (0, 0, 0)
    col = collision_set([cube])

    # Probe starts at x = 1.2 — clear of the axis-aligned cube (max x = 1.0)
    probe = pf.ops.primitives.mesh_cube()
    probe.item().location = (2.2, 0, 0)
    assert not intersection_test(col, probe)

    # Rotate the cube 45° around Z: corner reaches x ≈ 1.414, entering the probe
    cube.item().rotation_euler = (0, 0, math.pi / 4)

    # Collision now expected — do NOT rebuild col
    assert intersection_test(col, probe)
