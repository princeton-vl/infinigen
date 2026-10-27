# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import collections.abc

import numpy as np
import procfunc as pf
import pytest

from infinigen2.curves import skirting_board_profile
from infinigen2.scenes.room import (
    room,
    skirting,
    wall_base,
    wall_cutouts,
    wall_mounts,
)
from infinigen2.shaders import functionality_lists
from infinigen2.util import mesh as mesh_util

WALL_GENERATORS = [
    wall_base.wall_plain_rand,
    wall_cutouts.wall_windows_rand,
    wall_cutouts.wall_painting_grid_rand,
    wall_mounts.wall_board_shelf_rand,
    wall_cutouts.wall_storage_shelf_rand,
    wall_cutouts.wall_cubby_rand,
    wall_mounts.wall_storage_flush_rand,
    wall_cutouts.wall_doors_rand,
    wall_cutouts.wall_full_window_rand,
    room.wall_arrangement_rand,
]


def _evaluated_world_extent(obj: pf.MeshObject) -> np.ndarray:
    item = obj.item()
    for modifier in item.modifiers:
        modifier.show_viewport = True
        if modifier.type == "SUBSURF":
            modifier.levels = modifier.render_levels
    lo, hi = mesh_util.evaluated_world_bbox(obj)
    return hi - lo


@pytest.mark.parametrize("func", WALL_GENERATORS, ids=lambda func: func.__name__)
def test_wall_generators_run_standalone(
    func: collections.abc.Callable[[pf.RNG], wall_base.WallResult],
    rng: pf.RNG,
) -> None:
    result = func(rng)

    assert result.wall_planes
    assert result.all_objects
    assert len(result.all_objects) == len({id(obj) for obj in result.all_objects})


def test_standalone_wall_has_vertical_metric_uvs(rng: pf.RNG) -> None:
    wall = wall_base._standalone_wall_rand(rng, width=1.75, height=2.25)
    uvs = pf.ops.attr.uv_coords(wall.obj)
    bbox_min, bbox_max = pf.ops.attr.bbox_min_max(wall.obj)
    extent = np.array(bbox_max) - np.array(bbox_min)

    assert np.allclose(uvs.max(axis=0) - uvs.min(axis=0), (wall.width, wall.height))
    assert np.allclose(extent, (0.0, 1.75, 2.25))
    assert np.allclose(pf.ops.attr.polygon_normals(wall.obj), ((1.0, 0.0, 0.0),))
    assert mesh_util.uv_winding_sign(wall.obj) == wall_base.SURFACE_UV_WINDING_SIGN


def test_wall_generator_accepts_precreased_wall(rng: pf.RNG) -> None:
    wall = wall_base._standalone_wall_rand(rng, width=3.0, height=2.5)
    pf.ops.attr.write_attribute(wall.obj, 0.0, "crease_edge", domain="EDGE")
    material = pf.Material(surface=pf.nodes.shader.principled_bsdf())

    result = wall_base.wall_plain_rand(rng, wall=wall, wall_material=material)
    creases = pf.ops.attr.read_attribute(
        result.wall_planes[0], "crease_edge", domain="EDGE"
    )

    assert np.all(creases == 1.0)


@pytest.mark.parametrize("func", WALL_GENERATORS, ids=lambda func: func.__name__)
def test_wall_generators_preserve_outer_extent_after_subdivision(
    func: collections.abc.Callable[..., wall_base.WallResult],
) -> None:
    rng = np.random.default_rng(42)
    wall = wall_base._standalone_wall_rand(rng, width=3.0, height=2.5)
    material = pf.Material(surface=pf.nodes.shader.principled_bsdf())

    result = func(rng, wall=wall, wall_material=material)
    extent = _evaluated_world_extent(result.wall_planes[0])

    np.testing.assert_allclose(extent[1:], (3.0, 2.5), atol=0.003)


@pytest.mark.parametrize(
    "func",
    [
        wall_cutouts.wall_doors_rand,
        wall_cutouts.wall_full_window_rand,
    ],
    ids=lambda func: func.__name__,
)
def test_wall_cutouts_fall_back_on_narrow_wall(
    func: collections.abc.Callable[..., wall_base.WallResult],
    rng: pf.RNG,
) -> None:
    wall = wall_base._standalone_wall_rand(rng, width=0.02, height=2.5)
    material = functionality_lists.wall_material_rand(
        np.random.default_rng(100), pf.nodes.shader.coord().uv
    )

    result = func(rng, wall=wall, wall_material=material)

    assert len(result.wall_planes) == 1
    assert result.decorations == {}


def test_wall_doors_add_invisible_affordance_colliders() -> None:
    rng = np.random.default_rng(42)
    wall = wall_base._standalone_wall_rand(rng, width=3.0, height=2.5)
    material = functionality_lists.wall_material_rand(
        np.random.default_rng(100), pf.nodes.shader.coord().uv
    )

    result = wall_cutouts.wall_doors_rand(rng, wall=wall, wall_material=material)

    doors = result.decorations["door"]
    assert result.colliders is not None
    affordances = [
        obj
        for obj in result.colliders.objs
        if obj.item().name.startswith("door_affordance")
    ]
    assert len(affordances) == len(doors)
    assert all(obj not in result.all_objects for obj in affordances)
    for collider in affordances:
        collider_min, collider_max = pf.ops.attr.bbox_min_max(
            collider, global_coords=False
        )
        collider_extent = collider_max - collider_min
        np.testing.assert_allclose(collider_extent[0], 2 * collider_extent[1])
        assert 0.85 <= collider_extent[1] <= 1.2
        np.testing.assert_allclose(collider_extent[2], 2.5)
        assert collider.item().hide_render
        assert collider.item().display_type == "WIRE"


def _wall_with_opening(opening_bottom: float) -> pf.MeshObject:
    rectangles = [
        (0.0, 1.0, 0.0, 2.5),
        (1.0, 2.0, 0.0, opening_bottom),
        (1.0, 2.0, 2.0, 2.5),
        (2.0, 3.0, 0.0, 2.5),
    ]
    vertices = []
    faces = []
    for y_min, y_max, z_min, z_max in rectangles:
        start = len(vertices)
        vertices.extend(
            [
                (0.0, y_min, z_min),
                (0.0, y_max, z_min),
                (0.0, y_max, z_max),
                (0.0, y_min, z_max),
            ]
        )
        faces.append((start, start + 1, start + 2, start + 3))
    return pf.ops.primitives.mesh_from_numpy(
        vertices=np.array(vertices), faces=np.array(faces)
    )


@pytest.mark.parametrize(
    "opening_bottom, profile_height, expected_under_opening",
    [(0.15, 0.12, True), (0.15, 0.18, False), (0.01, 0.12, False)],
    ids=["window-above-board", "window-overlaps-board", "door"],
)
def test_skirting_on_walls_follows_profile_height(
    opening_bottom: float, profile_height: float, expected_under_opening: bool
) -> None:
    wall = _wall_with_opening(opening_bottom)
    profile = skirting_board_profile.skirting_profile(height=profile_height, width=0.04)
    material = pf.Material(surface=pf.nodes.shader.principled_bsdf())

    skirt = skirting.skirting_on_walls_rand(
        np.random.default_rng(2), [wall], material, profile_curve=profile
    )[0]
    vertices = np.array([vertex.co for vertex in skirt.item().data.vertices])
    center_distance = np.abs(vertices[:, 1] - 1.5).min()

    assert bool(center_distance < 0.05) == expected_under_opening
