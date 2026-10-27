# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

import bpy
import numpy as np
import pytest

pytest.importorskip(
    "infinigen.perf.assets.objects.trees._cylinder_mesher",
    reason="needs INFINIGEN_INSTALL_PERF=True",
)

from infinigen.perf.assets.objects.trees.tessellate import (  # noqa: E402
    _catmull_clark_vertex_positions,
    retessellate_stems,
)


def _closed_tube_mesh(name, ring_count=3, profile_resolution=6):
    vertices = []
    for ring_index in range(ring_count):
        radius = 0.4 - 0.05 * ring_index
        z = 0.7 * ring_index
        for profile_index in range(profile_resolution):
            theta = 2 * np.pi * profile_index / profile_resolution
            vertices.append((radius * np.cos(theta), radius * np.sin(theta), z))
    faces = []
    for ring_index in range(ring_count - 1):
        for profile_index in range(profile_resolution):
            following = (profile_index + 1) % profile_resolution
            current = ring_index * profile_resolution
            next_ring = current + profile_resolution
            faces.append(
                (
                    current + profile_index,
                    current + following,
                    next_ring + following,
                    next_ring + profile_index,
                )
            )
    faces.append(tuple(range(profile_resolution - 1, -1, -1)))
    final_start = (ring_count - 1) * profile_resolution
    faces.append(tuple(final_start + i for i in range(profile_resolution)))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    return mesh


def test_catmull_clark_vertex_update_matches_cube_rule():
    mesh = bpy.data.meshes.new("tree-catmull-cube-test")
    mesh.from_pydata(
        [(x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        [],
        [
            (0, 1, 3, 2),
            (4, 6, 7, 5),
            (0, 4, 5, 1),
            (2, 3, 7, 6),
            (0, 2, 6, 4),
            (1, 5, 7, 3),
        ],
    )
    try:
        vertices = np.asarray([vertex.co for vertex in mesh.vertices])
        smoothed = _catmull_clark_vertex_positions(mesh, vertices)
        assert np.allclose(np.abs(smoothed), 5 / 9)
    finally:
        bpy.data.meshes.remove(mesh)


def test_tapered_tree_cylinder_bounds_edges_and_preserves_attributes():
    profile_resolution = 6
    mesh = _closed_tube_mesh("tree-retessellation-test", 3, profile_resolution)
    obj = bpy.data.objects.new("tree-retessellation-test", mesh)
    bpy.context.scene.collection.objects.link(obj)
    stem_id = mesh.attributes.new("stem_id", "INT", "POINT")
    stem_id.data.foreach_set("value", np.full(len(mesh.vertices), 17))
    skeleton_loc = mesh.attributes.new("skeleton_loc", "FLOAT_VECTOR", "POINT")
    skeleton_loc.data.foreach_set(
        "vector", np.asarray([vertex.co for vertex in mesh.vertices]).reshape(-1)
    )

    face_size = 0.11
    try:
        retessellate_stems(obj, face_size, profile_resolution)
        coordinates = np.asarray([vertex.co for vertex in obj.data.vertices])
        obj.data.calc_loop_triangles()
        triangles = np.asarray(
            [triangle.vertices for triangle in obj.data.loop_triangles]
        )
        triangle_coordinates = coordinates[triangles]
        lengths = np.stack(
            [
                np.linalg.norm(
                    triangle_coordinates[:, 1] - triangle_coordinates[:, 0], axis=1
                ),
                np.linalg.norm(
                    triangle_coordinates[:, 2] - triangle_coordinates[:, 1], axis=1
                ),
                np.linalg.norm(
                    triangle_coordinates[:, 0] - triangle_coordinates[:, 2], axis=1
                ),
            ],
            axis=1,
        )
        assert lengths.max() <= face_size + 1e-6
        assert obj.data.name == "tree-retessellation-test"
        assert {attribute.name for attribute in obj.data.attributes} >= {
            "stem_id",
            "skeleton_loc",
        }
        output_stem_id = np.empty(len(obj.data.vertices), dtype=np.int32)
        obj.data.attributes["stem_id"].data.foreach_get("value", output_stem_id)
        assert np.all(output_stem_id == 17)
        loop_totals = np.asarray([polygon.loop_total for polygon in obj.data.polygons])
        assert loop_totals.max() <= 4
        assert np.count_nonzero(loop_totals == 3) >= 2 * profile_resolution
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)


@pytest.mark.parametrize("flat", [True, False])
def test_retessellation_keeps_the_shading(flat):
    mesh = _closed_tube_mesh("tree-shading-test", 3, 6)
    obj = bpy.data.objects.new("tree-shading-test", mesh)
    bpy.context.scene.collection.objects.link(obj)
    mesh.shade_flat() if flat else mesh.shade_smooth()
    try:
        retessellate_stems(obj, 0.11, 6)
        smooth = np.empty(len(obj.data.polygons), dtype=bool)
        obj.data.polygons.foreach_get("use_smooth", smooth)
        assert not smooth.any() if flat else smooth.all()
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
