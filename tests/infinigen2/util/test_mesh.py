# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import math

import bmesh
import bpy
import numpy as np
import procfunc as pf
import pytest

from infinigen2.util import mesh as mesh_util


def _crease_values(geo: pf.ProcNode):
    obj = pf.nodes.to_mesh_object(geo)
    me = obj.item().data
    assert "crease_edge" in me.attributes
    return [d.value for d in me.attributes["crease_edge"].data]


def _creased_edges(mesh_node: pf.ProcNode, threshold_degrees: float):
    geo = mesh_util.crease_sharp(mesh_node, threshold_degrees=threshold_degrees)
    obj = pf.nodes.to_mesh_object(geo)
    me = obj.item().data

    assert "crease_edge" in me.attributes
    creased = [d.value > 0.5 for d in me.attributes["crease_edge"].data]

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.edges.ensure_lookup_table()
    assert len(list(bm.edges)) == len(creased)

    out = [
        (math.degrees(edge.calc_face_angle(0.0)), is_creased)
        for edge, is_creased in zip(bm.edges, creased)
    ]
    bm.free()
    return out


def test_crease_sharp_marks_edges_above_threshold():
    bpy.ops.wm.read_homefile(use_empty=True)
    cube = pf.nodes.geo.mesh_cube(size=(1.0, 1.0, 1.0)).mesh

    # Every box edge folds at 90deg: all crease below 90, none above.
    assert all(c for _, c in _creased_edges(cube, threshold_degrees=40.0))
    assert not any(c for _, c in _creased_edges(cube, threshold_degrees=100.0))


@pytest.mark.parametrize("vertices", [8, 16, 24])
def test_crease_sharp_cylinder_creases_rims_not_round_wall(vertices):
    bpy.ops.wm.read_homefile(use_empty=True)
    cyl = pf.nodes.geo.mesh_cylinder(vertices=vertices, radius=0.5, depth=1.0).mesh

    # Threshold 70 sits between the round-wall seams (<=45deg) and the 90deg cap rims.
    edges = _creased_edges(cyl, threshold_degrees=70.0)
    rims = [c for a, c in edges if a > 80]
    walls = [c for a, c in edges if a < 60]

    assert len(rims) == 2 * vertices
    assert all(rims)
    assert not any(walls)


def test_crease_sharp_flat_grid_creases_nothing():
    bpy.ops.wm.read_homefile(use_empty=True)
    grid = pf.nodes.geo.mesh_grid(
        size_x=1.0, size_y=1.0, vertices_x=4, vertices_y=4
    ).mesh

    assert not any(c for _, c in _creased_edges(grid, threshold_degrees=1.0))


def test_crease_by_angle_soft_ramp_midpoint():
    bpy.ops.wm.read_homefile(use_empty=True)
    cube = pf.nodes.geo.mesh_cube(size=(1.0, 1.0, 1.0)).mesh

    # All cube edges fold at exactly the 90deg threshold: soft band -> midpoint 0.5.
    geo = mesh_util.crease_by_angle(cube, threshold_degrees=90.0, softness_degrees=30.0)
    assert all(abs(v - 0.5) < 1e-4 for v in _crease_values(geo))


def test_crease_by_angle_matches_binary_when_band_narrow():
    bpy.ops.wm.read_homefile(use_empty=True)
    cyl = pf.nodes.geo.mesh_cylinder(vertices=16, radius=0.5, depth=1.0).mesh

    # Narrow band around 70deg reproduces crease_sharp: 90deg rims on, seams off.
    geo = mesh_util.crease_by_angle(cyl, threshold_degrees=70.0, softness_degrees=1.0)
    vals = _crease_values(geo)
    assert all(v > 0.99 or v < 0.01 for v in vals)
    assert sum(1 for v in vals if v > 0.5) == 2 * 16


def test_metric_box_uv_puts_each_face_long_axis_on_v():
    bpy.ops.wm.read_homefile(use_empty=True)
    cube = pf.nodes.geo.mesh_cube(size=(0.2, 0.4, 1.0)).mesh
    obj = pf.nodes.to_mesh_object(mesh_util.metric_box_uv(cube))
    mesh = obj.item().data
    uv = np.array([entry.uv[:] for entry in mesh.uv_layers["UVMap"].data])

    for polygon in mesh.polygons:
        face_uv = uv[list(polygon.loop_indices)]
        u_span, v_span = np.ptp(face_uv, axis=0)
        assert v_span > u_span


def test_box_places_anchor_and_preserves_grid_counts():
    bpy.ops.wm.read_homefile(use_empty=True)
    result = mesh_util.box(
        size=(2.0, 3.0, 4.0),
        location=(5.0, 6.0, 7.0),
        vertices_x=3,
        vertices_y=4,
        vertices_z=5,
    )
    mesh = pf.nodes.to_mesh_object(result).item().data
    positions = np.array([vertex.co[:] for vertex in mesh.vertices])

    np.testing.assert_array_equal(positions.min(axis=0), (4.0, 4.5, 5.0))
    np.testing.assert_array_equal(positions.max(axis=0), (6.0, 7.5, 9.0))
    np.testing.assert_allclose(np.unique(positions[:, 0]), (4.0, 5.0, 6.0))
    np.testing.assert_allclose(np.unique(positions[:, 1]), (4.5, 5.5, 6.5, 7.5))
    np.testing.assert_allclose(np.unique(positions[:, 2]), (5.0, 6.0, 7.0, 8.0, 9.0))
    assert [len(np.unique(positions[:, i])) for i in range(3)] == [3, 4, 5]
    assert "UVMap" in mesh.uv_layers


def test_box_places_fractional_anchor_and_stores_crease():
    bpy.ops.wm.read_homefile(use_empty=True)
    result = mesh_util.box(
        size=(2.0, 3.0, 4.0),
        location=(5.0, 6.0, 7.0),
        anchor=(0.1, 0.5, 1.0),
        crease=0.25,
    )
    mesh = pf.nodes.to_mesh_object(result).item().data
    positions = np.array([vertex.co[:] for vertex in mesh.vertices])

    np.testing.assert_allclose(positions.min(axis=0), (4.8, 4.5, 3.0))
    np.testing.assert_allclose(positions.max(axis=0), (6.8, 7.5, 7.0))
    assert all(item.value == 0.25 for item in mesh.attributes["crease_edge"].data)


def test_box_clamps_anchor_to_normalized_range():
    bpy.ops.wm.read_homefile(use_empty=True)
    result = mesh_util.box(size=(2.0, 3.0, 4.0), anchor=(0.0, 0.5, -1.0))
    mesh = pf.nodes.to_mesh_object(result).item().data
    positions = np.array([vertex.co[:] for vertex in mesh.vertices])

    np.testing.assert_allclose(positions.min(axis=0), (0.0, -1.5, 0.0))
    np.testing.assert_allclose(positions.max(axis=0), (2.0, 1.5, 4.0))


def test_fill_between_curves_stores_metric_uvs():
    bpy.ops.wm.read_homefile(use_empty=True)
    left = pf.nodes.geo.curve_line(start=(0, 0, 0), end=(0, 2, 0))
    right = pf.nodes.geo.curve_line(start=(3, 0, 0), end=(3, 2, 0))
    surface = mesh_util.fill_between_curves(left, right, n_points=5, n_rows=3)
    obj = pf.nodes.to_mesh_object(surface)
    uv = obj.item().data.uv_layers["UVMap"].data
    coords = np.array([entry.uv[:] for entry in uv])

    assert np.allclose(np.ptp(coords, axis=0), (2.0, 3.0))
