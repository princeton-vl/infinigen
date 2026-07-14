# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import math

import procfunc as pf
import pytest

from infinigen2.util import mesh as mesh_util


def _creased_edges(mesh_node: pf.ProcNode, threshold_degrees: float):
    geo = mesh_util.crease_sharp(mesh_node, threshold_degrees=threshold_degrees)
    angle = pf.nodes.geo.input_mesh_edge_angle().unsigned_angle
    geo = pf.nodes.geo.store_named_attribute(
        geometry=geo, name="face_angle", value=angle, domain="EDGE"
    )
    obj = pf.nodes.to_mesh_object(geo)
    me = obj.item().data

    assert "crease_edge" in me.attributes
    creased = [d.value > 0.5 for d in me.attributes["crease_edge"].data]
    angles = [math.degrees(d.value) for d in me.attributes["face_angle"].data]
    return list(zip(angles, creased))


def test_crease_sharp_marks_edges_above_threshold():
    cube = pf.nodes.geo.mesh_cube(size=(1.0, 1.0, 1.0)).mesh

    # Every box edge folds at 90deg: all crease below 90, none above.
    assert all(c for _, c in _creased_edges(cube, threshold_degrees=40.0))
    assert not any(c for _, c in _creased_edges(cube, threshold_degrees=100.0))


@pytest.mark.parametrize("vertices", [8, 16, 24])
def test_crease_sharp_cylinder_creases_rims_not_round_wall(vertices):
    cyl = pf.nodes.geo.mesh_cylinder(vertices=vertices, radius=0.5, depth=1.0).mesh

    # Threshold 70 sits between the round-wall seams (<=45deg) and the 90deg cap rims.
    edges = _creased_edges(cyl, threshold_degrees=70.0)
    rims = [c for a, c in edges if a > 80]
    walls = [c for a, c in edges if a < 60]

    assert len(rims) == 2 * vertices
    assert all(rims)
    assert not any(walls)


def test_crease_sharp_flat_grid_creases_nothing():
    grid = pf.nodes.geo.mesh_grid(
        size_x=1.0, size_y=1.0, vertices_x=4, vertices_y=4
    ).mesh

    assert not any(c for _, c in _creased_edges(grid, threshold_degrees=1.0))
