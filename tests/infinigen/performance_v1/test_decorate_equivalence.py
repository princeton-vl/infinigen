# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Optimized decorate helpers (CORE) against the default, each run both ways on one
mesh, since some equivalences depend on Blender's behavior."""

import bmesh
import bpy
import numpy as np
import pytest

from infinigen.assets.utils import decorate
from infinigen.core.util import blender as butil

perf_decorate = pytest.importorskip(
    "infinigen.perf.assets.utils.decorate", reason="needs infinigen.perf"
)


def _new_mesh(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [list(f) for f in faces])
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def grid(nx=6, ny=5):
    verts = [(x, y, 0.0) for y in range(ny) for x in range(nx)]
    faces = [
        (j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i)
        for j in range(ny - 1)
        for i in range(nx - 1)
    ]
    return verts, faces


def grid_with_hole():
    verts, faces = grid()
    # drop a face from the middle, so there is an interior boundary too
    del faces[len(faces) // 2]
    return verts, faces


def closed_cube():
    verts = [(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)]
    faces = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    return verts, faces


def strip():
    return [(i, 0.0, 0.0) for i in range(4)] + [(i, 1.0, 0.0) for i in range(4)], [
        (i, i + 1, i + 5, i + 4) for i in range(3)
    ]


CASES = {
    "grid": grid,
    "grid_with_hole": grid_with_hole,
    "closed_cube": closed_cube,
    "strip": strip,
}


@pytest.fixture
def default_path(monkeypatch):
    """Use the default of every gated `decorate` function, whatever the run selected."""
    for name, fn in list(vars(decorate).items()):
        if callable(fn) and hasattr(fn, "resolve") and hasattr(fn, "default"):
            monkeypatch.setattr(decorate, name, fn.default)


@pytest.fixture(autouse=True)
def _clean():
    # an empty file, but not factory preferences: read_factory_settings
    # unregisters every add-on, which later tests in the same session need
    bpy.ops.wm.read_homefile(use_empty=True)
    yield


def _boundary_by_operators(obj):
    """Edges and vertices that `select_all` + `region_to_loop` select."""
    with butil.ViewportMode(obj, "EDIT"):
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.region_to_loop()
    with butil.ViewportMode(obj, "EDIT"):
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        verts = np.array([v.select for v in bm.verts], dtype=bool)
        edges = np.array([e.select for e in bm.edges], dtype=bool)
    return edges, verts


KNOWN_DIFFERENT = {"grid_with_hole", "strip"}


@pytest.mark.parametrize("case", sorted(CASES))
def test_mesh_boundary_against_region_to_loop(case):
    """Vertices always match `region_to_loop`'s; edges do outside `KNOWN_DIFFERENT`."""
    verts, faces = CASES[case]()
    obj = _new_mesh(f"boundary_{case}", verts, faces)
    want_edges, want_verts = _boundary_by_operators(obj)

    obj2 = _new_mesh(f"boundary_{case}_b", verts, faces)
    got_edges, got_verts = perf_decorate.mesh_boundary(obj2)

    # distance2boundary seeds its search from these vertices, so they must
    # match even where the edges do not
    np.testing.assert_array_equal(
        got_verts, want_verts, err_msg=f"{case}: boundary vertices"
    )

    if case in KNOWN_DIFFERENT:
        assert not np.array_equal(got_edges, want_edges), (
            f"{case} now agrees on edges too; the opt-in gate may no longer be needed"
        )
        return
    np.testing.assert_array_equal(got_edges, want_edges)


def _distance_by_bfs(obj):
    """`distance2boundary` as a breadth-first search from `region_to_loop`."""
    with butil.ViewportMode(obj, "EDIT"):
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.region_to_loop()
    with butil.ViewportMode(obj, "EDIT"):
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        distance = np.full(len(obj.data.vertices), -100.0)
        queue = set(v.index for v in bm.verts if v.select)
        d = 0
        while True:
            if queue:
                distance[list(queue)] = d
            next_queue = set()
            for i in queue:
                v = bm.verts[i]
                for e in v.link_edges:
                    next_queue.add(e.other_vert(v).index)
            queue = set(i for i in next_queue if distance[i] < 0)
            if not queue:
                break
            d += 1
    distance[distance < 0] = 0
    distance /= max(d, 1)
    return distance


@pytest.mark.parametrize("case", sorted(CASES))
def test_default_distance2boundary_is_the_operator_route(case, default_path):
    verts, faces = CASES[case]()
    want = _distance_by_bfs(_new_mesh(f"dist_{case}", verts, faces))
    got = decorate.distance2boundary(_new_mesh(f"dist_{case}_b", verts, faces))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


@pytest.mark.parametrize("case", sorted(CASES))
def test_default_distance2boundary_leaves_the_same_selection(case, default_path):
    verts, faces = CASES[case]()
    ref = _new_mesh(f"sel_{case}", verts, faces)
    _distance_by_bfs(ref)
    want = {
        domain: np.array([e.select for e in getattr(ref.data, domain)], dtype=bool)
        for domain in ("vertices", "edges", "polygons")
    }

    obj = _new_mesh(f"sel_{case}_b", verts, faces)
    decorate.distance2boundary(obj)
    for domain, expected in want.items():
        got = np.array([e.select for e in getattr(obj.data, domain)], dtype=bool)
        np.testing.assert_array_equal(got, expected, err_msg=f"{case}: {domain}")


@pytest.mark.parametrize("case", sorted(CASES))
def test_bulk_reads_match_per_element_reads(case, default_path):
    verts, faces = CASES[case]()
    obj = _new_mesh(f"read_{case}", verts, faces)
    mesh = obj.data

    np.testing.assert_array_equal(
        decorate.read_co(obj),
        np.array([tuple(v.co) for v in mesh.vertices], dtype=np.float32).reshape(-1, 3),
    )
    np.testing.assert_array_equal(
        decorate.read_edges(obj),
        np.array([tuple(e.vertices) for e in mesh.edges]).reshape(-1, 2),
    )
    if len(mesh.polygons):
        np.testing.assert_allclose(
            decorate.read_area(obj),
            np.array([p.area for p in mesh.polygons], dtype=np.float32),
            rtol=1e-6,
        )
        np.testing.assert_array_equal(
            decorate.read_material_index(obj),
            np.array([p.material_index for p in mesh.polygons]),
        )


def test_bulk_write_matches_per_element_write(default_path):
    verts, faces = grid()
    a = _new_mesh("write_a", verts, faces)
    b = _new_mesh("write_b", verts, faces)
    new = np.asarray(decorate.read_co(a), dtype=np.float64) + 0.37

    decorate.write_co(a, new)
    for v, co in zip(b.data.vertices, new):
        v.co = co

    np.testing.assert_array_equal(decorate.read_co(a), decorate.read_co(b))
