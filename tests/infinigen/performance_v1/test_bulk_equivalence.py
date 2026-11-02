# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Optimized CORE helpers against references written out here, since their
equivalence depends on Blender's behavior."""

import bpy  # noqa: I001  -- bpy must be imported before bmesh
import bmesh
import numpy as np
import pytest

from infinigen.core.nodes.nodegroups import transfer_attributes
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
    verts = [(float(x), float(y), 0.0) for y in range(ny) for x in range(nx)]
    faces = [
        (j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i)
        for j in range(ny - 1)
        for i in range(nx - 1)
    ]
    return verts, faces


def cube():
    verts = [
        (float(x), float(y), float(z)) for z in (0, 1) for y in (0, 1) for x in (0, 1)
    ]
    faces = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    return verts, faces


CASES = {"grid": grid, "cube": cube}


@pytest.fixture(autouse=True)
def _clean():
    # an empty file, but not factory preferences: read_factory_settings
    # unregisters every add-on, which later tests in the same session need
    bpy.ops.wm.read_homefile(use_empty=True)
    yield


def _state(obj):
    m = obj.data
    co = np.empty(len(m.vertices) * 3, np.float32)
    m.vertices.foreach_get("co", co)
    return {
        "counts": (len(m.vertices), len(m.edges), len(m.polygons), len(m.loops)),
        "co": co,
        "edges": sorted(tuple(sorted(e.vertices)) for e in m.edges),
        "polys": sorted(tuple(p.vertices) for p in m.polygons),
        "sel_v": np.array([v.select for v in m.vertices], bool),
        "sel_e": np.array([e.select for e in m.edges], bool),
        "sel_p": np.array([p.select for p in m.polygons], bool),
    }


def _assert_same(a, b, what):
    assert a["counts"] == b["counts"], f"{what}: element counts"
    np.testing.assert_array_equal(a["co"], b["co"], err_msg=f"{what}: coordinates")
    assert a["edges"] == b["edges"], f"{what}: edges"
    assert a["polys"] == b["polys"], f"{what}: polygons"
    for k in ("sel_v", "sel_e", "sel_p"):
        np.testing.assert_array_equal(a[k], b[k], err_msg=f"{what}: {k}")


def _remove_vertices_edit_mode(obj, to_delete):
    """`remove_vertices` as an edit-mode round trip."""
    to_delete = np.nonzero(to_delete)[0]
    with butil.ViewportMode(obj, "EDIT"):
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.verts[i] for i in to_delete])
        bmesh.update_edit_mesh(obj.data)
    return obj


def _remove_edges_edit_mode(obj, to_delete):
    to_delete = np.nonzero(to_delete)[0]
    with butil.ViewportMode(obj, "EDIT"):
        bm = bmesh.from_edit_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        bmesh.ops.delete(
            bm, geom=[bm.edges[i] for i in to_delete], context="EDGES_FACES"
        )
        bmesh.update_edit_mesh(obj.data)
    return obj


@pytest.mark.parametrize("case", sorted(CASES))
def test_remove_vertices_matches_the_edit_mode_round_trip(case):
    verts, faces = CASES[case]()
    mask = np.zeros(len(verts), bool)
    mask[::3] = True

    want = _state(
        _remove_vertices_edit_mode(_new_mesh(f"rv_{case}_a", verts, faces), mask)
    )
    got = _state(
        perf_decorate.remove_vertices(_new_mesh(f"rv_{case}_b", verts, faces), mask)
    )
    _assert_same(got, want, f"remove_vertices[{case}]")


@pytest.mark.parametrize("case", sorted(CASES))
def test_remove_edges_matches_the_edit_mode_round_trip(case):
    verts, faces = CASES[case]()
    obj_a = _new_mesh(f"re_{case}_a", verts, faces)
    mask = np.zeros(len(obj_a.data.edges), bool)
    mask[::4] = True

    want = _state(_remove_edges_edit_mode(obj_a, mask))
    got = _state(
        perf_decorate.remove_edges(_new_mesh(f"re_{case}_b", verts, faces), mask)
    )
    _assert_same(got, want, f"remove_edges[{case}]")


def _with_uvs(name):
    verts, faces = grid()
    obj = _new_mesh(name, verts, faces)
    layer = obj.data.uv_layers.new()
    obj.data.uv_layers.active = layer
    rng = np.random.RandomState(0)
    uv = rng.uniform(size=len(obj.data.loops) * 2).astype(np.float32)
    layer.data.foreach_set("uv", uv)
    return obj


def _uvs_to_attribute_per_loop(obj, name="uv_map"):
    """Per-loop UV gather; a vertex with several loops keeps the last."""
    n = len(obj.data.vertices)
    data = np.zeros((n, 3), dtype=np.float32)
    for loop in obj.data.loops:
        u, v = obj.data.uv_layers.active.data[loop.index].uv
        data[loop.vertex_index] = u, v, 0
    attr = obj.data.attributes.new(name, type="FLOAT_VECTOR", domain="POINT")
    attr.data.foreach_set("vector", data.reshape(-1))
    return attr


def test_uvs_to_attribute_matches_the_per_loop_gather():
    a, b = _with_uvs("uv_a"), _with_uvs("uv_b")
    # interior grid vertices have four loops, so this exercises last-loop-wins
    assert max(np.bincount([loop.vertex_index for loop in a.data.loops])) > 1

    _uvs_to_attribute_per_loop(a)
    transfer_attributes.uvs_to_attribute(b)

    want = np.empty(len(a.data.vertices) * 3, np.float32)
    got = np.empty(len(b.data.vertices) * 3, np.float32)
    a.data.attributes["uv_map"].data.foreach_get("vector", want)
    b.data.attributes["uv_map"].data.foreach_get("vector", got)
    np.testing.assert_array_equal(got, want)


def test_attribute_to_uvs_matches_the_per_loop_write():
    a, b = _with_uvs("back_a"), _with_uvs("back_b")
    for obj in (a, b):
        transfer_attributes.uvs_to_attribute(obj, "src")

    n = len(a.data.vertices)
    data = np.empty(n * 3, np.float32)
    a.data.attributes["src"].data.foreach_get("vector", data)
    data = data.reshape((n, 3))
    a.data.uv_layers.active = a.data.uv_layers.new()
    for loop in a.data.loops:
        u, v, _ = data[loop.vertex_index]
        a.data.uv_layers.active.data[loop.index].uv = (u, v)

    transfer_attributes.attribute_to_uvs(b, "src")

    want = np.empty(len(a.data.loops) * 2, np.float32)
    got = np.empty(len(b.data.loops) * 2, np.float32)
    a.data.uv_layers.active.data.foreach_get("uv", want)
    b.data.uv_layers.active.data.foreach_get("uv", got)
    np.testing.assert_array_equal(got, want)
