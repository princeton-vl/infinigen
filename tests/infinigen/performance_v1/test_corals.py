# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

import bpy
import numpy as np

import infinigen
from infinigen.assets.utils import reaction_diffusion as rd
from infinigen.assets.utils.object import new_icosphere
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.perf.assets.objects.corals import laplacian as opt_laplacian
from infinigen.perf.assets.utils import reaction_diffusion as opt_rd


def _weights(obj, name):
    """Weight per vertex in vertex group `name`, 0 where it is unassigned."""
    group = obj.vertex_groups[name].index
    out = np.zeros(len(obj.data.vertices), np.float32)
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == group:
                out[v.index] = g.weight
    return out


def _weight_fn(co):
    return 0.5 + 0.5 * np.sin(7 * co[:, 0]) * np.cos(5 * co[:, 1])


def _run(selected, steps_impl=None):
    butil.clear_scene()
    infinigen.use_optimized(selected)
    obj = new_icosphere(subdivisions=3)
    saved = opt_rd._steps_impl
    if steps_impl is not None:
        opt_rd._steps_impl = steps_impl
    try:
        with FixedSeed(5):
            rd.reaction_diffusion(obj, _weight_fn, steps=200)
    finally:
        opt_rd._steps_impl = saved
    return {n: _weights(obj, n) for n in ("A", "B", "LA", "LB")}


def test_reaction_diffusion_matches_the_default():
    try:
        default = _run(False)
        compiled = _run(["corals"])
        arrays = _run(["corals"], steps_impl=opt_rd._steps_numpy)
    finally:
        infinigen.use_optimized(False)
    for name in default:
        assert np.array_equal(default[name], compiled[name]), name
        assert np.array_equal(default[name], arrays[name]), name


def _positions(obj):
    co = np.empty(len(obj.data.vertices) * 3, np.float32)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def _faces(obj):
    totals = np.empty(len(obj.data.polygons), np.int32)
    obj.data.polygons.foreach_get("loop_total", totals)
    verts = np.empty(len(obj.data.loops), np.int32)
    obj.data.loops.foreach_get("vertex_index", verts)
    return totals, verts


def test_cauliflower_subdivision_matches_the_modifier():
    butil.clear_scene()
    obj = new_icosphere(subdivisions=2)
    co = _positions(obj)
    co *= 1 + 0.1 * np.sin(9 * co[:, :1])  # break the symmetry
    obj.data.vertices.foreach_set("co", co.reshape(-1))
    obj.data.update()
    reference = butil.deep_clone_obj(obj)
    butil.modify_mesh(reference, "SUBSURF", levels=1, render_levels=1)
    opt_laplacian.subdivide_limit(obj)

    assert np.array_equal(_faces(obj)[0], _faces(reference)[0])
    assert np.array_equal(_faces(obj)[1], _faces(reference)[1])
    np.testing.assert_allclose(_positions(obj), _positions(reference), atol=1e-6)
    assert bpy.data.meshes.get(obj.data.name) is not None
