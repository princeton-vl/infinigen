# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

import bpy
import numpy as np
import pytest

pytest.importorskip(
    "infinigen.perf.assets.utils._native_fields",
    reason="needs the native kernels (INFINIGEN_INSTALL_PERF=True)",
)

import infinigen  # noqa: E402
from infinigen.assets.materials.plant.bark_random import BarkRandom  # noqa: E402
from infinigen.core.util import blender as butil  # noqa: E402
from infinigen.core.util.math import FixedSeed  # noqa: E402
from infinigen.perf.assets.utils import surface_fields  # noqa: E402


class _Factory:
    """Tree factory stub; trees call `BarkRandom.apply` unbound with it as `self`."""

    def __init__(self, adapt_mesh_method):
        self.adapt_mesh_method = adapt_mesh_method


def _trunk():
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, depth=2, radius=0.2)
    obj = bpy.context.object
    n = len(obj.data.vertices)
    co = np.empty(n * 3, np.float32)
    obj.data.vertices.foreach_get("co", co)
    axis = co.reshape(-1, 3).copy()
    axis[:, :2] = 0
    for name, values in (("skeleton_loc", axis), ("parent_skeleton_loc", axis * 0)):
        attr = obj.data.attributes.new(name, "FLOAT_VECTOR", "POINT")
        attr.data.foreach_set("vector", values.reshape(-1))
    return obj


def _positions(obj):
    co = np.empty(len(obj.data.vertices) * 3, np.float32)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def test_native_trunk_bark_is_deferred_then_displaced():
    butil.clear_scene()
    infinigen.use_optimized(["trees"])
    obj = _trunk()
    with FixedSeed(3):
        BarkRandom.apply(_Factory("stem_tessellate"), obj)
    assert surface_fields.has_pending(obj)
    assert not obj.modifiers  # no geometry-node displacement
    before = _positions(obj)
    assert surface_fields.apply_pending(obj)
    after = _positions(obj)
    assert np.isfinite(after).all() and not np.array_equal(before, after)
    captured = np.empty(before.size, np.float32)
    obj.data.attributes["initial_position"].data.foreach_get("vector", captured)
    assert np.array_equal(captured.reshape(-1, 3), before)
    assert len(obj.data.materials) == 1


def test_other_trunks_keep_the_default_bark():
    butil.clear_scene()
    infinigen.use_optimized(["trees"])
    obj = _trunk()
    with FixedSeed(3):
        BarkRandom.apply(_Factory("subdivide"), obj)
    assert not surface_fields.has_pending(obj)
    assert obj.modifiers  # the default geometry-node displacement


def test_pending_bark_follows_the_pieces_of_a_split():
    butil.clear_scene()
    infinigen.use_optimized(["trees"])
    whole, piece_a, piece_b = _trunk(), _trunk(), _trunk()
    with FixedSeed(3):
        BarkRandom.apply(_Factory("stem_tessellate"), whole)
    surface_fields.transfer_pending(whole, piece_a, piece_b)
    assert not surface_fields.has_pending(whole)
    assert surface_fields.apply_pending(piece_a)
    assert surface_fields.apply_pending(piece_b)


def test_realized_twig_leaves_keep_their_own_surface():
    """Leaf faces in a realized twig mesh never get the bark material."""
    pytest.importorskip(
        "infinigen.perf.assets.objects.trees._cylinder_mesher",
        reason="needs the native cylinder mesher (INFINIGEN_INSTALL_PERF=True)",
    )
    from infinigen.assets.objects.trees.generate import (
        make_twig_collection,
        random_species,
    )

    butil.clear_scene()
    infinigen.use_optimized(["trees"])
    with FixedSeed(4):
        (_, twig_params, leaf_params), _ = random_species(season="summer")
        col = make_twig_collection(
            4,
            twig_params,
            leaf_params,
            BarkRandom,
            2,
            1,
            "leaf_broadleaf",
            season="summer",
        )
    twigs = [o for o in col.objects if o.type == "MESH"]
    assert twigs
    for twig in twigs:
        mesh = twig.data
        bark = [
            i
            for i, m in enumerate(mesh.materials)
            if m and m.name.startswith("shader_random_bark")
        ]
        assert bark and len(mesh.materials) > len(bark)  # leaf materials survive
        mat = np.empty(len(mesh.polygons), np.int64)
        mesh.polygons.foreach_get("material_index", mat)
        vein = np.empty(len(mesh.vertices))
        mesh.attributes["vein value"].data.foreach_get("value", vein)
        loops = np.empty(len(mesh.loops), np.int64)
        mesh.loops.foreach_get("vertex_index", loops)
        loop_face = np.repeat(
            np.arange(len(mesh.polygons)), [len(p.vertices) for p in mesh.polygons]
        )
        leafy = np.zeros(len(mesh.polygons), bool)
        np.logical_or.at(leafy, loop_face, vein[loops] > 0)
        assert not (np.isin(mat, bark) & leafy).any(), (
            "leaf faces carry the bark material"
        )
