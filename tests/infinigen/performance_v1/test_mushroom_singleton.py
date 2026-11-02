# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

from unittest.mock import patch

import bpy
import numpy as np

from infinigen.core import init

# Importing the mushroom modules sets up add-ons these tests do not need.
# Patch only for these imports: pytest imports every test module during
# collection, so a module-level assignment would leak into unrelated tests.
with patch.object(init, "require_blender_addon", lambda *a, **k: False):
    from infinigen.assets.objects.mushroom import generate
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


# The class the gate substitutes once mushroom is selected; tests/conftest.py
# restores the selection after each test.
def _mushroom_factory_cls():
    import infinigen
    from infinigen.assets.objects.mushroom.generate import MushroomFactory

    infinigen.use_optimized(["mushroom"])
    cls = MushroomFactory.resolve()
    if cls is MushroomFactory:
        # the gate keeps the default when the kernels are not built
        _backend_unavailable()
    return cls


def _backend_unavailable():
    import pytest

    pytest.skip("the mushroom backend needs INFINIGEN_INSTALL_PERF=True")


def test_singleton_keypoints_do_not_change_placement():
    factory = _mushroom_factory_cls()(41001)
    with FixedSeed(902):
        points = np.random.uniform(-2, 2, (73, 3))
    for maker in (factory.cluster_make, factory.directional_make):
        with FixedSeed(17):
            expected = maker([points])
            rng_expected = np.random.get_state()
        with (
            FixedSeed(17),
            patch.object(
                generate.kdtree,
                "KDTree",
                side_effect=AssertionError("Singleton built KD tree"),
            ),
        ):
            actual = maker([np.zeros((1, 3))])
            rng_actual = np.random.get_state()
        for a, b in zip(expected, actual):
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
        assert rng_expected[0] == rng_actual[0]
        assert rng_expected[1].tobytes() == rng_actual[1].tobytes()
        assert rng_expected[2:] == rng_actual[2:]


def test_placement_builds_no_proxy():
    """Placement reads each mushroom's own surface, with no clone or voxel remesh."""

    class Growth:
        def create_asset(self, **kwargs):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6)
            return bpy.context.object

    cls = _mushroom_factory_cls()
    factory = cls.__new__(cls)
    import infinigen.perf.assets.objects.mushroom.factories as optimized

    factory.factory = Growth()
    factory.tolerant_length = 0
    for count in (1, 2):
        with (
            patch.object(optimized.np.random, "randint", return_value=count),
            patch.object(
                butil, "modify_mesh", side_effect=AssertionError("proxy remesh")
            ),
        ):
            objects, points = factory.build_mushrooms(0, 0.01)
        assert len(objects) == len(points) == count
        for obj in objects:
            butil.delete(obj)


def test_scatter_reuses_a_singleton_cluster():
    """`find_closest` works on the keypoints of a one-mushroom cluster."""

    class Growth:
        def create_asset(self, **kwargs):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6)
            return bpy.context.object

    cls = _mushroom_factory_cls()
    factory = cls.__new__(cls)
    import infinigen.perf.assets.objects.mushroom.factories as optimized

    factory.factory = Growth()
    factory.tolerant_length = 0
    with patch.object(optimized.np.random, "randint", return_value=1):
        objects, points = factory.build_mushrooms(0, 0.01)
    with patch.object(
        type(factory), "radius", new_callable=lambda: property(lambda self: 1.0)
    ):
        with FixedSeed(3):
            rotations = list(np.random.uniform(0, np.pi, (3, 3)))
            starts = list(np.random.uniform(-1, 1, (3, 3)))
            directions = list(np.eye(3))
            locations, _, scales = factory.find_closest(
                [points[0]] * 3, rotations, starts, directions
            )
    assert len(locations) == len(scales) == 3
    for obj in objects:
        butil.delete(obj)


def test_optimized_stems_never_use_the_spin_operator():
    """Both webs avoid Blender's spin operator, whose output varies between runs."""
    from infinigen.assets.objects.mushroom import stem as default_stem
    from infinigen.assets.utils import draw

    factory = _mushroom_factory_cls()(41001)
    stem = factory.factory.stem_factory

    def operator_spin(*args, **kwargs):
        raise AssertionError("the optimized stem called the spin operator")

    for builder in (stem.build_solid_web, stem.build_hollow_web):
        stem.web_builder = builder
        with (
            FixedSeed(5),
            patch.object(draw, "spin", operator_spin),
            patch.object(default_stem, "spin", operator_spin),
        ):
            obj = stem.create_asset(face_size=0.02)
        butil.delete(obj)


if __name__ == "__main__":
    butil.clear_scene()
    test_singleton_keypoints_do_not_change_placement()
    test_placement_builds_no_proxy()
    test_scatter_reuses_a_singleton_cluster()
    test_optimized_stems_never_use_the_spin_operator()
    print("MUSHROOM_SINGLETON_TESTS_PASS", flush=True)
