# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from collections.abc import Callable

import procfunc as pf
import pytest

from infinigen2.objects import bowl, plant_pot, random_primitives, vase
from infinigen2.objects.lamp import LampResult
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.room import decoration_objects

SMALL_OBJECT_GENERATORS = {
    func.__name__: func
    for func in (
        bowl.bowl_rand,
        plant_pot.plant_pot_small_rand,
        vase.cup_rand,
    )
}


def _resolve_label(label: str) -> tuple[object, object | None]:
    if "_effect_" not in label:
        return SMALL_OBJECT_GENERATORS.get(label), None
    primitive, _, effect = label.partition("_effect_")
    return (
        getattr(random_primitives, primitive, None),
        getattr(random_primitives, f"effect_{effect}", None),
    )


def test_small_objects_pool_has_unique_stems_and_sampler_labels(rng: pf.RNG) -> None:
    pool = decoration_objects.decoration_collection_primitives_and_real_rand(rng)
    data_names = [obj.item().data.name for obj in pool]
    labels = [decoration_objects._smallobj_label(name) for name in data_names]

    assert len(data_names) == len(set(data_names))

    for label in labels:
        assert "." not in label, label
        primitive, effect = _resolve_label(label)
        assert callable(primitive), label
        assert effect is None or callable(effect), label


def test_small_objects_are_never_labelled_after_the_wrapper(rng: pf.RNG) -> None:
    pool = decoration_objects.decoration_collection_primitives_and_real_rand(rng)
    labels = {decoration_objects._smallobj_label(obj.item().data.name) for obj in pool}

    assert random_primitives.primitive_with_effect_rand.__name__ not in labels
    assert random_primitives.primitive_rand.__name__ not in labels
    assert len(labels) > 1


def test_small_objects_label_survives_alias_copy_suffix() -> None:
    label = "cube_rand_effect_twist"
    assert decoration_objects._smallobj_label(f"{label}_004.003") == label


@pytest.mark.parametrize(
    "scatter",
    [
        decoration_objects.scatter_small_objects_on_containers,
        decoration_objects.scatter_small_objects_on_support_tops,
    ],
)
def test_small_object_scatter_collection_default_and_override(
    monkeypatch: pytest.MonkeyPatch, rng: pf.RNG, scatter: Callable
) -> None:
    calls: list[pf.RNG] = []

    def collection_rand(_rng: pf.RNG) -> pf.Collection:
        calls.append(_rng)
        return pf.Collection([pf.ops.primitives.mesh_cube(size=0.1)])

    monkeypatch.setattr(
        decoration_objects,
        "decoration_collection_primitives_rand",
        collection_rand,
    )
    target = pf.ops.primitives.mesh_cube(size=1.0)
    scatter(rng, [target], ccol.collision_set([target]), fraction=0.0)
    assert len(calls) == 1
    collection = pf.Collection([pf.ops.primitives.mesh_cube(size=0.1)])
    scatter(
        rng,
        [target],
        ccol.collision_set([target]),
        collection=collection,
        fraction=0.0,
    )
    assert len(calls) == 1


def test_rejected_surface_lamp_does_not_return_light(monkeypatch, rng: pf.RNG) -> None:
    lamp_result = LampResult(
        mesh=pf.ops.primitives.mesh_cube(size=0.2),
        light=pf.ops.primitives.point_lamp(energy=1000),
    )
    support = pf.ops.primitives.mesh_cube(size=1.0)
    monkeypatch.setattr(
        decoration_objects,
        "_sample_surface_collection",
        lambda *_: [lamp_result],
    )
    monkeypatch.setattr(decoration_objects, "retry_place", lambda *_, **__: None)

    result = decoration_objects.decorate_surface_objects_rand(
        rng,
        objects=[support],
        colliders=ccol.collision_set([]),
        support_tops=[support],
    )

    assert lamp_result.mesh not in result.all_objects
    assert lamp_result.light not in result.lights


def test_kept_surface_lamp_preserves_original_result(monkeypatch, rng: pf.RNG) -> None:
    lamp_result = LampResult(
        mesh=pf.ops.primitives.mesh_cube(size=0.2),
        light=pf.ops.primitives.point_lamp(energy=1000),
    )
    support = pf.ops.primitives.mesh_cube(size=1.0)
    seen = []
    monkeypatch.setattr(
        decoration_objects,
        "_sample_surface_collection",
        lambda *_: [lamp_result],
    )

    def keep_original(_rng, child, *_args, **_kwargs):
        seen.append(child)
        return child

    monkeypatch.setattr(decoration_objects, "retry_place", keep_original)
    result = decoration_objects.decorate_surface_objects_rand(
        rng,
        objects=[support],
        colliders=ccol.collision_set([]),
        support_tops=[support],
    )

    assert seen == [lamp_result]
    assert lamp_result.mesh in result.all_objects
    assert result.lights == [lamp_result.light]


def test_surface_decoration_centers_on_zero_extent_axis(
    monkeypatch: pytest.MonkeyPatch, rng: pf.RNG
) -> None:
    lamp_result = LampResult(
        mesh=pf.ops.primitives.mesh_cube(size=0.2),
        light=pf.ops.primitives.point_lamp(energy=1000),
    )
    support = pf.ops.primitives.mesh_cube(size=1.0)
    support.item().scale.y = 0.0
    placements: list[tuple[float, float]] = []

    def capture_placement(
        _rng: pf.RNG,
        _child: LampResult,
        _parents: list[pf.MeshObject],
        xy_frac: tuple[float, float],
    ) -> None:
        placements.append(xy_frac)

    monkeypatch.setattr(decoration_objects, "snap_on_top", capture_placement)
    decoration_objects._place_surface_decoration(rng, lamp_result, [support])

    assert placements[0][1] == 0.5
