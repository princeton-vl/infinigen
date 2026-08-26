# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import procfunc as pf

from infinigen2.objects import random_primitives
from infinigen2.scenes.room import room_small_objects


def _resolve_label(label: str) -> tuple[object, object]:
    primitive, _, effect = label.partition("_effect_")
    return (
        getattr(random_primitives, primitive, None),
        getattr(random_primitives, f"effect_{effect}", None),
    )


def test_small_objects_pool_has_unique_stems_and_sampler_labels(rng: pf.RNG) -> None:
    pool = room_small_objects.small_objects_collection_rand(rng)
    data_names = [obj.item().data.name for obj in pool]
    labels = [room_small_objects._smallobj_label(name) for name in data_names]

    assert len(data_names) == len(set(data_names))

    for label in labels:
        assert "." not in label, label
        primitive, effect = _resolve_label(label)
        assert callable(primitive), label
        assert callable(effect), label


def test_small_objects_are_never_labelled_after_the_wrapper(rng: pf.RNG) -> None:
    pool = room_small_objects.small_objects_collection_rand(rng)
    labels = {room_small_objects._smallobj_label(obj.item().data.name) for obj in pool}

    assert random_primitives.primitive_with_effect_rand.__name__ not in labels
    assert random_primitives.primitive_rand.__name__ not in labels
    assert len(labels) > 1


def test_small_objects_label_survives_alias_copy_suffix() -> None:
    label = "cube_rand_effect_twist"
    assert room_small_objects._smallobj_label(f"{label}_004.003") == label
