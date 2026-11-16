# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import inspect

import pytest

from infinigen.assets.composition import material_assignments


def wall_material_classes():
    lists = [
        material_assignments.wall,
        material_assignments.kitchen_wall,
        material_assignments.garage_wall,
        material_assignments.utility_wall,
        material_assignments.balcony_wall,
        material_assignments.bathroom_wall,
        material_assignments.warehouse_wall,
    ]
    seen = {}
    for weighted in lists:
        for cls, _ in weighted:
            seen[cls.__name__] = cls
    return list(seen.values())


@pytest.mark.parametrize("wall_cls", wall_material_classes())
def test_wall_material_accepts_room_walls_kwargs(wall_cls):
    # room_walls() gates the tile kwargs on whether generate() accepts **kwargs;
    # replicate that gate and assert the resulting call binds. Passing the tile
    # kwargs unconditionally raised TypeError for Concrete/Brick etc. (issue #505).
    sig = inspect.signature(wall_cls.generate)
    kwargs = dict(vertical=True, alternating=False, shape="square")
    if not any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ):
        kwargs = {}
    sig.bind(None, **kwargs)  # None stands in for the unbound `self`


def test_concrete_wall_rejects_tile_kwargs_without_gate():
    # Guards the premise of the gate: Concrete.generate has no **kwargs, so the
    # old unconditional room_walls() call crashed on garage/utility/warehouse walls.
    from infinigen.assets.materials.ceramic.concrete import Concrete

    sig = inspect.signature(Concrete.generate)
    with pytest.raises(TypeError):
        sig.bind(None, vertical=True, alternating=False, shape="square")
