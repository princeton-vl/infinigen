# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import procfunc as pf
import pytest

from infinigen2.exporters.realize_mesh import realize_scene
from infinigen2.scenes.placement.distribute import propagate_modifiers_to_instances

BASE_CUBE_VERTS = 8


def _template(name, levels):
    obj = pf.ops.primitives.mesh_cube(size=0.2)
    obj.item().name = obj.item().data.name = name
    pf.ops.modifier.subdivide_surface(obj, levels=levels, _skip_apply=True)
    return obj


def _instance(templates):
    points = pf.ops.primitives.mesh_grid(size=2, x_subdivisions=3, y_subdivisions=3)
    instances = pf.nodes.geo.instance_on_points(
        points=pf.nodes.geo.object_info(points, transform_space="RELATIVE").geometry,
        instance=pf.nodes.geo.collection_info(
            pf.Collection(templates), separate_children=True
        ),
        pick_instance=True,
    )
    aliases = pf.nodes.to_aliases(instances)
    propagate_modifiers_to_instances(templates, aliases)
    return aliases


def test_instances_keep_base_mesh_and_gain_modifier_copies():
    pf.ops.object.clear_scene()
    templates = [_template("tpl_a", 1), _template("tpl_b", 2)]
    aliases = _instance(templates)

    assert len(aliases) > 0
    levels_by_data = {}
    for alias in aliases:
        item = alias.item()
        assert len(item.data.vertices) == BASE_CUBE_VERTS
        mods = list(item.modifiers)
        assert [mod.type for mod in mods] == ["SUBSURF"]
        assert not mods[0].show_viewport
        assert mods[0].show_render
        levels_by_data.setdefault(item.data.name, set()).add(mods[0].levels)

    assert sorted(next(iter(v)) for v in levels_by_data.values()) == [1, 2]


def test_template_with_copy_suffix_still_matches():
    pf.ops.object.clear_scene()
    templates = [_template("tpl_a.003", 1)]
    aliases = _instance(templates)

    assert len(aliases) > 0
    for alias in aliases:
        assert [mod.levels for mod in alias.item().modifiers] == [1]


def test_ambiguous_template_stems_raise():
    pf.ops.object.clear_scene()
    templates = [_template("tpl_a.001", 1), _template("tpl_a.002", 2)]
    with pytest.raises(ValueError):
        propagate_modifiers_to_instances(templates, [])


def test_unknown_alias_stem_raises():
    pf.ops.object.clear_scene()
    templates = [_template("tpl_a", 1)]
    stranger = pf.ops.primitives.mesh_cube(size=0.2)
    stranger.item().data.name = "not_a_template"
    with pytest.raises(ValueError):
        propagate_modifiers_to_instances(templates, [stranger])


def test_realize_subdivides_shared_alias_data_once():
    pf.ops.object.clear_scene()
    templates = [_template("tpl_a", 1)]
    aliases = _instance(templates)
    realize_scene()

    datas = {alias.item().data.name for alias in aliases}
    assert len(datas) == 1
    for alias in aliases:
        assert len(alias.item().data.vertices) > BASE_CUBE_VERTS
        assert not list(alias.item().modifiers)
