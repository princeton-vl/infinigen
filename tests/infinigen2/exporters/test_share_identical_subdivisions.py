# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import procfunc as pf

from infinigen2.exporters import realize_mesh
from infinigen2.scenes.placement.distribute import propagate_modifiers_to_instances


def _template(name: str, levels: int) -> pf.MeshObject:
    obj = pf.ops.primitives.mesh_cube(size=0.2)
    obj.item().name = obj.item().data.name = name
    pf.ops.modifier.subdivide_surface(obj, levels=levels, _skip_apply=True)
    return obj


def _aliases_of(templates: list[pf.MeshObject]) -> list[pf.MeshObject]:
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


def test_grouped_aliases_share_one_baked_datablock():
    pf.ops.object.clear_scene()
    templates = [_template("tpl_a", 2)]
    aliases = _aliases_of(templates)
    realize_mesh.bake_shared_modifier_prefixes(aliases)

    assert len(aliases) > 1
    assert len({alias.item().data.as_pointer() for alias in aliases}) == 1
    for alias in aliases:
        assert not list(alias.item().modifiers)
        assert len(alias.item().data.vertices) > 8
    assert [mod.type for mod in templates[0].item().modifiers] == ["SUBSURF"]
    assert len(templates[0].item().data.vertices) == 8


def test_groups_stay_separate_per_template():
    pf.ops.object.clear_scene()
    templates = [_template("tpl_a", 1), _template("tpl_b", 2)]
    aliases = _aliases_of(templates)
    realize_mesh.bake_shared_modifier_prefixes(aliases)

    counts = {len(alias.item().data.vertices) for alias in aliases}
    assert len(counts) == 2
    assert len({alias.item().data.as_pointer() for alias in aliases}) == 2


def test_singleton_keeps_deferred_stack():
    pf.ops.object.clear_scene()
    lone = _template("lone", 2)
    realize_mesh.bake_shared_modifier_prefixes([lone])

    assert [mod.type for mod in lone.item().modifiers] == ["SUBSURF"]
    assert len(lone.item().data.vertices) == 8


def test_differing_stacks_on_shared_data_stay_deferred():
    pf.ops.object.clear_scene()
    a = _template("stack_a", 1)
    b = pf.ops.primitives.mesh_cube(size=0.2)
    b.item().data = a.item().data
    pf.ops.modifier.subdivide_surface(b, levels=3, _skip_apply=True)
    realize_mesh.bake_shared_modifier_prefixes([a, b])

    assert [mod.levels for mod in a.item().modifiers] == [1]
    assert [mod.levels for mod in b.item().modifiers] == [3]
    assert a.item().data.as_pointer() == b.item().data.as_pointer()


def test_second_run_is_a_noop():
    pf.ops.object.clear_scene()
    templates = [_template("tpl_a", 2)]
    aliases = _aliases_of(templates)
    objects = aliases
    realize_mesh.bake_shared_modifier_prefixes(objects)
    once = len(aliases[0].item().data.vertices)
    realize_mesh.bake_shared_modifier_prefixes(objects)

    assert len(aliases[0].item().data.vertices) == once
    assert len({alias.item().data.as_pointer() for alias in aliases}) == 1


def test_bake_respects_render_visibility():
    pf.ops.object.clear_scene()
    templates = [_template("tpl_a", 2)]
    aliases = _aliases_of(templates)
    for alias in aliases:
        assert not alias.item().modifiers[0].show_viewport
    realize_mesh.bake_shared_modifier_prefixes(aliases)

    for alias in aliases:
        assert len(alias.item().data.vertices) > 8


def test_bake_preserves_all_mesh_attributes():
    pf.ops.object.clear_scene()
    templates = [_template("tpl_a", 1)]
    aliases = _aliases_of(templates)
    source = aliases[0].item().data
    attribute = source.attributes.new("export_data", "FLOAT", "POINT")
    attribute.data[0].value = 0.75
    realize_mesh.bake_shared_modifier_prefixes(aliases)

    baked_attribute = aliases[0].item().data.attributes["export_data"]
    assert baked_attribute.data[0].value == 0.75


def test_shape_keys_keep_deferred_stack():
    pf.ops.object.clear_scene()
    templates = [_template("tpl_a", 1)]
    aliases = _aliases_of(templates)
    aliases[0].item().shape_key_add(name="Basis")
    realize_mesh.bake_shared_modifier_prefixes(aliases)

    for alias in aliases:
        assert [mod.type for mod in alias.item().modifiers] == ["SUBSURF"]
        assert alias.item().data.shape_keys is not None


def test_animated_modifiers_keep_deferred_stack():
    pf.ops.object.clear_scene()
    templates = [_template("tpl_a", 1)]
    aliases = _aliases_of(templates)
    aliases[0].item().modifiers[0].keyframe_insert(data_path="levels", frame=1)
    realize_mesh.bake_shared_modifier_prefixes(aliases)

    assert [mod.type for mod in aliases[0].item().modifiers] == ["SUBSURF"]


def test_differing_simplify_limits_keep_deferred_stack():
    pf.ops.object.clear_scene()
    templates = [_template("tpl_a", 1)]
    aliases = _aliases_of(templates)
    render = aliases[0].item().users_scene[0].render
    previous = (
        render.use_simplify,
        render.simplify_subdivision,
        render.simplify_subdivision_render,
    )
    try:
        render.use_simplify = True
        render.simplify_subdivision = 0
        render.simplify_subdivision_render = 2
        realize_mesh.bake_shared_modifier_prefixes(aliases)
    finally:
        (
            render.use_simplify,
            render.simplify_subdivision,
            render.simplify_subdivision_render,
        ) = previous

    for alias in aliases:
        assert [mod.type for mod in alias.item().modifiers] == ["SUBSURF"]


def test_shared_modifier_sequence_is_baked_in_order():
    pf.ops.object.clear_scene()
    a = pf.ops.primitives.mesh_cube(size=0.2)
    b = pf.ops.primitives.mesh_cube(size=0.2)
    b.item().data = a.item().data
    for obj in (a, b):
        obj.item().modifiers.new("shared bevel", "BEVEL").width = 0.01
        obj.item().modifiers.new("shared subdivision", "SUBSURF").levels = 1
    a.item().modifiers.new("suffix", "BOOLEAN")
    b.item().modifiers.new("suffix", "NODES")

    realize_mesh.bake_shared_modifier_prefixes([a, b])

    assert a.item().data.as_pointer() == b.item().data.as_pointer()
    assert len(a.item().data.vertices) > 8
    assert [mod.type for mod in a.item().modifiers] == ["BOOLEAN"]
    assert [mod.type for mod in b.item().modifiers] == ["NODES"]


def test_context_dependent_prefix_stays_deferred():
    pf.ops.object.clear_scene()
    a = pf.ops.primitives.mesh_cube(size=0.2)
    b = pf.ops.primitives.mesh_cube(size=0.2)
    b.item().data = a.item().data
    for obj in (a, b):
        obj.item().modifiers.new("unsafe", "BOOLEAN")
        obj.item().modifiers.new("shared subdivision", "SUBSURF").levels = 1

    realize_mesh.bake_shared_modifier_prefixes([a, b])

    for obj in (a, b):
        assert [mod.type for mod in obj.item().modifiers] == ["BOOLEAN", "SUBSURF"]
