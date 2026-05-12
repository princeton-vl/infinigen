# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan


import bpy

# import pytest
import numpy as np
from mathutils import Vector

from infinigen.core import tagging
from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints.example_solver import state_def
from infinigen.core.constraints.example_solver.geometry import parse_scene, validity
from infinigen.core.util import blender as butil


def make_scene(loc2):
    """Create a scene with a table and a cup, and return the state."""
    butil.clear_scene()
    objs = {}

    table = butil.spawn_cube(scale=(5, 5, 1), name="table")
    cup = butil.spawn_cube(scale=(1, 1, 1), name="cup", location=loc2)

    for o in [table, cup]:
        butil.apply_transform(o)
        parse_scene.preprocess_obj(o)
        tagging.tag_canonical_surfaces(o)

    assert table.scale == Vector((1, 1, 1))
    assert cup.location != Vector((0, 0, 0))

    bpy.context.view_layer.update()

    objs["table"] = state_def.ObjectState(table)
    objs["cup"] = state_def.ObjectState(cup)
    objs["cup"].relations.append(
        state_def.RelationState(
            cl.StableAgainst({t.Subpart.Bottom}, {t.Subpart.Top}),
            target_name="table",
            child_plane_idx=0,
            parent_plane_idx=0,
        )
    )

    # butil.save_blend('test.blend')

    return state_def.State(objs=objs)


def make_scene_coplanar(loc2):
    """Create a scene with a table and a cup, and return the state."""
    butil.clear_scene()
    objs = {}

    table = butil.spawn_cube(scale=(5, 5, 1), name="table")
    cup = butil.spawn_cube(scale=(1, 1, 1), name="cup", location=loc2)

    for o in [table, cup]:
        butil.apply_transform(o)
        parse_scene.preprocess_obj(o)
        tagging.tag_canonical_surfaces(o)

    assert table.scale == Vector((1, 1, 1))
    assert cup.location != Vector((0, 0, 0))

    bpy.context.view_layer.update()

    objs["table"] = state_def.ObjectState(table)
    objs["cup"] = state_def.ObjectState(cup)
    objs["cup"].relations.append(
        state_def.RelationState(
            cl.StableAgainst({t.Subpart.Bottom}, {t.Subpart.Top}),
            target_name="table",
            child_plane_idx=0,
            parent_plane_idx=0,
        )
    )
    back = {t.Subpart.Back, -t.Subpart.Top, -t.Subpart.Front}
    back_coplanar_back = cl.CoPlanar(back, back, margin=0)

    objs["cup"].relations.append(
        state_def.RelationState(
            back_coplanar_back,
            target_name="table",
            child_plane_idx=0,
            parent_plane_idx=0,
        )
    )
    butil.save_blend("test.blend")

    return state_def.State(objs=objs)


def test_stable_against():
    # too low, intersects ground
    assert not validity.check_post_move_validity(make_scene((0, 0, 0.5)), "cup")

    # exactly touches surface
    assert validity.check_post_move_validity(make_scene((0, 0, 1)), "cup")

    # underneath
    assert not validity.check_post_move_validity(make_scene((0, 0, -3)), "cup")

    # exactly at corner
    assert validity.check_post_move_validity(make_scene((2, 2, 1)), "cup")

    # slightly over corner
    assert not validity.check_post_move_validity(make_scene((2.1, 2.1, 1)), "cup")

    # farr away
    assert not validity.check_post_move_validity(make_scene((4, 4, 0.5)), "cup")


def test_horizontal_stability():
    butil.clear_scene()
    objs = {}

    table = butil.spawn_cube(name="table")
    table.dimensions = (4, 10, 2)

    chair1 = butil.spawn_cube(name="chair1")
    chair1.dimensions = (2, 2, 3)
    chair1.location = (3, 3, 0)

    chair2 = butil.spawn_cube(name="chair2")
    chair2.dimensions = (2, 2, 3)
    chair2.location = (3, -3, 0)

    chair3 = butil.spawn_cube(name="chair3")
    chair3.dimensions = (2, 2, 3)
    chair3.location = (-3, 3, 0)

    chair4 = butil.spawn_cube(name="chair4")
    chair4.dimensions = (2, 2, 3)
    chair4.location = (-3, -3, 0)
    for o in [table, chair1, chair2, chair3, chair4]:
        butil.apply_transform(o)
        parse_scene.preprocess_obj(o)
        tagging.tag_canonical_surfaces(o)
    with butil.SelectObjects([table, chair1, chair2, chair3, chair4]):
        # rotate
        bpy.ops.transform.rotate(value=np.pi / 4, orient_axis="Z", orient_type="GLOBAL")
    # butil.save_blend('test.blend')
    bpy.context.view_layer.update()

    objs["table"] = state_def.ObjectState(table)
    objs["chair1"] = state_def.ObjectState(chair1)
    objs["chair2"] = state_def.ObjectState(chair2)
    objs["chair3"] = state_def.ObjectState(chair3)
    objs["chair4"] = state_def.ObjectState(chair4)
    objs["chair1"].relations.append(
        state_def.RelationState(
            cl.StableAgainst({t.Subpart.Back}, {t.Subpart.Front}, check_z=False),
            target_name="table",
            child_plane_idx=0,
            parent_plane_idx=0,
        )
    )
    objs["chair2"].relations.append(
        state_def.RelationState(
            cl.StableAgainst({t.Subpart.Back}, {t.Subpart.Front}, check_z=False),
            target_name="table",
            child_plane_idx=0,
            parent_plane_idx=0,
        )
    )
    objs["chair3"].relations.append(
        state_def.RelationState(
            cl.StableAgainst({t.Subpart.Front}, {t.Subpart.Back}, check_z=False),
            target_name="table",
            child_plane_idx=0,
            parent_plane_idx=0,
        )
    )
    objs["chair4"].relations.append(
        state_def.RelationState(
            cl.StableAgainst({t.Subpart.Front}, {t.Subpart.Back}, check_z=False),
            target_name="table",
            child_plane_idx=0,
            parent_plane_idx=0,
        )
    )
    state = state_def.State(objs=objs)
    assert validity.check_post_move_validity(state, "chair1")
    assert validity.check_post_move_validity(state, "chair2")
    assert validity.check_post_move_validity(state, "chair3")
    assert validity.check_post_move_validity(state, "chair4")

    # butil.save_blend('test.blend')


def test_coplanar():
    # Test case 1: Cup is stable against but not coplanar (should be invalid)
    assert not validity.check_post_move_validity(make_scene_coplanar((0, 0, 1)), "cup")

    # Test case 2: Cup is stable against and coplanar with the table (should be valid)
    assert validity.check_post_move_validity(make_scene_coplanar((-2, 0, 1)), "cup")

    # Test case 3: Cup is coplanar but not stable against (should be invalid)
    assert not validity.check_post_move_validity(
        make_scene_coplanar((-5.2, 0, 1)), "cup"
    )

    # Test case 4: Cup is neither stable against nor coplanar (should be invalid)
    assert not validity.check_post_move_validity(
        make_scene_coplanar((2, 2, 1.1)), "cup"
    )

    # Test case 5: Cup is at the back edge, stable against and coplanar (should be valid)
    assert validity.check_post_move_validity(make_scene_coplanar((-2, 2, 1)), "cup")

    # Test case 6: Cup is slightly off the back edge, not stable against but coplanar (should be invalid)
    assert not validity.check_post_move_validity(
        make_scene_coplanar((-2.1, 2, 1)), "cup"
    )

    # Test case 7: Cup is far from the table (should be invalid)
    assert not validity.check_post_move_validity(
        make_scene_coplanar((10, 10, 10)), "cup"
    )

    # Test case 8: Cup is inside the table, not stable against but coplanar (should be invalid)
    assert not validity.check_post_move_validity(make_scene_coplanar((-2, 0, 0)), "cup")

    print("All test cases for coplanar constraint passed successfully.")


if __name__ == "__main__":
    test_coplanar()
