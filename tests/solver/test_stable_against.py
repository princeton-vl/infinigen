# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan
    """Create a scene with a table and a cup, and return the state."""
    # butil.save_blend('test.blend')

def test_horizontal_stability():
    butil.clear_scene()
    objs = {}

    table = butil.spawn_cube(name='table') 
    table.dimensions = (4,10,2)

    chair1 = butil.spawn_cube(name='chair1')
    chair1.dimensions = (2,2,3)
    chair1.location = (3,3,0)

    chair2 = butil.spawn_cube(name='chair2')
    chair2.dimensions = (2,2,3)
    chair2.location = (3,-3,0)

    chair3 = butil.spawn_cube(name='chair3')
    chair3.dimensions = (2,2,3)
    chair3.location = (-3,3,0)

    chair4 = butil.spawn_cube(name='chair4')
    chair4.dimensions = (2,2,3)
    chair4.location = (-3,-3,0)
    for o in [table, chair1, chair2, chair3, chair4]:
        butil.apply_transform(o)
        parse_scene.preprocess_obj(o)
        tagging.tag_canonical_surfaces(o)
    with butil.SelectObjects([table, chair1, chair2, chair3, chair4]):
        # rotate
        bpy.ops.transform.rotate(value=np.pi/4, orient_axis='Z', orient_type='GLOBAL')
    # butil.save_blend('test.blend')
    bpy.context.view_layer.update()



    objs['table'] = state_def.ObjectState(table)
    objs['chair1'] = state_def.ObjectState(chair1)
    objs['chair2'] = state_def.ObjectState(chair2)
    objs['chair3'] = state_def.ObjectState(chair3)
    objs['chair4'] = state_def.ObjectState(chair4)
    objs['chair1'].relations.append(
        state_def.RelationState(
            target_name='table',
            child_plane_idx=0,
            parent_plane_idx=0
        )
    )
    objs['chair2'].relations.append(
        state_def.RelationState(
            target_name='table',
            child_plane_idx=0,
            parent_plane_idx=0
        )
    )
    objs['chair3'].relations.append(
        state_def.RelationState(
            target_name='table',
            child_plane_idx=0,
            parent_plane_idx=0
        )
    )
    objs['chair4'].relations.append(
        state_def.RelationState(
            target_name='table',
            child_plane_idx=0,
            parent_plane_idx=0
        )
    )
    state = state_def.State(objs=objs)
    assert validity.check_post_move_validity(state, 'chair1')
    assert validity.check_post_move_validity(state, 'chair2')
    assert validity.check_post_move_validity(state, 'chair3')
    assert validity.check_post_move_validity(state, 'chair4')

    # butil.save_blend('test.blend')

if __name__ == '__main__':
    test_horizontal_stability()