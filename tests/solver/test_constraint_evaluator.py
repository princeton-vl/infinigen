# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan
from itertools import chain
from functools import partial

from infinigen.core.constraints import (
    usage_lookup,
    example_solver as solver,
    constraint_language as cl
)
from infinigen.core.constraints.example_solver.state_def import state_from_dummy_scene, State, ObjectState
from infinigen.core.constraints.example_solver.propose_discrete import lookup_generator

from infinigen.assets.tables.dining_table import TableDiningFactory
from infinigen.assets.seating.chairs import ChairFactory

    butil.clear_scene()


    butil.clear_scene()
    butil.clear_scene()
    butil.clear_scene()
    constraints = []
    score_terms = []

    scene = cl.scene()
    problem = cl.Problem(constraints, score_terms)
    score_terms = []
    problem = cl.Problem(constraints, score_terms)


    score_terms = []
    problem = cl.Problem(constraints, score_terms)


def test_accessibility_monotonicity():
    butil.clear_scene()
    scores = []
        butil.clear_scene()
        obj_states = {}
        col = butil.get_collection("indoor_scene_test")
        chairs = butil.get_collection("chair")
        tables = butil.get_collection("table")    
        col.children.link(chairs)
        col.children.link(tables)

        chair = butil.spawn_cube(size=2, location=(0, 0, 0), name='chair1')
        butil.put_in_collection(chair, chairs)

        table = butil.spawn_cube(size=2, location=(2+dist, 0, 0), name='table1')
        butil.put_in_collection(table, tables)


        state = State(objs=obj_states)

        tagging.tag_canonical_surfaces(chair)
        tagging.tag_canonical_surfaces(table)

        constraints = []
        score_terms = []
        scene = cl.scene()

        problem = cl.Problem(constraints, score_terms)
        scores.append(res)
    print("nonaccessibility scores", scores)

def test_accessibility_side():
        butil.clear_scene()
        obj_states = {}
        col = butil.get_collection("indoor_scene_test")
        chairs = butil.get_collection("chair")
        tables = butil.get_collection("table")    
        col.children.link(chairs)
        col.children.link(tables)

        chair = butil.spawn_cube(size=2, location=(0, 0, 0), name='chair1')
        butil.put_in_collection(chair, chairs)

        table = butil.spawn_cube(size=2, location=(0, 2, 0), name='table1')
        butil.put_in_collection(table, tables)


        state = State(objs=obj_states)

        tagging.tag_canonical_surfaces(chair)
        tagging.tag_canonical_surfaces(table)

        constraints = []
        score_terms = []
        scene = cl.scene()

        score_terms += [cl.accessibility_cost(chair, table)]
        problem = cl.Problem(constraints, score_terms)
        print("nonaccessibility scores", res)
        assert np.isclose(res, 0)

def test_accessibility_angle():
    butil.clear_scene()
    scores = []
    for angle in [0, np.pi/4, np.pi/2, np.pi]:
        butil.clear_scene()
        obj_states = {}

        chair = butil.spawn_cube(size=2, location=(0, 0, 0), name='chair1')

        table = butil.spawn_sphere(radius = 1, location=(4*np.cos(angle), 4*np.sin(angle), 0), name='table1')
        print(table.location)


        state = State(objs=obj_states)

        tagging.tag_canonical_surfaces(chair)
        tagging.tag_canonical_surfaces(table)

        constraints = []
        score_terms = []
        scene = cl.scene()

        score_terms += [cl.accessibility_cost(chair, table)]
        problem = cl.Problem(constraints, score_terms)
        scores.append(res)
    print("nonaccessibility scores", scores)
    assert scores == sorted(scores, reverse=True)

def test_accessibility_volume():
    butil.clear_scene()
    scores = []
    for volume in [1, 2, 3, 4]:
        butil.clear_scene()
        obj_states = {}

        chair = butil.spawn_cube(size=2, location=(0, 0, 0), name='chair1')
        table = butil.spawn_sphere(radius=volume, location=(6, 0, 0), name='table1')


        state = State(objs=obj_states)

        tagging.tag_canonical_surfaces(chair)
        tagging.tag_canonical_surfaces(table)

        constraints = []
        score_terms = []
        scene = cl.scene()

        score_terms += [cl.accessibility_cost(chair, table)]
        problem = cl.Problem(constraints, score_terms)
        scores.append(res)
    print("nonaccessibility scores", scores)
    assert scores == sorted(scores)

# def test_accessibility_speed():
#     scores = []
#     butil.clear_scene()
#     obj_states = {}

#     chair = butil.spawn_cube(size=2, location=(0, 0, 0), name='chair1')
#     blocking_spheres = [butil.spawn_sphere(radius=1, location=(3+i, np.random.rand(), 0), name=f'sphere{i}') for i in range(100)]

#     for s in blocking_spheres:

#     state = State(objs=obj_states)

#     tagging.tag_canonical_surfaces(chair)
#     for s in blocking_spheres:
#         tagging.tag_canonical_surfaces(s)

#     constraints = []
#     score_terms = []
#     scene = cl.scene()

#     score_terms += [cl.accessibility_cost(chair, table)]
#     problem = cl.Problem(constraints, score_terms)
#     s = time()
#     print(time() - s)   
#     scores.append(res)
#     print("nonaccessibility scores", scores)
#     assert scores == sorted(scores)


def test_angle_alignment():
    butil.clear_scene()
    scores = []
    for angle in np.linspace(0, np.pi/2, 5):
        butil.clear_scene()
        obj_states = {}

        chair = butil.spawn_cube(size=1, location=(-3, 0, 0), name='chair1')
        table = butil.spawn_sphere(radius = 1, location=(0,0, 0), name='table1')
        # rotate chair by angle in z direction
        chair.rotation_euler = (0, 0, angle)


        state = State(objs=obj_states)

        tagging.tag_canonical_surfaces(chair)
        tagging.tag_canonical_surfaces(table)

        constraints = []
        score_terms = []
        scene = cl.scene()

        score_terms += [cl.angle_alignment_cost(chair, table)]
        problem = cl.Problem(constraints, score_terms)
        scores.append(res)
        # state.trimesh_scene.show()
    print("angle_alignment costs", scores)
    assert scores == sorted(scores)

def test_angle_alignment_multiple_objects():
    butil.clear_scene()
    scores = []
    
    for angle in np.linspace(0, np.pi/2, 5):
        butil.clear_scene()
        obj_states = {}
        
        chair = butil.spawn_cube(size=1, location=(-3, 0, 0), name='chair1')
        table1 = butil.spawn_sphere(radius=1, location=(0, 0, 0), name='table1')
        table2 = butil.spawn_sphere(radius=1, location=(3, 0, 0), name='table2')
        
        # Rotate chair by angle in z direction
        chair.rotation_euler = (0, 0, angle)
        
        
        state = State(objs=obj_states)
        
        tagging.tag_canonical_surfaces(chair)
        tagging.tag_canonical_surfaces(table1)
        tagging.tag_canonical_surfaces(table2)
        
        constraints = []
        score_terms = []
        scene = cl.scene()
        
        
        score_terms += [cl.angle_alignment_cost(chair, tables)]
        
        problem = cl.Problem(constraints, score_terms)
        scores.append(res)
    
    print("angle_alignment costs (multiple objects):", scores)
    assert scores == sorted(scores)

def test_angle_alignment_multiple_objects_varying_positions():
    butil.clear_scene()
    scores = []
    
    for i in range(5):
        butil.clear_scene()
        obj_states = {}
        
        chair = butil.spawn_cube(size=1, location=(-3, 0, 0), name='chair')
        
        table_positions = [
            (0, 0, 0),
            (3, 0, 0),
            (0, 3, 0),
            (-3, 2, 0),
            (3, 3, 0)
        ]
        
        tables = []
        for j, pos in enumerate(table_positions[:i+1], start=1):
            table = butil.spawn_sphere(radius=1, location=pos, name=f'table{j}')
            tables.append(table)
        
        
        state = State(objs=obj_states)
        
        tagging.tag_canonical_surfaces(chair)
        for table in tables:
            tagging.tag_canonical_surfaces(table)
        
        constraints = []
        score_terms = []
        scene = cl.scene()
        
        
        score_terms += [cl.angle_alignment_cost(chair_obj, table_objs)]
        
        problem = cl.Problem(constraints, score_terms)
        scores.append(res)
    
    print("angle_alignment costs (multiple objects, varying positions):", scores)
    assert scores == sorted(scores)

def test_angle_alignment_multipolygon_projection():
    butil.clear_scene()
    scores = []
    
    for i in range(5):
        butil.clear_scene()
        obj_states = {}
        
        chair = butil.spawn_cube(size=1, location=(-3, 0, 0), name='chair')
        
        # Create a complex object that may result in a multipolygon projection
        table_verts = [
            (-1, -1, 0),
            (1, -1, 0),
            (1, 1, 0),
            (-1, 1, 0),
            (0, 0, 1)
        ]
        table_faces = [
            (0, 1, 2, 3),
            (0, 1, 4),
            (1, 2, 4),
            (2, 3, 4),
            (3, 0, 4)
        ]
        
        table_mesh = bpy.data.meshes.new(name="TableMesh")
        table_obj = bpy.data.objects.new(name="Table", object_data=table_mesh)
        
        scene = bpy.context.scene
        scene.collection.objects.link(table_obj)
        
        table_mesh.from_pydata(table_verts, [], table_faces)
        table_mesh.update()
        
        table_obj.location = (0, 0, 0)
        
        # Rotate the table object based on the iteration
        chair.rotation_euler = (0, 0, i*np.pi/10)
        
        
        state = State(objs=obj_states)
        
        tagging.tag_canonical_surfaces(chair)
        tagging.tag_canonical_surfaces(table_obj)
        
        constraints = []
        score_terms = []
        scene = cl.scene()
        
        
        score_terms += [cl.angle_alignment_cost(chair_obj, table_objs)]
        
        problem = cl.Problem(constraints, score_terms)
        scores.append(res)
    
    print("angle_alignment costs (multipolygon projection):", scores)
    assert sorted(scores) == scores

def test_angle_alignment_tagged():
    butil.clear_scene()
    obj_states = {}

    chair = butil.spawn_cube(size=2, location=(5, 0, 0), name='chair1')
    table = butil.spawn_cube(size=2, location=(0,0, 0), name='table1')


    state = State(objs=obj_states)

    tagging.tag_canonical_surfaces(chair)
    tagging.tag_canonical_surfaces(table)

    table.rotation_euler[2] = np.pi/2

    constraints = []
    score_terms = []
    scene = cl.scene()

    problem = cl.Problem(constraints, score_terms)
    
    assert np.isclose(res, 0.5, atol=1e-3)

    butil.clear_scene()
    obj_states = {}

    chair = butil.spawn_cube(size=2, location=(5, 0, 0), name='chair1')
    table = butil.spawn_cube(size=2, location=(0,0, 0), name='table1')


    state = State(objs=obj_states)

    tagging.tag_canonical_surfaces(chair)
    tagging.tag_canonical_surfaces(table)

    constraints = []
    score_terms = []
    scene = cl.scene()

    problem = cl.Problem(constraints, score_terms)
    
    assert np.isclose(res, 0, atol=1e-3)


def test_focus_score():
    butil.clear_scene()
    scores = []
    for angle in np.linspace(0, np.pi/2, 5):
        butil.clear_scene()
        obj_states = {}

        chair = butil.spawn_cube(size=1, location=(-3, 0, 0), name='chair1')
        table = butil.spawn_sphere(radius = 1, location=(0,0, 0), name='table1')
        # rotate chair by angle in z direction
        chair.rotation_euler = (0, 0, angle)


        state = State(objs=obj_states)

        tagging.tag_canonical_surfaces(chair)
        tagging.tag_canonical_surfaces(table)

        constraints = []
        score_terms = []
        scene = cl.scene()

        score_terms += [cl.focus_score(chair, table)]
        problem = cl.Problem(constraints, score_terms)
        scores.append(res)
        # state.trimesh_scene.show()
    print("focus_score costs", scores)
    assert scores == sorted(scores)

    butil.clear_scene()

    butil.clear_scene()
    obj_states = {}

    chair = butil.spawn_cube(size=2, location=(0, 0, 2), name='chair1')
    table = butil.spawn_cube(size=10, location=(0,0, 0), name='table1')


    state = State(objs=obj_states)

    tagging.tag_canonical_surfaces(chair)
    tagging.tag_canonical_surfaces(table)

    constraints = []
    score_terms = []
    scene = cl.scene()

    
    
    problem = cl.Problem(constraints, score_terms)
    assert np.isclose(res, 4)
    constraints = []
    score_terms = []
    scene = cl.scene()

    # butil.save_blend('table_chair.blend')
    problem = cl.Problem(constraints, score_terms)
    assert np.isclose(res, 2)

    constraints = []
    score_terms = []
    scene = cl.scene()

    problem = cl.Problem(constraints, score_terms)
    assert np.isclose(res, 6)

def test_table():
    butil.clear_scene()

    used_as = home_asset_usage()
    usage_lookup.initialize_from_dict(used_as)

    butil.clear_scene()
    gen = TableDiningFactory(0)

    #     if fac in pholder_facs:
    #         obj = fac(0).spawn_placeholder(0, loc=(0,0,0), rot=(0,0,0))
    #     elif fac in asset_facs:
    #         obj = fac(0).spawn_asset(0, loc=(0,0,0), rot=(0,0,0))
    #     else: 
    #         raise ValueError()

    with butil.ViewportMode(obj, mode='EDIT'):
        butil.select(obj)
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')

    tagging.tag_canonical_surfaces(obj)
    # butil.save_blend('table.blend')


    #         if mask.sum() == 0:
    #             raise ValueError(
    #             )

def test_reflection_asymmetry():
    
    """
    create a bunch of chairs and a table. The chairs are reflected along the long part of the table. 
    check that the asymmetry score is 0
    """
    scores = []
    butil.clear_scene()
    obj_states = {}


    chairs = []
    for i in range(4):
        chairs.append(chair)
        # tagging.tag_canonical_surfaces(chair)
    chairs[0].location = (1, 1, 0)
    chairs[1].location = (1, -1, 0)
    chairs[2].location = (-1, 1, 0)
    chairs[3].location = (-1, -1, 0)

    chairs[0].rotation_euler = (0, 0, -np.pi/2)
    chairs[1].rotation_euler = (0, 0, np.pi/2)
    chairs[2].rotation_euler = (0, 0, -np.pi/2)
    chairs[3].rotation_euler = (0, 0, np.pi/2)

    bpy.context.view_layer.update()
    # chairs[0].rotation_euler = (0, 0, np.pi)

    # butil.save_blend('table_chairs.blend')

    state = State(objs=obj_states)

    # tagging.tag_canonical_surfaces(table)
    # for i in range(5):
    #     tagging.tag_canonical_surfaces(obj_states[f'chair{i}'].obj)

    constraints = []
    score_terms = []
    scene = cl.scene()

    problem = cl.Problem(constraints, score_terms)
    scores.append(res)
    print(res)
    assert np.isclose(res, 0, atol=1e-2)

    # assert the asymmetry increases as we gradually move one chair away from the table
    chairs[0].location = (1, 2, 0)
    bpy.context.view_layer.update()
    score_terms = []
    problem = cl.Problem(constraints, score_terms)
    scores.append(res)

    #assert the asymmetry increases if we rotate chair 0
    chairs[0].rotation_euler = (0, 0, 0)
    bpy.context.view_layer.update()
    score_terms = []
    problem = cl.Problem(constraints, score_terms)
    scores.append(res)

    print("asymmetry scores", scores)
    #assert monotonocity
    assert scores == sorted(scores)
    # assert it is strict
    assert (scores[0] < scores[1]) and scores[1] < scores[2]


def test_rotation_asymmetry():
    """
    create a bunch of chairs. The chairs are rotationally symmetric and then perturbed. 
    """
    scores = []
    butil.clear_scene()
    obj_states = {}

    chairs = []
    for i in range(6):
        chairs.append(chair)

    circle_locations_rotations = [((2*np.cos(i*np.pi/3), 2*np.sin(i*np.pi/3), 0),i*np.pi/3) for i in range(6)]
    np.random.shuffle(circle_locations_rotations)
    # put the chairs in a circle
    for i in range(6):
        chairs[i].location = circle_locations_rotations[i][0]
        chairs[i].rotation_euler = (0, 0, circle_locations_rotations[i][1])



    bpy.context.view_layer.update()

    state = State(objs=obj_states)


    constraints = []
    score_terms = []
    scene = cl.scene()

    problem = cl.Problem(constraints, score_terms)
    scores.append(res)
    assert np.isclose(res,0, atol=1e-2)

    # assert the asymmetry increases as we gradually move one chair from the circle
    chairs[0].location += Vector(np.random.rand(3))
    bpy.context.view_layer.update()
    score_terms = []
    problem = cl.Problem(constraints, score_terms)
    scores.append(res)

    #assert the asymmetry increases if we rotate chair 0
    bpy.context.view_layer.update()
    score_terms = []
    problem = cl.Problem(constraints, score_terms)
    scores.append(res)

    # do the same for another chair
    chairs[1].location += Vector(np.random.rand(3))
    bpy.context.view_layer.update()
    score_terms = []
    problem = cl.Problem(constraints, score_terms)
    scores.append(res)

    # assert monotonic
    # assert it is strict
    assert (scores[0] < scores[1]) and scores[1] < scores[2] and scores[2] < scores[3]
    print("asymmetry scores", scores)
    

def test_coplanarity():
    butil.clear_scene()
    obj_states = {}

    chair1 = butil.spawn_cube(size=2, location=(0, 0, 0), name='chair1')
    chair2 = butil.spawn_cube(size=2, location=(4, 0, 0), name='chair2')
    chair3 = butil.spawn_cube(size=2, location=(8, 0, 0), name='chair3')
    chair4 = butil.spawn_cube(size=2, location=(12, 0, 0), name='chair4')


    state = State(objs=obj_states)

    tagging.tag_canonical_surfaces(chair1)
    tagging.tag_canonical_surfaces(chair2)
    tagging.tag_canonical_surfaces(chair3)
    tagging.tag_canonical_surfaces(chair4)

    constraints = []
    score_terms = []
    scene = cl.scene()

    score_terms += [cl.coplanarity_cost(chairs)]
    problem = cl.Problem(constraints, score_terms)
    # print(res1)
    assert np.isclose(res1, 0, atol=1e-2)

    chair2.location = (4, 2, 0)
    bpy.context.view_layer.update()
    # butil.save_blend('test.blend')

    state = State(objs=obj_states)
    score_terms = []
    score_terms += [cl.coplanarity_cost(chairs)]
    problem = cl.Problem(constraints, score_terms)
    # print(res2)
    assert res2 > res1

    chair3.rotation_euler = (0, 0, np.pi/6)
    bpy.context.view_layer.update()
    state = State(objs=obj_states)
    score_terms = []
    score_terms += [cl.coplanarity_cost(chairs)]
    problem = cl.Problem(constraints, score_terms)
    assert res3 > res2

    chair2.dimensions = (2, 2, 4)
    bpy.context.view_layer.update()
    state = State(objs=obj_states)
    score_terms = []
    score_terms += [cl.coplanarity_cost(chairs)]
    problem = cl.Problem(constraints, score_terms)
    assert res4 > res3


    






if __name__ == '__main__':
    # test_min_dist()
    # test_reflection_asymmetry()
    # test_accessibility_speed()
    # test_coplanarity()
    test_angle_alignment_multipolygon_projection()
