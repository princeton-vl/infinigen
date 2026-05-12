# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

from functools import partial

# Authors: Karhan Kayan
import bpy
import numpy as np
import pytest
from mathutils import Vector

from infinigen.assets.objects.seating.chairs import ChairFactory
from infinigen.assets.objects.tables.dining_table import TableDiningFactory
from infinigen.assets.utils.bbox_from_mesh import bbox_mesh_from_hipoly
from infinigen.core import tagging
from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import usage_lookup
from infinigen.core.constraints.evaluator import evaluate
from infinigen.core.constraints.evaluator.node_impl import node_impls
from infinigen.core.constraints.example_solver.state_def import (
    ObjectState,
    State,
    state_from_dummy_scene,
)
from infinigen.core.util import blender as butil
from infinigen_examples.constraints.home import home_furniture_constraints
from infinigen_examples.constraints.semantics import home_asset_usage


def test_home_furniture_constraints_implemented():
    butil.clear_scene()

    cons = home_furniture_constraints()

    for node in cons.traverse():
        if node.__class__ in evaluate.SPECIAL_CASE_NODES:
            continue
        assert node.__class__ in node_impls


def make_chair_table():
    butil.clear_scene()
    col = butil.get_collection("indoor_scene_test")
    chairs = butil.get_collection("chair")
    tables = butil.get_collection("table")
    col.children.link(chairs)
    col.children.link(tables)

    chair = butil.spawn_cube(size=2, location=(0, 0, 0), name="chair1")
    butil.put_in_collection(chair, chairs)

    table = butil.spawn_cube(size=2, location=(3, 0, 0), name="table1")
    butil.put_in_collection(table, tables)

    return col


def test_parse_scene():
    butil.clear_scene()
    state = state_from_dummy_scene(make_chair_table())

    assert state.objs["chair1"].tags == {t.Semantics.Chair, t.SpecificObject("chair1")}
    assert state.objs["table1"].tags == {t.Semantics.Table, t.SpecificObject("table1")}


def test_eval_node():
    butil.clear_scene()

    state = state_from_dummy_scene(make_chair_table())
    eval = partial(evaluate.evaluate_node, state=state)

    scene = cl.scene()
    assert eval(scene) == {"chair1", "table1"}

    assert eval(scene.tagged({t.Semantics.Chair})) == {"chair1"}
    assert eval(scene.tagged({t.Semantics.Seating})) == set()
    assert eval(scene.tagged({t.Semantics.Chair}).count()) == 1


def test_min_dist():
    butil.clear_scene()

    col = make_chair_table()
    state = state_from_dummy_scene(col)

    constraints = []
    score_terms = []

    scene = cl.scene()
    chair = cl.tagged(scene, {t.Semantics.Chair})
    table = cl.tagged(scene, {t.Semantics.Table})
    sofa = cl.tagged(scene, {t.Semantics.Seating})

    score_terms += [cl.distance(chair, table)]
    problem = cl.Problem(constraints, score_terms)
    assert np.isclose(evaluate.evaluate_problem(problem, state).loss(), 1)

    score_terms = []
    score_terms += [cl.distance(table, chair) + 1]
    problem = cl.Problem(constraints, score_terms)
    assert np.isclose(evaluate.evaluate_problem(problem, state).loss(), 2)

    s = butil.spawn_cube(size=2, location=(-4, 0, 0), name="sofa1")
    sofas = butil.get_collection("seating")
    col.children.link(sofas)
    butil.put_in_collection(s, sofas)

    score_terms = []
    score_terms += [cl.distance(chair, sofa) + cl.distance(chair, table)]
    problem = cl.Problem(constraints, score_terms)
    state = state_from_dummy_scene(col)
    assert np.isclose(evaluate.evaluate_problem(problem, state).loss(), 3)

    butil.clear_scene()


def test_accessibility_monotonicity():
    butil.clear_scene()
    scores = []
    for dist in [1, 1.5, 2, 2.5]:
        butil.clear_scene()
        obj_states = {}
        col = butil.get_collection("indoor_scene_test")
        chairs = butil.get_collection("chair")
        tables = butil.get_collection("table")
        col.children.link(chairs)
        col.children.link(tables)

        chair = butil.spawn_cube(size=2, location=(0, 0, 0), name="chair1")
        chair.rotation_euler = (0, 0, 0.1)
        butil.put_in_collection(chair, chairs)

        table = butil.spawn_cube(size=2, location=(2 + dist, 0, 0), name="table1")
        butil.put_in_collection(table, tables)

        obj_states[chair.name] = ObjectState(chair, tags={t.Semantics.Chair})
        obj_states[table.name] = ObjectState(table, tags={t.Semantics.Table})

        state = State(objs=obj_states)

        tagging.tag_canonical_surfaces(chair)
        tagging.tag_canonical_surfaces(table)

        constraints = []
        score_terms = []
        scene = cl.scene()
        chair = cl.tagged(scene, {t.Semantics.Chair})
        table = cl.tagged(scene, {t.Semantics.Table})

        score_terms += [cl.accessibility_cost(chair, table, dist=3)]
        problem = cl.Problem(constraints, score_terms)
        res = evaluate.evaluate_problem(problem, state).loss()
        scores.append(res)

    print("nonaccessibility scores", scores)

    assert np.all(np.diff(scores) < 0)


def test_accessibility_side():
    butil.clear_scene()
    obj_states = {}
    col = butil.get_collection("indoor_scene_test")
    chairs = butil.get_collection("chair")
    tables = butil.get_collection("table")
    col.children.link(chairs)
    col.children.link(tables)

    chair = butil.spawn_cube(size=2, location=(0, 0, 0), name="chair1")
    butil.put_in_collection(chair, chairs)

    table = butil.spawn_cube(size=2, location=(0, 2, 0), name="table1")
    butil.put_in_collection(table, tables)

    obj_states[chair.name] = ObjectState(chair, tags={t.Semantics.Chair})
    obj_states[table.name] = ObjectState(table, tags={t.Semantics.Table})

    state = State(objs=obj_states)

    tagging.tag_canonical_surfaces(chair)
    tagging.tag_canonical_surfaces(table)

    constraints = []
    score_terms = []
    scene = cl.scene()
    chair = cl.tagged(scene, {t.Semantics.Chair})
    table = cl.tagged(scene, {t.Semantics.Table})

    score_terms += [cl.accessibility_cost(chair, table)]
    problem = cl.Problem(constraints, score_terms)
    res = evaluate.evaluate_problem(problem, state).loss()
    print("nonaccessibility scores", res)
    assert np.isclose(res, 0)


def test_accessibility_angle():
    butil.clear_scene()
    scores = []
    for angle in [0, np.pi / 4, np.pi / 2, np.pi]:
        butil.clear_scene()
        obj_states = {}

        chair = butil.spawn_cube(size=2, location=(0, 0, 0), name="chair1")

        table = butil.spawn_sphere(
            radius=1, location=(4 * np.cos(angle), 4 * np.sin(angle), 0), name="table1"
        )
        print(table.location)

        obj_states[chair.name] = ObjectState(chair, tags={t.Semantics.Chair})
        obj_states[table.name] = ObjectState(table, tags={t.Semantics.Table})

        state = State(objs=obj_states)

        tagging.tag_canonical_surfaces(chair)
        tagging.tag_canonical_surfaces(table)

        constraints = []
        score_terms = []
        scene = cl.scene()
        chair = cl.tagged(scene, {t.Semantics.Chair})
        table = cl.tagged(scene, {t.Semantics.Table})

        score_terms += [cl.accessibility_cost(chair, table)]
        problem = cl.Problem(constraints, score_terms)
        res = evaluate.evaluate_problem(problem, state).loss()
        scores.append(res)
    print("nonaccessibility scores", scores)
    assert scores == sorted(scores, reverse=True)


def test_accessibility_volume():
    butil.clear_scene()
    scores = []
    for volume in [1, 2, 3, 4]:
        butil.clear_scene()
        obj_states = {}

        chair = butil.spawn_cube(size=2, location=(0, 0, 0), name="chair1")
        table = butil.spawn_sphere(radius=volume, location=(6, 0, 0), name="table1")

        obj_states[chair.name] = ObjectState(chair, tags={t.Semantics.Chair})
        obj_states[table.name] = ObjectState(table, tags={t.Semantics.Table})

        state = State(objs=obj_states)

        tagging.tag_canonical_surfaces(chair)
        tagging.tag_canonical_surfaces(table)

        constraints = []
        score_terms = []
        scene = cl.scene()
        chair = cl.tagged(scene, {t.Semantics.Chair})
        table = cl.tagged(scene, {t.Semantics.Table})

        score_terms += [cl.accessibility_cost(chair, table)]
        problem = cl.Problem(constraints, score_terms)
        res = evaluate.evaluate_problem(problem, state).loss()
        scores.append(res)
    print("nonaccessibility scores", scores)
    assert scores == sorted(scores)


# def test_accessibility_speed():
#     scores = []
#     butil.clear_scene()
#     obj_states = {}

#     chair = butil.spawn_cube(size=2, location=(0, 0, 0), name='chair1')
#     blocking_spheres = [butil.spawn_sphere(radius=1, location=(3+i, np.random.rand(), 0), name=f'sphere{i}') for i in range(100)]

#     obj_states[chair.name] = ObjectState(chair, tags= {t.Semantics.Chair})
#     for s in blocking_spheres:
#         obj_states[s.name] = ObjectState(s, tags= {t.Semantics.Table})

#     state = State(objs=obj_states)

#     tagging.tag_canonical_surfaces(chair)
#     for s in blocking_spheres:
#         tagging.tag_canonical_surfaces(s)

#     constraints = []
#     score_terms = []
#     scene = cl.scene()
#     chair = cl.tagged(scene, {t.Semantics.Chair})
#     table = cl.tagged(scene, {t.Semantics.Table})

#     score_terms += [cl.accessibility_cost(chair, table)]
#     problem = cl.Problem(constraints, score_terms)
#     s = time()
#     res = evaluate.evaluate_problem(problem, state).score()
#     print(time() - s)
#     scores.append(res)
#     print("nonaccessibility scores", scores)
#     assert scores == sorted(scores)


def test_angle_alignment():
    butil.clear_scene()
    scores = []
    for angle in np.linspace(0, np.pi / 2, 5):
        butil.clear_scene()
        obj_states = {}

        chair = butil.spawn_cube(size=1, location=(-3, 0, 0), name="chair1")
        table = butil.spawn_sphere(radius=1, location=(0, 0, 0), name="table1")
        # rotate chair by angle in z direction
        chair.rotation_euler = (0, 0, angle)

        obj_states[chair.name] = ObjectState(chair, tags={t.Semantics.Chair})
        obj_states[table.name] = ObjectState(table, tags={t.Semantics.Table})

        state = State(objs=obj_states)

        tagging.tag_canonical_surfaces(chair)
        tagging.tag_canonical_surfaces(table)

        constraints = []
        score_terms = []
        scene = cl.scene()
        chair = cl.tagged(scene, {t.Semantics.Chair})
        table = cl.tagged(scene, {t.Semantics.Table})

        score_terms += [cl.angle_alignment_cost(chair, table)]
        problem = cl.Problem(constraints, score_terms)
        res = evaluate.evaluate_problem(problem, state).loss()
        scores.append(res)
        # state.trimesh_scene.show()
    print("angle_alignment costs", scores)
    assert scores == sorted(scores)


def test_angle_alignment_multiple_objects():
    butil.clear_scene()
    scores = []

    for angle in np.linspace(0, np.pi / 2, 5):
        butil.clear_scene()
        obj_states = {}

        chair = butil.spawn_cube(size=1, location=(-3, 0, 0), name="chair1")
        table1 = butil.spawn_sphere(radius=1, location=(0, 0, 0), name="table1")
        table2 = butil.spawn_sphere(radius=1, location=(3, 0, 0), name="table2")

        # Rotate chair by angle in z direction
        chair.rotation_euler = (0, 0, angle)

        obj_states[chair.name] = ObjectState(chair, tags={t.Semantics.Chair})
        obj_states[table1.name] = ObjectState(table1, tags={t.Semantics.Table})
        obj_states[table2.name] = ObjectState(table2, tags={t.Semantics.Table})

        state = State(objs=obj_states)

        tagging.tag_canonical_surfaces(chair)
        tagging.tag_canonical_surfaces(table1)
        tagging.tag_canonical_surfaces(table2)

        constraints = []
        score_terms = []
        scene = cl.scene()

        chair = cl.tagged(scene, {t.Semantics.Chair})
        tables = cl.tagged(scene, {t.Semantics.Table})

        score_terms += [cl.angle_alignment_cost(chair, tables)]

        problem = cl.Problem(constraints, score_terms)
        res = evaluate.evaluate_problem(problem, state).loss()
        scores.append(res)

    print("angle_alignment costs (multiple objects):", scores)
    assert scores == sorted(scores)


def test_angle_alignment_multiple_objects_varying_positions():
    butil.clear_scene()
    scores = []

    for i in range(5):
        butil.clear_scene()
        obj_states = {}

        chair = butil.spawn_cube(size=1, location=(-3, 0, 0), name="chair")

        table_positions = [(0, 0, 0), (3, 0, 0), (0, 3, 0), (-3, 2, 0), (3, 3, 0)]

        tables = []
        for j, pos in enumerate(table_positions[: i + 1], start=1):
            table = butil.spawn_sphere(radius=1, location=pos, name=f"table{j}")
            tables.append(table)
            obj_states[table.name] = ObjectState(table, tags={t.Semantics.Table})

        obj_states[chair.name] = ObjectState(chair, tags={t.Semantics.Chair})

        state = State(objs=obj_states)

        tagging.tag_canonical_surfaces(chair)
        for table in tables:
            tagging.tag_canonical_surfaces(table)

        constraints = []
        score_terms = []
        scene = cl.scene()

        chair_obj = cl.tagged(scene, {t.Semantics.Chair})
        table_objs = cl.tagged(scene, {t.Semantics.Table})

        score_terms += [cl.angle_alignment_cost(chair_obj, table_objs)]

        problem = cl.Problem(constraints, score_terms)
        res = evaluate.evaluate_problem(problem, state).loss()
        scores.append(res)

    print("angle_alignment costs (multiple objects, varying positions):", scores)
    assert scores == sorted(scores)


def test_angle_alignment_multipolygon_projection():
    butil.clear_scene()
    scores = []

    for i in range(5):
        butil.clear_scene()
        obj_states = {}

        chair = butil.spawn_cube(size=1, location=(-3, 0, 0), name="chair")

        # Create a complex object that may result in a multipolygon projection
        table_verts = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0), (0, 0, 1)]
        table_faces = [(0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]

        table_mesh = bpy.data.meshes.new(name="TableMesh")
        table_obj = bpy.data.objects.new(name="Table", object_data=table_mesh)

        scene = bpy.context.scene
        scene.collection.objects.link(table_obj)

        table_mesh.from_pydata(table_verts, [], table_faces)
        table_mesh.update()

        table_obj.location = (0, 0, 0)

        # Rotate the table object based on the iteration
        chair.rotation_euler = (0, 0, i * np.pi / 10)

        obj_states[chair.name] = ObjectState(chair, tags={t.Semantics.Chair})
        obj_states[table_obj.name] = ObjectState(table_obj, tags={t.Semantics.Table})

        state = State(objs=obj_states)

        tagging.tag_canonical_surfaces(chair)
        tagging.tag_canonical_surfaces(table_obj)

        constraints = []
        score_terms = []
        scene = cl.scene()

        chair_obj = cl.tagged(scene, {t.Semantics.Chair})
        table_objs = cl.tagged(scene, {t.Semantics.Table})

        score_terms += [cl.angle_alignment_cost(chair_obj, table_objs)]

        problem = cl.Problem(constraints, score_terms)
        res = evaluate.evaluate_problem(problem, state).loss()
        scores.append(res)

    print("angle_alignment costs (multipolygon projection):", scores)
    assert sorted(scores) == scores


def test_angle_alignment_tagged():
    butil.clear_scene()
    obj_states = {}

    chair = butil.spawn_cube(size=2, location=(5, 0, 0), name="chair1")
    table = butil.spawn_cube(size=2, location=(0, 0, 0), name="table1")

    obj_states[chair.name] = ObjectState(chair, tags={t.Semantics.Chair})
    obj_states[table.name] = ObjectState(table, tags={t.Semantics.Table})

    state = State(objs=obj_states)

    tagging.tag_canonical_surfaces(chair)
    tagging.tag_canonical_surfaces(table)

    table.rotation_euler[2] = np.pi / 2

    constraints = []
    score_terms = []
    scene = cl.scene()
    chair = cl.tagged(scene, {t.Semantics.Chair})
    table = cl.tagged(scene, {t.Semantics.Table})

    score_terms += [
        cl.angle_alignment_cost(chair, table, others_tags={t.Subpart.Front})
    ]
    problem = cl.Problem(constraints, score_terms)
    res = evaluate.evaluate_problem(problem, state).loss()

    assert np.isclose(res, 0.5, atol=1e-3)

    butil.clear_scene()
    obj_states = {}

    chair = butil.spawn_cube(size=2, location=(5, 0, 0), name="chair1")
    table = butil.spawn_cube(size=2, location=(0, 0, 0), name="table1")

    obj_states[chair.name] = ObjectState(chair, tags={t.Semantics.Chair})
    obj_states[table.name] = ObjectState(table, tags={t.Semantics.Table})

    state = State(objs=obj_states)

    tagging.tag_canonical_surfaces(chair)
    tagging.tag_canonical_surfaces(table)

    constraints = []
    score_terms = []
    scene = cl.scene()
    chair = cl.tagged(scene, {t.Semantics.Chair})
    table = cl.tagged(scene, {t.Semantics.Table})

    score_terms += [
        cl.angle_alignment_cost(chair, table, others_tags={t.Subpart.Front})
    ]
    problem = cl.Problem(constraints, score_terms)
    res = evaluate.evaluate_problem(problem, state).loss()

    assert np.isclose(res, 0, atol=1e-3)


def test_focus_score():
    butil.clear_scene()
    scores = []
    for angle in np.linspace(0, np.pi / 2, 5):
        butil.clear_scene()
        obj_states = {}

        chair = butil.spawn_cube(size=1, location=(-3, 0, 0), name="chair1")
        table = butil.spawn_sphere(radius=1, location=(0, 0, 0), name="table1")
        # rotate chair by angle in z direction
        chair.rotation_euler = (0, 0, angle)

        obj_states[chair.name] = ObjectState(chair, tags={t.Semantics.Chair})
        obj_states[table.name] = ObjectState(table, tags={t.Semantics.Table})

        state = State(objs=obj_states)

        tagging.tag_canonical_surfaces(chair)
        tagging.tag_canonical_surfaces(table)

        constraints = []
        score_terms = []
        scene = cl.scene()
        chair = cl.tagged(scene, {t.Semantics.Chair})
        table = cl.tagged(scene, {t.Semantics.Table})

        score_terms += [cl.focus_score(chair, table)]
        problem = cl.Problem(constraints, score_terms)
        res = evaluate.evaluate_problem(problem, state).loss()
        scores.append(res)
        # state.trimesh_scene.show()
    print("focus_score costs", scores)
    assert scores == sorted(scores)


@pytest.mark.skip
def test_viol_amounts():
    butil.clear_scene()

    def mk_state(n):
        butil.clear_scene()
        obj_states = {}

        for i in range(n):
            chair = butil.spawn_cube(size=1, location=(-3, 0, 0), name=f"chair{i}")
            obj_states[chair.name] = ObjectState(
                chair, tags={t.Semantics.Furniture, t.Semantics.Chair}
            )

        return State(objs=obj_states)

    cons = cl.Problem(
        [cl.scene().tagged(t.Semantics.Furniture).count().in_range(1, 3)], []
    )
    assert evaluate.evaluate_problem(cons, mk_state(0))[1] == 1
    assert evaluate.evaluate_problem(cons, mk_state(1))[1] == 0
    assert evaluate.evaluate_problem(cons, mk_state(3))[1] == 0
    assert evaluate.evaluate_problem(cons, mk_state(7))[1] == 4

    cons = cl.Problem([cl.scene().tagged(t.Semantics.Furniture).count() <= 3], [])
    assert evaluate.evaluate_problem(cons, mk_state(1))[1] == 0
    assert evaluate.evaluate_problem(cons, mk_state(3))[1] == 0
    assert evaluate.evaluate_problem(cons, mk_state(4))[1] == 1
    assert evaluate.evaluate_problem(cons, mk_state(6))[1] == 3

    cons = cl.Problem([cl.scene().tagged(t.Semantics.Furniture).count() >= 3], [])
    assert evaluate.evaluate_problem(cons, mk_state(0))[1] == 3
    assert evaluate.evaluate_problem(cons, mk_state(1))[1] == 2
    assert evaluate.evaluate_problem(cons, mk_state(3))[1] == 0
    assert evaluate.evaluate_problem(cons, mk_state(4))[1] == 0
    assert evaluate.evaluate_problem(cons, mk_state(6))[1] == 0

    cons = cl.Problem([cl.scene().tagged(t.Semantics.Furniture).count() == 3], [])
    assert evaluate.evaluate_problem(cons, mk_state(0))[1] == 3
    assert evaluate.evaluate_problem(cons, mk_state(1))[1] == 2
    assert evaluate.evaluate_problem(cons, mk_state(3))[1] == 0
    assert evaluate.evaluate_problem(cons, mk_state(4))[1] == 1
    assert evaluate.evaluate_problem(cons, mk_state(6))[1] == 3


def test_viol_integers():
    a = cl.constant(1)
    b = cl.constant(3)

    def violsingle(expr):
        return evaluate.evaluate_problem(cl.Problem([expr], []), State()).viol_count()

    assert violsingle(a < b) == 0
    assert violsingle(b > a) == 0
    assert violsingle(a <= b) == 0
    assert violsingle(b >= a) == 0

    assert violsingle(b <= b) == 0
    assert violsingle(b >= b) == 0

    assert violsingle(b < b) == 1
    assert violsingle(b > b) == 1

    assert violsingle(a == b) == 2
    assert violsingle(b <= a) == 2
    assert violsingle(a >= b) == 2

    assert violsingle(b < a) == 3
    assert violsingle(a > b) == 3

    assert violsingle(a >= (b * 2)) == 5


def test_min_dist_tagged():
    butil.clear_scene()
    obj_states = {}

    chair = butil.spawn_cube(size=2, location=(0, 0, 2), name="chair1")
    table = butil.spawn_cube(size=10, location=(0, 0, 0), name="table1")

    obj_states[chair.name] = ObjectState(chair, tags={t.Semantics.Chair})
    obj_states[table.name] = ObjectState(table, tags={t.Semantics.Table})

    state = State(objs=obj_states)

    tagging.tag_canonical_surfaces(chair)
    tagging.tag_canonical_surfaces(table)

    constraints = []
    score_terms = []
    scene = cl.scene()
    chair = cl.tagged(scene, {t.Semantics.Chair})
    table = cl.tagged(scene, {t.Semantics.Table})

    score_terms += [cl.distance(chair, table, others_tags={t.Subpart.Front})]
    problem = cl.Problem(constraints, score_terms)
    res = evaluate.evaluate_problem(problem, state).loss()
    assert np.isclose(res, 4)

    constraints = []
    score_terms = []
    scene = cl.scene()
    # chair = cl.tagged(scene, {t.Semantics.Chair})
    # table = cl.tagged(scene, {t.Semantics.Table})

    # butil.save_blend('table_chair.blend')
    score_terms += [cl.distance(chair, table, others_tags={t.Subpart.Top})]
    problem = cl.Problem(constraints, score_terms)
    res = evaluate.evaluate_problem(problem, state).loss()
    assert np.isclose(res, 2)

    constraints = []
    score_terms = []
    scene = cl.scene()
    # chair = cl.tagged(scene, {t.Semantics.Chair})
    # table = cl.tagged(scene, {t.Semantics.Table})

    score_terms += [cl.distance(chair, table, others_tags={t.Subpart.Bottom})]
    problem = cl.Problem(constraints, score_terms)
    res = evaluate.evaluate_problem(problem, state).loss()
    assert np.isclose(res, 6)


def test_table():
    butil.clear_scene()

    used_as = home_asset_usage()
    usage_lookup.initialize_from_dict(used_as)

    butil.clear_scene()
    gen = TableDiningFactory(0)
    obj = bbox_mesh_from_hipoly(gen, 0, use_pholder=False)

    #     if fac in pholder_facs:
    #         obj = fac(0).spawn_placeholder(0, loc=(0,0,0), rot=(0,0,0))
    #     elif fac in asset_facs:
    #         obj = fac(0).spawn_asset(0, loc=(0,0,0), rot=(0,0,0))
    #     else:
    #         raise ValueError()

    with butil.ViewportMode(obj, mode="EDIT"):
        butil.select(obj)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")

    tagging.tag_canonical_surfaces(obj)
    tagging.extract_tagged_faces(obj, {t.Subpart.Front})
    tagging.extract_tagged_faces(obj, {t.Subpart.Top})
    tagging.extract_tagged_faces(obj, {t.Subpart.Bottom})
    tagging.extract_tagged_faces(obj, {t.Subpart.Back})
    # butil.save_blend('table.blend')

    #     obj_tags = tagging.union_object_tags(obj)

    #     for tag in [t.Semantics.Back, t.Semantics.Bottom, t.Semantics.SupportSurface]:
    #         mask = tagging.tagged_face_mask(obj, {tag})
    #         if mask.sum() == 0:
    #             obj_tags = tagging.union_object_tags(obj)
    #             raise ValueError(
    #                 f'{obj.name=} has nothing tagged for {tag=}. {obj_tags=}'
    #             )


def test_reflection_asymmetry():
    """
    create a bunch of chairs and a table. The chairs are reflected along the long part of the table.
    check that the asymmetry score is 0
    """
    scores = []
    butil.clear_scene()
    obj_states = {}

    table = bbox_mesh_from_hipoly(TableDiningFactory(0), 0, use_pholder=False)
    obj_states[table.name] = ObjectState(table, tags={t.Semantics.Table})

    chairs = []
    for i in range(4):
        chair = bbox_mesh_from_hipoly(ChairFactory(0), i, use_pholder=False)
        obj_states[chair.name] = ObjectState(chair, tags={t.Semantics.Chair})
        chairs.append(chair)
        # tagging.tag_canonical_surfaces(chair)
        # tagging.extract_tagged_faces(chair, {t.Semantics.Front})
    chairs[0].location = (1, 1, 0)
    chairs[1].location = (1, -1, 0)
    chairs[2].location = (-1, 1, 0)
    chairs[3].location = (-1, -1, 0)

    chairs[0].rotation_euler = (0, 0, -np.pi / 2)
    chairs[1].rotation_euler = (0, 0, np.pi / 2)
    chairs[2].rotation_euler = (0, 0, -np.pi / 2)
    chairs[3].rotation_euler = (0, 0, np.pi / 2)

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
    table_tagged = cl.tagged(scene, {t.Semantics.Table})
    chairs_tagged = cl.tagged(scene, {t.Semantics.Chair})

    score_terms += [cl.reflectional_asymmetry(chairs_tagged, table_tagged)]
    problem = cl.Problem(constraints, score_terms)
    res = evaluate.evaluate_problem(problem, state).loss()
    scores.append(res)
    print(res)
    assert np.isclose(res, 0, atol=1e-2)

    # assert the asymmetry increases as we gradually move one chair away from the table
    chairs[0].location = (1, 2, 0)
    bpy.context.view_layer.update()
    score_terms = []
    score_terms += [cl.reflectional_asymmetry(chairs_tagged, table_tagged)]
    problem = cl.Problem(constraints, score_terms)
    res = evaluate.evaluate_problem(problem, state).loss()
    scores.append(res)

    # assert the asymmetry increases if we rotate chair 0
    chairs[0].rotation_euler = (0, 0, 0)
    bpy.context.view_layer.update()
    score_terms = []
    score_terms += [cl.reflectional_asymmetry(chairs_tagged, table_tagged)]
    problem = cl.Problem(constraints, score_terms)
    res = evaluate.evaluate_problem(problem, state).loss()
    scores.append(res)

    print("asymmetry scores", scores)
    # assert monotonocity
    assert scores == sorted(scores)
    # assert it is strict
    assert (scores[0] < scores[1]) and scores[1] < scores[2]


@pytest.mark.skip
def test_rotation_asymmetry():
    """
    create a bunch of chairs. The chairs are rotationally symmetric and then perturbed.
    """
    scores = []
    butil.clear_scene()
    obj_states = {}

    chairs = []
    for i in range(6):
        chair = bbox_mesh_from_hipoly(ChairFactory(0), i, use_pholder=False)
        obj_states[chair.name] = ObjectState(chair, tags={t.Semantics.Chair})
        chairs.append(chair)

    circle_locations_rotations = [
        ((2 * np.cos(i * np.pi / 3), 2 * np.sin(i * np.pi / 3), 0), i * np.pi / 3)
        for i in range(6)
    ]
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
    chairs_tagged = cl.tagged(scene, {t.Semantics.Chair})

    score_terms += [cl.rotational_asymmetry(chairs_tagged)]
    problem = cl.Problem(constraints, score_terms)
    res = evaluate.evaluate_problem(problem, state).loss()
    scores.append(res)
    assert np.isclose(res, 0, atol=1e-2)

    # assert the asymmetry increases as we gradually move one chair from the circle
    chairs[0].location += Vector(np.random.rand(3))
    bpy.context.view_layer.update()
    score_terms = []
    score_terms += [cl.rotational_asymmetry(chairs_tagged)]
    problem = cl.Problem(constraints, score_terms)
    res = evaluate.evaluate_problem(problem, state).loss()
    scores.append(res)

    # assert the asymmetry increases if we rotate chair 0
    chairs[0].rotation_euler = (0, 0, np.random.rand(1))
    bpy.context.view_layer.update()
    score_terms = []
    score_terms += [cl.rotational_asymmetry(chairs_tagged)]
    problem = cl.Problem(constraints, score_terms)
    res = evaluate.evaluate_problem(problem, state).loss()
    scores.append(res)

    # do the same for another chair
    chairs[1].location += Vector(np.random.rand(3))
    bpy.context.view_layer.update()
    score_terms = []
    score_terms += [cl.rotational_asymmetry(chairs_tagged)]
    problem = cl.Problem(constraints, score_terms)
    res = evaluate.evaluate_problem(problem, state).loss()
    scores.append(res)

    # assert monotonic
    # assert scores == sorted(scores) # warning: heisenbug. disabled by araistrick

    # assert it is strict
    assert (scores[0] < scores[1]) and scores[1] < scores[2] and scores[2] < scores[3]
    print("asymmetry scores", scores)


def test_coplanarity():
    butil.clear_scene()
    obj_states = {}

    chair1 = butil.spawn_cube(size=2, location=(0, 0, 0), name="chair1")
    chair2 = butil.spawn_cube(size=2, location=(4, 0, 0), name="chair2")
    chair3 = butil.spawn_cube(size=2, location=(8, 0, 0), name="chair3")
    chair4 = butil.spawn_cube(size=2, location=(12, 0, 0), name="chair4")

    obj_states[chair1.name] = ObjectState(chair1, tags={t.Semantics.Chair})
    obj_states[chair2.name] = ObjectState(chair2, tags={t.Semantics.Chair})
    obj_states[chair3.name] = ObjectState(chair3, tags={t.Semantics.Chair})
    obj_states[chair4.name] = ObjectState(chair4, tags={t.Semantics.Chair})

    state = State(objs=obj_states)

    tagging.tag_canonical_surfaces(chair1)
    tagging.tag_canonical_surfaces(chair2)
    tagging.tag_canonical_surfaces(chair3)
    tagging.tag_canonical_surfaces(chair4)

    constraints = []
    score_terms = []
    scene = cl.scene()
    chairs = cl.tagged(scene, {t.Semantics.Chair})

    score_terms += [cl.coplanarity_cost(chairs)]
    problem = cl.Problem(constraints, score_terms)
    res1 = evaluate.evaluate_problem(problem, state).loss()
    # print(res1)
    assert np.isclose(res1, 0, atol=1e-2)

    chair2.location = (4, 2, 0)
    bpy.context.view_layer.update()
    # butil.save_blend('test.blend')

    state = State(objs=obj_states)
    score_terms = []
    score_terms += [cl.coplanarity_cost(chairs)]
    problem = cl.Problem(constraints, score_terms)
    res2 = evaluate.evaluate_problem(problem, state).loss()
    # print(res2)
    assert res2 > res1

    chair3.rotation_euler = (0, 0, np.pi / 6)
    bpy.context.view_layer.update()
    state = State(objs=obj_states)
    score_terms = []
    score_terms += [cl.coplanarity_cost(chairs)]
    problem = cl.Problem(constraints, score_terms)
    res3 = evaluate.evaluate_problem(problem, state).loss()
    assert res3 > res2

    chair2.dimensions = (2, 2, 4)
    bpy.context.view_layer.update()
    state = State(objs=obj_states)
    score_terms = []
    score_terms += [cl.coplanarity_cost(chairs)]
    problem = cl.Problem(constraints, score_terms)
    res4 = evaluate.evaluate_problem(problem, state).loss()
    assert res4 > res3


def test_evaluate_problem_scalar_ops():
    state = State(objs={})

    one = cl.constant(1)
    two = cl.constant(2)
    three = cl.constant(3)

    def e(x):
        return evaluate.evaluate_problem(cl.Problem({}, {repr(x): x}), state).loss()

    assert e(two) == 2
    assert e(one + two) == 3
    assert e(one - two) == -1
    assert e(two * three) == 6
    assert e(two / three) == 2 / 3
    assert e(two**three) == 8

    assert e(two == two) == 1
    assert e(two == one) == 0
    assert e(two >= two) == 1
    assert e(two >= three) == 0
    assert e(two > one) == 1
    assert e(two > two) == 0
    assert e(two <= two) == 1
    assert e(two <= one) == 0
    assert e(two < three) == 1
    assert e(two < two) == 0
    assert e(two != one) == 1
    assert e(two != two) == 0

    assert e(cl.max_expr(one, two)) == 2
    assert e(cl.min_expr(one, two)) == 1

    assert e(one.clamp_min(two)) == 2
    assert e(two.clamp_max(one)) == 1

    assert e(-one) == -1
    assert e((-one).abs()) == 1


def test_evaluate_hinge():
    state = State(objs={})

    def e(x):
        return evaluate.evaluate_problem(cl.Problem({}, {repr(x): x}), state).loss()

    one = cl.constant(1)
    two = cl.constant(2)

    assert e(cl.hinge(one, 0, 2)) == 0
    assert e(cl.hinge(one, 1, 2)) == 0
    assert e(cl.hinge(one, 0, 1)) == 0

    assert e(cl.hinge(one, 2, 3)) == 1
    assert e(cl.hinge(two, 0, 1.5)) == 0.5


if __name__ == "__main__":
    # test_min_dist()
    # test_min_dist_tagged()
    # test_reflection_asymmetry()
    # test_accessibility_speed()
    # test_coplanarity()
    test_angle_alignment_multipolygon_projection()
