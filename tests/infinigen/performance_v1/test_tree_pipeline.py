# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

import pytest

# Selecting the optimized trees here would leave them selected for other test
# modules, so run with --optimized=trees,placement; otherwise this skips.
from infinigen._perf_selection import selected

if not selected("TREES"):
    pytest.skip(
        "the optimized trees are not selected (--optimized=trees)",
        allow_module_level=True,
    )


import bpy
import numpy as np
import pytest

from infinigen.assets.objects.trees.generate import BushFactory, GenericTreeFactory

_tree_ops = pytest.importorskip(
    "infinigen.perf.assets.objects.trees._tree_ops",
    reason="needs INFINIGEN_INSTALL_PERF=True",
)
from infinigen.assets.objects.trees.tree import TreeVertices
from infinigen.assets.objects.trees.utils import mesh as tree_mesh
from infinigen.core.placement import placement
from infinigen.core.placement.detail import min_max_edgelen
from infinigen.core.util import blender as butil
from infinigen.perf.assets.objects.trees import generate as perf_generate
from infinigen.perf.assets.objects.trees.tree import (
    _filter_update_attractors,
    _grow_vertices_exact,
)


def _optimized_placement_loaded():
    """True when the optimized placement is selected."""
    return selected("PLACEMENT")


def _mesh_arrays(mesh):
    """Topology, positions and the non-internal attribute layout of a mesh."""

    def get(collection, prop, width, dtype):
        out = np.empty(len(collection) * width, dtype=dtype)
        collection.foreach_get(prop, out)
        return out.tolist()

    return {
        "vertices": get(mesh.vertices, "co", 3, np.float32),
        "edges": get(mesh.edges, "vertices", 2, np.int32),
        "loops": get(mesh.loops, "vertex_index", 1, np.int32),
        "loop_start": get(mesh.polygons, "loop_start", 1, np.int32),
        "loop_total": get(mesh.polygons, "loop_total", 1, np.int32),
        "attributes": sorted(
            (a.name, a.domain, a.data_type)
            for a in mesh.attributes
            if not a.name.startswith(".")
        ),
    }


def test_constant_time_instance_count_matches_evaluated_iteration():
    source = bpy.data.collections.new("instance-count-source")
    source_objects = [
        bpy.data.objects.new(f"instance-count-source-{index}", None)
        for index in range(3)
    ]
    for obj in source_objects:
        source.objects.link(obj)
    instances = []
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        expected = sum(item.is_instance for item in depsgraph.object_instances)
        assert butil.count_instance() == expected

        for index in range(2):
            obj = bpy.data.objects.new(f"instance-count-{index}", None)
            obj.instance_type = "COLLECTION"
            obj.instance_collection = source
            bpy.context.scene.collection.objects.link(obj)
            instances.append(obj)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        expected = sum(item.is_instance for item in depsgraph.object_instances)
        assert expected == 6
        assert butil.count_instance() == expected

        source.hide_viewport = True
        depsgraph = bpy.context.evaluated_depsgraph_get()
        expected = sum(item.is_instance for item in depsgraph.object_instances)
        assert expected == 0
        assert butil.count_instance() == expected
    finally:
        for obj in instances:
            bpy.data.objects.remove(obj, do_unlink=True)
        for obj in source_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(source)


def test_tree_vertices_append_matches_numpy_semantics():
    tree = TreeVertices(np.array([[0, 0, 0]], dtype=np.int64))
    expected = np.array([[0, 0, 0]], dtype=np.int64)
    chunks = [
        (np.array([[0.25, 1.5, -2.0]]), [0], 2),
        (np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]), [1, 2], None),
    ]
    for vertices, parents, level in chunks:
        expected = np.append(expected, vertices, axis=0)
        tree.append(vertices, parents, level)
        assert tree.vtxs.dtype == expected.dtype
        assert np.array_equal(tree.vtxs, expected)

    assert tree.parent == [-1, 0, 1, 2]
    assert tree.level == [0, 2, 0, 0]


def test_tree_attractor_state_does_not_need_cumulative_deltas():
    attractors = np.array([[0.2, 0.0, 0.0], [2.0, 0.1, 0.0], [4.0, 0.0, 0.0]])
    batches = [
        np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        np.array([[1.9, 0.0, 0.0], [4.2, 0.0, 0.0]]),
    ]
    threshold = 0.15
    current_min = np.full(len(attractors), 10.0)
    current_match = np.full(len(attractors), -1, dtype=int)
    all_vertices = []

    for batch in batches:
        offset = sum(len(vertices) for vertices in all_vertices)
        attractors, current_min, current_match = _filter_update_attractors(
            attractors,
            batch,
            threshold,
            current_min,
            current_match,
            offset,
        )
        all_vertices.append(batch)

        vertices = np.concatenate(all_vertices)
        selected_deltas = attractors - vertices[current_match]
        selected_distances = np.linalg.norm(selected_deltas, axis=1)
        assert np.array_equal(selected_distances, current_min)

    assert np.array_equal(current_match, np.array([0, 3]))


def test_native_nearest_distances_are_bit_exact():
    rng = np.random.default_rng(8841)
    for attractor_count, vertex_count in ((1, 1), (7, 3), (2000, 17), (2000, 113)):
        attractors = np.ascontiguousarray(rng.normal(size=(attractor_count, 3)))
        vertices = np.ascontiguousarray(rng.normal(size=(vertex_count, 3)))
        expected = np.linalg.norm(attractors[:, None] - vertices[None], axis=-1)
        distances, closest = _tree_ops.nearest_distances(attractors, vertices)
        assert distances.tobytes() == expected.min(1).tobytes()
        assert closest.tobytes() == expected.argmin(1).tobytes()


def test_native_filter_update_attractors_is_bit_exact():
    rng = np.random.RandomState(9821)
    for _ in range(100):
        attractors = np.ascontiguousarray(rng.randn(rng.randint(0, 80), 3))
        vertices = np.ascontiguousarray(rng.randn(rng.randint(1, 35), 3))
        threshold = float(rng.uniform(0, 2))
        current_minimum = np.ascontiguousarray(rng.uniform(0, 5, len(attractors)))
        current_match = np.ascontiguousarray(
            rng.randint(-1, len(vertices), len(attractors)), dtype=np.int64
        )
        offset = int(rng.randint(0, 500))

        minimum, closest = _tree_ops.nearest_distances(attractors, vertices)
        keep = minimum > threshold
        expected_attractors = attractors[keep]
        kept_minimum = minimum[keep]
        kept_closest = closest[keep]
        expected_minimum = current_minimum[keep]
        expected_match = current_match[keep]
        update = kept_minimum < expected_minimum
        expected_minimum[update] = kept_minimum[update]
        expected_match[update] = kept_closest[update] + offset

        actual = _tree_ops.filter_update_attractors(
            attractors,
            vertices,
            threshold,
            current_minimum,
            current_match,
            offset,
        )
        assert actual[0].tobytes() == expected_attractors.tobytes()
        assert actual[1].tobytes() == expected_minimum.tobytes()
        assert actual[2].tobytes() == expected_match.tobytes()


def test_native_grouped_means_are_bit_exact():
    rng = np.random.RandomState(857019)
    for attractor_count in (0, 1, 7, 31, 1000):
        attractors = np.ascontiguousarray(rng.randn(attractor_count, 3))
        vertices = np.ascontiguousarray(rng.randn(17, 3))
        matches = rng.randint(-1, len(vertices), size=attractor_count).astype(np.int64)
        matched, means = _tree_ops.grouped_means(attractors, vertices, matches)
        expected_matched = np.unique(matches[matches >= 0])
        expected_means = np.asarray(
            [
                (attractors[matches == index] - vertices[index]).mean(0)
                for index in expected_matched
            ],
            dtype=np.float64,
        ).reshape(-1, 3)
        assert matched.tobytes() == expected_matched.tobytes()
        assert means.tobytes() == expected_means.tobytes()


def test_batched_growth_preserves_scalar_numpy_bytes():
    rng = np.random.RandomState(73191)
    for case in range(100):
        count = rng.randint(1, 40)
        vertices = rng.randn(rng.randint(count, count + 50), 3)
        matched = rng.randint(0, len(vertices), size=count)
        means = rng.randn(count, 3)
        random_values = rng.randn(count, 4)
        step_size = float(rng.uniform(0.001, 2))
        direction_randomness = float(rng.uniform(0, 1))
        magnitude_randomness = float(rng.uniform(0, 1))
        pull_direction = None if case % 2 else rng.randn(3)

        expected = np.empty_like(means)
        for index, (vertex_index, direction, random_value) in enumerate(
            zip(matched, means, random_values)
        ):
            direction = direction / np.linalg.norm(direction)
            if pull_direction is not None:
                direction += pull_direction
                direction = direction / np.linalg.norm(direction)
            direction += random_value[:3] * direction_randomness
            distance = step_size * np.exp(random_value[3] * magnitude_randomness)
            expected[index] = vertices[vertex_index] + distance * direction

        actual = _grow_vertices_exact(
            means,
            matched,
            vertices,
            random_values,
            step_size,
            pull_direction,
            direction_randomness,
            magnitude_randomness,
        )
        assert actual.tobytes() == expected.tobytes()


def test_tree_mesh_creation_does_not_change_active_object():
    sentinel = bpy.data.objects.new("tree-active-object-sentinel", None)
    bpy.context.scene.collection.objects.link(sentinel)
    bpy.context.view_layer.objects.active = sentinel
    obj = tree_mesh.init_mesh("tree-active-object-test", [(0, 0, 0)], [], [])
    try:
        assert bpy.context.active_object is sentinel
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.objects.remove(sentinel, do_unlink=True)


def test_tree_mesh_can_be_materialized_before_collection_linking():
    obj = tree_mesh.init_mesh(
        "tree-unlinked-mesh-test", [(0, 0, 0)], [], [], coll=False
    )
    try:
        assert tuple(obj.users_collection) == ()
        assert len(obj.data.vertices) == 1
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)


def test_edge_only_tree_mesh_matches_from_pydata_bytes():
    vertices = np.array(
        ((0.0, 0.0, 0.0), (0.25, 1.0, -2.0), (3.0, -4.0, 0.5)),
        dtype=np.float64,
    )
    edges = np.array(((0, 1), (1, 2)), dtype=np.int64)
    expected = bpy.data.meshes.new("tree-mesh-from-pydata-test")
    expected.from_pydata(vertices, edges, ())
    actual_obj = tree_mesh.init_mesh(
        "tree-mesh-direct-test", vertices, edges, (), coll=False
    )
    try:
        assert _mesh_arrays(actual_obj.data) == _mesh_arrays(expected)
    finally:
        bpy.data.meshes.remove(expected)
        bpy.data.objects.remove(actual_obj, do_unlink=True)


def test_unlink_uses_exact_object_collection_memberships():
    obj = bpy.data.objects.new("tree-unlink-test", None)
    collections = [
        bpy.data.collections.new(f"tree-unlink-test-{index}") for index in range(3)
    ]
    for collection in collections:
        bpy.context.scene.collection.children.link(collection)
        collection.objects.link(obj)
    bpy.context.scene.collection.objects.link(obj)
    try:
        butil.unlink(obj)
        assert tuple(obj.users_collection) == ()
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        for collection in collections:
            bpy.data.collections.remove(collection)


def test_group_in_collection_batches_parent_and_child_in_traversal_order():
    parent = bpy.data.objects.new("tree-group-parent", None)
    child = bpy.data.objects.new("tree-group-child", None)
    bpy.context.scene.collection.objects.link(parent)
    bpy.context.scene.collection.objects.link(child)
    child.parent = parent
    collection = butil.group_in_collection([parent], "tree-group-test", reuse=False)
    try:
        assert tuple(collection.objects) == (parent, child)
        assert tuple(parent.users_collection) == (collection,)
        assert tuple(child.users_collection) == (collection,)
    finally:
        bpy.data.objects.remove(parent, do_unlink=True)
        bpy.data.objects.remove(child, do_unlink=True)
        bpy.data.collections.remove(collection)


def test_bulk_tree_edge_lengths_match_legacy_selection():
    mesh = bpy.data.meshes.new("tree-edge-length-test")
    vertices = [
        (0.0, 0.0, 0.0),
        (0.25, 1.0, 0.0),
        (2.0, 1.5, -0.5),
        (3.25, -1.0, 0.75),
        (5.0, 0.5, 1.25),
        (8.0, 2.0, -1.0),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    mesh.from_pydata(vertices, edges, [])

    try:
        legacy_vertices = np.array([vertex.co for vertex in mesh.vertices])
        legacy_edges = np.array([edge.vertices for edge in mesh.edges])
        legacy_lengths = np.linalg.norm(
            legacy_vertices[legacy_edges[:, 0]] - legacy_vertices[legacy_edges[:, 1]],
            axis=-1,
        )
        legacy_lengths.sort()
        expected = (
            legacy_lengths[len(legacy_lengths) // 4],
            legacy_lengths[-len(legacy_lengths) // 4],
        )

        assert min_max_edgelen(mesh) == expected
    finally:
        bpy.data.meshes.remove(mesh)


def test_unlinked_placeholder_cube_matches_blender_primitive_bytes():
    legacy = butil.spawn_cube(size=4)
    direct = perf_generate._new_unlinked_placeholder_cube()
    try:
        for collection, field, width, dtype in (
            ("vertices", "co", 3, np.float32),
            ("vertices", "normal", 3, np.float32),
            ("edges", "vertices", 2, np.int32),
            ("loops", "vertex_index", 1, np.int32),
            ("loops", "edge_index", 1, np.int32),
            ("polygons", "loop_start", 1, np.int32),
            ("polygons", "loop_total", 1, np.int32),
            ("polygons", "normal", 3, np.float32),
        ):
            legacy_items = getattr(legacy.data, collection)
            direct_items = getattr(direct.data, collection)
            legacy_values = np.empty(len(legacy_items) * width, dtype=dtype)
            direct_values = np.empty(len(direct_items) * width, dtype=dtype)
            legacy_items.foreach_get(field, legacy_values)
            direct_items.foreach_get(field, direct_values)
            assert np.array_equal(legacy_values, direct_values), (collection, field)

        legacy_uv = np.empty(len(legacy.data.uv_layers[0].data) * 2, np.float32)
        direct_uv = np.empty(len(direct.data.uv_layers[0].data) * 2, np.float32)
        legacy.data.uv_layers[0].data.foreach_get("uv", legacy_uv)
        direct.data.uv_layers[0].data.foreach_get("uv", direct_uv)
        assert np.array_equal(legacy_uv, direct_uv)
        assert direct.name.startswith("Cube")
        assert direct.data.name.startswith("Cube")
        assert not direct.users_collection
    finally:
        bpy.data.objects.remove(legacy, do_unlink=True)
        bpy.data.objects.remove(direct, do_unlink=True)


def test_tree_placeholder_cube_copies_keep_independent_equivalent_meshes(
    monkeypatch,
):
    def fail_operator_cube(*args, **kwargs):
        raise AssertionError("tree placeholder cubes must not use Blender operators")

    monkeypatch.setattr(butil, "spawn_cube", fail_operator_cube)
    factory = object.__new__(GenericTreeFactory)
    factory._placeholder_cube_template = None
    factory._last_placeholder_cube = None
    factory.coarse_mesh_placeholder = False
    first = perf_generate._spawn_placeholder_cube(factory)
    second = perf_generate._spawn_placeholder_cube(factory)

    try:
        assert tuple(first.users_collection) == (bpy.context.collection,)
        assert tuple(second.users_collection) == (bpy.context.collection,)
        assert first.data is not second.data
        assert first.data.users == second.data.users == 1

        for collection, field, width, dtype in (
            ("vertices", "co", 3, np.float32),
            ("edges", "vertices", 2, np.int32),
            ("loops", "vertex_index", 1, np.int32),
            ("polygons", "loop_start", 1, np.int32),
            ("polygons", "loop_total", 1, np.int32),
        ):
            left_items = getattr(first.data, collection)
            right_items = getattr(second.data, collection)
            left = np.empty(len(left_items) * width, dtype=dtype)
            right = np.empty(len(right_items) * width, dtype=dtype)
            left_items.foreach_get(field, left)
            right_items.foreach_get(field, right)
            assert np.array_equal(left, right), (collection, field)

        factory.finalize_placeholders([first, second])
        assert bpy.context.active_object is second
        assert not first.select_get()
        assert second.select_get()
    finally:
        bpy.data.objects.remove(first, do_unlink=True)
        bpy.data.objects.remove(second, do_unlink=True)


def test_bush_deferred_skeleton_is_attached_before_fine_generation(monkeypatch):
    factory = object.__new__(BushFactory)
    factory.factory_seed = 17
    placeholder = butil.spawn_cube(size=4)
    placeholder[perf_generate._DEFERRED_SKELETON_INDEX] = 9
    skeleton = butil.spawn_cube(size=1)
    monkeypatch.setattr(perf_generate, "_create_skeleton", lambda factory: skeleton)

    try:
        perf_generate._materialize_deferred_skeleton(factory, placeholder)

        assert perf_generate._DEFERRED_SKELETON_INDEX not in placeholder
        assert list(placeholder.children) == [skeleton]
        assert skeleton.parent is placeholder

        # Materialization is idempotent once the private marker is removed.
        perf_generate._materialize_deferred_skeleton(factory, placeholder)
        assert list(placeholder.children) == [skeleton]
    finally:
        for obj in (placeholder, skeleton):
            if obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)


def test_tree_population_batches_visibility_points(monkeypatch):
    if not _optimized_placement_loaded():
        pytest.skip(
            "population batching lives in the optimized placement package, "
            "which was already imported as the default one"
        )
    placeholders = [
        bpy.data.objects.new(f"TreeFactory(7).spawn_placeholder({i})", None)
        for i in range(3)
    ]
    point_counts = {placeholders[0]: 1, placeholders[1]: 2, placeholders[2]: 1}
    calls = []
    # the optimized placement imports get_placeholder_points by name, so patch
    # it there too
    from infinigen.perf.core.placement import placement as perf_placement

    for module in (placement, perf_placement):
        monkeypatch.setattr(
            module,
            "get_placeholder_points",
            lambda placeholder: np.zeros((point_counts[placeholder], 3)),
        )

    def compute(points, cameras, dist_max, vis_margin, verbose):
        calls.append(len(points))
        return (
            np.array([True, False, True, False]),
            np.array([3.0, 1e7, 5.0, 1e7]),
            np.array([0.25, 1e7, 0.5, 1e7]),
        )

    monkeypatch.setattr(placement.split_in_view, "compute_inview_distances", compute)
    try:
        targets = placement.filter_populate_targets(
            placeholders, [object()], dist_cull=100, vis_cull=2, verbose=False
        )
        assert calls == [4]
        assert targets == [
            (placeholders[0], 3.0, 0.25),
            (placeholders[1], 5.0, 0.5),
        ]
    finally:
        for placeholder in placeholders:
            bpy.data.objects.remove(placeholder)
