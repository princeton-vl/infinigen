# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import json
from pathlib import Path

import bpy
import numpy as np
import procfunc as pf
import pytest
from mathutils import Vector

from infinigen2.exporters.object_data import collect_object_data
from infinigen2.exporters.util.blender_render import configure_object_index_table
from infinigen2.exporters.util.format import (
    NON_CAMERA_TYPES,
    NON_FRAME_TYPES,
    ExportType,
)
from infinigen2.exporters.visualize_gt_boxes import (
    assert_object_data_matches_table,
    euler_to_matrix,
    object_names,
)


def _cubes(n: int) -> list[pf.MeshObject]:
    pf.ops.object.clear_scene()
    cubes = []
    for i in range(n):
        cube = pf.ops.primitives.mesh_cube(scale=(2, 2, 2))
        pf.ops.object.set_transform(cube, location=(float(i), 0.0, 0.0))
        cubes.append(cube)
    configure_object_index_table()
    return cubes


def _load(objects, tmp_path: Path, frame_start: int = 0, frame_end: int = 0):
    data = collect_object_data(objects, frame_start, frame_end)
    path = tmp_path / "object-data.npz"
    np.savez(path, **data)
    return np.load(path, allow_pickle=False)


def _world_corners(obj: bpy.types.Object) -> np.ndarray:
    return np.array([obj.matrix_world @ Vector(c) for c in obj.bound_box])


def test_object_data_is_per_scene_not_per_camera_or_frame():
    assert ExportType.OBJECT_DATA in NON_FRAME_TYPES
    assert ExportType.OBJECT_DATA in NON_CAMERA_TYPES


def test_every_field_has_one_row_per_object(tmp_path: Path):
    cubes = _cubes(3)
    data = _load(cubes, tmp_path, frame_start=2, frame_end=5)

    for key in data.files:
        if key in ("frame_start", "frame_end"):
            continue
        assert data[key].shape[0] == 3, key

    for key in ("location_meters", "rotation_euler_rad", "scale"):
        assert data[key].shape == (3, 3, 4)
    assert data["local_bbox_min"].shape == (3, 3, 4)
    assert int(data["frame_start"]) == 2
    assert int(data["frame_end"]) == 5


def test_collect_object_data_returns_in_memory_arrays():
    cubes = _cubes(2)
    data = collect_object_data(cubes, frame_start=2, frame_end=3)

    assert data["location_meters"].shape == (2, 3, 2)
    assert int(data["frame_start"]) == 2
    assert int(data["frame_end"]) == 3


def test_metadata_fields_describe_each_object(tmp_path: Path):
    cubes = _cubes(2)
    data = _load(cubes, tmp_path)

    assert object_names(data) == [c.item().name for c in cubes]
    assert [t.decode() for t in data["object_type"]] == ["MESH", "MESH"]
    assert [n.decode() for n in data["data_name"]] == [
        c.item().data.name for c in cubes
    ]
    assert sorted(data["data_id"]) == [0, 1]


def test_shared_mesh_data_gets_one_data_id(tmp_path: Path):
    pf.ops.object.clear_scene()
    original = pf.ops.primitives.mesh_cube()
    linked = bpy.data.objects.new("Linked", original.item().data)
    bpy.context.collection.objects.link(linked)
    configure_object_index_table()

    data = _load([original.item(), linked], tmp_path)
    assert data["data_id"][0] == data["data_id"][1]


def test_object_index_is_read_from_blender_not_row_order(tmp_path: Path):
    cubes = _cubes(3)
    reversed_cubes = list(reversed(cubes))
    data = _load(reversed_cubes, tmp_path)

    assert list(data["object_index"]) == [c.item().pass_index for c in reversed_cubes]
    assert object_names(data) == [c.item().name for c in reversed_cubes]
    assert data["location_meters"][:, 0, 0] == pytest.approx([2.0, 1.0, 0.0])


def test_rows_map_onto_the_object_index_table(tmp_path: Path):
    cubes = _cubes(3)
    table = ["none"] + [o.name for o in bpy.data.objects]
    data = _load(list(reversed(cubes)), tmp_path)

    for index, name in zip(data["object_index"], object_names(data), strict=True):
        assert table[index] == name

    table_json = tmp_path / "object-index-table.json"
    table_json.write_text(json.dumps(table))
    assert_object_data_matches_table(tmp_path / "object-data.npz", table_json)


def test_mismatched_table_is_rejected(tmp_path: Path):
    cubes = _cubes(2)
    _load(cubes, tmp_path)

    table_json = tmp_path / "object-index-table.json"
    table_json.write_text(json.dumps(["none", "SomethingElse", "Cube.001"]))
    with pytest.raises(AssertionError):
        assert_object_data_matches_table(tmp_path / "object-data.npz", table_json)


def test_unassigned_object_index_is_rejected(tmp_path: Path):
    pf.ops.object.clear_scene()
    cube = pf.ops.primitives.mesh_cube()
    with pytest.raises(ValueError, match="pass_index 0"):
        collect_object_data([cube], 0, 0)


def test_clashing_object_index_is_rejected(tmp_path: Path):
    cubes = _cubes(2)
    cubes[1].item().pass_index = cubes[0].item().pass_index
    with pytest.raises(ValueError, match="unique"):
        collect_object_data(cubes, 0, 0)


def test_stale_object_index_is_rejected(tmp_path: Path):
    cubes = _cubes(3)
    bpy.data.objects.remove(cubes[0].item(), do_unlink=True)
    with pytest.raises(ValueError, match="no longer points back"):
        collect_object_data(cubes[1:], 0, 0)


def test_export_without_stamping_is_rejected(tmp_path: Path):
    pf.ops.object.clear_scene()
    cubes = [pf.ops.primitives.mesh_cube() for _ in range(3)]
    with pytest.raises(ValueError, match="pass_index 0"):
        collect_object_data(cubes, 0, 0)


def test_pose_and_bbox_reconstruct_the_world_box(tmp_path: Path):
    pf.ops.object.clear_scene()
    cube = pf.ops.primitives.mesh_cube(scale=(2, 2, 2))
    pf.ops.object.set_transform(
        cube,
        location=(1.5, -2.0, 0.25),
        rotation_euler=(0.3, -0.7, 1.1),
        scale=(1.0, 2.0, 0.5),
    )
    configure_object_index_table()
    data = _load([cube], tmp_path)

    rotation = euler_to_matrix(data["rotation_euler_rad"][:, :, 0])[0]
    axes = np.stack(
        [data["local_bbox_min"][0, :, 0], data["local_bbox_max"][0, :, 0]]
    ).T
    corners_local = np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, 3)
    scaled = corners_local * data["scale"][0, :, 0]
    corners_world = scaled @ rotation.T + data["location_meters"][0, :, 0]

    expected = _world_corners(cube.item())
    assert np.sort(corners_world, axis=0) == pytest.approx(
        np.sort(expected, axis=0), abs=1e-4
    )


def test_parented_object_pose_is_world_space(tmp_path: Path):
    pf.ops.object.clear_scene()
    parent = pf.ops.primitives.mesh_cube()
    child = pf.ops.primitives.mesh_cube()
    pf.ops.object.set_transform(parent, location=(3.0, 0.0, 0.0), scale=(2.0, 2.0, 2.0))
    pf.ops.object.set_transform(child, location=(1.0, 0.0, 0.0))
    child.item().parent = parent.item()
    bpy.context.view_layer.update()
    configure_object_index_table()

    data = _load([child], tmp_path)
    assert data["location_meters"][0, :, 0] == pytest.approx(
        list(child.item().matrix_world.translation), abs=1e-5
    )
    assert data["scale"][0, :, 0] == pytest.approx([2.0, 2.0, 2.0], abs=1e-5)


def test_euler_stays_continuous_across_frames(tmp_path: Path):
    pf.ops.object.clear_scene()
    cube = pf.ops.primitives.mesh_cube()
    obj = cube.item()
    for frame_number in range(4):
        obj.rotation_euler = (0.0, 0.0, 1.7 * frame_number)
        obj.keyframe_insert("rotation_euler", frame=frame_number)
    configure_object_index_table()

    data = _load([cube], tmp_path, frame_start=0, frame_end=3)
    assert np.diff(data["rotation_euler_rad"][0, 2, :]) == pytest.approx(
        [1.7, 1.7, 1.7], abs=1e-4
    )


def test_frame_is_restored_after_export(tmp_path: Path):
    cubes = _cubes(1)
    bpy.context.scene.frame_set(7)
    _load(cubes, tmp_path, frame_start=0, frame_end=3)
    assert bpy.context.scene.frame_current == 7


def test_collect_object_data_with_no_objects(tmp_path: Path):
    pf.ops.object.clear_scene()
    data = _load([], tmp_path, frame_start=0, frame_end=1)
    assert data["location_meters"].shape == (0, 3, 2)
    assert object_names(data) == []
