# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import inspect
import json
from pathlib import Path

import bpy
import numpy as np
import procfunc as pf
import pytest

from infinigen2.exporters import render_cycles, render_eevee
from infinigen2.exporters.util.format import ExportType, RenderPass


def _configure_two_cubes_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

    left_cube = pf.ops.primitives.mesh_cube(scale=(2, 2, 2))
    right_cube = pf.ops.primitives.mesh_cube(scale=(2, 2, 2))

    left_cube.item().location = (-3, 0, 0)
    right_cube.item().location = (3, 0, 0)

    mat_left = pf.Material(
        surface=pf.nodes.shader.emission(
            color=(1, 0, 0, 1),
            strength=1.0,
        ),
    )
    pf.ops.object.set_material(left_cube, surface=mat_left.surface)
    left_cube.item().material_slots[0].material.name = "Mat_Left"

    mat_right = pf.Material(
        surface=pf.nodes.shader.emission(
            color=(0, 1, 0, 1),
            strength=1.0,
        ),
    )
    pf.ops.object.set_material(right_cube, surface=mat_right.surface)
    right_cube.item().material_slots[0].material.name = "Mat_Right"

    camera: pf.CameraObject = pf.ops.primitives.perspective_camera(
        focal_length_mm=50,
        sensor_width_mm=30,
        sensor_height_mm=30,
        clip_start=0.1,
        clip_end=1000,
    )
    camera.item().location = (0, 0, 12)
    camera.item().rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.camera = camera.item()

    light = pf.ops.primitives.point_lamp(energy=2000)
    light.item().location = (0, 0, 12)

    return (
        [left_cube, right_cube, light],
        camera,
        left_cube,
        right_cube,
        mat_left,
        mat_right,
    )


def _majority_label(arr: np.ndarray) -> int:
    flat = arr.reshape(-1)
    counts = np.bincount(flat.astype(np.int64))
    return int(np.argmax(counts))


@pytest.mark.slow
@pytest.mark.render_gpu
@pytest.mark.parametrize("method", ["cycles", "eevee"])
def test_object_and_material_indices(tmp_path, method, save_blend=False):
    objects, camera, left_cube, right_cube, mat_left, mat_right = (
        _configure_two_cubes_scene()
    )

    render_passes = [
        RenderPass(ExportType.OBJECT_INDEX, Path("object_%f.npy"), np.dtype(np.uint32)),
        RenderPass(
            ExportType.MATERIAL_INDEX, Path("material_%f.npy"), np.dtype(np.uint32)
        ),
    ]

    if method == "cycles":
        results = render_cycles.render_cycles(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=1,
            resolution=(256, 128),
        )
        results.update(
            render_cycles.render_cycles_ground_truth(
                objects=objects,
                camera=camera,
                output_folder=tmp_path,
                render_passes=render_passes,
                frame_start=1,
                frame_end=1,
                resolution=(256, 128),
            )
        )
    elif method == "eevee":
        results = render_eevee.render_eevee(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=1,
            resolution=(256, 128),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    obj_mask = np.load(results[ExportType.OBJECT_INDEX][0])
    mat_mask = np.load(results[ExportType.MATERIAL_INDEX][0])

    h, w = obj_mask.shape
    left_region = obj_mask[:, : w // 2]
    right_region = obj_mask[:, w // 2 :]

    left_obj_label = _majority_label(left_region)
    right_obj_label = _majority_label(right_region)

    assert left_obj_label != 0 and right_obj_label != 0
    assert left_obj_label != right_obj_label
    assert (obj_mask == 0).any()

    objects_table = tmp_path / camera.item().name / "object-index-table.json"
    assert objects_table.exists()
    obj_names = json.loads(objects_table.read_text())
    assert obj_names[0] == "none"

    left_obj_name = obj_names[left_obj_label]
    right_obj_name = obj_names[right_obj_label]

    assert left_cube.item().name in obj_names and right_cube.item().name in obj_names
    assert (
        left_obj_name == left_cube.item().name
        or right_obj_name == right_cube.item().name
    )
    assert set([left_obj_name, right_obj_name]) == set(
        [left_cube.item().name, right_cube.item().name]
    )

    left_region_m = mat_mask[:, : w // 2]
    right_region_m = mat_mask[:, w // 2 :]
    left_mat_label = _majority_label(left_region_m)
    right_mat_label = _majority_label(right_region_m)

    assert left_mat_label != 0 and right_mat_label != 0
    assert left_mat_label != right_mat_label

    materials_table = tmp_path / camera.item().name / "material-index-table.json"
    assert materials_table.exists()
    mat_names = json.loads(materials_table.read_text())
    assert mat_names[0] == "none"

    left_mat_name = mat_names[left_mat_label]
    right_mat_name = mat_names[right_mat_label]

    left_mat_actual = left_cube.item().material_slots[0].material
    right_mat_actual = right_cube.item().material_slots[0].material
    assert left_mat_actual.name in mat_names and right_mat_actual.name in mat_names
    assert set([left_mat_name, right_mat_name]) == set(
        [left_mat_actual.name, right_mat_actual.name]
    )

    if save_blend:
        blend_path = tmp_path / f"{inspect.currentframe().f_code.co_name}.blend"  # type: ignore
        pf.ops.file.save_blend(blend_path)
