# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import inspect
from pathlib import Path

import bpy
import numpy as np
import procfunc as pf
import pytest

from infinigen2.exporters import imu, render_cycles
from infinigen2.exporters.util.format import ExportType, RenderPass


@pytest.mark.slow
def test_export_imu_tum_and_camera(tmp_path, save_blend=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)

    cube = pf.ops.primitives.mesh_cube(scale=(2, 2, 2))

    camera: pf.CameraObject = pf.ops.primitives.perspective_camera(
        focal_length_mm=50,
        sensor_width_mm=30,
        sensor_height_mm=30,
        clip_start=0.1,
        clip_end=1000,
    )
    camera.item().location = (0, -5, 3)
    camera.item().rotation_euler = np.deg2rad((20, 0, 0))

    camera.item().keyframe_insert("location", frame=1)
    camera.item().location.x += 1
    camera.item().keyframe_insert("location", frame=2)

    cube.item().keyframe_insert("location", frame=1)
    cube.item().location.z += 1
    cube.item().keyframe_insert("location", frame=2)

    camera_rp = RenderPass(
        ExportType.CAMERA,
        Path("Camera/%c/camera_%f.npz"),
        np.dtype(np.float32),
    )

    imu_results = imu.save_imu(
        camera=camera,
        objects=[cube],
        output_folder=tmp_path,
        frame_start=1,
        frame_end=2,
    )
    render_results = render_cycles.render_cycles(
        objects=[cube],
        camera=camera,
        output_folder=tmp_path,
        frame_start=1,
        frame_end=2,
        resolution=(256, 256),
        render_passes=[camera_rp],
    )
    results = {**imu_results, **render_results}

    assert ExportType.CAM_IMU_TUM_TRAJ in results
    assert ExportType.OBJ_IMU_TUM_TRAJ in results
    assert ExportType.CAMERA in results

    cam_imu_paths = results[ExportType.CAM_IMU_TUM_TRAJ]
    obj_imu_paths = results[ExportType.OBJ_IMU_TUM_TRAJ]
    cam_paths = results[ExportType.CAMERA]

    assert len(cam_imu_paths) >= 2
    assert len(obj_imu_paths) >= 2
    assert len(cam_paths) == 2

    for p in cam_imu_paths + obj_imu_paths + cam_paths:
        assert p.exists()

    with open([p for p in cam_imu_paths if p.name.endswith("_imu.txt")][0], "r") as f:
        cam_imu_lines = [ln for ln in f.read().strip().splitlines() if ln]
    with open([p for p in cam_imu_paths if p.name.endswith("_tum.txt")][0], "r") as f:
        cam_tum_lines = [ln for ln in f.read().strip().splitlines() if ln]
    with open([p for p in obj_imu_paths if p.name.endswith("_imu.txt")][0], "r") as f:
        obj_imu_lines = [ln for ln in f.read().strip().splitlines() if ln]
    with open([p for p in obj_imu_paths if p.name.endswith("_tum.txt")][0], "r") as f:
        obj_tum_lines = [ln for ln in f.read().strip().splitlines() if ln]

    assert len(cam_imu_lines) >= 3
    assert len(cam_tum_lines) >= 3
    assert len(obj_imu_lines) >= 3
    assert len(obj_tum_lines) >= 3

    bpy.context.scene.frame_set(1)
    cam_npz_1 = np.load(cam_paths[0])
    T_saved_1 = cam_npz_1["T"]
    HW = cam_npz_1["HW"]
    assert tuple(HW.tolist()) == (256, 256)

    T_expected_1 = np.asarray(camera.item().matrix_world, dtype=np.float64) @ np.diag(
        (1.0, -1.0, -1.0, 1.0)
    )
    assert np.allclose(T_saved_1, T_expected_1, atol=1e-6)

    bpy.context.scene.frame_set(2)
    cam_npz_2 = np.load(cam_paths[1])
    T_saved_2 = cam_npz_2["T"]
    T_expected_2 = np.asarray(camera.item().matrix_world, dtype=np.float64) @ np.diag(
        (1.0, -1.0, -1.0, 1.0)
    )
    assert np.allclose(T_saved_2, T_expected_2, atol=1e-6)

    if save_blend:
        blend_path = tmp_path / f"{inspect.currentframe().f_code.co_name}.blend"  # type: ignore
        pf.ops.file.save_blend(blend_path)


def _imu_tum_rows_at_fps(tmp_path, fps, fps_base):
    """Export a 2*fps-frame motion at the given frame rate and return (imu, tum) rows."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps, scene.render.fps_base = fps, fps_base

    obj = bpy.data.objects.new("probe", None)
    scene.collection.objects.link(obj)
    end = 2 * fps
    # x eases 0 -> 2 m (bezier), z-rotation turns 0 -> 2 rad (linear), over frames 0..end
    for frame, x, rz in [(0, 0.0, 0.0), (end, 2.0, 2.0)]:
        obj.location = (x, 0, 0)
        obj.rotation_euler = (0, 0, rz)
        obj.keyframe_insert("location", frame=frame)
        obj.keyframe_insert("rotation_euler", frame=frame)
    for fc in obj.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = (
                "LINEAR" if fc.data_path == "rotation_euler" else "BEZIER"
            )

    paths = imu.save_imu_tum_files(tmp_path, [obj], 0, end)

    def read_rows(suffix):
        path = next(p for p in paths if p.name.endswith(suffix))
        return np.loadtxt(path, comments="#")

    return read_rows("_imu.txt"), read_rows("_tum.txt")


@pytest.mark.parametrize("fps, fps_base", [(24, 1.0), (8, 1.0), (30, 1.001)])
def test_imu_tum_units_independent_of_fps(tmp_path, fps, fps_base):
    """IMU/TUM timestamps are seconds and IMU values are per-second at any frame rate."""
    imu_rows, tum_rows = _imu_tum_rows_at_fps(tmp_path, fps, fps_base)
    seconds_per_frame = fps_base / fps
    duration = 2 * fps * seconds_per_frame

    # timestamps are seconds, not frame numbers
    expected_t = np.arange(len(imu_rows)) * seconds_per_frame
    assert np.allclose(imu_rows[:, 0], expected_t)
    assert np.allclose(tum_rows[:, 0], expected_t)

    # quarter of the way through the motion (s = 0.25)
    row = fps // 2
    # 2 rad linear over the duration -> constant 2 / duration rad/s
    assert imu_rows[row, 3] == pytest.approx(2 / duration, abs=1e-4)
    # bezier ease x(s) = 2(3s^2 - 2s^3), s = t / duration -> x'' = 2(6 - 12s) / duration^2
    assert imu_rows[row, 4] == pytest.approx(6 / duration**2, abs=1e-3)
