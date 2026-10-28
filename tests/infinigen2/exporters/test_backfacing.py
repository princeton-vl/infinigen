# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import bpy
import numpy as np
import procfunc as pf
import pytest

from infinigen2.exporters.render_error_check.backfacing import check_backfacing


def _cube_and_camera(
    flipped: bool, object_scale: tuple[float, float, float]
) -> tuple[pf.MeshObject, pf.CameraObject]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    cube = pf.ops.primitives.mesh_cube(scale=(2, 2, 2))
    cube.item().scale = object_scale
    if flipped:
        cube.item().data.flip_normals()
    camera = pf.ops.primitives.perspective_camera(
        focal_length_mm=50, sensor_width_mm=30, sensor_height_mm=30
    )
    camera.item().location = (6, -6, 6)
    camera.item().rotation_euler = np.deg2rad((54, 0, 45))
    return cube, camera


@pytest.mark.parametrize("object_scale", [(1, 1, 1), (-1, 1, 1)])
@pytest.mark.parametrize("flipped", [False, True])
def test_check_backfacing_flags_only_inverted_closed_mesh(
    tmp_path: Path, flipped: bool, object_scale: tuple[float, float, float]
) -> None:
    _, camera = _cube_and_camera(flipped, object_scale)
    result = check_backfacing(
        camera=camera, output_folder=tmp_path, resolution=(48, 48)
    )

    report = json.loads((tmp_path / camera.item().name / "backfacing.json").read_text())
    frame = report["frames"][0]
    assert frame["hit_pixels"] > 0
    if flipped:
        assert frame["bad_pixels"] == frame["hit_pixels"]
        assert len(result[next(iter(result))]) == 1
    else:
        assert frame["bad_pixels"] == 0
        assert frame["image"] is None
