# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

from pathlib import Path

import bpy
import numpy as np
import procfunc as pf

from infinigen2.exporters import render_cycles
from infinigen2.exporters.util.format import ExportType, RenderPass


def test_camera_export_basic(tmp_path: Path):
    bpy.ops.wm.read_factory_settings(use_empty=True)

    cube = pf.ops.primitives.mesh_cube(scale=(2, 2, 2))
    camera: pf.CameraObject = pf.ops.primitives.perspective_camera(
        focal_length_mm=50,
        sensor_width_mm=30,
        sensor_height_mm=30,
        clip_start=0.1,
        clip_end=1000,
    )
    camera.item().location = (6, -6, 6)
    camera.item().rotation_euler = np.deg2rad((54, 0, 45))

    results = render_cycles.render_cycles(
        objects=[cube],
        camera=camera,
        output_folder=tmp_path,
        frame_start=1,
        frame_end=1,
        resolution=(256, 256),
        render_passes=[
            RenderPass(
                ExportType.CAMERA,
                Path("Camera/%c/camera_%f.npz"),
                np.dtype(np.float32),
            )
        ],
    )

    paths = results[ExportType.CAMERA]
    assert len(paths) == 1
    assert paths[0].exists()

    data = np.load(paths[0])
    K = data["K"]
    T = data["T"]
    HW = data["HW"]

    assert K.shape == (3, 3)
    assert T.shape == (4, 4)
    assert HW.shape == (2,)
    assert np.allclose(T[3], np.array([0, 0, 0, 1]))
    assert tuple(HW) == (256, 256)

    assert np.allclose(np.array(camera.item().location), T[:3, 3])
