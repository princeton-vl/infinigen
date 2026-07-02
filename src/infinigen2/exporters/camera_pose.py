# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lahav Lipson, Lingjie Mei: original Infinigen v1 camera/export code (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/core/placement/camera.py)
# - Alexander Raistrick: port to v2

from pathlib import Path

import bpy
import numpy as np
import procfunc as pf

from infinigen2.exporters.util.format import ExportType
from infinigen2.util.camera_projection import get_camera_parameters

__all__ = [
    "save_camera_poses",
]

CAMERA_PATH_TEMPLATE = Path("%c/camera.npz")


def save_camera_poses(
    camera: pf.CameraObject,
    output_folder: Path,
    frame_start: int,
    frame_end: int,
    path_template: Path = CAMERA_PATH_TEMPLATE,
) -> dict[ExportType, list[Path]]:
    stacked = "%f" not in str(path_template)
    cam_name = camera.item().name
    frames = list(range(frame_start, frame_end + 1))
    per_frame = []
    for frame_number in frames:
        bpy.context.scene.frame_set(frame_number)
        per_frame.append(get_camera_parameters(camera=camera))

    if stacked:
        stacked_params = {k: np.stack([d[k] for d in per_frame]) for k in per_frame[0]}
        result_path = Path(str(output_folder / path_template).replace("%c", cam_name))
        result_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(result_path, **stacked_params)
        return {ExportType.CAMERA: [result_path]}

    frame_paths = []
    for frame_number, params in zip(frames, per_frame, strict=False):
        result_path = Path(
            str(output_folder / path_template)
            .replace("%f", f"{frame_number:04d}")
            .replace("%c", cam_name)
        )
        result_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(result_path, **params)
        frame_paths.append(result_path)
    return {ExportType.CAMERA: frame_paths}
