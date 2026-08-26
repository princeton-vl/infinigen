# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lahav Lipson: original Infinigen v1 3D box visualization (https://github.com/princeton-vl/infinigen/blob/main/infinigen/tools/ground_truth/bounding_boxes_3d.py)
# - Alexander Raistrick: port to v2

import itertools
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from imageio.v3 import imread, imwrite
from mathutils import Euler

from infinigen2.exporters.visualize_gt_passes import index_colors
from infinigen2.util.camera_projection import project_points_from_parameters

__all__ = [
    "assert_object_data_matches_table",
    "euler_to_matrix",
    "object_names",
    "overlay_3d_boxes",
    "visualize_object_boxes",
]

logger = logging.getLogger(__name__)

# corner order matches itertools.product([0, 1], repeat=3): idx = 4*bx + 2*by + bz
_BOX_CORNER_SIGNS = np.array(list(itertools.product([0, 1], repeat=3)))
_BOX_EDGES = [
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
]  # fmt: skip

# below this camera-space depth the projection diverges and draws garbage across the frame
NEAR_PLANE = 1e-3

NpzData = dict[str, np.ndarray | np.integer] | np.lib.npyio.NpzFile


def object_names(objdata: np.lib.npyio.NpzFile) -> list[str]:
    return [n.decode() for n in objdata["object_name"]]


def assert_object_data_matches_table(object_data_npz: Path, table_json: Path) -> None:
    """Fail if a row's object_index no longer points at that row's object in the
    object-index table, which happens when objects are created or deleted between
    writing the two."""
    objdata = np.load(object_data_npz, allow_pickle=False)
    table = json.loads(Path(table_json).read_text())

    wrong = []
    for index, name in zip(objdata["object_index"], object_names(objdata), strict=True):
        if index < len(table) and table[index] == name:
            continue
        wrong.append(f"{name!r} claims index {index}")

    if wrong:
        raise AssertionError(
            f"{object_data_npz} rows do not match {table_json} "
            f"({len(table)} entries): {wrong[:5]}"
        )


def euler_to_matrix(euler_rad: np.ndarray) -> np.ndarray:
    """Rotation matrices (N,3,3) for XYZ eulers (N,3), matching Blender's Euler.to_matrix."""
    return np.array([Euler(euler, "XYZ").to_matrix() for euler in euler_rad])


def _draw_box_edges(
    img: np.ndarray, uv: np.ndarray, depth: np.ndarray, color: tuple, thickness: int
) -> None:
    for i, j in _BOX_EDGES:
        if depth[i] < NEAR_PLANE or depth[j] < NEAR_PLANE:
            continue
        if not np.isfinite(uv[[i, j]]).all():
            continue
        pt_i = tuple(np.round(uv[i]).astype(int))
        pt_j = tuple(np.round(uv[j]).astype(int))
        cv2.line(img, pt_i, pt_j, color, thickness, cv2.LINE_AA)


def _frame_index(objdata: NpzData, frame_number: int) -> int:
    frame_start = int(objdata["frame_start"])
    frame_end = int(objdata["frame_end"])
    if not (frame_start <= frame_number <= frame_end):
        raise ValueError(
            f"{frame_number=} outside object-data range [{frame_start}, {frame_end}]"
        )
    return frame_number - frame_start


def overlay_3d_boxes(
    rgb_path: Path,
    objdata: NpzData,
    cams: NpzData,
    frame_number: int,
    output_path: Path,
    thickness: int = 1,
) -> Path:
    """Project every object's 3D bbox into the camera and draw its 12 edges over rgb_path.

    Reads the location/rotation/scale pose and local bbox from objdata and K/T from cams.
    Each box is coloured by its object_index using the same palette as
    the object-index segmentation visualization, so the two line up by eye. Rows with no
    pose on this frame are NaN and are skipped.
    """
    img = np.ascontiguousarray(imread(rgb_path)[..., :3])
    idx = _frame_index(objdata, frame_number)

    K = cams["K"][idx] if cams["K"].ndim == 3 else cams["K"]
    cam_to_world = cams["T"][idx] if cams["T"].ndim == 3 else cams["T"]

    location = objdata["location_meters"][:, :, idx]
    rotation = euler_to_matrix(np.nan_to_num(objdata["rotation_euler_rad"][:, :, idx]))
    scale = objdata["scale"][:, :, idx]
    bbox_min = objdata["local_bbox_min"][:, :, idx]
    bbox_max = objdata["local_bbox_max"][:, :, idx]
    palette = index_colors(np.asarray(objdata["object_index"]).reshape(-1, 1))
    colors = [tuple(int(c) for c in row) for row in palette]

    boxes = zip(location, rotation, scale, bbox_min, bbox_max, colors, strict=True)
    for loc, rot, obj_scale, bmin, bmax, box_color in boxes:
        if np.isnan([loc, obj_scale, bmin, bmax]).any():
            continue
        corners = (bmin + _BOX_CORNER_SIGNS * (bmax - bmin)) * obj_scale
        corners_world = corners @ rot.T + loc
        projected = project_points_from_parameters(corners_world, K, cam_to_world)
        _draw_box_edges(img, projected[:, :2], projected[:, 2], box_color, thickness)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(output_path, img)
    return output_path


def visualize_object_boxes(
    rgb_frames: list[Path],
    object_data_npz: Path,
    camera_npz: Path,
    output_folder: Path,
    table_json: Path | None = None,
) -> list[Path]:
    """3D-bbox-over-rgb overlay for every frame in rgb_frames."""
    if table_json is not None:
        assert_object_data_matches_table(object_data_npz, table_json)

    output_folder.mkdir(parents=True, exist_ok=True)
    out_paths = []
    with (
        np.load(object_data_npz, allow_pickle=False) as objdata,
        np.load(camera_npz, allow_pickle=False) as cams,
    ):
        for frame_path in rgb_frames:
            out_path = output_folder / f"{frame_path.stem}_bbox3d.png"
            out_paths.append(
                overlay_3d_boxes(
                    frame_path, objdata, cams, int(frame_path.stem), out_path
                )
            )
    return out_paths
