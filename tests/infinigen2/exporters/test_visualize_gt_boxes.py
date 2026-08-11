# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from pathlib import Path

import numpy as np
import pytest
from imageio.v3 import imread, imwrite
from mathutils import Euler

from infinigen2.exporters.util.format import ExportType
from infinigen2.exporters.visualize_gt import visualize_gt
from infinigen2.exporters.visualize_gt_boxes import euler_to_matrix, overlay_3d_boxes
from infinigen2.exporters.visualize_gt_passes import index_colors


def _synthetic_scene(tmp_path: Path, n: int = 2) -> tuple[Path, Path, Path]:
    height, width = 240, 320
    rgb = tmp_path / "0000.png"
    imwrite(rgb, np.zeros((height, width, 3), np.uint8))

    location = np.zeros((n, 3, 1), np.float32)
    for i in range(n):
        location[i, :, 0] = (-0.6 + 1.2 * i, 0.0, 4.0)

    objdata = tmp_path / "object-data.npz"
    np.savez(
        objdata,
        frame_start=np.int32(0),
        frame_end=np.int32(0),
        location_meters=location,
        rotation_euler_rad=np.zeros((n, 3, 1), np.float32),
        scale=np.ones((n, 3, 1), np.float32),
        local_bbox_min=np.full((n, 3, 1), -0.5, np.float32),
        local_bbox_max=np.full((n, 3, 1), 0.5, np.float32),
        object_index=np.arange(1, n + 1, dtype=np.int32),
        object_name=np.array([f"obj{i}" for i in range(n)], dtype="S63"),
        object_type=np.array(["MESH"] * n, dtype="S63"),
        data_name=np.array([f"mesh{i}" for i in range(n)], dtype="S63"),
        data_id=np.arange(n, dtype=np.int32),
    )

    camera = tmp_path / "camera.npz"
    intrinsics = np.array(
        [[200.0, 0, width / 2], [0, 200.0, height / 2], [0, 0, 1]], np.float32
    )
    np.savez(camera, K=intrinsics, T=np.eye(4, dtype=np.float32))
    return rgb, objdata, camera


def test_euler_to_matrix_matches_blender():
    eulers = np.array([[0.0, 0.0, 0.0], [0.3, -0.7, 1.1], [-2.0, 1.4, 0.2]])
    expected = [np.array(Euler(tuple(e), "XYZ").to_matrix()) for e in eulers]
    assert euler_to_matrix(eulers) == pytest.approx(np.array(expected), abs=1e-6)


def test_overlay_3d_boxes_draws_edges(tmp_path: Path):
    rgb, objdata, camera = _synthetic_scene(tmp_path)
    out = overlay_3d_boxes(
        rgb, dict(np.load(objdata)), dict(np.load(camera)), 0, tmp_path / "out.png"
    )
    drawn = np.asarray(imread(out))
    assert int((drawn[..., 1] > 100).sum()) > 200


def test_overlay_3d_boxes_skips_rows_without_a_pose(tmp_path: Path):
    rgb, objdata, camera = _synthetic_scene(tmp_path, n=1)
    data = dict(np.load(objdata, allow_pickle=False))
    data["location_meters"][0, :, 0] = np.nan
    np.savez(objdata, **data)

    out = overlay_3d_boxes(rgb, data, dict(np.load(camera)), 0, tmp_path / "nopose.png")
    assert int((np.asarray(imread(out)) > 0).sum()) == 0


def test_overlay_3d_boxes_colors_by_object_index(tmp_path: Path):
    rgb, objdata, camera = _synthetic_scene(tmp_path, n=1)
    data = dict(np.load(objdata, allow_pickle=False))
    data["object_index"] = np.array([7], np.int32)
    np.savez(objdata, **data)

    out = overlay_3d_boxes(
        rgb, data, dict(np.load(camera)), 0, tmp_path / "colored.png"
    )
    drawn = np.asarray(imread(out)).reshape(-1, 3)
    # the box is drawn anti-aliased over black, so only fully covered pixels hit full value
    brightest = drawn[drawn.sum(-1).argmax()]
    expected = index_colors(np.array([[7]]))[0]
    assert np.abs(brightest.astype(int) - expected.astype(int)).max() < 20


def test_overlay_3d_boxes_skips_geometry_behind_camera(tmp_path: Path):
    rgb, objdata, camera = _synthetic_scene(tmp_path, n=1)
    data = dict(np.load(objdata, allow_pickle=False))
    data["location_meters"][0, 2, 0] = -4.0
    np.savez(objdata, **data)

    out = overlay_3d_boxes(rgb, data, dict(np.load(camera)), 0, tmp_path / "behind.png")
    drawn = np.asarray(imread(out))
    assert int((drawn[..., 1] > 100).sum()) == 0


def test_overlay_3d_boxes_rejects_a_frame_it_has_no_pose_for(tmp_path: Path):
    rgb, objdata, camera = _synthetic_scene(tmp_path)
    with pytest.raises(ValueError):
        overlay_3d_boxes(
            rgb,
            dict(np.load(objdata)),
            dict(np.load(camera)),
            5,
            tmp_path / "missing.png",
        )


def test_visualize_gt_draws_boxes_for_object_data_outside_its_output_folder(
    tmp_path: Path,
):
    """object-data.npz is per-scene while visualize_gt runs per-camera, so the box
    overlay has to find it through the exports dict rather than beside its output."""
    camera_folder = tmp_path / "Camera"
    camera_folder.mkdir()
    rgb, objdata, camera = _synthetic_scene(tmp_path)
    rgb = rgb.rename(camera_folder / rgb.name)

    exports = {
        ExportType.IMAGE: [rgb],
        ExportType.CAMERA: [camera],
        ExportType.OBJECT_DATA: [objdata],
    }
    paths = visualize_gt(exports, camera_folder)[ExportType.VISUALIZATIONS]
    assert len(paths) == 1
    assert int((np.asarray(imread(paths[0]))[..., 1] > 100).sum()) > 200


def test_visualize_gt_skips_boxes_without_object_data(tmp_path: Path):
    rgb, _, camera = _synthetic_scene(tmp_path)
    exports = {ExportType.IMAGE: [rgb], ExportType.CAMERA: [camera]}
    assert visualize_gt(exports, tmp_path)[ExportType.VISUALIZATIONS] == []
