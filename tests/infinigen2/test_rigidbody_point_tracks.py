import cv2
import numpy as np
import pytest

from infinigen2.exporters.rigidbody_point_tracks import (
    _uncovered_centers,
    object_pose_matrices,
    rigid_body_point_tracks,
    save_point_tracks,
    visualize_point_tracks,
)
from infinigen2.exporters.util.format import ExportType

HEIGHT = 96
WIDTH = 128
K = np.array([[100.0, 0.0, 64.0], [0.0, 100.0, 48.0], [0.0, 0.0, 1.0]])


def _object_data(frames: int) -> dict:
    location = np.zeros((2, 3, frames), dtype=np.float32)
    location[0, 2] = 5.0
    location[1, 2] = 3.0
    return {
        "object_index": np.array([1, 2], dtype=np.int32),
        "location_meters": location,
        "rotation_euler_rad": np.zeros((2, 3, frames), dtype=np.float32),
        "scale": np.ones((2, 3, frames), dtype=np.float32),
        "local_bbox_min": np.full((2, 3, frames), -0.5, dtype=np.float32),
        "local_bbox_max": np.full((2, 3, frames), 0.5, dtype=np.float32),
        "frame_start": np.int32(0),
        "frame_end": np.int32(frames - 1),
    }


def _scene_frames(frames: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    depth_THW = []
    seg_THW = []
    for frame in range(frames):
        depth = np.full((HEIGHT, WIDTH), 5.0, dtype=np.float32)
        seg = np.ones((HEIGHT, WIDTH), dtype=np.uint32)
        if frame == 1:
            depth[36:60, 52:76] = 3.0
            seg[36:60, 52:76] = 2
        depth_THW.append(depth)
        seg_THW.append(seg)
    return depth_THW, seg_THW


def _tracks(objdata: dict, frames: int, **kwargs) -> dict[str, np.ndarray]:
    depth_THW, seg_THW = _scene_frames(frames)
    K_T33 = np.stack([K] * frames)
    cam_T44 = np.stack([np.eye(4)] * frames)
    matrices_OT44 = object_pose_matrices(objdata)
    return rigid_body_point_tracks(
        matrices_OT44, K_T33, cam_T44, depth_THW, seg_THW, **kwargs
    )


def test_point_tracks_reappear_after_occlusion():
    tracks = _tracks(_object_data(3), 3)
    center = np.abs(tracks["uv"][:, :, 0] - [64.0, 48.0]) < 8.0
    far_center = (tracks["object_index"] == 1) & (tracks["birth_frame"] == 0)
    far_center &= center.all(axis=1)

    assert far_center.any()
    assert not tracks["visible"][far_center, 1].any()
    assert tracks["visible"][far_center, 2].any()


def test_user_seed_points_replace_the_grid():
    seed_uv = np.array([[64.0, 48.0], [30.0, 30.0], [100.0, 70.0]])

    tracks = _tracks(_object_data(3), 3, seed_uv_Q2=seed_uv)

    assert len(tracks["uv"]) == len(seed_uv)
    assert (tracks["birth_frame"] == 0).all()
    assert np.allclose(tracks["uv"][:, :, 0], seed_uv, atol=1.0)


def test_scale_animation_rejected_as_nonrigid():
    objdata = _object_data(3)
    objdata["scale"][0, :, 2] = 1.5

    with pytest.raises(ValueError, match="rigid"):
        _tracks(objdata, 3)


def test_deforming_vertices_rejected_as_nonrigid():
    objdata = _object_data(3)
    objdata["local_bbox_max"][0, 2, 2] = 0.8

    with pytest.raises(ValueError, match="deform"):
        object_pose_matrices(objdata)


def test_save_point_tracks_from_exports(tmp_path):
    frames = 3
    depth_THW, seg_THW = _scene_frames(frames)
    object_data = tmp_path / "object-data.npz"
    camera = tmp_path / "camera.npz"
    np.savez(object_data, **_object_data(frames))
    np.savez(camera, K=np.stack([K] * frames), T=np.stack([np.eye(4)] * frames))
    depth_paths = []
    seg_paths = []
    for frame in range(frames):
        depth_paths.append(tmp_path / f"depth_{frame:04d}.npy")
        seg_paths.append(tmp_path / f"object_{frame:04d}.npy")
        np.save(depth_paths[-1], depth_THW[frame])
        np.save(seg_paths[-1], seg_THW[frame])
    exports = [
        {ExportType.OBJECT_DATA: [object_data], ExportType.CAMERA: [camera]},
        {ExportType.DEPTH: depth_paths, ExportType.OBJECT_INDEX: seg_paths},
    ]

    result = save_point_tracks(exports, tmp_path)

    tracks = np.load(result[ExportType.POINT_TRAJECTORIES][0])
    assert tracks["uv"].shape[-1] == frames
    assert len(tracks["uv"])
    assert int(tracks["frame_start"]) == 0


def test_save_point_tracks_requires_the_gt_exports(tmp_path):
    with pytest.raises(ValueError, match="depth"):
        save_point_tracks([{}], tmp_path)


def _write_rgb_frames(tmp_path, frames: int) -> list:
    paths = []
    for frame in range(frames):
        path = tmp_path / f"{frame:04d}.png"
        cv2.imwrite(str(path), np.zeros((16, 16, 3), dtype=np.uint8))
        paths.append(path)
    return paths


def test_point_track_tails_are_opt_in(tmp_path):
    rgb_frames = _write_rgb_frames(tmp_path, 2)
    tracks = {
        "uv": np.array([[[2.0, 13.0], [8.0, 8.0]]], dtype=np.float32),
        "visible": np.array([[True, True]]),
        "birth_frame": np.array([0], dtype=np.int32),
    }

    plain = visualize_point_tracks(rgb_frames, tracks, tmp_path / "plain", dot_radius=1)
    tailed = visualize_point_tracks(
        rgb_frames, tracks, tmp_path / "tailed", dot_radius=1, tails=True
    )

    assert not cv2.imread(str(plain[-1]))[8, 8].any()
    assert cv2.imread(str(tailed[-1]))[8, 8].any()


def test_point_track_tails_default_to_two_frames():
    assert visualize_point_tracks.__defaults__[-1] == 2


def test_dynamic_objects_get_double_seed_density():
    static = _uncovered_centers(np.empty((0, 2)), np.zeros((HEIGHT, WIDTH), dtype=bool))
    dynamic = _uncovered_centers(np.empty((0, 2)), np.ones((HEIGHT, WIDTH), dtype=bool))

    assert len(static) == 7 * 12
    assert len(dynamic) == 14 * 24
    assert len(dynamic) == 4 * len(static)
