# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

"""Track rigid-body surface points across frames from rendered depth and object index.

Inputs come from `object_data.collect_object_data` (saved as object-data.npz),
`camera_pose.save_camera_poses`, and the DEPTH / OBJECT_INDEX render passes.
Compute tracks post-facto from a render's file outputs, then save or visualize them::

    objdata = np.load(folder / "object-data.npz")
    cams = np.load(folder / cam / "camera.npz")
    depth_THW = [np.load(p) for p in sorted((folder / cam).glob("depth_*.npy"))]
    seg_THW = [np.load(p) for p in sorted((folder / cam).glob("object_*.npy"))]
    tracks = rigid_body_point_tracks(
        object_pose_matrices(objdata), cams["K"], cams["T"], depth_THW, seg_THW,
        frame_start=int(objdata["frame_start"]),
    )
    np.savez(folder / cam / "point-tracks.npz", **tracks)
    rgb_frames = sorted((folder / cam).glob("[0-9]*.png"))
    visualize_point_tracks(rgb_frames, tracks, folder / "tracks_vis")
"""

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from infinigen2.exporters.util.flow_vis import flow_uv_to_colors
from infinigen2.exporters.util.format import ExportType

__all__ = [
    "object_pose_matrices",
    "rigid_body_point_tracks",
    "save_point_tracks",
    "visualize_point_tracks",
]


def object_pose_matrices(
    objdata: "dict[str, np.ndarray] | np.lib.npyio.NpzFile",
) -> np.ndarray:
    """Per-frame object-to-world matrices_OT44 from collect_object_data output,
    indexed by segmentation slot. Rejects objects whose vertices deform."""
    _require_static_local_bboxes(objdata)
    index_N = objdata["object_index"]
    location_N3T = objdata["location_meters"]
    n_slots = int(index_N.max()) + 1
    matrices_OT44 = np.tile(np.eye(4), (n_slots, location_N3T.shape[-1], 1, 1))
    for row, slot in enumerate(index_N):
        euler_T3 = objdata["rotation_euler_rad"][row].T
        rotation_T33 = Rotation.from_euler("xyz", euler_T3).as_matrix()
        scale_T3 = objdata["scale"][row].T
        matrices_OT44[slot, :, :3, :3] = rotation_T33 * scale_T3[:, None, :]
        matrices_OT44[slot, :, :3, 3] = location_N3T[row].T
    return matrices_OT44


def _require_rigid(matrices_OT44: np.ndarray) -> None:
    scale_OT3 = np.linalg.norm(matrices_OT44[:, :, :3, :3], axis=-2)
    span_O3 = scale_OT3.max(axis=1) - scale_OT3.min(axis=1)
    stretchy_O = np.flatnonzero((span_O3 > 1e-4).any(axis=1))
    if len(stretchy_O):
        raise ValueError(
            f"Point tracks assume rigid bodies, objects {stretchy_O} change scale"
        )


def _require_static_local_bboxes(objdata: np.lib.npyio.NpzFile) -> None:
    for name in ("local_bbox_min", "local_bbox_max"):
        span_N3 = objdata[name].max(axis=2) - objdata[name].min(axis=2)
        deforming_N = (span_N3 > 1e-4).any(axis=1)
        if not deforming_N.any():
            continue
        slots = objdata["object_index"][deforming_N]
        raise ValueError(
            f"Point tracks assume rigid bodies, objects {slots} deform,"
            f" their {name} moves over frames"
        )


def _sample(image_HW: np.ndarray, uv_P2: np.ndarray) -> np.ndarray:
    return image_HW[uv_P2[:, 1].astype(int), uv_P2[:, 0].astype(int)]


def _unproject(
    uv_P2: np.ndarray,
    depth_HW: np.ndarray,
    K_33: np.ndarray,
    cam_to_world_44: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z_P = _sample(depth_HW, uv_P2)
    rays_P3 = np.concatenate([uv_P2 + 0.5, np.ones((len(uv_P2), 1))], axis=1)
    cam_P3 = (rays_P3 @ np.linalg.inv(K_33).T) * z_P[:, None]
    world_P3 = cam_P3 @ cam_to_world_44[:3, :3].T + cam_to_world_44[:3, 3]
    valid_P = np.isfinite(z_P) & (z_P > 0.05) & (z_P < 1e6)
    return world_P3, valid_P


def _visible(
    uv_P2: np.ndarray,
    track_depth_P: np.ndarray,
    depth_HW: np.ndarray,
    seg_HW: np.ndarray,
    object_P: np.ndarray,
) -> np.ndarray:
    height, width = depth_HW.shape
    visible_P = np.isfinite(uv_P2).all(axis=1) & (track_depth_P > 0.05)
    visible_P &= (uv_P2[:, 0] >= 0) & (uv_P2[:, 0] < width)
    visible_P &= (uv_P2[:, 1] >= 0) & (uv_P2[:, 1] < height)
    limit_2 = [width - 1, height - 1]
    pixel_P2 = np.clip(np.nan_to_num(uv_P2).round().astype(int), 0, limit_2)
    seen_depth_P = depth_HW[pixel_P2[:, 1], pixel_P2[:, 0]]
    seen_object_P = seg_HW[pixel_P2[:, 1], pixel_P2[:, 0]]
    visible_P &= np.isfinite(seen_depth_P) & (seen_depth_P > 0.97 * track_depth_P)
    return visible_P & (seen_object_P == object_P)


def _moving_objects(matrices_OT44: np.ndarray) -> np.ndarray:
    span_O44 = matrices_OT44.max(axis=1) - matrices_OT44.min(axis=1)
    return (span_O44 > 1e-4).any(axis=(1, 2))


def _dynamic_mask(seg_HW: np.ndarray, moving_O: np.ndarray) -> np.ndarray:
    known_HW = (seg_HW >= 0) & (seg_HW < len(moving_O))
    mask_HW = np.zeros(seg_HW.shape, dtype=bool)
    mask_HW[known_HW] = moving_O[seg_HW[known_HW]]
    return mask_HW


def _grid_cell(
    uv_P2: np.ndarray, height: int, width: int, rows: int, cols: int
) -> tuple[np.ndarray, np.ndarray]:
    col_P = (uv_P2[:, 0] / width * cols).astype(int).clip(0, cols - 1)
    row_P = (uv_P2[:, 1] / height * rows).astype(int).clip(0, rows - 1)
    return row_P, col_P


def _uncovered_centers(uv_visible_P2: np.ndarray, dynamic_HW: np.ndarray) -> np.ndarray:
    height, width = dynamic_HW.shape
    uncovered = []
    for rows, cols, dynamic in ((7, 12, False), (14, 24, True)):
        ys_R = np.linspace(24, height - 25, rows).round()
        xs_C = np.linspace(24, width - 25, cols).round()
        grid_x_RC, grid_y_RC = np.meshgrid(xs_C, ys_R)
        centers_Q2 = np.stack([grid_x_RC.ravel(), grid_y_RC.ravel()], axis=-1)
        centers_Q2 = centers_Q2[_sample(dynamic_HW, centers_Q2) == dynamic]
        counts_RC = np.zeros((rows, cols), dtype=int)
        np.add.at(counts_RC, _grid_cell(uv_visible_P2, height, width, rows, cols), 1)
        empty_Q = counts_RC[_grid_cell(centers_Q2, height, width, rows, cols)] == 0
        uncovered.append(centers_Q2[empty_Q])
    return np.concatenate(uncovered)


def _spawn_centers(
    seed_uv_Q2: np.ndarray | None,
    frame: int,
    uv_visible_P2: np.ndarray,
    dynamic_HW: np.ndarray,
) -> np.ndarray:
    if seed_uv_Q2 is None:
        return _uncovered_centers(uv_visible_P2, dynamic_HW)
    if frame == 0:
        return np.asarray(seed_uv_Q2, dtype=np.float64)
    return np.zeros((0, 2))


def _spawn(
    centers_Q2: np.ndarray,
    matrices_O44: np.ndarray,
    K_33: np.ndarray,
    cam_to_world_44: np.ndarray,
    depth_HW: np.ndarray,
    seg_HW: np.ndarray,
    trackable_O: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_Q3, valid_Q = _unproject(centers_Q2, depth_HW, K_33, cam_to_world_44)
    object_Q = _sample(seg_HW, centers_Q2).astype(np.int64)
    valid_Q &= (object_Q >= 0) & (object_Q < len(trackable_O))
    valid_Q &= trackable_O[object_Q.clip(0, len(trackable_O) - 1)]
    inverse_P44 = np.linalg.inv(matrices_O44[object_Q[valid_Q]])
    local_P3 = np.einsum("pij,pj->pi", inverse_P44[:, :3, :3], world_Q3[valid_Q])
    return local_P3 + inverse_P44[:, :3, 3], object_Q[valid_Q]


def _track_frame(
    local_P3: np.ndarray,
    object_P: np.ndarray,
    matrices_O44: np.ndarray,
    K_33: np.ndarray,
    cam_to_world_44: np.ndarray,
    depth_HW: np.ndarray,
    seg_HW: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix_P44 = matrices_O44[object_P]
    world_P3 = np.einsum("pij,pj->pi", matrix_P44[:, :3, :3], local_P3)
    world_P3 += matrix_P44[:, :3, 3]
    world_to_cam_44 = np.linalg.inv(cam_to_world_44)
    cam_P3 = world_P3 @ world_to_cam_44[:3, :3].T + world_to_cam_44[:3, 3]
    uvw_P3 = cam_P3 @ K_33.T
    behind_P1 = np.abs(uvw_P3[:, [2]]) < 1e-9
    uv_P2 = uvw_P3[:, :2] / np.where(behind_P1, np.nan, uvw_P3[:, [2]])
    visible_P = _visible(uv_P2, cam_P3[:, 2], depth_HW, seg_HW, object_P)
    return uv_P2, cam_P3[:, 2], visible_P


def _stack_frames(
    per_frame: list[tuple[np.ndarray, np.ndarray, np.ndarray]], n_tracks: int
) -> dict[str, np.ndarray]:
    n_frames = len(per_frame)
    uv_P2T = np.full((n_tracks, 2, n_frames), np.nan, dtype=np.float32)
    depth_PT = np.full((n_tracks, n_frames), np.nan, dtype=np.float32)
    visible_PT = np.zeros((n_tracks, n_frames), dtype=bool)
    for frame, (uv_Q2, depth_Q, visible_Q) in enumerate(per_frame):
        count = len(uv_Q2)
        uv_P2T[:count, :, frame] = uv_Q2
        depth_PT[:count, frame] = depth_Q
        visible_PT[:count, frame] = visible_Q
    return {"uv": uv_P2T, "depth_meters": depth_PT, "visible": visible_PT}


def rigid_body_point_tracks(
    matrices_OT44: np.ndarray,
    K_T33: np.ndarray,
    cam_to_world_T44: np.ndarray,
    depth_THW: np.ndarray,
    object_index_THW: np.ndarray,
    seed_uv_Q2: np.ndarray | None = None,
    frame_start: int = 0,
    trackable_O: np.ndarray | None = None,
    max_tracks: int = 16_000,
) -> dict[str, np.ndarray]:
    _require_rigid(matrices_OT44)
    n_frames = len(depth_THW)
    counts = {
        "matrices": matrices_OT44.shape[1],
        "K": len(K_T33),
        "cam_to_world": len(cam_to_world_T44),
        "object_index": len(object_index_THW),
    }
    if any(count != n_frames for count in counts.values()):
        raise ValueError(
            f"Point tracks need every input for all {n_frames} frames, got {counts}"
        )
    if trackable_O is None:
        trackable_O = np.ones(len(matrices_OT44), dtype=bool)
    moving_O = _moving_objects(matrices_OT44)

    local_P3 = np.zeros((0, 3))
    object_P = np.zeros(0, dtype=np.int64)
    birth_P = np.zeros(0, dtype=np.int32)
    per_frame = []
    for frame in range(n_frames):
        depth_HW = np.asarray(depth_THW[frame], dtype=np.float64)
        seg_HW = np.asarray(object_index_THW[frame], dtype=np.int64)
        frame_args = (matrices_OT44[:, frame], K_T33[frame], cam_to_world_T44[frame])
        frame_args += (depth_HW, seg_HW)
        tracked = _track_frame(local_P3, object_P, *frame_args)
        uv_P2, _, visible_P = tracked
        centers_Q2 = np.zeros((0, 2))
        if len(local_P3) < max_tracks:
            dynamic_HW = _dynamic_mask(seg_HW, moving_O)
            centers_Q2 = _spawn_centers(seed_uv_Q2, frame, uv_P2[visible_P], dynamic_HW)
        if not len(centers_Q2):
            per_frame.append(tracked)
            continue
        new_local_Q3, new_object_Q = _spawn(centers_Q2, *frame_args, trackable_O)
        local_P3 = np.concatenate([local_P3, new_local_Q3])
        object_P = np.concatenate([object_P, new_object_Q])
        new_birth_Q = np.full(len(new_object_Q), frame, dtype=np.int32)
        birth_P = np.concatenate([birth_P, new_birth_Q])
        per_frame.append(_track_frame(local_P3, object_P, *frame_args))

    tracks = _stack_frames(per_frame, len(local_P3))
    tracks["birth_frame"] = birth_P
    tracks["object_index"] = object_P.astype(np.int32)
    tracks["frame_start"] = np.int32(frame_start)
    tracks["frame_end"] = np.int32(frame_start + n_frames - 1)
    return tracks


def _stacked_camera(camera_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    cams = [np.load(path) for path in sorted(camera_paths)]
    K_T33 = np.stack([c["K"] for c in cams]) if len(cams) > 1 else cams[0]["K"]
    cam_T44 = np.stack([c["T"] for c in cams]) if len(cams) > 1 else cams[0]["T"]
    if K_T33.ndim == 2:
        K_T33 = K_T33[None]
        cam_T44 = cam_T44[None]
    return K_T33, cam_T44


def _merge_exports(
    exports: list[dict[ExportType, list[Path]]],
) -> dict[ExportType, list[Path]]:
    merged: dict[ExportType, list[Path]] = {}
    for export in exports:
        for export_type, paths in export.items():
            merged.setdefault(export_type, []).extend(paths)
    return merged


def save_point_tracks(
    exports: dict[ExportType, list[Path]] | list[dict[ExportType, list[Path]]],
    output_folder: Path,
    path: Path = Path("point-tracks.npz"),
) -> dict[ExportType, list[Path]]:
    if isinstance(exports, list):
        exports = _merge_exports(exports)

    required = (
        ExportType.OBJECT_DATA,
        ExportType.CAMERA,
        ExportType.DEPTH,
        ExportType.OBJECT_INDEX,
    )
    missing = [t.value for t in required if not exports.get(t)]
    if missing:
        raise ValueError(f"Point tracks need {missing} exported before them")

    objdata = np.load(exports[ExportType.OBJECT_DATA][0], allow_pickle=False)
    K_T33, cam_T44 = _stacked_camera(exports[ExportType.CAMERA])
    depth_THW = [np.load(path) for path in sorted(exports[ExportType.DEPTH])]
    seg_THW = [np.load(path) for path in sorted(exports[ExportType.OBJECT_INDEX])]
    tracks = rigid_body_point_tracks(
        object_pose_matrices(objdata),
        K_T33,
        cam_T44,
        depth_THW,
        seg_THW,
        frame_start=int(objdata["frame_start"]),
    )

    result_path = Path(output_folder) / path
    result_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(result_path, **tracks)
    return {ExportType.POINT_TRAJECTORIES: [result_path]}


def _track_colors(
    uv_P2T: np.ndarray, birth_P: np.ndarray, width: int, height: int
) -> np.ndarray:
    start_P2 = np.nan_to_num(uv_P2T[np.arange(len(uv_P2T)), :, birth_P])
    u_P = (start_P2[:, 0] - width / 2) / (width / 2)
    v_P = (start_P2[:, 1] - height / 2) / (height / 2)
    scale_P = 1 / np.maximum(np.hypot(u_P, v_P), 1e-6)
    return flow_uv_to_colors(
        (u_P * scale_P)[None, :], (v_P * scale_P)[None, :], convert_to_bgr=True
    )[0]


def _draw_tail(
    image_HW3: np.ndarray,
    positions_L2: np.ndarray,
    visible_L: np.ndarray,
    color_3: np.ndarray,
) -> None:
    for index in range(len(positions_L2) - 1):
        start_2, end_2 = positions_L2[index], positions_L2[index + 1]
        drawable = visible_L[index : index + 2].all()
        drawable = drawable and np.isfinite([start_2, end_2]).all()
        if not drawable or np.linalg.norm(end_2 - start_2) > 700:
            continue
        fade = (index + 1) / max(len(positions_L2) - 1, 1)
        cv2.line(
            image_HW3,
            tuple(np.round(start_2).astype(int)),
            tuple(np.round(end_2).astype(int)),
            (color_3 * fade).astype(int).tolist(),
            1,
            cv2.LINE_AA,
        )


def visualize_point_tracks(
    rgb_frames: list[Path],
    tracks: dict[str, np.ndarray],
    output_folder: Path,
    dot_radius: int = 6,
    dim: float = 0.55,
    tails: bool = False,
    tail_frames: int = 2,
) -> list[Path]:
    """Track-dot-over-rgb overlay for every frame in rgb_frames."""
    uv_P2T = tracks["uv"]
    visible_PT = tracks["visible"]
    birth_P = tracks["birth_frame"]
    if len(rgb_frames) != uv_P2T.shape[-1]:
        raise ValueError(
            f"Track frames {uv_P2T.shape[-1]} do not match images {len(rgb_frames)}"
        )
    height, width = cv2.imread(str(rgb_frames[0])).shape[:2]
    colors_P3 = _track_colors(uv_P2T, birth_P, width, height)
    output_folder.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for frame, frame_path in enumerate(rgb_frames):
        image_HW3 = cv2.imread(str(frame_path)).astype(np.float32)
        image_HW3 = (image_HW3 * dim).astype(np.uint8)
        tail_tracks = np.flatnonzero(birth_P <= frame) if tails else []
        for index in tail_tracks:
            first = max(frame - tail_frames, int(birth_P[index]))
            tail_uv_L2 = uv_P2T[index, :, first : frame + 1].T
            tail_visible_L = visible_PT[index, first : frame + 1]
            _draw_tail(image_HW3, tail_uv_L2, tail_visible_L, colors_P3[index])
        on_P = visible_PT[:, frame]
        for point_2, color_3 in zip(
            uv_P2T[on_P, :, frame], colors_P3[on_P], strict=True
        ):
            center = tuple(np.round(point_2).astype(int))
            cv2.circle(image_HW3, center, dot_radius, color_3.tolist(), -1, cv2.LINE_AA)
        out_path = output_folder / f"{frame_path.stem}_tracks.png"
        cv2.imwrite(str(out_path), image_HW3)
        out_paths.append(out_path)
    return out_paths
