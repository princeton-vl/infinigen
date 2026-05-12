import numpy as np
import procfunc as pf

import infinigen_v2.generators.scenes.collision_collection as ccol

from .util import AcceptPred, _place_camera_in_bbox, total_bbox


@pf.tracer.grammar
def monocular_camera_in_bbox_distribution(
    rng: pf.RNG,
    objects: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    frame_start: int = 1,
    frame_end: int = 1,
    margin: float = 0.05,
    max_tries: int = 100,
    focal_length_mm: float = 15,
    accept_pred: AcceptPred | None = None,
) -> list[pf.CameraObject]:
    camera = pf.ops.primitives.perspective_camera(focal_length_mm=focal_length_mm)
    camera.item().name = "Camera"
    _place_camera_in_bbox(
        rng,
        camera,
        total_bbox(objects),
        colliders,
        frame_start,
        frame_end,
        margin,
        max_tries,
        accept_pred=accept_pred,
    )
    return [camera]


@pf.tracer.grammar
def monocular_360_camera_distribution(
    objects: list[pf.MeshObject],
    camera: pf.CameraObject | None = None,
    center: tuple[float, float] | None = None,
    radius: float | None = None,
    height: float | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
    focal_length_mm: float = 15,
    total_angle_rad: float = 2 * np.pi,
) -> list[pf.CameraObject]:
    if camera is None:
        camera = pf.ops.primitives.perspective_camera(focal_length_mm=focal_length_mm)

    all_min, all_max = total_bbox(objects)
    dims = all_max - all_min
    if center is None:
        center = (all_min[0] + all_max[0]) / 2, (all_min[1] + all_max[1]) / 2
    if radius is None:
        radius = min(dims[0], dims[1]) * 0.4
    if height is None:
        height = (all_min[2] + all_max[2]) / 2

    n_frames = max(frame_end - frame_start + 1, 1)
    angles = np.linspace(0, total_angle_rad, n_frames, endpoint=False)
    for t, a in enumerate(angles):
        camera.item().location = (
            center[0] + radius * np.sin(a),
            center[1] - radius * np.cos(a),
            height,
        )
        camera.item().rotation_euler = (np.pi / 2, 0, a)
        camera.item().keyframe_insert("location", frame=frame_start + t)
        camera.item().keyframe_insert("rotation_euler", frame=frame_start + t)

    return [camera]
