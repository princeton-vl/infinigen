# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lahav Lipson, Lingjie Mei: original camera matrices
# - Alex Raistrick: refactor,

import logging

import bpy
import numpy as np
import procfunc as pf
from mathutils import Matrix

import infinigen_v2.generators.scenes.collision_collection as ccol

logger = logging.getLogger(__name__)


def adjust_camera_sensor(camera: pf.CameraObject):
    W = bpy.context.scene.render.resolution_x
    H = bpy.context.scene.render.resolution_y
    sensor_width = 18 * (W / H)
    assert sensor_width.is_integer(), (18, W, H)

    data = camera.item().data
    data.sensor_height = 18
    data.sensor_width = int(sensor_width)


def bpy_resolution() -> tuple[int, int]:
    scale = bpy.context.scene.render.resolution_percentage / 100
    return (
        bpy.context.scene.render.resolution_x * scale,
        bpy.context.scene.render.resolution_y * scale,
    )


def get_calibration_matrix_K_from_blender(
    camera: pf.CameraObject,
) -> np.ndarray:
    """
    Build intrinsic camera parameters from Blender camera data

    Based on https://blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model

    Args:
        camera: Blender camera object

    Returns:
        K: 3x3 intrinsic camera matrix

    """

    camd = camera.item().data

    W, H = bpy_resolution()
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    f_in_mm = camd.lens

    if sensor_width_in_mm / sensor_height_in_mm != W / H:
        vals = f"{(sensor_width_in_mm, sensor_height_in_mm, W, H)=}"
        raise ValueError(
            f"Camera sensor has not been properly configured, you probably need to call camera.adjust_camera_sensor on it. {vals}"
        )

    pixel_aspect_ratio = (
        bpy.context.scene.render.pixel_aspect_x
        / bpy.context.scene.render.pixel_aspect_y
    )
    if camd.sensor_fit == "VERTICAL":
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = W / sensor_width_in_mm / pixel_aspect_ratio  # pixels per milimeter
        s_v = H / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        s_u = W / sensor_width_in_mm
        s_v = H * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = W / 2
    v_0 = H / 2
    skew = 0  # only use rectangular pixels

    K = Matrix(((alpha_u, skew, u_0), (0, alpha_v, v_0), (0, 0, 1)))
    return K


# Returns camera rotation and translation matrices from Blender.


def get_3x4_RT_matrix_from_blender(
    camera: pf.CameraObject,
) -> np.ndarray:
    """
    Get camera rotation and translation matrices from Blender.

    We return a camera matrix in the usual computer vision convention: +x is horizontal, +y is down, +z is forward.

    This is different than blender's camera typical convention, which is +x is horizontal, +y is up, -z is forward.

    Args:
        camera: Blender camera object

    Returns:
        RT: 3x4 camera matrix
    """

    blender_convention_to_compvis = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

    location, rotation = camera.item().matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    T_world2bcam = -1 * R_world2bcam @ location

    R_world2cv = blender_convention_to_compvis @ R_world2bcam
    T_world2cv = blender_convention_to_compvis @ T_world2bcam

    RT = Matrix(
        (
            R_world2cv[0][:] + (T_world2cv[0],),
            R_world2cv[1][:] + (T_world2cv[1],),
            R_world2cv[2][:] + (T_world2cv[2],),
        )
    )
    return RT


def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam)
    RT = get_3x4_RT_matrix_from_blender(cam)
    K = np.array(K)
    RT = np.array(RT)
    return K @ RT, K, RT


def project_points(
    camera: pf.CameraObject,
    points_N3: np.ndarray,
) -> np.ndarray:
    """
    Project points onto the camera image plane.
    """

    P, K, RT = get_3x4_P_matrix_from_blender(camera)

    points_hom_N4 = np.concatenate([points_N3, np.ones((len(points_N3), 1))], -1)
    projected = points_hom_N4 @ np.array(P).T
    assert projected.shape == (len(points_N3), 3)

    projected[:, :2] /= projected[:, [-1]]
    return projected


def is_projection_within_image(
    projected: np.ndarray,
    resolution: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Check if the projection is within the image.
    """
    u, v, d = projected.T

    if resolution is None:
        resolution = (
            bpy.context.scene.render.resolution_x,
            bpy.context.scene.render.resolution_y,
        )

    res = (
        (u >= 0)  # comment so ruff doesnt collapse this block
        * (u < resolution[0])
        * (v >= 0)
        * (v < resolution[1])
        * (d > 0)
    )
    return res


def camera_depth_raycast(
    cam: pf.CameraObject,
    colliders: ccol.CollisionSet,
    meshgridcoords: np.ndarray,
) -> np.ndarray:
    # meshgridcoords: (N, 2) normalized coords in [-1, 1] x [-1, 1]
    # dir_cam = (u * sensor_w/2, v * sensor_h/2, -focal_len) in mm, rotated to world
    # matches infinigen1 get_sensor_coords convention
    bpy.context.view_layer.update()
    cam_item = cam.item()
    cam_pos = np.array(cam_item.matrix_world.translation)
    rot = np.array(cam_item.matrix_world.to_3x3())
    half_w = cam_item.data.sensor_width / 2
    half_h = cam_item.data.sensor_height / 2
    f = cam_item.data.lens

    depths = np.full(len(meshgridcoords), np.inf)
    for i, (u, v) in enumerate(meshgridcoords):
        dir_cam = np.array([u * half_w, v * half_h, -f])
        dir_world = rot @ (dir_cam / np.linalg.norm(dir_cam))
        locs, _, _ = ccol.raycast(
            colliders,
            cam_pos.reshape(1, 3),
            dir_world.reshape(1, 3),
        )
        if len(locs) > 0:
            depths[i] = np.linalg.norm(locs[0] - cam_pos)
    return depths


def get_camera_parameters(
    camera: pf.CameraObject,
    frame: int | None = None,
    use_dof: bool | None = False,
) -> dict:
    if frame is not None:
        bpy.context.scene.frame_set(frame)
    if use_dof is not None:
        camera.item().data.dof.use_dof = use_dof
    K = get_calibration_matrix_K_from_blender(camera)
    T = np.asarray(camera.item().matrix_world, dtype=np.float64) @ np.diag(
        (1.0, -1.0, -1.0, 1.0)
    )
    HW = np.array(
        (
            bpy.context.scene.render.resolution_y,
            bpy.context.scene.render.resolution_x,
        )
    )
    return {"K": np.asarray(K, dtype=np.float64), "T": T, "HW": HW}
