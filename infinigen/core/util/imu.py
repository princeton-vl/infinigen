# Copyright (c) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Dylan Li: primary author

import logging
from math import ceil, floor
from pathlib import Path

import bpy
import numpy as np

logger = logging.getLogger(__name__)


def solve_cubic(c0, c1, c2, c3):
    ZERO = float(-1.0e-10)
    ONE = float(1.000001)

    roots = np.roots([c3, c2, c1, c0])
    return [
        root.real
        for root in roots
        if abs(root.imag) < 1e-10 and ZERO <= root.real <= ONE
    ]


def bezier_zeros(x, q0, q1, q2, q3):
    c0 = q0 - x
    c1 = 3.0 * (q1 - q0)
    c2 = 3.0 * (q0 - 2.0 * q1 + q2)
    c3 = q3 - q0 + 3.0 * (q1 - q2)

    return solve_cubic(c0, c1, c2, c3)


def y_of_bezier(t, cy):
    """
    Returns the position of a cubic Bezier curve at t where t is in [0,1]
    """

    return (
        (1 - t) ** 3 * cy[0]
        + 3 * t * (1 - t) ** 2 * cy[1]
        + 3 * (t**2) * (1 - t) * cy[2]
        + t**3 * cy[3]
    )


def v_of_bezier(t, cx, cy):
    """
    Returns the velocity of a cubic Bezier curve at t where t is in [0,1]
    """

    dy_dt = (
        -3 * (1 - t) ** 2 * cy[0]
        + 3 * (3 * t**2 - 4 * t + 1) * cy[1]
        + 3 * (2 * t - 3 * t**2) * cy[2]
        + 3 * t**2 * cy[3]
    )
    dx_dt = (
        -3 * (1 - t) ** 2 * cx[0]
        + 3 * (3 * t**2 - 4 * t + 1) * cx[1]
        + 3 * (2 * t - 3 * t**2) * cx[2]
        + 3 * t**2 * cx[3]
    )

    return dy_dt / dx_dt


def a_of_bezier(t, cx, cy):
    """
    Returns the acceleration of a cubic Bezier curve at t where t is in [0,1]
    """

    A = (
        (cy[2] - 2 * cy[1] + cy[0]) * cx[3]
        + (-cy[3] + 3 * cy[1] - 2 * cy[0]) * cx[2]
        + (2 * cy[3] - 3 * cy[2] + cy[0]) * cx[1]
        + (-cy[3] + 2 * cy[2] - cy[1]) * cx[0]
    )
    B = (
        (cy[1] - cy[0]) * cx[3]
        + (3 * cy[0] - 3 * cy[1]) * cx[2]
        + (-cy[3] + 3 * cy[2] - 2 * cy[0]) * cx[1]
        + (cy[3] - 3 * cy[2] + 2 * cy[1]) * cx[0]
    )
    C = (cy[1] - cy[0]) * cx[2] + (cy[0] - cy[2]) * cx[1] + (cy[2] - cy[1]) * cx[0]
    D = cx[3] - 3 * cx[2] + 3 * cx[1] - cx[0]
    E = 2 * cx[2] - 4 * cx[1] + 2 * cx[0]
    F = cx[1] - cx[0]
    dx_dt = (
        -3 * (1 - t) ** 2 * cx[0]
        + 3 * (3 * t**2 - 4 * t + 1) * cx[1]
        + 3 * (2 * t - 3 * t**2) * cx[2]
        + 3 * t**2 * cx[3]
    )

    return -2 * (A * t**2 + B * t + C) / (D * t**2 + E * t + F) ** 2 / dx_dt


def data_from_keyframes(keyframe_points, frame_start, frame_end, for_acceleration):
    """
    Return imu (acceleration or rotational velocity) and location data for an fcurve
    """
    frame_start = int(ceil(frame_start))
    frame_end = int(floor(frame_end))

    def f_to_i(frame, is_ceil):
        f = min(max(frame, frame_start), frame_end) - frame_start
        return ceil(f) if is_ceil else floor(f)

    def i_to_f(index):
        return index + frame_start

    data = np.zeros(frame_end - frame_start + 1)
    locs = np.zeros(frame_end - frame_start + 1)
    locs[: f_to_i(keyframe_points[0].co[0], False)] = keyframe_points[0].co[1]
    locs[f_to_i(keyframe_points[-1].co[0], True) :] = keyframe_points[-1].co[1]

    for i in range(len(keyframe_points) - 1):
        kfs = keyframe_points[i].co[0]
        kfe = keyframe_points[i + 1].co[0]
        kls = keyframe_points[i].co[1]
        kle = keyframe_points[i + 1].co[1]

        if kfs > frame_end or kfe < frame_start:
            continue

        if keyframe_points[i].interpolation == "CONSTANT":
            locs[f_to_i(kfs, True) : f_to_i(kfe, False) + 1] = kls
            data[f_to_i(kfs, True) : f_to_i(kfe, False) + 1] = 0

        elif keyframe_points[i].interpolation == "LINEAR":
            slope = (kle - kls) / (kfe - kfs)
            for j in range(f_to_i(kfs, True), f_to_i(kfe, False) + 1):
                locs[j] = kls + (i_to_f(j) - kfs) * slope
                data[j] = 0 if for_acceleration else slope

        elif keyframe_points[i].interpolation == "BEZIER":
            # control points
            cx, cy = np.zeros(4), np.zeros(4)
            cx[0], cy[0] = kfs, kls
            cx[-1], cy[-1] = kfe, kle

            if keyframe_points[i].handle_right[0] > kfe:
                cx[1] = kfe
                cy[1] = kls + (kfe - kfs) * (
                    (keyframe_points[i].handle_right[1] - kls)
                    / (keyframe_points[i].handle_right[0] - kfs)
                )
            else:
                cx[1] = keyframe_points[i].handle_right[0]
                cy[1] = keyframe_points[i].handle_right[1]

            if keyframe_points[i + 1].handle_left[0] < kfs:
                cx[2] = kfs
                cy[2] = kle + (kfs - kfe) * (
                    (keyframe_points[i + 1].handle_left[1] - kle)
                    / (keyframe_points[i + 1].handle_left[0] - kfe)
                )
            else:
                cx[2] = keyframe_points[i + 1].handle_left[0]
                cy[2] = keyframe_points[i + 1].handle_left[1]

            for j in range(f_to_i(kfs, True), f_to_i(kfe, False) + 1):
                roots = bezier_zeros(i_to_f(j), cx[0], cx[1], cx[2], cx[3])
                if len(roots) == 0:
                    raise Exception(
                        "Bezier interpolation failed at frame {}".format(i_to_f(j))
                    )
                data[j] = (
                    a_of_bezier(roots[0], cx, cy)
                    if for_acceleration
                    else v_of_bezier(roots[0], cx, cy)
                )
                locs[j] = y_of_bezier(roots[0], cy)

        else:
            raise Exception(
                "Keyframe interpolation mode {} not a supported mode: constant, linear, bezier".format(
                    keyframe_points[i].interpolation
                )
            )

    return data, locs


def get_imu_tum_data(object, start, end):
    """
    Returns imu and tum data of camera in file formatted strings
    """

    if object.animation_data is None:
        raise Exception(f"{object.name} has no animation data")

    if object.animation_data.action is None:
        raise Exception(f"{object.name} has no action")

    start = ceil(start)
    end = floor(end)
    length = end - start + 1

    if length <= 1:
        raise Exception(f"trajectory duration is too short: {length} frames")

    old_rotation = object.rotation_mode
    object.rotation_mode = "XYZ"

    ax, ay, az, rvx, rvy, rvz = (
        np.zeros(length),
        np.zeros(length),
        np.zeros(length),
        np.zeros(length),
        np.zeros(length),
        np.zeros(length),
    )
    x, y, z, rx, ry, rz = None, None, None, None, None, None

    # find data
    for curve in object.animation_data.action.fcurves:
        if curve.data_path == "location":
            if curve.array_index == 0:
                ax, x = data_from_keyframes(curve.keyframe_points, start, end, True)
            elif curve.array_index == 1:
                ay, y = data_from_keyframes(curve.keyframe_points, start, end, True)
            elif curve.array_index == 2:
                az, z = data_from_keyframes(curve.keyframe_points, start, end, True)

        elif curve.data_path == "rotation_euler":
            if curve.array_index == 0:
                rvx, rx = data_from_keyframes(curve.keyframe_points, start, end, False)
            elif curve.array_index == 1:
                rvy, ry = data_from_keyframes(curve.keyframe_points, start, end, False)
            elif curve.array_index == 2:
                rvz, rz = data_from_keyframes(curve.keyframe_points, start, end, False)

        else:
            raise Exception("Unsupported action fcurve: {}".format(curve.data_path))

    SMALL = float(1e-4)

    # check data accuracy
    for i in range(length):
        bpy.context.scene.frame_set(i + start)
        if x is not None:
            if abs(x[i] - object.location[0]) > SMALL:
                raise Exception(
                    "Bezier interpolation innacurate at frame {} for x translation {} vs {}".format(
                        i, x[i], object.location[0]
                    )
                )
        if y is not None:
            if abs(y[i] - object.location[1]) > SMALL:
                raise Exception(
                    "Bezier interpolation innacurate at frame {} for y translation {} vs {}".format(
                        i, y[i], object.location[1]
                    )
                )
        if z is not None:
            if abs(z[i] - object.location[2]) > SMALL:
                raise Exception(
                    "Bezier interpolation innacurate at frame {} for z translation {} vs {}".format(
                        i, z[i], object.location[2]
                    )
                )
        if rx is not None:
            if abs(rx[i] - object.rotation_euler[0]) > SMALL:
                raise Exception(
                    "Bezier interpolation innacurate at frame {} for x Euler rotation {} vs {}".format(
                        i, rx[i], object.rotation_euler[0]
                    )
                )
        if ry is not None:
            if abs(ry[i] - object.rotation_euler[1]) > SMALL:
                raise Exception(
                    "Bezier interpolation innacurate at frame {} for y Euler rotation {} vs {}".format(
                        i, ry[i], object.rotation_euler[1]
                    )
                )
        if rz is not None:
            if abs(rz[i] - object.rotation_euler[2]) > SMALL:
                raise Exception(
                    "Bezier interpolation innacurate at frame {} for z Euler rotation {} vs {}".format(
                        i, rz[i], object.rotation_euler[2]
                    )
                )

    imu_text = []
    tum_text = []
    object.rotation_mode = "QUATERNION"

    # format data
    for i in range(length):
        bpy.context.scene.frame_set(i + start)
        imu_text.append(
            " ".join(
                [
                    str(i + start),
                    str(rvx[i]),
                    str(rvy[i]),
                    str(rvz[i]),
                    str(ax[i]),
                    str(ay[i]),
                    str(az[i]),
                ]
            )
        )
        tum_text.append(
            " ".join(
                [
                    str(i + start),
                    str(object.location[0]),
                    str(object.location[1]),
                    str(object.location[2]),
                    str(object.rotation_quaternion[1]),
                    str(object.rotation_quaternion[2]),
                    str(object.rotation_quaternion[3]),
                    str(object.rotation_quaternion[0]),
                ]
            )
        )

    object.rotation_mode = old_rotation
    return "\n".join(imu_text), "\n".join(tum_text)


def save_imu_tum_files(
    output_folder,
    objects: list[bpy.types.Object],
    start=bpy.context.scene.frame_start,
    end=bpy.context.scene.frame_end,
):
    """
    Write imu and tum data to output files for each object.
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    anim_objects = [x for x in objects if x.animation_data is not None]
    if len(anim_objects) == 0:
        logger.warning("save_imu_tum_files: No animation data in given objects")
        return

    for i in range(len(objects)):
        try:
            imu_text, tum_text = get_imu_tum_data(anim_objects[i], start, end)
            name = anim_objects[i].name.replace("/", "_")
        except Exception as e:
            logger.warning(
                f"Error when saving imu/tum data for {anim_objects[i].name}: {e}"
            )
            continue
        imu_file_name = output_folder / f"{name}_imu.txt"
        tum_file_name = output_folder / f"{name}_tum.txt"

        with open(imu_file_name, "w") as imu_file:
            imu_file.write(
                "#Format: timestamp angular velocity(x y z) linear acceleration(x y z)\n"
            )
            imu_file.write(imu_text)
            print(f"saved {imu_file_name}")
            logger.info(f"Saved IMU data for {anim_objects[i].name}")

        with open(tum_file_name, "w") as tum_file:
            tum_file.write("#Format: timestamp position(x y z) rotation(x y z w)\n")
            tum_file.write(tum_text)
            logger.info(f"Saved TUM data for {anim_objects[i].name}")
