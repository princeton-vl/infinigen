# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alejandro Newell, Lahav Lipson


import numpy as np


def get_cos_sin(angle, convert_to_rad=False):
    if convert_to_rad:
        angle = angle * np.pi / 180
    return np.cos(angle), np.sin(angle)


def rodrigues_rot(vec, axis, angle, convert_to_rad=False):
    axis = axis / np.linalg.norm(axis)
    cs, sn = get_cos_sin(angle, convert_to_rad)
    return vec * cs + sn * np.cross(axis, vec) + axis * np.dot(axis, vec) * (1 - cs)


def yaw_clockwise(current_dir, angle_magnitude):
    axis = np.array((0, 0, 1))
    return rodrigues_rot(current_dir, axis, -angle_magnitude, True)


def pitch_up(current_dir, angle_magnitude):
    axis = np.array((0, 1, 0))
    return rodrigues_rot(current_dir, axis, -angle_magnitude, True)


def increment_step(current_dir, amount):
    mag = np.sqrt(np.sum(np.power(current_dir, 2)))
    unit_dir = current_dir / mag
    output = current_dir + unit_dir*amount
    return output if np.sum(np.power(output, 2)) > 0 else current_dir
