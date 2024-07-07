# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei: primary author
# - Karhan Kayan: bug fixes

import gin
import numpy as np

from infinigen.core.util.random import random_general as rg


def make_np(xs):
    return [np.array(x) for x in xs]


@gin.configurable
def global_params(
    unit=0.5,
    segment_margin=1.2,
    wall_thickness=("uniform", 0.2, 0.3),
    wall_height=("uniform", 2.7, 3.8),
):
    wall_thickness = rg(wall_thickness)
    wall_height = rg(wall_height)
    return {
        "unit": unit,
        "segment_margin": segment_margin,
        "wall_thickness": wall_thickness,
        "wall_height": wall_height,
    }


UNIT, SEGMENT_MARGIN, WALL_THICKNESS, WALL_HEIGHT = make_np(global_params().values())


@gin.configurable
def door_params(door_width=("uniform", 0.85, 1), door_size=("uniform", 2.0, 2.4)):
    door_width = rg(door_width)
    assert door_width > 0
    door_margin = (door_width + WALL_THICKNESS) / 2
    door_size = rg(door_size)
    return {
        "door_width": door_width,
        "door_margin": door_margin,
        "door_size": door_size,
    }


DOOR_WIDTH, DOOR_MARGIN, DOOR_SIZE = make_np(door_params().values())


@gin.configurable
def window_params(
    max_window_length=("uniform", 6, 8),
    window_height=("uniform", 0.4, 1.2),
    window_margin=("uniform", 0.2, 0.6),
):
    max_window_length = rg(max_window_length)
    window_height = rg(window_height)
    window_margin = rg(window_margin)
    window_size = WALL_HEIGHT - WALL_THICKNESS - window_height - window_margin
    assert window_size > 0
    return {
        "max_window_length": max_window_length,
        "window_height": window_height,
        "window_margin": window_margin,
        "window_size": window_size,
    }


MAX_WINDOW_LENGTH, WINDOW_HEIGHT, WINDOW_MARGIN, WINDOW_SIZE = make_np(
    window_params().values()
)


@gin.configurable
def staircase_params(staircase_snap=("uniform", 0.8, 1.2)):
    return {"staircase_snap": rg(staircase_snap)}


STAIRCASE_SNAP = make_np(staircase_params().values())


def init_global_params():
    ys = make_np(global_params().values())
    xs = UNIT, SEGMENT_MARGIN, WALL_THICKNESS, WALL_HEIGHT
    for x, y in zip(xs, ys):
        x.fill(y)


def init_door_params():
    ys = make_np(door_params().values())
    xs = DOOR_WIDTH, DOOR_MARGIN, DOOR_SIZE
    for x, y in zip(xs, ys):
        x.fill(y)


def init_window_params():
    ys = make_np(window_params().values())
    xs = MAX_WINDOW_LENGTH, WINDOW_HEIGHT, WINDOW_MARGIN, WINDOW_SIZE
    for x, y in zip(xs, ys):
        x.fill(y)


def initialize_constants():
    init_global_params()
    init_door_params()
    init_window_params()
