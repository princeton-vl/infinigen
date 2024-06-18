# Copyright (c) Princeton University.

import gin
import numpy as np

from infinigen.core.util.random import random_general as rg


def make_np(xs):
    return [np.array(x) for x in xs]


@gin.configurable
def global_params(unit=.5, segment_margin=1.2, wall_thickness=('uniform', .2, .3),
    wall_thickness = rg(wall_thickness)
    wall_height = rg(wall_height)
    return {
        'unit': unit,
        'segment_margin': segment_margin,
        'wall_thickness': wall_thickness,
        'wall_height': wall_height
    }


UNIT, SEGMENT_MARGIN, WALL_THICKNESS, WALL_HEIGHT = make_np(global_params().values())


@gin.configurable
    assert door_width > 0
    door_margin = (door_width + WALL_THICKNESS) / 2
    door_size = rg(door_size)
    return {'door_width': door_width, 'door_margin': door_margin, 'door_size': door_size, }


DOOR_WIDTH, DOOR_MARGIN, DOOR_SIZE = make_np(door_params().values())


@gin.configurable
    max_window_length = rg(max_window_length)
    window_height = rg(window_height)
    window_margin = rg(window_margin)
    window_size = WALL_HEIGHT - WALL_THICKNESS - window_height - window_margin
    assert window_size > 0
    return {
        'max_window_length': max_window_length,
        'window_height': window_height,
        'window_margin': window_margin,
        'window_size': window_size,
    }


MAX_WINDOW_LENGTH, WINDOW_HEIGHT, WINDOW_MARGIN, WINDOW_SIZE = make_np(window_params().values())


@gin.configurable
def staircase_params(staircase_snap=('uniform', .8, 1.2)):
    return {'staircase_snap': rg(staircase_snap)}


STAIRCASE_SNAP = make_np(staircase_params().values())


    ys = make_np(global_params().values())
    xs = UNIT, SEGMENT_MARGIN, WALL_THICKNESS, WALL_HEIGHT
    for x, y in zip(xs, ys):
        x.fill(y)
    ys = make_np(door_params().values())
    xs = DOOR_WIDTH, DOOR_MARGIN, DOOR_SIZE
    for x, y in zip(xs, ys):
        x.fill(y)
    ys = make_np(window_params().values())
    xs = MAX_WINDOW_LENGTH, WINDOW_HEIGHT, WINDOW_MARGIN, WINDOW_SIZE
    for x, y in zip(xs, ys):
        x.fill(y)
