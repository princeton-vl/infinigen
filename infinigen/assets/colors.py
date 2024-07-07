# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Stamatis Alexandropoulos, Meenal Parakh: original version
# - Alexander Raistrick: Refactor into MixtureOfGaussians class

import numpy as np
from numpy.random import normal, uniform

# ruff: noqa: F401
from infinigen.core.util.color import hex2rgba, hsv2rgba, rgb2hsv
from infinigen.core.util.random import (
    log_uniform,
    mixture_of_gaussian,
    weighted_sample,
    wrap_gaussian,
)


def sofa_fabric_hsv():
    return mixture_of_gaussian(
        means=np.array([[0.1, 0.25, 0.47], [0.5, 0.7, 0.47], [0.65, 0.15, 0.47]]),
        stds=np.array(
            [[0.007, 0.028, 0.23], [0.014, 0.014, 0.23], [0.021, 0.0084, 0.23]]
        ),
        weights=[0.5, 0.3, 0.2],
        clamp_min=[0.0, 0.0, 0.1],
        clamp_max=[1.0, 1.0, 0.88],
    )


def leather_hsv():
    return mixture_of_gaussian(
        means=np.array([[0.07, 0.45, 0.40], [0.6, 0.3, 0.40]]),
        stds=0.7 * np.array([[0.0035, 0.0063, 0.2], [0.0105, 0.028, 0.2]]),
        clamp_min=[0, 0, 0.06],
        clamp_max=[1, 1, 0.93],
        weights=[0.7, 0.3],
    )


# def leather_hsv():
#     return (
#         uniform(0.04, 0.07),
#         uniform(0.80, 1.0),
#         uniform(0.1, 0.6),
#     )


def linen_hsv():
    return mixture_of_gaussian(
        means=np.array([[0.12, 0.5, 0.55], [0.6, 0.4, 0.55], [0.9, 0.2, 0.55]]),
        std=np.array([[0.007, 0.084, 0.2], [0.007, 0.063, 0.2], [0.007, 0.014, 0.2]]),
        weights=[0.8, 0.15, 0.05],
        clamp_min=[0, 0, 0.15],
        clamp_max=[1, 1, 0.86],
    )


def velvet_hsv():
    return mixture_of_gaussian(
        means=np.array([[0.52, 0.45, 0.35]]),
        std=np.array([[0.14, 0.14, 0.18]]),
        weights=[1.0],
        clamp_min=[0, 0, 0.11],
        clamp_max=[1, 1, 0.70],
    )


def bedding_sheet_hsv():
    return mixture_of_gaussian(
        means=np.array([[0.1, 0.4, 0.66], [0.6, 0.2, 0.17]]),
        std=np.array([[0.007, 0.56, 0.17], [0.021, 0.014, 0.17]]),
        weights=[0.9, 0.1],
        clamp_min=[0, 0, 0.15],
        clamp_max=[1, 1, 0.94],
    )


def petal_hsv():
    return (
        wrap_gaussian(0.95, 1.2),
        uniform(0.2, 0.85),
        uniform(0.2, 0.75),
    )


def gem_hsv():
    return (
        uniform(0, 1),
        0.85,
        uniform(0.5, 1),
    )


def plant_green():
    return (
        uniform(0.25, 0.33),
        normal(0.65, 0.03),
        uniform(0.1, 0.45),
    )


def plant_pink():
    return (
        wrap_gaussian(0.88, 0.06),
        normal(0.6, 0.05),
        normal(0.8, 0.05),
    )


def plant_white():
    return (
        wrap_gaussian(0.0, 0.06),
        uniform(0.0, 0.2),
        normal(0.95, 0.02),
    )


def plant_red():
    return (
        wrap_gaussian(0.0, 0.05),
        normal(0.9, 0.03),
        normal(0.6, 0.05),
    )


def plant_yellow():
    return (
        wrap_gaussian(0.15, 0.005),
        normal(0.95, 0.02),
        normal(0.9, 0.02),
    )


def tree_petal_hsv():
    r = uniform()

    if r < 0.4:
        return plant_pink()
    elif r < 0.6:
        return plant_white()
    elif r < 0.8:
        return plant_red()
    else:
        return plant_yellow()


def fog_hsv():
    return (
        uniform(0, 1),
        uniform(0, 0.2),
        uniform(0.8, 1),
    )


def water_hsv():
    return (
        uniform(0.2, 0.6),
        normal(0.5, 0.1),
        uniform(0.7, 1),
    )


def darker_water_hsv():
    return (
        uniform(0.2, 0.6),
        normal(0.5, 0.1),
        uniform(0.2, 0.3),
    )


def under_water_hsv():
    return (
        uniform(0.5, 0.7),
        uniform(0.7, 0.95),
        uniform(0.7, 1),
    )


def eye_schlera_hsv():
    return (
        uniform(0.05, 0.15),
        uniform(0.2, 0.8),
        uniform(0.05, 0.5),
    )


def eye_pupil_hsv():
    return (
        uniform(0, 1),
        uniform(0.1, 0.9),
        uniform(0.1, 0.9),
    )


def beak_hsv():
    return (
        uniform(0, 0.13),
        uniform(0, 0.9),
        uniform(0.1, 0.6),
    )


def fur_hsv():
    return (
        uniform(0, 0.11),
        uniform(0.5, 0.95),
        uniform(0.02, 0.9),
    )


def pine_needle_hsv():
    return (
        wrap_gaussian(0.05, 0.02),
        uniform(0.5, 0.93),
        uniform(0.045, 0.4),
    )


def wet_sand_hsv():
    return (
        uniform(0.05, 0.1),
        uniform(0.65, 0.7),
        uniform(0.05, 0.15),
    )


def dry_sand_hsv():
    return (
        uniform(0.05, 0.1),
        uniform(0.65, 0.7),
        uniform(0.15, 0.25),
    )


def concrete_hsv():
    return (
        uniform(0.0, 1.0),
        uniform(0.02, 0.12),
        uniform(0.3, 0.9),
    )


def textile_hsv():
    return (
        uniform(0, 1),
        uniform(0.15, 0.7),
        uniform(0.1, 0.3),
    )


def fabric_hsv():
    return (
        uniform(0, 1),
        uniform(0.3, 0.8),
        uniform(0.6, 0.9),
    )


wood_colors = [
    # TODO: these discrete colors are discouraged and should be changed to a normal distribution / mixture of gaussians over HSV values
    0x4C2F27,
    0x69432D,
    0x371803,
    0x7F4040,
    0xCC9576,
    0x9E8170,
    0x3D2B1F,
    0x8D6A58,
    0x8B3325,
    0x79443C,
    0x88540B,
    0x9B5F43,
    0x4E3828,
    0x4E3828,
    0xC09A6B,
    0x944536,
    0x3F0110,
    0x773C12,
    0x6E4E37,
    0x5C4033,
    0x5C4033,
    0x3C3034,
    0x96704C,
    0x371B1A,
    0x483B32,
    0x43141A,
    0x471713,
    0xC3B090,
    0x6B4423,
    0x674D46,
    0x5D2E1A,
    0x331C1F,
    0x7A5640,
    0xB99984,
    0x71543D,
    0x8F4B28,
    0x491A00,
    0x836446,
    0x7F461B,
    0x6A3208,
    0x724115,
    0xA0522B,
    0x832A0C,
    0x371B1A,
    0xC7A373,
    0x483B32,
    0x635147,
    0x664228,
    0x5C5248,
]


def bark_hsv():
    hexval = np.random.choice(wood_colors)
    return rgb2hsv(hex2rgba(hexval)[:-1])


plain_base_colors = [
    (0xFDD017, 0.5),
    (0xC0C0C0, 1),
    (0x8C7853, 1),
    (0xB87333, 0.5),
    (0xB5A642, 0.5),
    (0xBDBAAE, 1),
    (0xA9ACB6, 1),
    (0xB6AFA9, 1),
]

natural_base_colors = [
    (0xC0C0C0, 1),
    (0x8C7853, 1),
    (0xBDBAAE, 1),
    (0xA9ACB6, 1),
    (0xB6AFA9, 1),
]


def metal_plain_hsv():
    hexval = weighted_sample(plain_base_colors)
    h, s, v = rgb2hsv(hex2rgba(hexval)[:-1])
    return (
        h + uniform(-0.1, 0.1),
        s + uniform(-0.1, 0.1),
        v * log_uniform(0.5, 0.2),
    )


def metal_natural_hsv():
    hexval = weighted_sample(natural_base_colors)
    h, s, v = rgb2hsv(hex2rgba(hexval)[:-1])
    return (
        h + uniform(-0.1, 0.1),
        s + uniform(-0.1, 0.1),
        v * log_uniform(0.5, 0.2),
    )


def metal_bw_hsv():
    return (
        uniform(0, 1),
        uniform(0.0, 0.2),
        log_uniform(0.01, 0.2),
    )


def metal_bw_natural_hsv():
    return metal_bw_hsv() if uniform() < 0.5 else metal_natural_hsv()


def metal_hsv():
    if uniform() < 0.2:
        return metal_natural_hsv()
    return (
        uniform(0, 1),
        uniform(0.3, 0.6),
        log_uniform(0.02, 0.5),
    )
