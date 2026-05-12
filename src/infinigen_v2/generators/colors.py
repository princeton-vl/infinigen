# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Stamatis Alexandropoulos, Meenal Parakh: infinigen indoors version
# - Alexander Raistrick: refactor, unify all color distributions in repo into this file

import logging
from typing import Tuple

import numpy as np
import procfunc as pf

# ruff: noqa: F401

logger = logging.getLogger(__name__)


def sofa_fabric_color(rng: np.random.Generator) -> pf.Color:
    hsv = pf.random.mixture_of_gaussian(
        rng=rng,
        means=np.array([[0.1, 0.25, 0.47], [0.5, 0.7, 0.47], [0.65, 0.15, 0.47]]),
        stds=np.array(
            [[0.007, 0.028, 0.23], [0.014, 0.014, 0.23], [0.021, 0.0084, 0.23]]
        ),
        weights=[0.5, 0.3, 0.2],
        low=np.array([0.0, 0.0, 0.1]),
        high=np.array([1.0, 1.0, 0.88]),
    )
    return pf.color.hsv_color(hsv=hsv)


def leather_color(rng: np.random.Generator) -> pf.Color:
    hsv = pf.random.mixture_of_gaussian(
        rng=rng,
        means=np.array([[0.07, 0.45, 0.40], [0.6, 0.3, 0.40]]),
        stds=0.7 * np.array([[0.0035, 0.0063, 0.2], [0.0105, 0.028, 0.2]]),
        low=np.array([0, 0, 0.06]),
        high=np.array([1, 1, 0.93]),
        weights=[0.7, 0.3],
    )
    return pf.color.hsv_color(hsv=hsv)


def linen_color(rng: np.random.Generator) -> pf.Color:
    hsv = pf.random.mixture_of_gaussian(
        rng=rng,
        means=np.array([[0.12, 0.5, 0.55], [0.6, 0.4, 0.55], [0.9, 0.2, 0.55]]),
        stds=np.array([[0.007, 0.084, 0.2], [0.007, 0.063, 0.2], [0.007, 0.014, 0.2]]),
        weights=[0.8, 0.15, 0.05],
        low=np.array([0, 0, 0.15]),
        high=np.array([1, 1, 0.86]),
    )
    return pf.color.hsv_color(hsv=hsv)


def velvet_color(rng: np.random.Generator) -> pf.Color:
    hsv = pf.random.mixture_of_gaussian(
        rng=rng,
        means=np.array([[0.52, 0.45, 0.35]]),
        stds=np.array([[0.14, 0.14, 0.18]]),
        weights=[1.0],
        low=np.array([0, 0, 0.11]),
        high=np.array([1, 1, 0.70]),
    )
    return pf.color.hsv_color(hsv=hsv)


def bedding_sheet_color(rng: np.random.Generator) -> pf.Color:
    hsv = pf.random.mixture_of_gaussian(
        rng=rng,
        means=np.array([[0.1, 0.4, 0.66], [0.6, 0.2, 0.17]]),
        stds=np.array([[0.007, 0.56, 0.17], [0.021, 0.014, 0.17]]),
        weights=[0.9, 0.1],
        low=np.array([0, 0, 0.15]),
        high=np.array([1, 1, 0.94]),
    )
    return pf.color.hsv_color(hsv=hsv)


def petal_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.wrap_gaussian(rng, 0.95, 1.2, 0, 1)
    saturation = pf.random.uniform(rng, 0.2, 0.85)
    value = pf.random.uniform(rng, 0.2, 0.75)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def gem_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0, 1)
    saturation = 0.85
    value = pf.random.uniform(rng, 0.5, 1)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def plant_green_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0.25, 0.33)
    saturation = pf.random.normal(rng, 0.65, 0.03)
    value = pf.random.uniform(rng, 0.1, 0.45)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def plant_pink_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.wrap_gaussian(rng, 0.88, 0.06, 0, 1)
    saturation = pf.random.normal(rng, 0.6, 0.05)
    value = pf.random.normal(rng, 0.8, 0.05)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def plant_white_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.wrap_gaussian(rng, 0.0, 0.06, 0, 1)
    saturation = pf.random.uniform(rng, 0.0, 0.2)
    value = pf.random.normal(rng, 0.95, 0.02)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def plant_red_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.wrap_gaussian(rng, 0.0, 0.05, 0, 1)
    saturation = pf.random.normal(rng, 0.9, 0.03)
    value = pf.random.normal(rng, 0.6, 0.05)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def plant_yellow_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.wrap_gaussian(rng, 0.15, 0.005, 0, 1)
    saturation = pf.random.normal(rng, 0.95, 0.02)
    value = pf.random.normal(rng, 0.9, 0.02)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def tree_petal_color(rng: np.random.Generator) -> pf.Color:
    options = [
        (plant_pink_color, 0.4),
        (plant_white_color, 0.2),
        (plant_red_color, 0.2),
        (plant_yellow_color, 0.2),
    ]
    return pf.control.choice(rng, options)(rng)


def bedding_sheet_color_alt(rng: np.random.Generator) -> pf.Color:
    hsv = pf.random.mixture_of_gaussian(
        rng=rng,
        means=np.array([[0.1, 0.4, 0.66], [0.6, 0.2, 0.17]]),
        stds=np.array([[0.007, 0.56, 0.17], [0.021, 0.014, 0.17]]),
        weights=[0.9, 0.1],
        low=np.array([0, 0, 0.15]),
        high=np.array([1, 1, 0.94]),
    )
    return pf.color.hsv_color(hsv=hsv)


def petal_color_alt(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.wrap_gaussian(rng, 0.95, 1.2, 0, 1)
    saturation = pf.random.uniform(rng, 0.2, 0.85)
    value = pf.random.uniform(rng, 0.2, 0.75)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def gem_color_alt(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0, 1)
    saturation = 0.85
    value = pf.random.uniform(rng, 0.5, 1)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def plant_green_color_alt(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0.25, 0.33)
    saturation = pf.random.normal(rng, 0.65, 0.03)
    value = pf.random.uniform(rng, 0.1, 0.45)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def plant_pink_color_alt(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.wrap_gaussian(rng, 0.88, 0.06, 0, 1)
    saturation = pf.random.normal(rng, 0.6, 0.05)
    value = pf.random.normal(rng, 0.8, 0.05)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def plant_white_color_alt(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.wrap_gaussian(rng, 0.0, 0.06, 0, 1)
    saturation = pf.random.uniform(rng, 0.0, 0.2)
    value = pf.random.normal(rng, 0.95, 0.02)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def plant_red_color_alt(rng: np.random.Generator) -> pf.Color:
    logger.debug(f"RED ALT {rng=}")
    hue = pf.random.wrap_gaussian(rng, 0.0, 0.05, 0, 1)
    saturation = pf.random.normal(rng, 0.9, 0.03)
    value = pf.random.normal(rng, 0.6, 0.05)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def plant_yellow_color_alt(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.wrap_gaussian(rng, 0.15, 0.005, 0, 1)
    saturation = pf.random.normal(rng, 0.95, 0.02)
    value = pf.random.normal(rng, 0.9, 0.02)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def tree_petal_color_alt(rng: np.random.Generator) -> pf.Color:
    options = [
        (plant_pink_color_alt, 0.4),
        (plant_white_color_alt, 0.2),
        (plant_red_color_alt, 0.2),
        (plant_yellow_color_alt, 0.2),
    ]
    return pf.control.choice(rng, options)(rng)


def fog_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0, 1)
    saturation = pf.random.uniform(rng, 0, 0.2)
    value = pf.random.uniform(rng, 0.8, 1)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def water_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0.2, 0.6)
    saturation = pf.random.normal(rng, 0.5, 0.1)
    value = pf.random.uniform(rng, 0.7, 1)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def darker_water_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0.2, 0.6)
    saturation = pf.random.normal(rng, 0.5, 0.1)
    value = pf.random.uniform(rng, 0.2, 0.3)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def under_water_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0.5, 0.7)
    saturation = pf.random.uniform(rng, 0.7, 0.95)
    value = pf.random.uniform(rng, 0.7, 1)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def eye_schlera_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0.05, 0.15)
    saturation = pf.random.uniform(rng, 0.2, 0.8)
    value = pf.random.uniform(rng, 0.05, 0.5)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def eye_pupil_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0, 1)
    saturation = pf.random.uniform(rng, 0.1, 0.9)
    value = pf.random.uniform(rng, 0.1, 0.9)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def beak_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0, 0.13)
    saturation = pf.random.uniform(rng, 0, 0.9)
    value = pf.random.uniform(rng, 0.1, 0.6)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def fur_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0.05, 0.15)
    saturation = pf.random.uniform(rng, 0.2, 0.8)
    value = pf.random.uniform(rng, 0.05, 0.5)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def pine_needle_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0.25, 0.33)
    saturation = pf.random.normal(rng, 0.65, 0.03)
    value = pf.random.uniform(rng, 0.1, 0.45)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def wet_sand_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0.1, 0.15)
    saturation = pf.random.uniform(rng, 0.2, 0.4)
    value = pf.random.uniform(rng, 0.5, 0.7)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def dry_sand_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0.1, 0.15)
    saturation = pf.random.uniform(rng, 0.1, 0.3)
    value = pf.random.uniform(rng, 0.7, 0.9)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def concrete_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0, 0.1)
    saturation = pf.random.uniform(rng, 0, 0.1)
    value = pf.random.uniform(rng, 0.7, 0.9)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def textile_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0, 1)
    saturation = pf.random.uniform(rng, 0.2, 0.8)
    value = pf.random.uniform(rng, 0.2, 0.8)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def fabric_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0, 1)
    saturation = pf.random.uniform(rng, 0.2, 0.8)
    value = pf.random.uniform(rng, 0.2, 0.8)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def _unhex(colors: list[tuple[int, float]]) -> list[tuple[pf.Color, float]]:
    return [(pf.color.hex_color(h), w) for h, w in colors]


_WOOD_COLORS = _unhex(
    [
        # TODO: these discrete colors are discouraged and should be changed to a normal distribution / mixture of gaussians over HSV values
        (0x4C2F27, 1),
        (0x69432D, 1),
        (0x371803, 1),
        (0x7F4040, 1),
        (0xCC9576, 1),
        (0x9E8170, 1),
        (0x3D2B1F, 1),
        (0x8D6A58, 1),
        (0x8B3325, 1),
        (0x79443C, 1),
        (0x88540B, 1),
        (0x9B5F43, 1),
        (0x4E3828, 1),
        (0x4E3828, 1),
        (0xC09A6B, 1),
        (0x944536, 1),
        (0x3F0110, 1),
        (0x773C12, 1),
        (0x6E4E37, 1),
        (0x5C4033, 1),
        (0x5C4033, 1),
        (0x3C3034, 1),
        (0x96704C, 1),
        (0x371B1A, 1),
        (0x483B32, 1),
        (0x43141A, 1),
        (0x471713, 1),
        (0xC3B090, 1),
        (0x6B4423, 1),
        (0x674D46, 1),
        (0x5D2E1A, 1),
        (0x331C1F, 1),
        (0x7A5640, 1),
        (0xB99984, 1),
        (0x71543D, 1),
        (0x8F4B28, 1),
        (0x491A00, 1),
        (0x836446, 1),
        (0x7F461B, 1),
        (0x6A3208, 1),
        (0x724115, 1),
        (0xA0522B, 1),
        (0x832A0C, 1),
        (0x371B1A, 1),
        (0xC7A373, 1),
        (0x483B32, 1),
        (0x635147, 1),
        (0x664228, 1),
        (0x5C5248, 1),
    ]
)


def bark_color(rng: np.random.Generator) -> pf.Color:
    return pf.control.choice(rng, _WOOD_COLORS)


def wood_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0.02, 0.08)
    saturation = pf.random.uniform(rng, 0.2, 0.8)
    value = pf.random.log_uniform(rng, 0.1, 0.6)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


_METAL_PLAIN_BASE_COLORS = _unhex(
    [
        (0xFDD017, 0.5),
        (0xC0C0C0, 1),
        (0x8C7853, 1),
        (0xB87333, 0.5),
        (0xB5A642, 0.5),
        (0xBDBAAE, 1),
        (0xA9ACB6, 1),
        (0xB6AFA9, 1),
    ]
)


_METAL_NATURAL_BASE_COLORS = _unhex(
    [
        (0xC0C0C0, 1),
        (0x8C7853, 1),
        (0xBDBAAE, 1),
        (0xA9ACB6, 1),
        (0xB6AFA9, 1),
    ]
)


def metal_plain_color(rng: np.random.Generator) -> pf.Color:
    return pf.control.choice(rng, _METAL_PLAIN_BASE_COLORS)


def metal_natural_color(rng: np.random.Generator) -> pf.Color:
    return pf.control.choice(rng, _METAL_NATURAL_BASE_COLORS)


def metal_bw_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0, 1)
    saturation = pf.random.uniform(rng, 0.0, 0.2)
    value = pf.random.log_uniform(rng, 0.01, 0.2)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def metal_bw_natural_color(rng: np.random.Generator) -> pf.Color:
    options = [(metal_bw_color, 0.5), (metal_natural_color, 0.5)]
    return pf.control.choice(rng, options)(rng)


def alternate_metal_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0, 1)
    saturation = pf.random.uniform(rng, 0.3, 0.6)
    value = pf.random.log_uniform(rng, 0.02, 0.5)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def metal_color(rng: np.random.Generator) -> pf.Color:
    options = [(metal_natural_color, 0.2), (alternate_metal_color, 0.8)]
    return pf.control.choice(rng, options)(rng)


def white_color(rng: np.random.Generator) -> pf.Color:
    hue = 0.0
    saturation = pf.random.uniform(rng, 0, 0.1)
    value = pf.random.uniform(rng, 0.8, 1)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def ice_color(rng: np.random.Generator) -> pf.Color:
    hue = pf.random.uniform(rng, 0.59, 0.69)
    saturation = pf.random.uniform(rng, 0.22, 0.42)
    value = pf.random.uniform(rng, 0.85, 1.0)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


# =============================================================================
# Granite Mineral Colors
# Based on real granite geology: feldspar, quartz, mica, hornblende
# =============================================================================


def granite_feldspar_color(rng: np.random.Generator) -> pf.Color:
    """Feldspar color for granite - the most varied mineral determining overall granite color.

    Real feldspar colors:
    - K-feldspar (orthoclase): pink, salmon, red, peach
    - Plagioclase: white, cream, light gray
    - Rare: amazonite (green), labradorite (blue)
    """
    hsv = pf.random.mixture_of_gaussian(
        rng=rng,
        means=np.array(
            [
                [0.0, 0.35, 0.55],  # K-feldspar pink/salmon (most common)
                [0.05, 0.1, 0.75],  # Plagioclase white/cream
                [0.08, 0.45, 0.4],  # K-feldspar deeper salmon/peach
                [0.0, 0.1, 0.6],  # Plagioclase light gray
                [0.02, 0.55, 0.35],  # K-feldspar red (less common)
                [0.45, 0.35, 0.45],  # Rare: amazonite green
                [0.6, 0.25, 0.5],  # Rare: labradorite blue-gray
            ]
        ),
        stds=np.array(
            [
                [0.015, 0.1, 0.12],
                [0.02, 0.05, 0.1],
                [0.02, 0.1, 0.1],
                [0.03, 0.05, 0.1],
                [0.01, 0.1, 0.08],
                [0.03, 0.1, 0.1],
                [0.04, 0.08, 0.1],
            ]
        ),
        weights=[0.3, 0.25, 0.2, 0.15, 0.05, 0.03, 0.02],
        low=np.array([0.0, 0.0, 0.2]),
        high=np.array([1.0, 0.8, 0.9]),
    )
    return pf.color.hsv_color(hsv=hsv)


def granite_quartz_color(rng: np.random.Generator) -> pf.Color:
    """Quartz color for granite - typically light, translucent mineral.

    Real quartz colors:
    - Milky white/translucent (most common)
    - Smoky gray/brown
    - Rose quartz (rare pink tint)
    """
    hsv = pf.random.mixture_of_gaussian(
        rng=rng,
        means=np.array(
            [
                [0.0, 0.02, 0.8],  # Milky white/clear (most common)
                [0.08, 0.08, 0.65],  # Light gray
                [0.07, 0.12, 0.4],  # Smoky gray/brown
                [0.95, 0.15, 0.7],  # Rose quartz (rare)
            ]
        ),
        stds=np.array(
            [
                [0.05, 0.02, 0.1],
                [0.02, 0.04, 0.1],
                [0.02, 0.05, 0.1],
                [0.02, 0.05, 0.08],
            ]
        ),
        weights=[0.5, 0.3, 0.15, 0.05],
        low=np.array([0.0, 0.0, 0.2]),
        high=np.array([1.0, 0.3, 0.95]),
    )
    return pf.color.hsv_color(hsv=hsv)


def granite_mica_color(rng: np.random.Generator) -> pf.Color:
    """Mica color for granite - shiny flaky mineral.

    Real mica colors:
    - Biotite: black to dark brown (most common in granite)
    - Muscovite: silvery/golden (less common)
    - Phlogopite: bronze/brown
    """
    hsv = pf.random.mixture_of_gaussian(
        rng=rng,
        means=np.array(
            [
                [0.08, 0.4, 0.08],  # Biotite black-brown (most common)
                [0.0, 0.15, 0.05],  # Biotite nearly black
                [0.12, 0.5, 0.4],  # Muscovite golden
                [0.08, 0.55, 0.25],  # Phlogopite bronze
                [0.0, 0.0, 0.12],  # Pure black flakes
            ]
        ),
        stds=np.array(
            [
                [0.02, 0.12, 0.04],
                [0.03, 0.08, 0.03],
                [0.02, 0.12, 0.1],
                [0.02, 0.1, 0.08],
                [0.02, 0.02, 0.04],
            ]
        ),
        weights=[0.35, 0.25, 0.15, 0.15, 0.1],
        low=np.array([0.0, 0.0, 0.01]),
        high=np.array([1.0, 0.8, 0.55]),
    )
    return pf.color.hsv_color(hsv=hsv)


def granite_hornblende_color(rng: np.random.Generator) -> pf.Color:
    """Hornblende color for granite - dark mineral giving granite its speckled appearance.

    Real hornblende/amphibole colors:
    - Black to dark gray (most common)
    - Dark green (common in some granite types)
    - Dark brown
    """
    hsv = pf.random.mixture_of_gaussian(
        rng=rng,
        means=np.array(
            [
                [0.0, 0.1, 0.04],  # Near black (most common)
                [0.35, 0.35, 0.08],  # Dark green
                [0.08, 0.35, 0.1],  # Dark brown
                [0.4, 0.25, 0.05],  # Very dark green-black
            ]
        ),
        stds=np.array(
            [
                [0.05, 0.08, 0.025],
                [0.04, 0.1, 0.04],
                [0.03, 0.1, 0.04],
                [0.03, 0.08, 0.025],
            ]
        ),
        weights=[0.45, 0.25, 0.2, 0.1],
        low=np.array([0.0, 0.0, 0.005]),
        high=np.array([1.0, 0.6, 0.2]),
    )
    return pf.color.hsv_color(hsv=hsv)
