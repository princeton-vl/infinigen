import numpy as np

from infinigen.assets import colors
from infinigen.assets.materials.utils import common
from infinigen.core.util.color import hsv2rgba

from . import (
    brushed_metal,
    galvanized_metal,
    grained_and_polished_metal,
    hammered_metal,
)


def get_shader():
    return np.random.choice(
        [
            brushed_metal.shader_brushed_metal,
            galvanized_metal.shader_galvanized_metal,
            grained_and_polished_metal.shader_grained_metal,
            hammered_metal.shader_hammered_metal,
        ]
    )


def apply(obj, selection=None, color_hsv=None, **kwargs):
    if color_hsv is None:
        color_hsv = colors.metal_hsv()
    shader = get_shader()
    common.apply(obj, shader, selection, base_color=hsv2rgba(color_hsv), **kwargs)
