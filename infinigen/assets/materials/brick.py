# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from collections.abc import Iterable

import numpy as np
from numpy.random import uniform

from infinigen.assets.materials import common
from infinigen.assets.utils.object import new_plane
from infinigen.assets.utils.uv import unwrap_normal
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.util import blender as butil
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.random import log_uniform


def shader_brick(nw: NodeWrangler, height=None, **kwargs):
    if height is None:
        height = log_uniform(0.07, 0.12)
    uv_map = nw.new_node(Nodes.UVMap)

    front_color, back_color = [
        hsv2rgba(uniform(0, 0.05), uniform(0.8, 1), log_uniform(0.02, 0.5))
        for _ in range(2)
    ]
    mortar_color = hsv2rgba(uniform(0, 0.05), uniform(0.2, 0.5), log_uniform(0.02, 0.8))
    dark_color = hsv2rgba(uniform(0, 0.05), uniform(0.8, 1), log_uniform(0.005, 0.02))
    noise = nw.new_node(
        Nodes.NoiseTexture,
        [uv_map],
        input_kwargs={"Scale": uniform(40, 50), "Detail": uniform(15, 20)},
    )
    color = nw.new_node(
        Nodes.BrickTexture,
        [uv_map, front_color, back_color, mortar_color],
        input_kwargs={
            "Scale": 1,
            "Row Height": height,
            "Brick Width": height * log_uniform(1.2, 2.5),
            "Mortar Size": height * log_uniform(0.04, 0.08),
            "Mortar Smooth": noise,
        },
    ).outputs["Color"]
    noise = nw.new_node(
        Nodes.MusgraveTexture, [uv_map], input_kwargs={"Scale": uniform(2, 5)}
    )
    color = nw.new_node(
        Nodes.MixRGB,
        [nw.scalar_multiply(log_uniform(0.5, 1.0), noise), color, dark_color],
        attrs={"blend_type": "DARKEN"},
    )

    roughness = nw.build_float_curve(
        nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": 50}),
        [(0, 0.5), (1, 1.0)],
    )

    offset = nw.scalar_add(
        nw.scalar_multiply(nw.scalar_sub(color, 0.5), uniform(0.01, 0.04)),
        nw.scalar_multiply(
            nw.new_node(Nodes.MusgraveTexture, [uv_map], input_kwargs={"Scale": 50}),
            uniform(0.0, 0.01),
        ),
    )
    bump = nw.new_node(Nodes.Bump, input_kwargs={"Height": offset})
    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={"Roughness": roughness, "Base Color": color, "Normal": bump},
    )
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": principled_bsdf})


def apply(obj, selection=None, height=None, **kwargs):
    for o in obj if isinstance(obj, Iterable) else [obj]:
        unwrap_normal(o, selection, axis_="z")
    common.apply(obj, shader_brick, selection, height, **kwargs)


def make_sphere():
    obj = new_plane()
    obj.rotation_euler[0] = np.pi / 2
    butil.apply_transform(obj, True)
    return obj
