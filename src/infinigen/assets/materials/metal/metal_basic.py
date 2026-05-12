# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from numpy.random import uniform

from infinigen.assets import colors
from infinigen.assets.materials.utils import common
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler


def shader_metal(nw: NodeWrangler, color_hsv=None, **kwargs):
    position = nw.new_node(Nodes.TextureCoord).outputs["Object"]
    roughness = nw.build_float_curve(
        nw.new_node(
            Nodes.NoiseTexture, [position], input_kwargs={"Scale": uniform(10, 25)}
        ),
        [(0, uniform(0, 0.2)), (1, uniform(0.4, 0.7))],
    )
    color_hsv = color_hsv or colors.metal_hsv()
    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Metallic": 1.0,
            "Specular IOR Level": uniform(0.5, 1.0),
            "Base Color": colors.hsv2rgba(color_hsv),
            "Roughness": roughness,
        },
    )
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": principled_bsdf})


class MetalBasic:
    shader = shader_metal

    def generate(self, selection=None, color_hsv=None, **kwargs):
        color_hsv = color_hsv or colors.metal_hsv()
        return surface.shaderfunc_to_material(shader_metal, color_hsv)

    def apply(self, obj, selection=None, **kwargs):
        common.apply(obj, shader_metal, selection, **kwargs)

    __call__ = generate
