# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from numpy.random import uniform

from infinigen.assets.materials import common
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler


def shader_metal(nw: NodeWrangler, color=None, **kwargs):
    position = nw.new_node(Nodes.TextureCoord).outputs["Object"]
    roughness = nw.build_float_curve(
        nw.new_node(
            Nodes.NoiseTexture, [position], input_kwargs={"Scale": uniform(10, 25)}
        ),
        [(0, uniform(0, 0.2)), (1, uniform(0.4, 0.7))],
    )
    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Metallic": 1.0,
            "Specular IOR Level": uniform(0.5, 1.0),
            "Base Color": color,
            "Roughness": roughness,
        },
    )
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": principled_bsdf})


def apply(obj, selection=None, **kwargs):
    from infinigen.assets.materials.metal import sample_metal_color

    color = sample_metal_color(**kwargs)
    common.apply(obj, shader_metal, selection, color, **kwargs)
