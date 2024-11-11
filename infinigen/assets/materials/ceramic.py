# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from numpy.random import uniform

from infinigen.assets.materials import common
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.random import log_uniform


def shader_ceramic(
    nw: NodeWrangler, clear=False, roughness_min=0, roughness_max=0.8, **kwargs
):
    if uniform(0, 1) < 0.8 and not clear:
        color = hsv2rgba(uniform(0, 1), uniform(0.2, 0.4), log_uniform(0.3, 0.6))
    else:
        color = hsv2rgba(0, 0, log_uniform(0.3, 0.6))

    roughness = nw.build_float_curve(
        nw.musgrave(log_uniform(20, 40)), [(0, roughness_min), (1, roughness_max)]
    )
    clearcoat_roughness = nw.build_float_curve(
        nw.musgrave(log_uniform(20, 40)), [(0, roughness_min), (1, roughness_max)]
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Roughness": roughness,
            "Coat Weight": 1,
            "Coat Roughness": clearcoat_roughness,
            "Specular IOR Level": 1,
            "Base Color": color,
            "Subsurface Weight": uniform(0.02, 0.05),
            "Subsurface Radius": (0.02, 0.02, 0.02),
        },
    )

    displacement = nw.new_node(
        "ShaderNodeDisplacement",
        input_kwargs={
            "Height": nw.scalar_multiply(
                log_uniform(0.001, 0.005),
                nw.new_node(
                    Nodes.NoiseTexture, input_kwargs={"Scale": log_uniform(20, 40)}
                ),
            ),
            "Midlevel": 0.0000,
        },
    )

    nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf, "Displacement": displacement},
    )


def apply(obj, selection=None, clear=False, **kwargs):
    common.apply(obj, shader_ceramic, selection, clear, **kwargs)
