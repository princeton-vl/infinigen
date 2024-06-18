# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from numpy.random import uniform

from infinigen.core.util.color import hsv2rgba
from infinigen.assets.materials import common
from infinigen.core.util.random import log_uniform
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler


def shader_ceramic(nw: NodeWrangler, clear=False, roughness_min=0, roughness_max=.8, **kwargs):
    if uniform(0, 1) < .8 and not clear:
        color = hsv2rgba(uniform(0, 1), uniform(.2, .4), log_uniform(.3, .6))
    else:
        color = hsv2rgba(0, 0, log_uniform(.3, .6))

    roughness = nw.build_float_curve(nw.musgrave(log_uniform(20, 40)), [(0, roughness_min), (1, roughness_max)])
    clearcoat_roughness = nw.build_float_curve(nw.musgrave(log_uniform(20, 40)),
                                               [(0, roughness_min), (1, roughness_max)])
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
        "Roughness": roughness,
        'Clearcoat': 1,
        'Clearcoat Roughness': clearcoat_roughness,
        'Specular': 1,
        'Base Color': color,
        'Subsurface': uniform(.02, .05),
        'Subsurface Radius': (.02, .02, .02)
    })

    displacement = nw.new_node('ShaderNodeDisplacement', input_kwargs={
        'Height': nw.scalar_multiply(log_uniform(.001, .005), nw.new_node(Nodes.NoiseTexture,
                                                       input_kwargs={'Scale': log_uniform(20, 40)})),
        'Midlevel': 0.0000
    })

    nw.new_node(Nodes.MaterialOutput, input_kwargs={'Surface': principled_bsdf, 'Displacement': displacement})


def apply(obj, selection=None, clear=False, **kwargs):
    common.apply(obj, shader_ceramic, selection, clear, **kwargs)
