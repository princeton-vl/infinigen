# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han

from numpy.random import uniform, normal , randint
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category, hsv2rgba
from infinigen.core import surface


def shader_spider_plant(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    main_hsv_color = (uniform(0.18, 0.36), uniform(0.70, 0.90), uniform(0.2, 0.3))
    main_color = hsv2rgba(main_hsv_color)

    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
                                  input_kwargs={'Base Color': main_color, 'Subsurface IOR': 1.01,
                                                'Roughness': 2.0})

    translucent_bsdf = nw.new_node(Nodes.TranslucentBSDF,
                                   input_kwargs={'Color': main_color})

    mix_shader = nw.new_node(Nodes.MixShader,
                             input_kwargs={1: principled_bsdf, 2: translucent_bsdf})

    material_output = nw.new_node(Nodes.MaterialOutput,
                                  input_kwargs={'Surface': mix_shader})


def apply(obj, selection=None, **kwargs):
    surface.add_material(obj, shader_spider_plant, selection=selection)