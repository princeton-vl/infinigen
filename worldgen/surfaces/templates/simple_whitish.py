# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han
# Date Signed: June 15, 2023

from numpy.random import uniform as U, normal as N, randint
from nodes.node_wrangler import Nodes, NodeWrangler
from nodes import node_utils
from nodes.color import color_category
from surfaces import surface


def shader_simple_white(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    def noise():
        return nw.new_node(Nodes.NoiseTexture, attrs={'noise_dimensions': '4D'},
                           input_kwargs={'W': U(0, 100), 'Scale': N(60, 25), 'Detail': U(0, 10), 'Roughness': U(0, 1),
                                         'Distortion': U(0, 3)})

    rough = nw.new_node(Nodes.MapRange, attrs={'interpolation_type': 'SMOOTHSTEP'},
                        input_kwargs={'Value': noise(), 3: U(0.1, 0.8), 4: U(0.1, 0.8)})
    v = U(0.7, 1.0)
    base_color = (v * (1. + N(0, 0.05)), v * (1. + N(0, 0.05)), v * (1. + N(0, 0.05)), 1.)
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
                                  input_kwargs={'Base Color': base_color, 'Roughness': rough.outputs["Result"]})

    material_output = nw.new_node(Nodes.MaterialOutput,
                                  input_kwargs={'Surface': principled_bsdf})


def apply(obj, selection=None, **kwargs):
    surface.add_material(obj, shader_simple_white, selection=selection)