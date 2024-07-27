# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma
# Acknowledgement: This file draws inspiration from https://physbam.stanford.edu/cs448x/old/Procedural_Noise(2f)Perlin_Noise.html

import numpy as np
from numpy.random import uniform

from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


def shader_material_001(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    geometry = nw.new_node(Nodes.NewGeometry)

    mapping = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": geometry.outputs["Position"],
            "Scale": (20.0000, 20.0000, 20.0000),
        },
    )

    roughness = nw.new_node(Nodes.Value, label="roughness ~ U(0.7,0.9)")
    roughness.outputs[0].default_value = uniform(0.7, 0.9)

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "Scale": 0.1000,
            "Detail": 9.0000,
            "Roughness": roughness,
            "Distortion": 0.2000,
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: noise_texture.outputs["Fac"], 1: 20.0000},
        attrs={"operation": "MULTIPLY"},
    )

    random_plane_angle = uniform(0, 2 * np.pi)

    dot_product = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: mapping,
            1: (np.cos(random_plane_angle), np.sin(random_plane_angle), 0.0000),
        },
        attrs={"operation": "DOT_PRODUCT"},
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply, 1: dot_product.outputs["Value"]}
    )

    sine = nw.new_node(Nodes.Math, input_kwargs={0: add}, attrs={"operation": "SINE"})

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: sine, 1: 1.0000})

    darkness = nw.new_node(Nodes.Value, label="darkness ~ U(0,1)")
    darkness.outputs[0].default_value = uniform(0.0, 1.0)

    map_range = nw.new_node(
        Nodes.MapRange, input_kwargs={"Value": darkness, 3: 0.2000, 4: 0.3000}
    )

    power = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_1, 1: map_range.outputs["Result"]},
        attrs={"operation": "POWER"},
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF, input_kwargs={"Base Color": power}
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf},
        attrs={"is_active_output": True},
    )


def apply(obj, selection=None, **kwargs):
    surface.add_material(obj, shader_material_001, selection=selection)
