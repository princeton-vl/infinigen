# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=wTzk9T06gdw by Ryan King Art

import numpy as np
from numpy.random import uniform

from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


def shader_material(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    geometry = nw.new_node(Nodes.NewGeometry)

    mapping = nw.new_node(
        Nodes.Mapping, input_kwargs={"Vector": geometry.outputs["Position"]}
    )

    roughness = nw.new_node(Nodes.Value, label="roughness ~ U(0.5,0.7)")
    roughness.outputs[0].default_value = uniform(0.5, 0.7)

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "Scale": 2.0000,
            "Detail": 9.0000,
            "Roughness": roughness,
        },
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
        Nodes.VectorMath,
        input_kwargs={
            0: noise_texture.outputs["Color"],
            1: dot_product.outputs["Value"],
        },
    )

    voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture, input_kwargs={"Vector": add.outputs["Vector"]}
    )

    colorramp = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": voronoi_texture.outputs["Distance"]}
    )
    colorramp.color_ramp.elements[0].position = uniform(0.4, 0.5)
    colorramp.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    colorramp.color_ramp.elements[1].position = 0.9600
    colorramp.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": colorramp.outputs["Color"],
            "Metallic": 0.5000,
            "Roughness": 0.0000,
        },
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf},
        attrs={"is_active_output": True},
    )


def apply(obj, selection=None, **kwargs):
    surface.add_material(obj, shader_material, selection=selection)
