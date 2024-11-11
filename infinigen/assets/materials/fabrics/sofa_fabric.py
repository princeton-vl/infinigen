# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from numpy.random import uniform

from infinigen.assets.materials import common
from infinigen.assets.utils.uv import unwrap_faces
from infinigen.core.nodes import Nodes, NodeWrangler
from infinigen.core.util.color import color_category


def shader_sofa_fabric(nw: NodeWrangler, scale=1, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute, attrs={"attribute_name": "UVMap"})
    attribute = nw.new_node(
        Nodes.Mapping, [attribute], input_kwargs={"Scale": [scale] * 3}
    )

    rgb = nw.new_node(Nodes.RGB)
    rgb.outputs[0].default_value = color_category("fabric")

    brightness_contrast = nw.new_node(
        "ShaderNodeBrightContrast",
        input_kwargs={"Color": rgb, "Bright": uniform(-0.1500, -0.05)},
    )

    brick_texture = nw.new_node(
        Nodes.BrickTexture,
        input_kwargs={
            "Vector": attribute.outputs["Vector"],
            "Color1": rgb,
            "Color2": brightness_contrast,
            "Scale": 276.9800,
            "Mortar Size": 0.0100,
            "Mortar Smooth": 1.0000,
            "Bias": 0.5000,
            "Row Height": 0.1000,
        },
        attrs={"offset": 0.5479, "squash_frequency": 1},
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": brick_texture.outputs["Color"],
            "Roughness": 0.8624,
            "Sheen Weight": 1.0000,
        },
    )

    displacement = nw.new_node(
        Nodes.Displacement, input_kwargs={"Height": brick_texture.outputs["Fac"]}
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled_bsdf, "Displacement": displacement},
        attrs={"is_active_output": True},
    )


def apply(obj, selection=None, **kwargs):
    unwrap_faces(obj, selection)
    common.apply(obj, shader_sofa_fabric, selection, **kwargs)
