# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Mingzhe Wang, Alex Raistrick


from numpy.random import uniform as U

from infinigen.assets.materials.utils.surface_utils import (
    sample_range,
    sample_ratio,
)
from infinigen.assets.utils.nodegroups.shader import nodegroup_color_mask
from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.color import hsv2rgba


def shader_giraffe_attr(nw: NodeWrangler, rand=True, **input_kwargs):
    # Code generated using version 2.4.3 of the node_transpiler

    attribute = nw.new_node(Nodes.Attribute, attrs={"attribute_name": "local_pos"})

    noise_texture = nw.new_node(
        Nodes.NoiseTexture, input_kwargs={"Vector": attribute.outputs["Color"]}
    )

    mix = nw.new_node(
        Nodes.MixRGB,
        input_kwargs={
            "Fac": 0.9,
            "Color1": noise_texture.outputs["Color"],
            "Color2": attribute.outputs["Color"],
        },
    )

    mapping = nw.new_node(Nodes.Mapping, input_kwargs={"Vector": mix})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 10.0
    if rand:
        value.outputs[0].default_value = sample_ratio(
            value.outputs[0].default_value, 0.5, 2
        )

    voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={"Vector": mapping, "Scale": value},
        attrs={"voronoi_dimensions": "2D"},
    )

    voronoi_texture_4 = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={"Vector": mapping, "Scale": value},
        attrs={"voronoi_dimensions": "2D", "feature": "SMOOTH_F1"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: voronoi_texture.outputs["Distance"],
            1: voronoi_texture_4.outputs["Distance"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    less_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: sample_range(0.04, 0.08) if rand else 0.07},
        attrs={"operation": "LESS_THAN"},
    )

    colorramp_1 = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": less_than})
    colorramp_1.color_ramp.elements[0].position = 0.2545
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.2886
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    group = nw.new_node(nodegroup_color_mask().name)

    colorramp = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": group})
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.9301, 0.5647, 0.3372, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (0.9755, 1.0, 0.9096, 1.0)
    if rand:
        colorramp.color_ramp.elements[0].color = hsv2rgba(
            (U(0.02, 0.06), U(0.4, 0.8), U(0.15, 0.7))
        )
        colorramp.color_ramp.elements[1].color = hsv2rgba(
            (U(0.02, 0.06), U(0.4, 0.8), U(0.15, 0.7))
        )

    mix_1 = nw.new_node(
        Nodes.MixRGB,
        input_kwargs={
            "Fac": colorramp_1.outputs["Color"],
            "Color1": colorramp.outputs["Color"],
            "Color2": hsv2rgba((U(0.02, 0.06), U(0.4, 0.9), U(0.04, 0.1))),
        },
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={"Base Color": mix_1},
        attrs={"subsurface_method": "BURLEY"},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput, input_kwargs={"Surface": principled_bsdf}
    )


class Giraffe:
    def apply(self, obj, geo_kwargs=None, shader_kwargs=None, **kwargs):
        surface.add_material(
            obj, shader_giraffe_attr, reuse=False, input_kwargs=shader_kwargs
        )
