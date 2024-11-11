# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo
# Acknowledgement: This file draws inspiration https://www.youtube.com/watch?v=82smQvoh0GE by Mix CG Arts

from numpy.random import uniform

from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.random import log_uniform


@node_utils.to_nodegroup(
    "nodegroup_hammered_metal", singleton=False, type="ShaderNodeTree"
)
def nodegroup_hammered_metal(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketColor", "Base Color", (0.8000, 0.8000, 0.8000, 1.0000)),
            ("NodeSocketFloat", "Scale", 0.0000),
            ("NodeSocketFloat", "Seed", 0.0000),
        ],
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": group_input.outputs["Base Color"],
            "Metallic": 1.0000,
            "Specular IOR Level": 0.0000,
            "Roughness": 0.1000,
        },
    )

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Scale"], 1: 20.0000},
        attrs={"operation": "MULTIPLY"},
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "W": group_input.outputs["Seed"],
            "Scale": multiply,
            "Detail": 15.0000,
            "Roughness": 0.4000,
            "Distortion": 0.2000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: 0.0100,
            6: texture_coordinate.outputs["Object"],
            7: noise_texture.outputs["Color"],
        },
        attrs={"clamp_factor": False, "data_type": "RGBA"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Scale"], 1: 300.0000},
        attrs={"operation": "MULTIPLY"},
    )

    voronoi_texture_1 = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={
            "Vector": mix.outputs[2],
            "W": group_input.outputs["Seed"],
            "Scale": multiply_1,
            "Smoothness": 0.2000,
        },
        attrs={"voronoi_dimensions": "4D", "feature": "SMOOTH_F1"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: voronoi_texture_1.outputs["Distance"],
            1: group_input.outputs["Scale"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    power = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_2, 1: 2.5000},
        attrs={"operation": "POWER"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: power, 1: log_uniform(0.001, 0.003)},
        attrs={"operation": "MULTIPLY"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "BSDF": principled_bsdf,
            "Displacement": multiply_3,
            "tmp_viewer": voronoi_texture_1.outputs["Color"],
        },
        attrs={"is_active_output": True},
    )


def shader_hammered_metal(
    nw: NodeWrangler, scale=None, base_color=None, seed=None, **kwargs
):
    # Code generated using version 2.6.4 of the node_transpiler
    if seed is None:
        seed = uniform(-1000.0, 1000.0)
    if base_color is None:
        from infinigen.assets.materials.metal import sample_metal_color

        base_color = sample_metal_color(**kwargs)
    if scale is None:
        scale = log_uniform(0.8, 1.2)

    group = nw.new_node(
        nodegroup_hammered_metal().name,
        input_kwargs={"Base Color": base_color, "Scale": scale, "Seed": seed},
    )

    displacement = nw.new_node(
        "ShaderNodeDisplacement",
        input_kwargs={"Height": group.outputs["Displacement"], "Midlevel": 0.0000},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": group.outputs["BSDF"], "Displacement": displacement},
        attrs={"is_active_output": True},
    )


def apply(obj, selection=None, **kwargs):
    surface.add_material(
        obj, shader_hammered_metal, selection=selection, input_kwargs=kwargs
    )
