# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import numpy as np
from numpy.random import uniform

from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.color import hsv2rgba, rgb2hsv
from infinigen.core.util.random import log_uniform


def perturb(hsv):
    return np.array(
        [
            hsv[0] + uniform(-0.02, 0.02),
            hsv[1] + uniform(-0.2, 0.2),
            hsv[2] * log_uniform(0.5, 2.0),
        ]
    )


def shader_wood(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: texture_coordinate.outputs["Object"]},
        attrs={"operation": "SCALE"},
    )

    vector_rotate = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={"Vector": scale.outputs["Vector"]},
        attrs={"rotation_type": "EULER_XYZ"},
    )

    mapping_2 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={"Vector": vector_rotate, "Scale": (5.0000, 100.0000, 100.0000)},
    )

    seed = nw.new_node(Nodes.Value, label="seed")
    seed.outputs[0].default_value = 0.0000

    musgrave_texture_2 = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": mapping_2,
            "W": seed,
            "Scale": 10.0000,
            "Detail": 15.0000,
            "Dimension": 7.0000,
        },
        attrs={"musgrave_dimensions": "4D"},
    )

    map_range_2 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": musgrave_texture_2, 3: 1.0000, 4: -1.0000},
    )

    mapping_1 = nw.new_node(Nodes.Mapping, input_kwargs={"Vector": vector_rotate})

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping_1,
            "W": seed,
            "Scale": 0.5000,
            "Detail": 1.0000,
            "Distortion": 1.1000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    musgrave_texture_1 = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "W": seed,
            "Scale": noise_texture_1.outputs["Fac"],
            "Detail": 15.0000,
            "Dimension": 0.2000,
            "Lacunarity": 2.4000,
        },
        attrs={"musgrave_dimensions": "4D"},
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": musgrave_texture_1, 3: -1.4000, 4: 1.5000},
    )

    map_range_1 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": map_range.outputs["Result"], 3: 1.0000, 4: 0.5000},
    )

    mapping = nw.new_node(
        Nodes.Mapping,
        input_kwargs={"Vector": vector_rotate, "Scale": (0.1500, 1.0000, 0.1500)},
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "W": seed,
            "Detail": 5.0000,
            "Distortion": 1.0000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    musgrave_texture = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": noise_texture.outputs["Fac"],
            "W": seed,
            "Scale": 4.0000,
            "Detail": 10.0000,
            "Dimension": 0.0000,
        },
        attrs={"musgrave_dimensions": "4D"},
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={6: noise_texture.outputs["Fac"], 7: musgrave_texture},
        attrs={"data_type": "RGBA"},
    )

    mix_1 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 0.9000, 6: map_range_1.outputs["Result"], 7: mix.outputs[2]},
        attrs={"data_type": "RGBA", "blend_type": "MULTIPLY"},
    )

    mix_2 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: 0.9500, 6: map_range_2.outputs["Result"], 7: mix_1.outputs[2]},
        attrs={"data_type": "RGBA", "blend_type": "MULTIPLY"},
    )

    rgb = nw.new_node(Nodes.RGB)
    rgb.outputs[0].default_value = hsv2rgba(perturb(rgb2hsv(0.0242, 0.0056, 0.0027)))

    rgb_1 = nw.new_node(Nodes.RGB)
    rgb_1.outputs[0].default_value = hsv2rgba(perturb(rgb2hsv(0.5089, 0.2122, 0.0685)))

    mix_3 = nw.new_node(
        Nodes.Mix,
        input_kwargs={0: mix_2.outputs[2], 6: rgb, 7: rgb_1},
        attrs={"data_type": "RGBA"},
    )

    displacement = nw.new_node(
        Nodes.Displacement,
        input_kwargs={"Height": mix_2.outputs[2], "Midlevel": 0.0, "Scale": 0.05},
    )

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={"Base Color": mix_3.outputs[2]},
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={
            "Surface": principled_bsdf,
            "Displacement": displacement,
        },
        attrs={"is_active_output": True},
    )


class TableWood:
    shader = shader_wood

    def generate(self):
        return surface.shaderfunc_to_material(shader_wood)

    __call__ = generate
