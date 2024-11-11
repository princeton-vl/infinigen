# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Meenal Parakh
# Acknowledgement: This file draws inspiration from following sources:

# https://www.youtube.com/watch?v=Aa8gf1pwb4E by Riley Brown
# https://www.youtube.com/watch?v=EQ149bMtKRA by Christopher Fraser
# https://www.youtube.com/watch?v=lDbsHpqKgoI by The DiNusty Empire
# https://www.youtube.com/watch?v=bLRwf2rZiAs by DECODED
# https://www.youtube.com/watch?v=NnlaIizA_AQ by Aryan
# https://www.youtube.com/watch?v=_wEXl3LncAc by diivja


import logging
from collections.abc import Iterable

from numpy.random import choice, uniform

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

logger = logging.getLogger(__name__)


def get_edge_wear_params():
    return {
        "_worn_off_opacity": uniform(0.01),
        "_worn_off_radius": uniform(0.005, 0.01),
        "_scratch_radius": uniform(0.01, 0.03),
        "_worn_off_mask_randomness": uniform(2.5, 3.0),
        "_edge_base_color_hue": uniform(0.0, 1.0),
        "_edge_base_color_whiteness": uniform(0.1, 0.6),
        "_scratch_mask_randomness": choice([uniform(0.1, 5.0), uniform(1.0, 10.0)]),
        "_scratch_density": choice([uniform(1.5, 10.0)]),
        "_scratch_opacity": uniform(0.5, 1.0),
    }


def shader_edge_tear_free_node_group(
    nw: NodeWrangler,
    original_bsdf,
    original_displacement,
    _worn_off_opacity=0.5,
    _worn_off_radius=0.015,
    _scratch_radius=0.01,
    _worn_off_mask_randomness=2.0,
    _edge_base_color_hue=1.0,
    _edge_base_color_whiteness=0.1,
    _scratch_mask_randomness=0.5,
    _scratch_density=5.0,
    _scratch_opacity=0.2,
):
    scratch_opacity = nw.new_node(Nodes.Value)
    scratch_opacity.outputs[0].default_value = _scratch_opacity

    scratch_mask_randomness = nw.new_node(Nodes.Value)
    scratch_mask_randomness.outputs[0].default_value = _scratch_mask_randomness

    scratch_radius = nw.new_node(Nodes.Value)
    scratch_radius.outputs[0].default_value = _scratch_radius

    worn_off_opacity = nw.new_node(Nodes.Value)
    worn_off_opacity.outputs[0].default_value = _worn_off_opacity

    paint_worn_off_radius = nw.new_node(Nodes.Value)
    paint_worn_off_radius.outputs[0].default_value = _worn_off_radius

    worn_off_mask_randomness = nw.new_node(Nodes.Value)
    worn_off_mask_randomness.outputs[0].default_value = _worn_off_mask_randomness

    edge_base_color_whiteness = nw.new_node(Nodes.Value)
    edge_base_color_whiteness.outputs[0].default_value = _edge_base_color_whiteness

    edge_base_color_hue = nw.new_node(Nodes.Value)
    edge_base_color_hue.outputs[0].default_value = _edge_base_color_hue

    scratch_density = nw.new_node(Nodes.Value)
    scratch_density.outputs[0].default_value = _scratch_density

    texture_coordinate = nw.new_node(Nodes.TextureCoord)

    mapping = nw.new_node(
        Nodes.Mapping, input_kwargs={"Vector": texture_coordinate.outputs["Object"]}
    )

    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "Scale": scratch_mask_randomness,
            "Detail": 1.0000,
        },
    )

    color_ramp = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": noise_texture.outputs["Fac"]}
    )
    color_ramp.color_ramp.elements[0].position = 0.4436
    color_ramp.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    color_ramp.color_ramp.elements[1].position = 0.5345
    color_ramp.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]

    bevel = nw.new_node(
        "ShaderNodeBevel",
        input_kwargs={"Radius": scratch_radius},
        attrs={"samples": 20},
    )

    geometry = nw.new_node(Nodes.NewGeometry)

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bevel, 1: geometry.outputs["Normal"]},
        attrs={"operation": "SUBTRACT"},
    )

    absolute = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract.outputs["Vector"]},
        attrs={"operation": "ABSOLUTE"},
    )

    color_ramp_1 = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": absolute})
    color_ramp_1.color_ramp.elements[0].position = 0.0691
    color_ramp_1.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    color_ramp_1.color_ramp.elements[1].position = 0.1564
    color_ramp_1.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: color_ramp.outputs["Color"], 1: color_ramp_1.outputs["Color"]},
        attrs={"use_clamp": True, "operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: scratch_opacity, 1: multiply},
        attrs={"use_clamp": True, "operation": "MULTIPLY"},
    )

    bevel_1 = nw.new_node(
        "ShaderNodeBevel",
        input_kwargs={"Radius": paint_worn_off_radius},
        attrs={"samples": 20},
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bevel_1, 1: geometry.outputs["Normal"]},
        attrs={"operation": "SUBTRACT"},
    )

    absolute_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1.outputs["Vector"]},
        attrs={"operation": "ABSOLUTE"},
    )

    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping,
            "Scale": worn_off_mask_randomness,
            "Detail": 1.0000,
        },
    )

    color_ramp_2 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": noise_texture_1.outputs["Fac"]}
    )
    color_ramp_2.color_ramp.elements[0].position = 0.0764
    color_ramp_2.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    color_ramp_2.color_ramp.elements[1].position = 0.5709
    color_ramp_2.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: absolute_1, 1: color_ramp_2.outputs["Color"]},
        attrs={"use_clamp": True, "operation": "MULTIPLY"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: worn_off_opacity, 1: multiply_2},
        attrs={"use_clamp": True, "operation": "MULTIPLY"},
    )

    color_ramp_3 = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": multiply_3})
    color_ramp_3.color_ramp.elements[0].position = 0.0000
    color_ramp_3.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    color_ramp_3.color_ramp.elements[1].position = 0.7782
    color_ramp_3.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]

    combine_color = nw.new_node(
        Nodes.CombineColor,
        input_kwargs={"Red": edge_base_color_hue, "Green": 0.7733, "Blue": 0.0100},
        attrs={"mode": "HSV"},
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: edge_base_color_whiteness,
            6: combine_color,
            7: (0.02, 0.02, 0.02, 1.0000),
        },
        attrs={"clamp_result": True, "data_type": "RGBA", "clamp_factor": False},
    )

    reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": mix.outputs[2]})

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": reroute,
            "Metallic": 0.3745,
            "Specular IOR Level": 0.0000,
            "Roughness": 0.1436,
        },
    )

    mix_shader = nw.new_node(
        Nodes.MixShader,
        input_kwargs={
            "Fac": color_ramp_3.outputs["Color"],
            1: original_bsdf,
            2: principled_bsdf,
        },
    )

    mapping_1 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "Scale": (10.0000, 1.0000, 1.0000),
        },
    )

    voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={"Vector": mapping_1, "Scale": scratch_density},
        attrs={"feature": "DISTANCE_TO_EDGE"},
    )

    mapping_2 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "Scale": (1.0000, 10.0000, 1.0000),
        },
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: scratch_density, "Scale": 2.0000},
        attrs={"operation": "SCALE"},
    )

    voronoi_texture_1 = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={"Vector": mapping_2, "Scale": scale.outputs["Vector"]},
        attrs={"feature": "DISTANCE_TO_EDGE"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: voronoi_texture.outputs["Distance"],
            1: voronoi_texture_1.outputs["Distance"],
        },
        attrs={"use_clamp": True, "operation": "MULTIPLY"},
    )

    color_ramp_6 = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": multiply_4})
    color_ramp_6.color_ramp.elements[0].position = 0.0000
    color_ramp_6.color_ramp.elements[0].color = [1.0000, 1.0000, 1.0000, 1.0000]
    color_ramp_6.color_ramp.elements[1].position = 0.0073
    color_ramp_6.color_ramp.elements[1].color = [0.0000, 0.0000, 0.0000, 1.0000]

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: color_ramp_1.outputs["Color"],
            1: color_ramp_6.outputs["Color"],
        },
        attrs={"use_clamp": True, "operation": "MULTIPLY"},
    )

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply, 1: multiply_5},
        attrs={"use_clamp": True, "operation": "MULTIPLY"},
    )

    principled_bsdf_1 = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": reroute,
            "Metallic": 0.3855,
            "Specular IOR Level": 0.0000,
            "Roughness": 0.0000,
        },
    )

    mix_shader_1 = nw.new_node(
        Nodes.MixShader,
        input_kwargs={"Fac": multiply_1, 1: mix_shader, 2: principled_bsdf_1},
    )

    # add operation
    scale_multiply6 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply_6, "Scale": 2.0000},
        attrs={"operation": "SCALE"},
    )

    if original_displacement is None:
        total_displacement = scale_multiply6
    else:
        total_displacement = nw.new_node(
            Nodes.Math,
            input_kwargs={0: original_displacement, 1: scale_multiply6},
            attrs={"operation": "ADD", "use_clamp": True},
        )

    return mix_shader_1, total_displacement


MARKER_LABEL = "wear_tear"


def apply_over(obj, selection=None, **shader_kwargs):
    # get all materials
    # https://blenderartists.org/t/finding-out-if-an-object-has-a-material/512570/6
    materials = obj.data.materials.items()
    if len(materials) == 0:
        logging.warning(
            f"No material exist for {obj.name}! Scratches can only be applied over some existing material."
        )
        return

    if len(shader_kwargs) == 0:
        logging.debug("Obtaining Randomized Scratch Parameters")
        shader_kwargs = get_edge_wear_params()

    for material_name, material in materials:
        # get material node tree
        # https://blender.stackexchange.com/questions/240278/how-to-access-shader-node-via-python-script
        material_node_tree = material.node_tree

        if any([node.label == MARKER_LABEL for node in material_node_tree.nodes]):
            continue

        nw = NodeWrangler(material_node_tree)

        result = nw.find("ShaderNodeOutputMaterial")
        if len(result) == 0:
            logger.warning(
                "No Material Output Node found in the object's materials! Returning"
            )
            continue

        # get nodes and links connected to specific inputs
        # https://blender.stackexchange.com/questions/5462/is-it-possible-to-find-the-nodes-connected-to-a-node-in-python
        initial_bsdf = result[0].inputs["Surface"].links[0].from_node
        displacement_links = result[0].inputs["Displacement"].links

        if len(displacement_links) == 0:
            initial_displacement = None
        else:
            initial_displacement = result[0].inputs["Displacement"].links[0].from_node

        final_bsdf, final_displacement = shader_edge_tear_free_node_group(
            nw, initial_bsdf, initial_displacement, **shader_kwargs
        )
        # connecting nodes
        # https://blender.stackexchange.com/questions/101820/how-to-add-remove-links-to-existing-or-new-nodes-using-python
        material_node_tree.links.new(final_bsdf.outputs[0], result[0].inputs["Surface"])
        material_node_tree.links.new(
            final_displacement.outputs[0], result[0].inputs["Displacement"]
        )

        final_bsdf.label = MARKER_LABEL

    return


def apply(obj):
    if not isinstance(obj, Iterable):
        obj = [obj]
    for o in obj:
        apply_over(o)
