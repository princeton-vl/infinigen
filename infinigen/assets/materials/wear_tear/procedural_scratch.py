# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Meenal Parakh
# Acknowledgement: This file draws inspiration from following sources:

# https://www.youtube.com/watch?v=l0whu3494_c by Ryan King Art
# https://www.youtube.com/watch?v=qMCuDjXjsZ0 by Ryan King Art
# https://www.youtube.com/watch?v=L3SvNpjIERs by Sina Sinaie
# https://www.youtube.com/watch?v=0B-lexp10jk by Holmes Motion
# https://www.youtube.com/watch?v=ewq69iNRdmQ by Ryan King Art
# https://www.youtube.com/watch?v=MH8iutCKtYc by ChuckCG


import logging
from collections.abc import Iterable

from numpy.random import uniform

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.random import log_uniform

logger = logging.getLogger(__name__)


def get_scratch_params():
    return {
        "angle1": uniform(10.0, 80.0),
        "angle2": uniform(-80.0, -10.0),
        "scratch_scale": log_uniform(5, 80),
        "scratch_mask_ratio": log_uniform(0.01, 0.9),
        "scratch_mask_noise": log_uniform(5, 40),
        "scratch_depth": log_uniform(0.1, 1.0),
    }


def scratch_shader(
    nw: NodeWrangler,
    original_bsdf,
    angle1=45.0,
    angle2=-20.0,
    scratch_scale=20.0,
    scratch_mask_ratio=0.8,
    scratch_mask_noise=10.0,
    scratch_depth=0.1,
):
    # Code generated using version 2.6.5 of the node_transpiler

    texture_coordinate_1 = nw.new_node(Nodes.TextureCoord)

    n_angle1 = nw.new_node(Nodes.Value)
    n_angle1.outputs[0].default_value = angle1

    n_angle2 = nw.new_node(Nodes.Value)
    n_angle2.outputs[0].default_value = angle2

    n_scratch_scale = nw.new_node(Nodes.Value)
    n_scratch_scale.outputs[0].default_value = scratch_scale

    n_scratch_mask_ratio = nw.new_node(Nodes.Value)
    n_scratch_mask_ratio.outputs[0].default_value = scratch_mask_ratio

    n_scratch_mask_noise = nw.new_node(Nodes.Value)
    n_scratch_mask_noise.outputs[0].default_value = scratch_mask_noise

    n_scratch_depth = nw.new_node(Nodes.Value)
    n_scratch_depth.outputs[0].default_value = scratch_depth

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": n_angle1})

    mapping_1 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate_1.outputs["Object"],
            "Rotation": combine_xyz,
            "Scale": (25.0000, 1.0000, 1.0000),
        },
        attrs={"vector_type": "TEXTURE"},
    )

    noise_texture_3 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping_1,
            "Scale": n_scratch_scale,
            "Detail": 15.0000,
            "Roughness": 0.0000,
            "Distortion": 22.8000,
        },
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": n_angle2})

    mapping_2 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate_1.outputs["Object"],
            "Rotation": combine_xyz_1,
            "Scale": (25.0000, 1.0000, 1.0000),
        },
        attrs={"vector_type": "TEXTURE"},
    )

    noise_texture_5 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping_2,
            "Scale": n_scratch_scale,
            "Detail": 15.0000,
            "Roughness": 0.0000,
            "Distortion": 22.8000,
        },
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: noise_texture_3.outputs["Fac"],
            1: noise_texture_5.outputs["Fac"],
        },
    )

    mapping_3 = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            "Vector": texture_coordinate_1.outputs["Object"],
            "Rotation": (0.1588, -0.5742, 0.1920),
        },
        attrs={"vector_type": "TEXTURE"},
    )

    noise_texture_6 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": mapping_3,
            "Scale": n_scratch_mask_noise,
            "Detail": 1.0000,
        },
    )

    color_ramp_2 = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": noise_texture_6.outputs["Fac"]}
    )
    color_ramp_2.color_ramp.elements[0].position = 0.4109
    color_ramp_2.color_ramp.elements[0].color = [0.0000, 0.0000, 0.0000, 1.0000]
    color_ramp_2.color_ramp.elements[1].position = 1.0000
    color_ramp_2.color_ramp.elements[1].color = [1.0000, 1.0000, 1.0000, 1.0000]

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: n_scratch_mask_ratio, 1: color_ramp_2.outputs["Color"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: multiply}, attrs={"use_clamp": True}
    )

    map_range = nw.new_node(
        Nodes.MapRange, input_kwargs={"Value": add_1, 1: 0.7000, 2: 0.7200, 4: 0.9000}
    )
    # scaled_scratch = nw.new_node(Nodes.Math, input_kwargs={0: n_scratch_depth, 1: map_range.outputs["Result"]}, attrs={'operation': 'MULTIPLY'})

    # material_output = nw.new_node(Nodes.MaterialOutput,
    #                               input_kwargs={'Surface': original_bsdf, 'Displacement': scaled_scratch},
    #                               attrs={'is_active_output': True})

    # return material_output

    bump = nw.new_node(
        Nodes.Bump,
        input_kwargs={
            "Strength": n_scratch_depth,
            "Height": map_range.outputs["Result"],
        },
    )
    return {"Normal": bump}


def find_normal_input(bsdf):
    for i, o in enumerate(bsdf.inputs):
        if o.name == "Normal":
            return i
    logger.debug(f"Normal not found for {bsdf}")
    return None


MARKER_LABEL = "scratch"


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
        shader_kwargs = get_scratch_params()

    for material_name, material in materials:
        # get material node tree
        # https://blender.stackexchange.com/questions/240278/how-to-access-shader-node-via-python-script
        material_node_tree = material.node_tree

        if any([n.label == MARKER_LABEL for n in material_node_tree.nodes]):
            logging.warning(f"Scratch already applied to {material_name}! Skipping")
            continue

        nw = NodeWrangler(material_node_tree)

        result = nw.find_recursive("ShaderNodeBsdf")
        if len(result) == 0:
            logging.debug("No BSDF found in the object's materials! Returning")
            continue

        nw_bsdf, bsdf = result[-1]
        # final_bsdf = scratch_shader(nw_bsdf, bsdf, **shader_kwargs)

        if "Normal" in bsdf.inputs.keys():
            if len(nw_bsdf.find_from(bsdf.inputs["Normal"])) == 0:
                bump = scratch_shader(nw_bsdf, None, **shader_kwargs)["Normal"]

                # connecting nodes: https://blender.stackexchange.com/questions/101820/how-to-add-remove-links-to-existing-or-new-nodes-using-python
                nw_bsdf.links.new(bump.outputs[0], bsdf.inputs["Normal"])

        nw_bsdf.label = MARKER_LABEL


def apply(obj):
    if not isinstance(obj, Iterable):
        obj = [obj]
    for o in obj:
        apply_over(o)
