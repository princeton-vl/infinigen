# Copyright (C) 2024, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Yiming Zuo

from functools import partial

import numpy as np
from numpy.random import uniform

from infinigen.assets.materials import ceramic, tile
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.color import hsv2rgba

from .basket_weave import nodegroup_basket_weave
from .brick import nodegroup_birck
from .cheveron import nodegroup_cheveron
from .diamond import nodegroup_diamond
from .herringbone import nodegroup_herringbone
from .hexagon import nodegroup_hexagon
from .shell import nodegroup_shell
from .spanish_bound import nodegroup_spanish_bond
from .star import nodegroup_star
from .triangle import nodegroup_triangle

tile_pattern_dict = {
    "basket_weave": nodegroup_basket_weave,
    "brick": nodegroup_birck,
    "cheveron": nodegroup_cheveron,
    "diamond": nodegroup_diamond,
    "herringbone": nodegroup_herringbone,
    "hexagon": nodegroup_hexagon,
    "shell": nodegroup_shell,
    "spanish_bound": nodegroup_spanish_bond,
    "star": nodegroup_star,
    "triangle": nodegroup_triangle,
}


@node_utils.to_nodegroup("nodegroup_tile", singleton=False, type="ShaderNodeTree")
def nodegroup_tile(nw: NodeWrangler, scale=None, vertical=False, **kwargs):
    # Code generated using version 2.6.5 of the node_transpiler

    rotation = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315])
    if scale is None:
        scale = uniform(7.5, 15)
    flip = np.random.choice([-1.0, 1.0])
    border = uniform(0.01, 0.05)
    flatness = uniform(0.9, 0.95)
    subtile_number = np.random.choice([1, 2])
    aspect_ratio = np.random.choice([2, 3, 4, 5])

    # random color type
    color_pattern = np.random.choice([1, 2, 3])

    tile_pattern = np.random.choice(
        [
            "basket_weave",
            "brick",
            "cheveron",
            "diamond",
            "herringbone",
            "hexagon",
            "shell",
            "spanish_bound",
            "star",
            "triangle",
        ]
    )

    # texture_coordinate = nw.new_node(Nodes.TextureCoord)

    # vec, normal = map(nw.new_node(Nodes.TextureCoord).outputs.get, ["Object", "Normal"])
    vec = nw.new_node(Nodes.TextureCoord).outputs["Object"]
    normal = nw.new_node(Nodes.ShaderNodeNormalMap).outputs["Normal"]

    if vertical:
        vec = nw.combine(
            nw.separate(nw.vector_math("CROSS_PRODUCT", vec, normal))[-1],
            nw.separate(vec)[-1],
            0,
        )

    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = float(rotation)

    radians = nw.new_node(
        Nodes.Math, input_kwargs={0: value_1}, attrs={"operation": "RADIANS"}
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": radians})

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = scale

    value_2 = nw.new_node(Nodes.Value)
    value_2.outputs[0].default_value = flip

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: value, 1: value_2}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": value, "Y": multiply, "Z": value}
    )

    mapping = nw.new_node(
        Nodes.Mapping,
        input_kwargs={"Vector": vec, "Rotation": combine_xyz_2, "Scale": combine_xyz},
    )

    tile_nodegroup = tile_pattern_dict[tile_pattern]
    group_4 = nw.new_node(
        tile_nodegroup().name,
        input_kwargs={
            "Coordinate": mapping,
            "Subtiles Number": subtile_number,
            "Aspect Ratio": aspect_ratio,
            "border": border,
            "Flatness": flatness,
        },
    )

    if color_pattern == 1:
        tile_color = group_4.outputs["Tile Color"]
    elif color_pattern == 2:
        tile_color = group_4.outputs["Tile Type 1"]
    elif color_pattern == 3:
        tile_color = group_4.outputs["Tile Type 2"]

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mask": group_4.outputs["Result"], "Tile Color": tile_color},
        attrs={"is_active_output": True},
    )


# use this to generate mask and color pattern
def shader_raw_tiles(nw: NodeWrangler, **kwargs):
    group = nw.new_node(nodegroup_tile(**kwargs).name)
    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": group.outputs["Mask"]},
        attrs={"is_active_output": True},
    )


# def apply(obj, selection=None, **kwargs):
#     surface.add_material(obj, shader_raw_tiles, selection=selection, input_kwargs=kwargs)


# this applies masks on existing materials
def tile_of_material(
    nw: NodeWrangler,
    shader_function,
    vertical=False,
    displacement_scale=0.001,
    **kwargs,
):
    def get_connected_links(nw, input_socket):
        links = [l for l in nw.links if l.to_socket == input_socket]
        return links

    shader_node = shader_function(nw)

    links_to_output = [
        link for link in nw.links if (link.to_node.bl_idname == Nodes.MaterialOutput)
    ]

    if len(links_to_output) == 0:
        # add an output node

        principled_bsdf = nw.find(Nodes.PrincipledBSDF)[0]
        material_output = nw.new_node(
            Nodes.MaterialOutput,
            input_kwargs={"Surface": principled_bsdf},
            attrs={"is_active_output": True},
        )
        links_to_output = [
            link
            for link in nw.links
            if (link.to_node.bl_idname == Nodes.MaterialOutput)
        ]

    # get the BSDF socket
    links_to_surface = get_connected_links(
        nw, links_to_output[0].to_node.inputs["Surface"]
    )
    displacement_out_socket = links_to_output[0].to_node.inputs["Displacement"]
    links_to_displacement = get_connected_links(nw, displacement_out_socket)

    color_value = np.random.choice([0.0, 1.0])  # black or white
    seam_color = hsv2rgba((0.0, 0.0, color_value))

    seam_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": seam_color,
            "Specular IOR Level": 0.0000,
            "Roughness": 0.9000,
        },
    )

    if len(links_to_surface) == 1 and len(links_to_displacement) <= 1:
        # mix shader with tile
        tile_node = nw.new_node(nodegroup_tile(vertical=vertical, **kwargs).name)

        original_bsdf = links_to_surface[0].from_socket

        binary_mask = nw.new_node(
            Nodes.Math,
            input_kwargs={0: tile_node.outputs["Mask"], 1: 0.1000},
            attrs={"operation": "GREATER_THAN"},
        )

        mix_shader = nw.new_node(
            Nodes.MixShader,
            input_kwargs={"Fac": binary_mask, 1: seam_bsdf, 2: original_bsdf},
        )

        nw.links.new(links_to_surface[0].to_socket, mix_shader.outputs["Shader"])

        # displacement
        displacement_tile = nw.new_node(
            Nodes.Displacement,
            input_kwargs={
                "Height": tile_node.outputs["Mask"],
                "Scale": displacement_scale,
            },
        )

        # mix displacement
        if len(links_to_displacement) == 0:
            nw.links.new(
                displacement_out_socket,
                displacement_tile.outputs["Displacement"],
            )

        else:
            original_displacement = links_to_displacement[0].from_socket

            add = nw.new_node(
                Nodes.VectorMath,
                input_kwargs={0: original_displacement, 1: displacement_tile},
            )

            nw.links.new(displacement_out_socket, add.outputs["Vector"])


def apply(obj, selection=None, vertical=False, **kwargs):
    shader_funcs = tile.get_shader_funcs()
    funcs, weights = zip(*shader_funcs)
    weights = np.array(weights) / sum(weights)
    shader_func = np.random.choice(funcs, p=weights)
    name = shader_func.__name__

    low, high = sorted(uniform(0.1, 0.7, 2))
    if shader_func is ceramic.shader_ceramic:
        shader_func = partial(
            ceramic.shader_ceramic,
            roughness_min=low,
            roughness_max=high,
        )

    surface.add_material(
        obj,
        tile_of_material,
        selection=selection,
        input_kwargs=({"shader_function": shader_func, "vertical": vertical} | kwargs),
    )
