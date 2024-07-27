# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

from functools import partial

# Authors: Lingjie Mei
from inspect import signature

import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.object import new_cube
from infinigen.core.nodes import Nodes, NodeWrangler
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform

from . import ceramic, common
from .utils.surface_utils import perturb_coordinates


def mix_shader(nw, base_shader, offset, rotations, mortar, alternating, selections):
    n = len(selections) + 1
    seeds = (
        np.random.randint(0, 1e7, n) if alternating else [np.random.randint(1e7)] * n
    )
    shaders, disps = [], []
    darken_factor = uniform(0.4, 1.0)
    for i, seed in enumerate(seeds):
        with FixedSeed(seed):
            kwargs = {}
            names = signature(base_shader).parameters
            if "random_seed" in names:
                kwargs["random_seed"] = np.random.randint(1e7)
            if "w" in names:
                kwargs["w"] = offset
            if "hscale" in names:
                if i % 2 == 0:
                    kwargs["hscale"] = log_uniform(20, 30)
                    kwargs["vscale"] = 0.01
                else:
                    kwargs["hscale"] = 0.01
                    kwargs["vscale"] = log_uniform(20, 30)
            base_shader(nw, **kwargs)
            bsdfs = nw.find("Bsdf")
            n = nw.nodes[-1]
            if len(bsdfs) > 0:
                bsdf = bsdfs[-1]
                links = nw.find_from(bsdf.inputs[0])
                if len(links) > 0:
                    color = links[0].from_socket
                else:
                    color = bsdf.inputs[0].default_value
                color = nw.new_node(
                    Nodes.MixRGB,
                    input_kwargs={
                        0: darken_factor,
                        6: color,
                        7: nw.scalar_sub(1, mortar),
                    },
                    attrs={"blend_type": "MULTIPLY", "data_type": "RGBA"},
                ).outputs[2]
                nw.connect_input(color, bsdf.inputs[0])
            match type(n).__name__:
                case Nodes.GroupOutput:
                    shaders.append(nw.find_from(n.inputs[0])[0].from_socket)
                    disp_links = nw.find_from(n.inputs[1])
                    disps.append(
                        disp_links[0].from_socket if len(disp_links) > 0 else None
                    )
                    nw.nodes.remove(n)
                case (
                    Nodes.PrincipledBSDF
                    | Nodes.GlassBSDF
                    | Nodes.GlossyBSDF
                    | Nodes.TranslucentBSDF
                    | Nodes.TransparentBSDF
                    | Nodes.TranslucentBSDF
                ):
                    shaders.append(n.outputs[0])
                    disps.append(None)
                case _:
                    n = nw.find(Nodes.MaterialOutput)[-1]
                    shaders.append(nw.find_from(n.inputs["Surface"])[0].from_socket)
                    disp_links = nw.find_from(n.inputs["Displacement"])
                    disps.append(
                        disp_links[0].from_socket if len(disp_links) > 0 else None
                    )
    shader = shaders[0]
    disp = disps[0]
    rotation = rotations[0]
    for sel, sh, dis, rot in zip(selections, shaders[1:], disps[1:], rotations[1:]):
        shader = nw.new_node(Nodes.MixShader, [sel, shader, sh])
        disp = nw.new_node(
            Nodes.Mix,
            input_kwargs={"Factor": sel, "A": disp, "B": dis},
            attrs={"data_type": "VECTOR"},
        )
        rotation = nw.new_node(
            Nodes.Mix,
            input_kwargs={"Factor": sel, "A": rotation, "B": rot},
            attrs={"data_type": "FLOAT"},
        )
    for node in nw.find(Nodes.TextureCoord)[1:] + nw.find(Nodes.NewGeometry):
        perturb_coordinates(nw, node, offset, rotation)
    disp = nw.add(
        disp,
        nw.new_node(
            Nodes.Displacement,
            input_kwargs={
                "Height": nw.scalar_multiply(mortar, -uniform(0.01, 0.02)),
                "Midlevel": 0.0,
            },
        ),
    )
    nw.new_node(
        Nodes.MaterialOutput, input_kwargs={"Surface": shader, "Displacement": disp}
    )


def shader_square_tile(
    nw: NodeWrangler, base_shader, vertical=False, alternating=None, scale=1, **kwargs
):
    if alternating is None:
        alternating = uniform() < 0.75
    size = log_uniform(0.2, 0.4)
    vec = nw.new_node(Nodes.TextureCoord).outputs["Object"]
    normal = nw.new_node(Nodes.ShaderNodeNormalMap).outputs["Normal"]
    if vertical:
        vec = nw.combine(
            nw.separate(nw.vector_math("CROSS_PRODUCT", vec, normal))[-1],
            nw.separate(vec)[-1],
            0,
        )
    rotation = np.pi / 4 if uniform() < 0.3 else 0
    vec = nw.new_node(
        Nodes.Mapping,
        [
            vec,
            uniform(0, 1, 3),
            (0, 0, np.pi / 2 * np.random.randint(4) + rotation),
            [scale] * 3,
        ],
    )
    vec = nw.combine(*nw.separate(vec)[:2], 0)
    offset, mortar = map(
        nw.new_node(
            Nodes.BrickTexture,
            [vec],
            input_kwargs={
                "Scale": 1 / size,
                "Row Height": 1,
                "Brick Width": 1,
                "Mortar Size": uniform(0.005, 0.01),
                "Color2": (0, 0, 0, 1),
            },
            attrs={"offset": 0.0, "offset_frequency": 1},
        ).outputs.get,
        ["Color", "Fac"],
    )
    selections = [
        nw.new_node(
            Nodes.CheckerTexture, [vec, (0, 0, 0, 1), (1, 1, 1, 1), 1 / size]
        ).outputs[1]
    ]
    rotations = np.pi / 2 * np.arange(2)
    mix_shader(nw, base_shader, offset, rotations, mortar, alternating, selections)


def shader_rectangle_tile(
    nw: NodeWrangler, base_shader, vertical=False, alternating=None, scale=1, **kwargs
):
    if alternating is None:
        alternating = uniform() < 0.75
    size = log_uniform(0.2, 0.4)
    vec = nw.new_node(Nodes.TextureCoord).outputs["Object"]
    normal = nw.new_node(Nodes.ShaderNodeNormalMap).outputs["Normal"]
    if vertical:
        vec = nw.combine(
            nw.separate(nw.vector_math("CROSS_PRODUCT", vec, normal))[-1],
            nw.separate(vec)[-1],
            0,
        )
    vec = nw.new_node(
        Nodes.Mapping,
        [
            vec,
            uniform(0, 1, 3),
            (0, 0, np.pi / 2 * np.random.randint(4)),
            [scale, scale * log_uniform(1.3, 2), scale],
        ],
    )
    vec = nw.combine(*nw.separate(vec)[:2], 0)
    offset, mortar = map(
        nw.new_node(
            Nodes.BrickTexture,
            [vec],
            input_kwargs={
                "Scale": 1 / size,
                "Row Height": 1,
                "Brick Width": 1,
                "Mortar Size": uniform(0.005, 0.01),
                "Color2": (0, 0, 0, 1),
            },
            attrs={"offset": 0.0, "offset_frequency": 1},
        ).outputs.get,
        ["Color", "Fac"],
    )
    selections = [
        nw.new_node(
            Nodes.CheckerTexture, [vec, (0, 0, 0, 1), (1, 1, 1, 1), 1 / size]
        ).outputs[1]
    ]
    rotations = np.pi / 2 * np.arange(2)
    mix_shader(nw, base_shader, offset, rotations, mortar, alternating, selections)


def shader_hexagon_tile(
    nw: NodeWrangler, base_shader, vertical=False, alternating=None, scale=1, **kwargs
):
    if alternating is None:
        alternating = uniform() < 0.6
    size = log_uniform(0.15, 0.3)
    vec = nw.new_node(Nodes.TextureCoord).outputs["Object"]
    normal = nw.new_node(Nodes.ShaderNodeNormalMap).outputs["Normal"]
    if vertical:
        vec = nw.combine(
            nw.separate(nw.vector_math("CROSS_PRODUCT", vec, normal))[-1],
            nw.separate(vec)[-1],
            0,
        )
    vec = nw.new_node(
        Nodes.Mapping,
        [vec, uniform(0, 1, 3), (0, 0, np.pi / 2 * np.random.randint(4)), [scale] * 3],
    )
    qs = []
    for n in (
        np.array(
            [[1 / np.sqrt(3), -1 / 3, 0], [0, 2 / 3, 0], [-1 / np.sqrt(3), -1 / 3, 0]]
        )
        / size
    ):
        qs.append(nw.vector_math("DOT_PRODUCT", vec, n))
    qs_ = [nw.math("ROUND", q) for q in qs]
    qs_diff = [nw.math("ABSOLUTE", nw.scalar_sub(q, q_)) for q, q_ in zip(qs, qs_)]
    coords = []
    for i in range(3):
        coords.append(
            nw.new_node(
                Nodes.Mix,
                [
                    nw.scalar_multiply(
                        nw.math("GREATER_THAN", qs_diff[i], qs_diff[(i + 1) % 3]),
                        nw.math("GREATER_THAN", qs_diff[i], qs_diff[(i + 2) % 3]),
                    ),
                    None,
                    qs_[i],
                    nw.scalar_sub(0, nw.scalar_add(qs_[(i + 1) % 3], qs_[(i + 2) % 3])),
                ],
            )
        )
    offset = nw.combine(coords[0], coords[1], coords[2])
    i = np.random.randint(3)
    fraction = nw.math(
        "FRACT",
        nw.scalar_divide(
            nw.scalar_add(nw.scalar_sub(coords[i], coords[(i + 1) % 3]), 0.5), 3
        ),
    )
    diffs = [nw.math("ABSOLUTE", nw.scalar_sub(q, c)) for q, c in zip(qs, coords)]
    max_dist = nw.math(
        "MAXIMUM",
        nw.math(
            "MAXIMUM",
            nw.scalar_add(diffs[0], diffs[1]),
            nw.scalar_add(diffs[1], diffs[2]),
        ),
        nw.scalar_add(diffs[2], diffs[0]),
    )
    mortar = nw.math("GREATER_THAN", max_dist, 1 - uniform(0.005, 0.01) / size / 2)
    rotations = np.pi * 2 / 3 * np.arange(3)
    mix_shader(
        nw,
        base_shader,
        offset,
        rotations,
        mortar,
        alternating,
        [
            nw.math("LESS_THAN", fraction, 2 / 3),
            nw.math("LESS_THAN", fraction, 1 / 3),
        ],
    )


def shader_staggered_tile(
    nw: NodeWrangler,
    base_shader,
    vertical=False,
    alternating=None,
    scale=1,
    vertical_scale=None,
    **kwargs,
):
    horizontal_scale = scale * log_uniform(2.0, 3.5)
    if vertical_scale is None:
        vertical_scale = horizontal_scale * log_uniform(0.05, 0.2)

    vec = nw.new_node(Nodes.TextureCoord).outputs["Object"]
    normal = nw.new_node(Nodes.ShaderNodeNormalMap).outputs["Normal"]
    if vertical:
        vec = nw.combine(
            nw.separate(nw.vector_math("CROSS_PRODUCT", vec, normal))[-1],
            nw.separate(vec)[-1],
            0,
        )
    vec = nw.new_node(Nodes.Mapping, [vec, uniform(0, 1, 3)])
    vec = nw.add(vec, nw.combine(0, nw.scalar_divide(0.5, horizontal_scale), 0))

    offset, mortar = map(
        nw.new_node(
            Nodes.BrickTexture,
            input_kwargs={
                "Vector": vec,
                "Color2": (0, 0, 0, 1.0000),
                "Scale": 1.0000,
                "Mortar Size": uniform(0.005, 0.01),
                "Mortar Smooth": 1.0000,
                "Bias": -0.5000,
                "Brick Width": nw.scalar_divide(1, vertical_scale),
                "Row Height": nw.scalar_divide(1, horizontal_scale),
            },
            attrs={"squash_frequency": 1},
        ).outputs.get,
        ["Color", "Fac"],
    )
    mix_shader(nw, base_shader, offset, [0], mortar, alternating, [])


def shader_crossed_tile(
    nw: NodeWrangler,
    base_shader,
    vertical=False,
    alternating=None,
    scale=1,
    n=None,
    **kwargs,
):
    n = np.random.randint(4, 8)
    vec = nw.new_node(Nodes.TextureCoord).outputs["Object"]
    normal = nw.new_node(Nodes.ShaderNodeNormalMap).outputs["Normal"]
    if vertical:
        vec = nw.combine(
            nw.separate(nw.vector_math("CROSS_PRODUCT", vec, normal))[-1],
            nw.separate(vec)[-1],
            0,
        )
    vec = nw.new_node(
        Nodes.Mapping,
        [vec, uniform(0, 1, 3), (0, 0, np.pi / 2 * np.random.randint(4)), [scale] * 3],
    )
    x, y, z = nw.separate(vec)
    x_ = nw.scalar_sub(
        x, nw.scalar_divide(nw.math("FLOOR", nw.scalar_multiply(y, n)), n)
    )
    vec = nw.combine(x_, y, 0)
    offset, mortar = map(
        nw.new_node(
            Nodes.BrickTexture,
            input_kwargs={
                "Vector": vec,
                "Color2": (0, 0, 0, 1.0000),
                "Scale": 1.0000,
                "Mortar Size": uniform(0.005, 0.01),
                "Brick Width": 1,
                "Row Height": 1 / n,
            },
            attrs={"squash_frequency": 1, "offset": 0},
        ).outputs.get,
        ["Color", "Fac"],
    )
    vec_ = nw.combine(
        nw.scalar_sub(
            y,
            nw.scalar_divide(
                nw.scalar_add(nw.math("FLOOR", nw.scalar_multiply(x, n)), 1), n
            ),
        ),
        nw.scalar_sub(0, x),
        0,
    )

    offset_, mortar_ = map(
        nw.new_node(
            Nodes.BrickTexture,
            input_kwargs={
                "Vector": vec_,
                "Color2": (0, 0, 0, 1.0000),
                "Scale": 1.0000,
                "Mortar Size": uniform(0.005, 0.01),
                "Brick Width": 1,
                "Row Height": 1 / n,
            },
            attrs={"squash_frequency": 1, "offset": 0},
        ).outputs.get,
        ["Color", "Fac"],
    )
    selection = nw.math(
        "LESS_THAN",
        nw.scalar_sub(
            nw.scalar_divide(x_, 2), nw.math("FLOOR", nw.scalar_divide(x_, 2))
        ),
        0.5,
    )
    offset = nw.new_node(
        Nodes.Mix,
        input_kwargs={"Factor": selection, "A": offset, "B": offset_},
        attrs={"data_type": "FLOAT"},
    )
    mortar = nw.new_node(
        Nodes.Mix,
        input_kwargs={"Factor": selection, "A": mortar, "B": mortar_},
        attrs={"data_type": "FLOAT"},
    )

    mix_shader(
        nw, base_shader, offset, [0, np.pi / 2], mortar, alternating, [selection]
    )


def shader_composite_tile(
    nw: NodeWrangler, base_shader, vertical=False, alternating=None, scale=1, **kwargs
):
    if alternating is None:
        alternating = uniform() < 0.75
    size = log_uniform(0.2, 0.4)
    vec = nw.new_node(Nodes.TextureCoord).outputs["Object"]
    normal = nw.new_node(Nodes.ShaderNodeNormalMap).outputs["Normal"]
    if vertical:
        vec = nw.combine(
            nw.separate(nw.vector_math("CROSS_PRODUCT", vec, normal))[-1],
            nw.separate(vec)[-1],
            0,
        )
    vec = nw.new_node(
        Nodes.Mapping,
        [vec, uniform(0, 1, 3), (0, 0, np.pi / 2 * np.random.randint(8)), [scale] * 3],
    )
    vec = nw.combine(*nw.separate(vec)[:2], 0)

    selections = [
        nw.new_node(
            Nodes.CheckerTexture, [vec, (0, 0, 0, 1), (1, 1, 1, 1), 1 / size]
        ).outputs[1]
    ]
    rotations = np.pi / 2 * np.arange(2)

    mortar_size = uniform(0.002, 0.005)
    stride = np.random.randint(4, 7)
    offset_h, mortar_h = map(
        nw.new_node(
            Nodes.BrickTexture,
            input_kwargs={
                "Vector": vec,
                "Color2": (0, 0, 0, 1.0000),
                "Scale": 1.0000,
                "Mortar Size": mortar_size,
                "Mortar Smooth": 1.0000,
                "Brick Width": size / stride,
                "Row Height": 1000,
            },
            attrs={"squash_frequency": 1},
        ).outputs.get,
        ["Color", "Fac"],
    )
    offset_v, mortar_v = map(
        nw.new_node(
            Nodes.BrickTexture,
            input_kwargs={
                "Vector": vec,
                "Color2": (0, 0, 0, 1.0000),
                "Scale": 1.0000,
                "Mortar Size": mortar_size,
                "Mortar Smooth": 1.0000,
                "Brick Width": 1000,
                "Row Height": size / stride,
            },
            attrs={"squash_frequency": 1},
        ).outputs.get,
        ["Color", "Fac"],
    )
    mortar = nw.new_node(
        Nodes.Mix,
        input_kwargs={"Factor": selections[0], "A": mortar_h, "B": mortar_v},
        attrs={"data_type": "FLOAT"},
    )
    offset = nw.new_node(
        Nodes.Mix,
        input_kwargs={"Factor": selections[0], "A": offset_h, "B": offset_v},
        attrs={"data_type": "VECTOR"},
    )
    mix_shader(nw, base_shader, offset, rotations, mortar, alternating, selections)


def get_shader_funcs():
    from . import bone, ceramic, cobble_stone, dirt, stone
    from .table_materials import shader_marble
    from .woods.wood import shader_wood

    return [
        (bone.shader_bone, 1),
        (cobble_stone.shader_cobblestone, 1),
        (ceramic.shader_ceramic, 4),
        (dirt.shader_dirt, 1),
        (stone.shader_stone, 1),
        (shader_marble, 2),
        (shader_wood, 5),
    ]


def apply(
    obj,
    selection=None,
    vertical=False,
    shader_func=None,
    scale=None,
    alternating=None,
    shape=None,
    **kwargs,
):
    funcs, weights = zip(*get_shader_funcs())
    weights = np.array(weights) / sum(weights)
    if shader_func is None:
        shader_func = np.random.choice(funcs, p=weights)
    name = shader_func.__name__

    if scale is None:
        scale = log_uniform(1.0, 2.0)

    if shader_func == ceramic.shader_ceramic:
        low = uniform(0.1, 0.3)
        high = uniform(0.6, 0.8)
        shader_func = partial(
            ceramic.shader_ceramic, roughness_min=low, roughness_max=high
        )
    match shape:
        case "square":
            method = shader_square_tile
        case "rectangle":
            method = shader_rectangle_tile
        case "hexagon":
            method = shader_hexagon_tile
        case "staggered":
            method = shader_staggered_tile
        case "crossed":
            method = shader_crossed_tile
        case "composite":
            method = shader_composite_tile
        case _:
            method = np.random.choice(
                [
                    shader_hexagon_tile,
                    shader_square_tile,
                    shader_rectangle_tile,
                    shader_staggered_tile,
                    shader_crossed_tile,
                ]
            )

    return common.apply(
        obj,
        method,
        selection,
        shader_func,
        vertical,
        alternating,
        name=f"{name}_{method.__name__}_tile",
        scale=scale,
        **kwargs,
    )


def make_sphere():
    return new_cube()
