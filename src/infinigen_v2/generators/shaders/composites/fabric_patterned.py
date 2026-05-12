import procfunc as pf
from procfunc.nodes import types as t

from infinigen_v2.generators.shaders.masks.tile_shapes import (
    TileShapeResult,
    tile_coord_transform_distribution,
    tile_mask_distribution,
)
from infinigen_v2.generators.shaders.materials.fabric import (
    fabric_color_distribution,
    fabric_distribution,
)


def fabric_patterned_colors_distribution(
    rng: pf.RNG,
) -> tuple[pf.Color, pf.Color, pf.Color]:
    rngs = rng.spawn(5)
    val_plain = pf.random.clip_gaussian(rngs[0], 0.4, 0.3, 0.1, 0.9)
    val_bold = pf.random.clip_gaussian(rngs[1], 0.2, 0.1, 0.1, 0.4)
    sat = pf.random.clip_gaussian(rngs[1], 0.4, 0.2, 0.1, 0.5)
    plain_color1 = fabric_color_plain_distribution(
        rngs[2], saturation=sat, value=val_plain
    )
    plain_color2 = fabric_color_plain_distribution(
        rngs[3], saturation=sat, value=val_plain
    )
    color_bold = fabric_color_distribution(rngs[4], value=val_bold)
    return (plain_color1, plain_color2, color_bold)


def patterned_color_distribution(
    rng,
    color1: t.SocketOrVal[pf.Color],
    color2: t.SocketOrVal[pf.Color],
    color3: t.SocketOrVal[pf.Color],
    tile_mask_result: TileShapeResult,
):
    mask = tile_mask_result.result.astype(dtype=float) > 0.1
    mask1 = tile_mask_result.tile_type_1.astype(dtype=float) > 0.1
    mask2 = tile_mask_result.tile_type_2.astype(dtype=float) > 0.1

    color_mixed1 = pf.control.choice(
        rng,
        [
            (color1, 1),
            (pf.nodes.func.mix_rgb(factor=mask1, a=color1, b=color2), 1),
            (pf.nodes.func.mix_rgb(factor=mask, a=color1, b=color3), 1),
        ],
    )
    color = pf.control.choice(
        rng,
        [
            (color_mixed1, 1),
            (pf.nodes.func.mix_rgb(factor=mask2, a=color_mixed1, b=color3), 1),
        ],
    )
    return color


def fabric_patterned_distribution(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    color1: t.SocketOrVal[pf.Color] | None = None,
    color2: t.SocketOrVal[pf.Color] | None = None,
    color3: t.SocketOrVal[pf.Color] | None = None,
    tile_mask: TileShapeResult | None = None,
    scale: float = 20.0,
    translucency: float | None = None,
) -> pf.Material:
    rngs = rng.spawn(9)

    if tile_mask is None:
        scale = pf.random.uniform(rngs[0], 10.0, 50)
        tile_mask = tile_mask_distribution(
            rngs[0],
            tile_coord_transform_distribution(rngs[1], vector, scale=scale),
        )
    mask = tile_mask.result.astype(dtype=float) > 0.1

    if color1 is None:
        color1, color2, color3 = fabric_patterned_colors_distribution(rngs[2])

    color = patterned_color_distribution(rngs[5], color1, color2, color3, tile_mask)

    small_indent = pf.random.uniform(rngs[6], 0.0, 0.0025)

    displacement_offset = pf.control.choice(
        rngs[7],
        [
            (0, 2),
            (mask * small_indent, 1),
            (mask * -small_indent, 1),
        ],
    )

    res = fabric_distribution(
        rngs[8], vector=vector, base_color=color, translucency=translucency
    )

    displacement = res.displacement + pf.nodes.shader.displacement(
        height=displacement_offset, midlevel=0.0
    )

    return pf.Material(
        surface=res.surface,
        displacement=displacement,
    )


def fabric_color_plain_distribution(
    rng: pf.RNG,
    hue: t.SocketOrVal[float] | None = None,
    saturation: t.SocketOrVal[float] | None = None,
    value: t.SocketOrVal[float] | None = None,
) -> pf.Color:
    if hue is None:
        hue = pf.random.uniform(rng, 0.0, 1.0)
    if saturation is None:
        saturation = pf.random.clip_gaussian(rng, 0.6, 0.32, 0.0, 0.7)
    if value is None:
        value = pf.random.uniform(rng, 0.0, 1.0)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)
