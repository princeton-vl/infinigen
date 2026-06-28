import numpy as np
import procfunc as pf
from procfunc.nodes import types as t

from infinigen_v2.generators.shaders.masks.tile_shapes import (
    TileShapeResult,
    tile_coord_transform_distribution,
    tile_mask_distribution,
)
from infinigen_v2.generators.shaders.materials import (
    brick_concrete,
    ceramic,
    concrete,
    granite,
    gravel_concrete,
    marble,
    stone_smooth,
    terrazzo,
)


def ceramic_colored_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
) -> pf.Material:
    rng_color, rng_ceramic = rng.spawn(2)
    hue = pf.random.uniform(rng_color, 0.0, 1.0)
    saturation = pf.random.clip_gaussian(rng_color, 0.6, 0.32, 0.0, 0.7)
    value = pf.random.clip_gaussian(rng_color, 0.6, 0.3, 0.0, 1.0)
    color = pf.color.hsv_color(hue=hue, saturation=saturation, value=value)
    return ceramic.ceramic_distribution(rng_ceramic, vector, color=color)


def tile_indoor_wall_material_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
):
    """
    Randomly selects a material distribution for indoor wall tiles.

    Args:
        rng (pf.RNG): The random number generator.
        vector (pf.ProcNode[pf.Vector]): The input vector for the material.

    Returns:
        pf.Material: The material shader.
    """

    rng_choice, rng_func = rng.spawn(2)
    func = pf.control.choice(
        rng_choice,
        [
            (marble.marble_distribution, 2.0),
            (terrazzo.terrazzo_distribution, 0.1),
            (terrazzo.terrazzo_monocolor_distribution, 0.5),
            (ceramic.ceramic_distribution, 1.5),
            (ceramic_colored_distribution, 3),
            (concrete.concrete_distribution, 0.2),
            (granite.granite_smooth_distribution, 1.0),
        ],
    )
    return func(rng_func, vector)


def tile_indoor_ground_material_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
):
    """
    Randomly selects a material distribution for indoor ground tiles.

    Args:
        rng (pf.RNG): The random number generator.
        vector (pf.ProcNode[pf.Vector]): The input vector for the material.

    Returns:
        pf.Material: The material shader.
    """
    rng_choice, rng_func = rng.spawn(2)
    func = pf.control.choice(
        rng_choice,
        [
            (marble.marble_distribution, 2.5),
            (terrazzo.terrazzo_black_monocolor_distribution, 1.0),
            (terrazzo.terrazzo_distribution, 0.2),
            (stone_smooth.stone_smooth_distribution, 0.1),
            (ceramic_colored_distribution, 2.0),
            (concrete.concrete_distribution, 0.5),
            (granite.granite_smooth_distribution, 1.0),
            (granite.granite_distribution, 0.5),
        ],
    )
    return func(rng_func, vector)


def tile_outdoor_wall_material_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
):
    """
    Randomly selects a material distribution for outdoor wall tiles.

    Args:
        rng (pf.RNG): The random number generator.
        vector (pf.ProcNode[pf.Vector]): The input vector for the material.

    Returns:
        pf.Material: The material shader.
    """
    rng_choice, rng_func = rng.spawn(2)
    func = pf.control.choice(
        rng_choice,
        [
            (granite.granite_distribution, 1.0),
            (concrete.concrete_distribution, 1.0),
            (gravel_concrete.gravel_concrete_distribution, 0.2),
            (brick_concrete.brick_concrete_distribution, 1.0),
            (stone_smooth.stone_smooth_distribution, 0.5),
        ],
    )
    return func(rng_func, vector)


def tile_outdoor_ground_material_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
):
    """
    Randomly selects a material distribution for outdoor ground tiles.

    Args:
        rng (pf.RNG): The random number generator.
        vector (pf.ProcNode[pf.Vector]): The input vector for the material.

    Returns:
        pf.Material: The material shader.
    """
    rng_choice, rng_func = rng.spawn(2)
    func = pf.control.choice(
        rng_choice,
        [
            (granite.granite_distribution, 1.0),
            (concrete.concrete_distribution, 1.0),
            (gravel_concrete.gravel_concrete_distribution, 0.5),
            (brick_concrete.brick_concrete_distribution, 1.0),
            (stone_smooth.stone_smooth_distribution, 1.0),
        ],
    )
    return func(rng_func, vector)


def tile_material_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
):
    """
    Randomly selects a material distribution for tiles from a set of types.

    Args:
        rng (pf.RNG): The random number generator.
        vector (pf.ProcNode[pf.Vector]): The input vector for the material.

    Returns:
        pf.Material: The material shader.
    """
    rc, rv = rng.spawn(2)
    func = pf.control.choice(
        rc,
        [
            (tile_indoor_wall_material_distribution, 1.0),
            (tile_outdoor_wall_material_distribution, 1.0),
            (tile_indoor_ground_material_distribution, 1.0),
            (tile_outdoor_ground_material_distribution, 1.0),
        ],
    )
    return func(rv, vector)


def tile_generate(
    vector: pf.ProcNode[pf.Vector],
    tile: pf.Material,
    grout: pf.Material,
    tile_mask: TileShapeResult,
) -> pf.Material:
    """
    Generates a tile shader by mixing tile and grout shaders based on a mask.

    Args:
        vector (pf.ProcNode[pf.Vector]): The input vector.
        tile (pf.Material): The tile material shader.
        grout (pf.Material): The grout material shader.
        tile_mask (TileShapeResult): The tile mask result containing mask and tile types.

    Returns:
        pf.Material: The combined tile and grout shader.
    """
    mask = tile_mask.mask.astype(dtype=float) > 0.1
    shader = pf.nodes.shader.mix_shader(factor=mask, b=tile.surface, a=grout.surface)
    displacement = pf.nodes.math.mix(
        factor=mask, b=tile.displacement, a=grout.displacement
    )
    return pf.Material(
        surface=shader,
        displacement=displacement,
    )


def tile_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector] | None = None,
    tile: pf.Material | None = None,
    grout: pf.Material | None = None,
    tile_mask: TileShapeResult | None = None,
    scale: t.SocketOrVal[float] | None = None,
) -> pf.Material:
    """
    Generates a tile shader with randomized parameters if not provided.

    Args:
        rng (pf.RNG): The random number generator.
        vector (pf.ProcNode[pf.Vector], optional): The input vector. Defaults to texture coordinates.
        tile (pf.Material, optional): The tile material. Defaults to random.
        grout (pf.Material, optional): The grout material. Defaults to brick_concrete grout.
        tile_mask (TileShapeResult, optional): The tile mask. Defaults to random.
        scale (t.SocketOrVal[float], optional): The scale of the tile. Defaults to random.

    Returns:
        pf.Material: The generated tile shader.
    """
    rngs = rng.spawn(6)
    if vector is None:
        vector = pf.nodes.shader.coord().uv
    if scale is None:
        scale = pf.random.log_uniform(rngs[5], 15, 25)
    if tile_mask is None:
        tile_mask = tile_mask_distribution(
            rngs[0],
            tile_coord_transform_distribution(rngs[0], vector, scale=scale),
        )
    if tile is None:
        tile = tile_material_distribution(
            rngs[3], shifted_vector_distribution(rngs[3], vector, tile_mask)
        )
    if grout is None:
        grout_displacement = pf.random.uniform(rngs[4], 0.005, 0.015)
        grout = brick_concrete.brick_concrete_grout_distribution(
            rngs[4], vector, displacement_additional_height=-grout_displacement
        )
    return tile_generate(
        vector,
        tile,
        grout,
        tile_mask,
    )


def shifted_vector_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    tile_mask: TileShapeResult,
) -> pf.ProcNode[pf.Vector]:
    is_second = tile_mask.tile_type_1 + tile_mask.tile_type_2 * pf.random.uniform(
        rng, 0.5, 1.0
    )
    translation_offset = (
        pf.random.uniform(rng, 0.5, 1.0),
        pf.random.uniform(rng, 0.5, 1.0),
        pf.random.uniform(rng, 0.5, 1.0),
    )
    rotation_offset = (
        pf.random.uniform(rng, 0, np.pi * 2),
        pf.random.uniform(rng, 0, np.pi * 2),
        0,
    )
    return pf.nodes.shader.mapping(
        vector,
        location=pf.nodes.math.vector_scale(translation_offset, is_second),
        rotation=pf.nodes.math.vector_scale(rotation_offset, is_second),
    )


def tile_indoor_wall_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector] | None = None,
    tile: pf.Material | None = None,
    grout: pf.Material | None = None,
    tile_mask: TileShapeResult | None = None,
    scale: t.SocketOrVal[float] | None = None,
) -> pf.Material:
    """
    Generates a tile shader for indoor walls with randomized parameters.

    Args:
        rng (pf.RNG): The random number generator.
        vector (pf.ProcNode[pf.Vector], optional): The input vector. Defaults to texture coordinates.
        tile (pf.Material, optional): The tile material. Defaults to random.
        grout (pf.Material, optional): The grout material. Defaults to brick_concrete grout.
        tile_mask (TileShapeResult, optional): The tile mask. Defaults to random.
        scale (t.SocketOrVal[float], optional): The scale of the tile. Defaults to random.

    Returns:
        pf.Material: The generated tile shader.
    """
    rngs = rng.spawn(6)
    if vector is None:
        vector = pf.nodes.shader.coord().uv
    if scale is None:
        scale = pf.random.clip_gaussian(rngs[5], 15, 10, 7, 40)
    if tile_mask is None:
        tile_mask = tile_mask_distribution(
            rngs[0],
            tile_coord_transform_distribution(rngs[0], vector, scale=scale),
        )
    if tile is None:
        tile = tile_indoor_wall_material_distribution(
            rngs[3], shifted_vector_distribution(rngs[3], vector, tile_mask)
        )
    if grout is None:
        grout_displacement = pf.random.uniform(rngs[4], 0.005, 0.015)
        grout = brick_concrete.brick_concrete_grout_distribution(
            rngs[4], vector, displacement_additional_height=-grout_displacement
        )
    return tile_generate(
        vector,
        tile,
        grout,
        tile_mask,
    )


def tile_indoor_ground_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector] | None = None,
    tile: pf.Material | None = None,
    grout: pf.Material | None = None,
    tile_mask: TileShapeResult | None = None,
    scale: t.SocketOrVal[float] | None = None,
) -> pf.Material:
    """
    Generates a tile shader for indoor ground with randomized parameters.

    Args:
        rng (pf.RNG): The random number generator.
        vector (pf.ProcNode[pf.Vector], optional): The input vector. Defaults to texture coordinates.
        tile (pf.Material, optional): The tile material. Defaults to random.
        grout (pf.Material, optional): The grout material. Defaults to brick_concrete grout.
        tile_mask (TileShapeResult, optional): The tile mask. Defaults to random.
        scale (t.SocketOrVal[float], optional): The scale of the tile. Defaults to random.

    Returns:
        pf.Material: The generated tile shader.
    """
    rngs = rng.spawn(6)
    if vector is None:
        vector = pf.nodes.shader.coord().uv
    if scale is None:
        scale = pf.random.clip_gaussian(rngs[5], 7, 5, 1, 25)
    if tile_mask is None:
        mask_coord = tile_coord_transform_distribution(rngs[0], vector, scale=scale)
        tile_mask = tile_mask_distribution(rngs[0], mask_coord)
    if tile is None:
        tile_coord = shifted_vector_distribution(rngs[3], vector, tile_mask)
        tile = tile_indoor_ground_material_distribution(rngs[3], tile_coord)
    if grout is None:
        grout = brick_concrete.brick_concrete_grout_distribution(
            rngs[4], vector, displacement_additional_height=-0.02
        )
    return tile_generate(
        vector,
        tile,
        grout,
        tile_mask,
    )


def tile_outdoor_wall_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector] | None = None,
    tile: pf.Material | None = None,
    grout: pf.Material | None = None,
    tile_mask: TileShapeResult | None = None,
    scale: t.SocketOrVal[float] | None = None,
) -> pf.Material:
    """
    Generates a tile shader for outdoor walls with randomized parameters.

    Args:
        rng (pf.RNG): The random number generator.
        vector (pf.ProcNode[pf.Vector], optional): The input vector. Defaults to texture coordinates.
        tile (pf.Material, optional): The tile material. Defaults to random.
        grout (pf.Material, optional): The grout material. Defaults to brick_concrete grout.
        tile_mask (TileShapeResult, optional): The tile mask. Defaults to random.
        scale (t.SocketOrVal[float], optional): The scale of the tile. Defaults to random.

    Returns:
        pf.Material: The generated tile shader.
    """
    rngs = rng.spawn(6)
    if vector is None:
        vector = pf.nodes.shader.coord().uv
    if scale is None:
        scale = pf.random.log_uniform(rngs[5], 5, 10)
    if tile_mask is None:
        tile_mask = tile_mask_distribution(
            rngs[0],
            tile_coord_transform_distribution(rngs[0], vector, scale=scale),
        )
    if tile is None:
        tile = tile_outdoor_wall_material_distribution(
            rngs[3], shifted_vector_distribution(rngs[3], vector, tile_mask)
        )
    if grout is None:
        grout = brick_concrete.brick_concrete_grout_distribution(
            rngs[4], vector, displacement_additional_height=-0.02
        )
    return tile_generate(
        vector,
        tile,
        grout,
        tile_mask,
    )


def tile_outdoor_ground_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector] | None = None,
    tile: pf.Material | None = None,
    grout: pf.Material | None = None,
    tile_mask: TileShapeResult | None = None,
    scale: t.SocketOrVal[float] | None = None,
) -> pf.Material:
    """
    Generates a tile shader for outdoor ground with randomized parameters.

    Args:
        rng (pf.RNG): The random number generator.
        vector (pf.ProcNode[pf.Vector], optional): The input vector. Defaults to texture coordinates.
        tile (pf.Material, optional): The tile material. Defaults to random.
        grout (pf.Material, optional): The grout material. Defaults to brick_concrete grout.
        tile_mask (TileShapeResult, optional): The tile mask. Defaults to random.
        scale (t.SocketOrVal[float], optional): The scale of the tile. Defaults to random.

    Returns:
        pf.Material: The generated tile shader.
    """
    rngs = rng.spawn(6)
    if vector is None:
        vector = pf.nodes.shader.coord().uv
    if scale is None:
        scale = pf.random.log_uniform(rngs[5], 5, 10)
    if tile_mask is None:
        tile_mask = tile_mask_distribution(
            rngs[0],
            tile_coord_transform_distribution(rngs[0], vector, scale=scale),
        )
    if tile is None:
        tile = tile_outdoor_ground_material_distribution(
            rngs[3], shifted_vector_distribution(rngs[3], vector, tile_mask)
        )
    if grout is None:
        grout = brick_concrete.brick_concrete_grout_distribution(
            rngs[4], vector, displacement_additional_height=-0.02
        )
    return tile_generate(
        vector,
        tile,
        grout,
        tile_mask,
    )
