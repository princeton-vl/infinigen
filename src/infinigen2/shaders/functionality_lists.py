# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import math
from functools import partial

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.base_materials import (
    carpet,
    ceramic,
    concrete,
    fabric,
    glass_colored,
    glass_no_refraction,
    granite,
    gravel_concrete,
    marble,
    metal_brushed,
    paint,
    plastic,
    stone_smooth,
    wood_grain,
)
from infinigen2.shaders.composites import (
    bricks,
    fabric_patterned,
    paint_overlay,
    tiles,
    wood_planks,
)
from infinigen2.shaders.composites.scratches_overlay import (
    scratches_overlay_rand,
)
from infinigen2.shaders.composites.splats_overlay import (
    splats_base_material_rand,
    splats_overlay_rand,
)
from infinigen2.shaders.masks import cracks, splats
from infinigen2.shaders.masks.tile_shapes import (
    tile_coord_transform_rand,
    tile_mask_rand,
)

__all__ = [
    "art_color_rand",
    "art_pattern_material_rand",
    "ceiling_material_rand",
    "floor_material_rand",
    "furniture_fabric",
    "furniture_material_rand",
    "glass_material_rand",
    "mirror_material_rand",
    "paint_flaked_rand",
    "paint_wall_rand",
    "rug_material_rand",
    "skirt_material_rand",
    "table_top_material_rand",
    "uv_maybe_rotate",
    "uv_maybe_rotate_90",
    "wall_material_rand",
]


def uv_maybe_rotate(rng: pf.RNG, vector, rotation_z=None):
    if rotation_z is None:
        (rng,) = rng.spawn(1)
        rotation_z = pf.control.choice(
            rng,
            [
                (0.0, 12),
                (math.pi / 2, 1),
                (math.pi / 4, 1),
                (-math.pi / 2, 1),
                (-math.pi / 4, 1),
                (pf.random.uniform(rng, 0.0, 2 * math.pi), 4),
            ],
        )
    return pf.nodes.shader.mapping(
        vector=vector,
        location=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, rotation_z),
        scale=(1.0, 1.0, 1.0),
    )


def uv_maybe_rotate_90(rng: pf.RNG, vector):
    rotation_z = pf.control.choice(rng, [(0.0, 1), (math.pi / 2, 1)])
    return uv_maybe_rotate(rng, vector, rotation_z=rotation_z)


def table_top_material_rand(rng: pf.RNG, vec) -> pf.Material:
    rng_uv, rng_choice, rng_mat, rng_wear_choice, rng_wear = rng.spawn(5)
    vec = uv_maybe_rotate_90(rng_uv, vec)
    material_func = pf.control.choice(
        rng_choice,
        [
            (wood_grain.wood_grain_rand, 1.0),
            (wood_planks.wood_planks_rand, 1.5),
            (marble.marble_rand, 1.0),
            (metal_brushed.metal_brushed_linear_rand, 0.25),
            (metal_brushed.metal_brushed_radial_rand, 0.25),
            (ceramic.ceramic_rand, 1.0),
            (granite.granite_smooth_rand, 1.0),
            (glass_colored.glass_colored_rand, 0.5),
        ],
    )
    material = material_func(rng_mat, vec)
    wear = pf.control.choice(
        rng_wear_choice,
        [
            (lambda r, v, m: m, 3.0),
            (scratches_overlay_rand, 1.0),
            (splats_overlay_rand, 1.0),
            # (paint_overlay.cracked_paint_overlay_rand, 0.5), # not realistic
        ],
    )
    return wear(rng_wear, vec, material)


def _glass_splats_gradient(rng: pf.RNG, vector, glass_height) -> pf.ProcNode[float]:
    rng, rng_splat = rng.spawn(2)
    uv = pf.nodes.shader.coord().uv
    reach = pf.random.uniform(rng, 0.2, 0.6)
    top_start = glass_height - reach * glass_height
    other_start = reach * glass_height
    res = pf.control.choice(
        rng, [((top_start, glass_height), 0.5), ((other_start, 0.0), 0.5)]
    )
    return splats.splats_mask_rand(
        rng=rng_splat,
        vector=vector,
        allow_gradient=True,
        gradient_fac=uv.x,
        gradient_start=res[0],
        gradient_end=res[1],
    )


def glass_material_rand(rng: pf.RNG, vec, glass_height=None) -> pf.Material:
    splats_vector = pf.nodes.shader.coord().object
    options = [
        (splats.splats_mask_rand, 1.0),
        (lambda *_, **__: splats.SplatsMaskResult(mask=0.0), 5.0),
    ]
    if glass_height is not None:
        options.append(
            (partial(_glass_splats_gradient, glass_height=glass_height), 2.0)
        )
    rng_mask_choice, rng_mask, rng_rough, rng_glass, rng_grime = rng.spawn(5)
    mask = pf.control.choice(rng_mask_choice, options)(
        rng=rng_mask, vector=splats_vector
    ).mask

    roughness = pf.random.clip_gaussian(rng_rough, 0.02, 0.03, 0.0, 0.3)
    glass = glass_no_refraction.glass_no_refraction_rand(
        rng_glass, vec, roughness=roughness
    )
    grime = splats_base_material_rand(rng_grime, splats_vector)
    surface = pf.nodes.shader.mix_shader(factor=mask, a=glass.surface, b=grime.surface)
    return pf.Material(surface=surface)


def furniture_material_rand(rng: pf.RNG, vec) -> pf.Material:
    rng_uv, rng_choice, rng_mat, rng_wear_choice, rng_wear = rng.spawn(5)
    vec = uv_maybe_rotate_90(rng_uv, vec)
    material_func = pf.control.choice(
        rng_choice,
        [
            (wood_grain.wood_grain_rand, 1.0),
            (wood_planks.wood_planks_rand, 1.0),
            (metal_brushed.metal_brushed_linear_rand, 0.3),
            (plastic.plastic_grayscale_rand, 0.5),
        ],
    )
    material = material_func(rng_mat, vec)
    wear = pf.control.choice(
        rng_wear_choice,
        [
            (lambda r, v, m: m, 3.0),
            (scratches_overlay_rand, 1.0),
            (splats_overlay_rand, 1.0),
            (paint_overlay.cracked_paint_overlay_rand, 0.5),
        ],
    )
    return wear(rng_wear, vec, material)


def _dark_scratches_overlay(rng: pf.RNG, vector, material: pf.Material) -> pf.Material:
    color = pf.color.hsv_color(
        hue=pf.random.uniform(rng, 0.0, 0.12),
        saturation=pf.random.clip_gaussian(rng, 0.15, 0.25, 0.0, 0.9),
        value=pf.random.uniform(rng, 0.0, 0.2),
    )
    scratch_shader = pf.nodes.shader.diffuse_bsdf(color=color, normal=(0.0, 0.0, 0.0))
    return scratches_overlay_rand(
        rng, vector, material, scale=0.6, scratch_shader=scratch_shader
    )


def furniture_fabric(
    rng: pf.RNG,
    vec,
    translucency: float | None = None,
    wear: bool = True,
) -> pf.Material:
    rng, rng_color, rng_choice, rng_mat, rng_wear_choice, rng_wear = rng.spawn(6)
    if translucency is None:
        translucency = pf.random.clip_gaussian(rng, 0.4, 0.2, 0.05, 0.8)
    value = pf.random.clip_gaussian(rng, 0.4, 0.3, 0.1, 0.9)
    color = fabric.fabric_color_rand(rng_color, value=value)

    plain = partial(
        fabric.fabric_translucent_rand,
        base_color=color,
        translucency=translucency,
    )
    opaque = partial(fabric.fabric_rand, base_color=color)
    patterned = partial(
        fabric_patterned.fabric_patterned_translucent_rand,
        translucency=translucency,
    )
    material_func = pf.control.choice(
        rng_choice,
        [
            (plain, 1.5),
            (patterned, 2.0),
            (opaque, 1.0),
        ],
    )
    material = material_func(rng_mat, vec)
    if not wear:
        return material
    wear_func = pf.control.choice(
        rng_wear_choice,
        [
            (lambda r, v, m: m, 2.0),
            (_dark_scratches_overlay, 1.5),
            (splats_overlay_rand, 2.0),
        ],
    )
    return wear_func(rng_wear, vec, material)


@pf.tracer.grammar
def paint_wall_rand(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    displacement_pct = pf.random.uniform(rng, 0.0, 0.8)
    paint_value = pf.random.clip_gaussian(rng, 0.5, 0.4, 0.1, 0.9)
    color = paint.paint_color_rand(rng, value=paint_value)
    return paint.paint_rand(
        rng, vector, displacement_pct=displacement_pct, base_color=color
    )


def paint_flaked_rand(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    """Paint coat cracked and flaked over a light base (planks/concrete), revealing
    the surface beneath through the crack mask. Dedicated wall composite, kept off
    heavy bases (e.g. brick) so the combined shader stays under budget."""
    rng_paint, rng_choice, rng_base, rng_cracks, rng_overlay = rng.spawn(5)
    paint_mat = paint_wall_rand(rng_paint, vector)
    base_material = pf.control.choice(
        rng_choice,
        [
            (wood_planks.wood_planks_rand, 1.0),
            (concrete.concrete_rand, 2.0),
        ],
    )
    base_material = base_material(rng_base, vector)
    mask = cracks.cracks_rand(
        rng_cracks,
        vector,
        displacement_a=base_material.displacement,
        displacement_b=paint_mat.displacement,
        height_threshold=0.0,
    ).mask
    return paint_overlay.paint_overlay_rand(
        rng_overlay, vector, material=base_material, paint=paint_mat, mask=mask
    )


def _layer_rand(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector], material: pf.Material
) -> pf.Material:
    rng_choice, rng_layer = rng.spawn(2)
    layer = pf.control.choice(
        rng_choice,
        [
            (scratches_overlay_rand, 1.0),
            (splats_overlay_rand, 1.0),
        ],
    )
    return layer(rng_layer, vector, material)


@pf.tracer.grammar
def wall_material_rand(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    rng_uv, rng_nonbrick, rng_brick, rng_choice, rng_func = rng.spawn(5)
    # walls: horizontal 60%, vertical 30% (split +/-), 45-deg snaps 10% (split +/-)
    rotation_z = pf.control.choice(
        rng_uv,
        [
            (0.0, 3.0),
            (math.pi / 2, 0.75),
            (-math.pi / 2, 0.75),
            (math.pi / 4, 0.25),
            (-math.pi / 4, 0.25),
        ],
    )
    vector = uv_maybe_rotate(rng_uv, vector, rotation_z=rotation_z)
    non_brick = pf.control.choice(
        rng_nonbrick,
        [
            (paint_wall_rand, 3.0),
            (wood_planks.wood_planks_rand, 1.5),
            (paint_flaked_rand, 1.0),
            (concrete.concrete_rand, 1.0),
            (stone_smooth.stone_smooth_rand, 0.5),
            (gravel_concrete.gravel_concrete_rand, 0.5),
            (granite.granite_rand, 0.5),
        ],
    )
    brick_tile = pf.control.choice(
        rng_brick,
        [
            (bricks.bricks_rand, 2.0),
            (bricks.bricks_paint_rand, 1.0),
            (bricks.bricks_pristine_rand, 0.5),
            (tiles.tile_indoor_wall_rand, 1.5),  # SVM stack overflow -> black
        ],
    )
    func = pf.control.choice(
        rng_choice,
        [
            (lambda r, v: non_brick(r, v), 3.0),
            (lambda r, v: brick_tile(r, v), 2.0),
            (lambda r, v: _layer_rand(r, v, non_brick(r, v)), 1.0),
        ],
    )
    return func(rng_func, vector)


@pf.tracer.grammar
def skirt_material_rand(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    rng_choice, rng_func = rng.spawn(2)
    func = pf.control.choice(
        rng_choice,
        [
            (paint.paint_rand, 1.0),
            (wood_grain.wood_grain_rand, 1.0),
        ],
    )
    return func(rng_func, vector)


@pf.tracer.grammar
def floor_material_rand(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    rng_uv, rng_layerable, rng_choice, rng_func = rng.spawn(4)
    vector = uv_maybe_rotate(rng_uv, vector)
    layerable = pf.control.choice(
        rng_layerable,
        [
            (concrete.concrete_rand, 1.0),
            (wood_planks.wood_planks_rand, 3.0),
            (carpet.carpet_rand, 2.0),
        ],
    )
    func = pf.control.choice(
        rng_choice,
        [
            (lambda r, v: layerable(r, v), 2.0),
            (tiles.tile_indoor_ground_rand, 2.0),
            (lambda r, v: _layer_rand(r, v, layerable(r, v)), 2.0),
        ],
    )
    return func(rng_func, vector)


@pf.tracer.grammar
def ceiling_material_rand(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    rng_uv, rng_choice, rng_func = rng.spawn(3)
    vector = uv_maybe_rotate(rng_uv, vector)
    func = pf.control.choice(
        rng_choice,
        [
            (concrete.concrete_rand, 1.0),
            (paint.paint_rand, 3.0),
            (wood_planks.wood_planks_rand, 0.5),
        ],
    )
    return func(rng_func, vector)


def art_color_rand(rng: pf.RNG) -> pf.Color:
    hue = pf.random.uniform(rng, 0.0, 1.0)
    value = pf.random.clip_gaussian(rng, 0.6, 0.5, 0.05, 0.95)
    sat = pf.random.clip_gaussian(rng, 0.7, 0.4, 0.05, 0.95)
    return pf.color.hsv_color(hue=hue, saturation=sat, value=value)


def art_pattern_material_rand(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector]
) -> pf.Material:
    r_tile, r_colors, r_choice, r_mat = rng.spawn(4)
    scale = pf.random.uniform(r_tile, 1.0, 20.0)

    tile_vec = tile_coord_transform_rand(r_tile, vector, scale=scale)
    tile_mask = tile_mask_rand(r_tile, tile_vec)

    base_color = fabric_patterned.patterned_color_rand(
        r_colors,
        color1=art_color_rand(r_colors),
        color2=art_color_rand(r_colors),
        color3=art_color_rand(r_colors),
        tile_mask_result=tile_mask,
    )
    func = pf.control.choice(
        r_choice,
        [
            (paint.paint_rand, 1.0),
            (fabric.fabric_rand, 1.0),
        ],
    )
    return func(r_mat, vector, base_color=base_color)


def rug_material_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.Material:
    rng_choice, rng_func = rng.spawn(2)
    func = pf.control.choice(
        rng_choice,
        [
            (fabric_patterned.fabric_patterned_rand, 3.0),
            (fabric.fabric_rand, 1.0),
            (lambda rng, vector, **_: carpet.carpet_rand(rng, vector), 2.0),
        ],
    )
    return func(rng_func, vector)


def mirror_material_rand(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    rng_rough, rng_splats, rng_scratches = rng.spawn(3)
    roughness = pf.random.clip_gaussian(rng_rough, 0.01, 0.05, 0.005, 0.2)
    surface = pf.nodes.shader.principled_bsdf(
        base_color=(1.0, 1.0, 1.0, 1.0),
        metallic=1.0,
        roughness=roughness,
        specular_ior_level=1.0,
        subsurface_anisotropy=0.0,
    )
    material = pf.Material(
        surface=surface,
        displacement=pf.nodes.math.constant((0.0, 0.0, 0.0)),
    )
    material = splats_overlay_rand(rng_splats, vector, material)
    return scratches_overlay_rand(rng_scratches, vector, material)
