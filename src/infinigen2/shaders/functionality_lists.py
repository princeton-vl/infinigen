# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import math
from collections.abc import Callable
from functools import partial

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.base_materials import (
    brick_concrete,
    carpet,
    ceramic,
    concrete,
    cracked_ground,
    dirt,
    fabric,
    glass_colored,
    glass_no_refraction,
    granite,
    gravel_concrete,
    ice,
    leather,
    marble,
    metal_brushed,
    metal_hammered,
    mud,
    paint,
    plastic,
    sand,
    sandstone,
    soil,
    stone,
    stone_smooth,
    terrazzo,
    wood_grain,
)
from infinigen2.shaders.composites import (
    bricks,
    fabric_patterned,
    fabric_wrinkled,
    paint_overlay,
    tiles,
    wall,
    wood_planks,
)
from infinigen2.shaders.composites.scratches_overlay import (
    scratches_overlay_rand,
)
from infinigen2.shaders.composites.splats_overlay import (
    splats_base_material_rand,
    splats_overlay_rand,
)
from infinigen2.shaders.displacements.wrinkles import wrinkles_rug_rand
from infinigen2.shaders.masks import cracks, graphicdesign, splats
from infinigen2.shaders.masks.tile_shapes import (
    tile_coord_transform_rand,
    tile_mask_rand,
)

__all__ = [
    "all_materials_rand",
    "boulder_material_rand",
    "cabinet_material_rand",
    "castor_wheel_material_rand",
    "ceiling_material_rand",
    "decorative_material_rand",
    "fabric_art_rand",
    "fabric_floor_rand",
    "fabric_general_rand",
    "fabric_sturdy_rand",
    "floor_material_rand",
    "furniture_material_rand",
    "furniture_surface_material_rand",
    "glass_material_rand",
    "kitchen_counter_rand",
    "mirror_material_rand",
    "paint_flaked_rand",
    "paint_patterned_rand",
    "paint_wall_rand",
    "rug_material_rand",
    "skirt_material_rand",
    "table_top_material_rand",
    "terrain_material_rand",
    "uv_maybe_rotate",
    "uv_maybe_rotate_90",
    "wall_material_rand",
    "wood_painted_rand",
]


@pf.tracer.grammar
def terrain_material_rand(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    rng_choice, rng_material = rng.spawn(2)
    material_func = pf.control.choice(
        rng_choice,
        [
            (cracked_ground.ground_cracked_rand, 1.0),
            (dirt.dirt_rand, 1.0),
            (granite.granite_rand, 1.0),
            (gravel_concrete.gravel_concrete_rand, 1.0),
            (ice.ice_rand, 1.0),
            (mud.mud_rand, 1.0),
            (sand.sand_rand, 1.0),
            (sandstone.sandstone_rand, 1.0),
            (soil.soil_rand, 1.0),
            (stone.stone_rand, 1.0),
            (stone_smooth.stone_smooth_rand, 1.0),
        ],
    )
    return material_func(rng_material, vector)


@pf.tracer.grammar
def boulder_material_rand(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    rng_choice, rng_material = rng.spawn(2)
    material_func = pf.control.choice(
        rng_choice,
        [
            (cracked_ground.ground_cracked_rand, 0.75),
            (granite.granite_rand, 4.0),
            (gravel_concrete.gravel_concrete_rand, 1.5),
            (sandstone.sandstone_rand, 2.5),
            (stone.stone_rand, 2.0),
            (stone_smooth.stone_smooth_rand, 4.0),
        ],
    )
    return material_func(rng_material, vector)


def uv_maybe_rotate(rng: pf.RNG, vector, rotation_z=None):
    if rotation_z is None:
        rng, rng_rotation = rng.spawn(2)
        rotation_z = pf.control.choice(
            rng_rotation,
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
    rng_rotation, rng_uv = rng.spawn(2)
    rotation_z = pf.control.choice(rng_rotation, [(0.0, 1), (math.pi / 2, 1)])
    return uv_maybe_rotate(rng_uv, vector, rotation_z=rotation_z)


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
            (plastic.plastic_translucent_rand, 0.25),
        ],
    )
    material = material_func(rng_mat, vec)
    wear = pf.control.choice(
        rng_wear_choice,
        [
            (lambda r, v, m: m, 3.0),
            (scratches_overlay_rand, 1.0),
            (splats_overlay_rand, 1.0),
            # (paint_overlay.paint_cracked_overlay_rand, 0.5), # not realistic
        ],
    )
    return wear(rng_wear, vec, material)


def kitchen_counter_rand(rng: pf.RNG, vec) -> pf.Material:
    rng_uv, rng_choice, rng_mat, rng_wear_choice, rng_wear = rng.spawn(5)
    vec = uv_maybe_rotate_90(rng_uv, vec)
    material_func = pf.control.choice(
        rng_choice,
        [
            (granite.granite_smooth_rand, 3.0),
            (granite.granite_rand, 1.0),
            (marble.marble_rand, 2.0),
            (ceramic.ceramic_rand, 1.5),
            (terrazzo.terrazzo_rand, 0.75),
            (stone_smooth.stone_smooth_rand, 0.75),
            (concrete.concrete_rand, 0.5),
            (wood_grain.wood_grain_rand, 1.0),
            (metal_brushed.metal_brushed_linear_rand, 0.8),
            (metal_hammered.metal_hammered_rand, 0.2),
            (plastic.plastic_opaque_rand, 0.5),
        ],
    )
    material = material_func(rng_mat, vec)
    wear = pf.control.choice(
        rng_wear_choice,
        [
            (lambda r, v, m: m, 4.0),
            (scratches_overlay_rand, 1.0),
            (splats_overlay_rand, 0.5),
        ],
    )
    return wear(rng_wear, vec, material)


def _glass_splats_gradient(rng: pf.RNG, vector, glass_height) -> pf.ProcNode[float]:
    rng, rng_splat, rng_side = rng.spawn(3)
    uv = pf.nodes.shader.coord().uv
    reach = pf.random.uniform(rng, 0.2, 0.6)
    top_start = glass_height - reach * glass_height
    other_start = reach * glass_height
    res = pf.control.choice(
        rng_side, [((top_start, glass_height), 0.5), ((other_start, 0.0), 0.5)]
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


def decorative_material_rand(rng: pf.RNG, vec) -> pf.Material:
    rng_choice, rng_mat = rng.spawn(2)
    material_func = pf.control.choice(
        rng_choice,
        [
            (metal_brushed.metal_brushed_linear_rand, 3.0),
            (metal_brushed.metal_brushed_radial_rand, 3.0),
            (metal_hammered.metal_hammered_rand, 2.0),
            (plastic.plastic_grayscale_rand, 1.0),
            (plastic.plastic_rand, 1.0),
            (plastic.plastic_translucent_rand, 1.0),
            (glass_colored.glass_colored_rand, 1.0),
            (wood_grain.wood_grain_rand, 0.5),
        ],
    )
    return material_func(rng_mat, vec)


def _furniture_material_func_rand(rng: pf.RNG) -> Callable[..., pf.Material]:
    return pf.control.choice(
        rng,
        [
            (wood_grain.wood_grain_rand, 1.0),
            (wood_planks.wood_planks_rand, 1.0),
            (metal_brushed.metal_brushed_linear_rand, 0.3),
            (plastic.plastic_grayscale_rand, 1.0),
        ],
    )


def wood_painted_rand(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    rng_color, rng_grain, rng_params = rng.spawn(3)
    color = paint.paint_color_rand(rng_color, saturation_power=3.0)
    grain_vector = pf.nodes.math.combine_xyz(x=vector.y, y=vector.x)
    grain = wood_grain.wood_grain_generator_rand(rng_grain, grain_vector).grain_map

    roughness = pf.random.uniform(rng_params, 0.15, 0.45)
    grain_roughness = pf.random.uniform(rng_params, 0.02, 0.08)
    grain_darken = pf.random.uniform(rng_params, 0.0, 0.04)
    grain_height = pf.random.uniform(rng_params, 0.00002, 0.00012)

    base_color = pf.nodes.color.mix_rgb(
        blend_type="MULTIPLY",
        factor=grain * grain_darken,
        a=color,
        b=pf.Color((0.5, 0.5, 0.5)),
    )
    surface = pf.nodes.shader.principled_bsdf(
        base_color=base_color,
        roughness=roughness + grain * grain_roughness,
        specular_ior_level=pf.random.uniform(rng_params, 0.4, 0.6),
    )
    displacement = pf.nodes.shader.displacement(
        height=grain * grain_height, midlevel=0.0
    )
    return pf.Material(surface=surface, displacement=displacement)


def cabinet_material_rand(rng: pf.RNG, vec: pf.ProcNode[pf.Vector]) -> pf.Material:
    rng_uv, rng_choice, rng_mat, rng_wear_choice, rng_wear = rng.spawn(5)
    vec = uv_maybe_rotate_90(rng_uv, vec)
    material_func = pf.control.choice(
        rng_choice,
        [
            (wood_painted_rand, 4.0),
            (wood_grain.wood_grain_rand, 2.5),
            (wood_planks.wood_planks_rand, 1.0),
            (plastic.plastic_grayscale_rand, 1.5),
            (metal_brushed.metal_brushed_linear_rand, 0.5),
        ],
    )
    material = material_func(rng_mat, vec)
    wear = pf.control.choice(
        rng_wear_choice,
        [
            (lambda r, v, m: m, 4.0),
            (scratches_overlay_rand, 1.0),
            (splats_overlay_rand, 0.5),
        ],
    )
    return wear(rng_wear, vec, material)


def furniture_material_rand(rng: pf.RNG, vec) -> pf.Material:
    rng_uv, rng_choice, rng_mat, rng_wear_choice, rng_wear = rng.spawn(5)
    vec = uv_maybe_rotate_90(rng_uv, vec)
    material_func = _furniture_material_func_rand(rng_choice)
    material = material_func(rng_mat, vec)
    wear = pf.control.choice(
        rng_wear_choice,
        [
            (lambda r, v, m: m, 3.0),
            (scratches_overlay_rand, 1.0),
            (splats_overlay_rand, 1.0),
            (paint_overlay.paint_cracked_overlay_rand, 0.25),
        ],
    )
    return wear(rng_wear, vec, material)


def _furniture_surface_material_func_rand(
    rng: pf.RNG,
) -> Callable[..., pf.Material]:
    return pf.control.choice(
        rng,
        [
            (furniture_material_rand, 2.0),
            (fabric_sturdy_rand, 1.0),
        ],
    )


def furniture_surface_material_rand(rng: pf.RNG, vec) -> pf.Material:
    rng_choice, rng_mat = rng.spawn(2)
    material_func = _furniture_surface_material_func_rand(rng_choice)
    return material_func(rng_mat, vec)


def castor_wheel_material_rand(
    rng: pf.RNG, vec, base_material: pf.Material
) -> pf.Material:
    """Caster wheel material; a hard decorative material, or inherits the base's."""
    rng_choice, rng_mat = rng.spawn(2)
    material_func = pf.control.choice(
        rng_choice,
        [
            (decorative_material_rand, 0.7),
            (lambda r, v: base_material, 0.3),
        ],
    )
    return material_func(rng_mat, vec)


def _dark_scratches_overlay(rng: pf.RNG, vector, material: pf.Material) -> pf.Material:
    rng, rng_overlay = rng.spawn(2)
    color = pf.color.hsv_color(
        hue=pf.random.uniform(rng, 0.0, 0.12),
        saturation=pf.random.clip_gaussian(rng, 0.15, 0.25, 0.0, 0.9),
        value=pf.random.uniform(rng, 0.0, 0.2),
    )
    scratch_shader = pf.nodes.shader.diffuse_bsdf(color=color)
    return scratches_overlay_rand(
        rng_overlay, vector, material, scale=0.6, scratch_shader=scratch_shader
    )


def fabric_art_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    translucency: float = 0.0,
) -> pf.Material:
    rng_color, rng_fabric = rng.spawn(2)
    color = graphicdesign.art_rand(rng_color, vector)
    return fabric.fabric_rand(
        rng_fabric,
        vector,
        base_color=color,
        translucency=translucency,
    )


def _fabric_finish_rand(rng: pf.RNG, vec, material: pf.Material) -> pf.Material:
    rng_choice, rng_finish = rng.spawn(2)
    finish_func = pf.control.choice(
        rng_choice,
        [
            (lambda r, v, m: m, 2.0),
            (fabric_wrinkled.wrinkles_small_overlay_rand, 2.0),
            (_dark_scratches_overlay, 1.0),
            (splats_overlay_rand, 1.0),
        ],
    )
    return finish_func(rng_finish, vec, material)


def _fabric_cloth_rand(
    rng: pf.RNG,
    vec: t.SocketOrVal[pf.Vector],
    translucency: float = 0.0,
) -> pf.Material:
    rng_choice, rng_mat = rng.spawn(2)
    material_func = pf.control.choice(
        rng_choice,
        [
            (fabric.fabric_rand, 2.5),
            (fabric_patterned.fabric_patterned_rand, 2.0),
            (fabric_art_rand, 0.5),
        ],
    )
    return material_func(rng_mat, vec, translucency=translucency)


def fabric_general_rand(
    rng: pf.RNG,
    vec: t.SocketOrVal[pf.Vector],
    translucency: float = 0.0,
) -> pf.Material:
    """Any cloth, optionally wrinkled and worn: bedding, drapery and lampshades."""
    rng_cloth, rng_finish = rng.spawn(2)
    material = _fabric_cloth_rand(rng_cloth, vec, translucency=translucency)
    return _fabric_finish_rand(rng_finish, vec, material)


def fabric_sturdy_rand(rng: pf.RNG, vec: t.SocketOrVal[pf.Vector]) -> pf.Material:
    """Upholstery for surfaces that are sat on: cloth, leather or carpet, optionally worn."""
    rng_choice, rng_mat, rng_finish = rng.spawn(3)
    material_func = pf.control.choice(
        rng_choice,
        [
            (_fabric_cloth_rand, 5.0),
            (leather.leather_rand, 0.75),
            (leather.leather_allcolor_rand, 0.75),
            (carpet.carpet_rand, 1.5),
        ],
    )
    material = material_func(rng_mat, vec)
    return _fabric_finish_rand(rng_finish, vec, material)


@pf.tracer.grammar
def paint_wall_rand(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    displacement_pct = pf.random.uniform(rng, 0.0, 0.8)
    paint_value = pf.random.clip_gaussian(rng, 0.5, 0.4, 0.02, 0.95)
    color = paint.paint_color_rand(rng, value=paint_value, saturation_power=0.9)
    return paint.paint_rand(
        rng, vector, displacement_pct=displacement_pct, base_color=color
    )


def paint_patterned_rand(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    r_size, r_tile, r_mask, r_c1, r_c2, r_c3, r_mix, r_paint, r_shrink = rng.spawn(9)
    tile_size = pf.random.uniform(r_size, 0.7, 2.0)
    # sometimes tighten the pattern up to 5x finer, occasionally another 4x on top
    pattern_shrink = pf.random.clip_gaussian(r_size, 1.0, 2.0, 1.0, 5.0)
    extra_shrink = pf.random.uniform(r_size, 1.5, 4.0)
    pattern_shrink = pattern_shrink * pf.control.choice(
        r_shrink, [(extra_shrink, 0.15), (1.0, 0.85)]
    )
    tile_size = tile_size / pattern_shrink
    tile_vec = tile_coord_transform_rand(r_tile, vector, scale=1.0 / tile_size)
    tile_mask = tile_mask_rand(r_mask, tile_vec)
    base_color = fabric_patterned.patterned_color_rand(
        r_mix,
        color1=paint.paint_color_rand(r_c1),
        color2=paint.paint_color_rand(r_c2),
        color3=paint.paint_color_rand(r_c3),
        tile_mask_result=tile_mask,
    )
    return paint.paint_rand(r_paint, vector, base_color=base_color)


def _art_patterned_paint_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.Material:
    rng_color, rng_paint = rng.spawn(2)
    color = graphicdesign.art_rand(rng_color, vector)
    return paint.paint_rand(rng_paint, vector, base_color=color)


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
def all_materials_rand(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    rng_uv, rng_nonlayerable, rng_layerable, rng_choice, rng_func = rng.spawn(5)
    vector = uv_maybe_rotate(rng_uv, vector)
    # granite and the brick/tile composites are bare-only: an overlay overflows the SVM stack
    nonlayerable = pf.control.choice(
        rng_nonlayerable,
        [
            (granite.granite_rand, 1.0),
            (bricks.bricks_rand, 2.0),
            (bricks.bricks_paint_rand, 1.0),
            (bricks.bricks_pristine_rand, 0.5),
            (tiles.tile_indoor_wall_rand, 2.0),
        ],
    )
    layerable = pf.control.choice(
        rng_layerable,
        [
            (brick_concrete.brick_concrete_rand, 1.0),
            (carpet.carpet_rand, 1.0),
            (ceramic.ceramic_rand, 1.0),
            (concrete.concrete_rand, 1.0),
            (fabric.fabric_rand, 1.0),
            (fabric_patterned.fabric_patterned_rand, 2.0),
            (fabric_wrinkled.fabric_wrinkled_rand, 1.0 / 3.0),
            (glass_colored.glass_colored_rand, 1.0),
            (granite.granite_smooth_rand, 1.0),
            (gravel_concrete.gravel_concrete_rand, 2.0),
            (marble.marble_rand, 2.0),
            (metal_brushed.metal_brushed_linear_rand, 1.0),
            (metal_brushed.metal_brushed_radial_rand, 1.0),
            (metal_hammered.metal_hammered_rand, 1.0),
            (paint.paint_rand, 2.0),
            (plastic.plastic_rand, 1.0),
            (plastic.plastic_translucent_rand, 1.0),
            (stone_smooth.stone_smooth_rand, 1.0),
            (terrazzo.terrazzo_rand, 1.0),
            (wood_grain.wood_grain_rand, 2.0),
            (wood_planks.wood_planks_rand, 3.0),
        ],
    )
    func = pf.control.choice(
        rng_choice,
        [
            (lambda r, v: nonlayerable(r, v), 1.5),
            (lambda r, v: layerable(r, v), 3.0),
            (lambda r, v: _layer_rand(r, v, layerable(r, v)), 1.0),
        ],
    )
    return func(rng_func, vector)


@pf.tracer.grammar
def wall_material_rand(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    rng_uv, rng_nonbrick, rng_brick, rng_choice, rng_func, rng_layerable, rng_rotate = (
        rng.spawn(7)
    )
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
    vector = uv_maybe_rotate(rng_rotate, vector, rotation_z=rotation_z)
    non_brick = pf.control.choice(
        rng_nonbrick,
        [
            (paint_wall_rand, 17.5),
            (paint_patterned_rand, 1.0),
            (_art_patterned_paint_rand, 0.5),
            (wood_planks.wood_planks_rand, 1.5),
            (paint_flaked_rand, 1.0),
            (wall.wall_flaked_tile_rand, 1.0),
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
            (tiles.tile_indoor_wall_rand, 3.0),  # SVM stack overflow -> black
        ],
    )
    # non_brick minus paint_flaked and raw granite: both overflow the SVM stack with an overlay
    layerable = pf.control.choice(
        rng_layerable,
        [
            (paint_wall_rand, 3.0),
            (wood_planks.wood_planks_rand, 1.5),
            (concrete.concrete_rand, 1.0),
            (stone_smooth.stone_smooth_rand, 0.5),
            (gravel_concrete.gravel_concrete_rand, 0.5),
        ],
    )
    func = pf.control.choice(
        rng_choice,
        [
            (lambda r, v: non_brick(r, v), 3.0),
            (lambda r, v: brick_tile(r, v), 2.0),
            (lambda r, v: _layer_rand(r, v, layerable(r, v)), 1.0),
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
            (tiles.tile_indoor_ground_rand, 2.7),
            (lambda r, v: _layer_rand(r, v, layerable(r, v)), 2.0),
        ],
    )
    return func(rng_func, vector)


def _paint_ceiling_rand(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    rng, r_color, r_paint = rng.spawn(3)
    saturation = pf.random.clip_gaussian(rng, 0.0, 0.1, 0.0, 0.3)
    color = paint.paint_color_rand(r_color, saturation=saturation)
    return paint.paint_rand(r_paint, vector, base_color=color)


@pf.tracer.grammar
def ceiling_material_rand(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    rng_uv, rng_choice, rng_func = rng.spawn(3)
    vector = uv_maybe_rotate(rng_uv, vector)
    func = pf.control.choice(
        rng_choice,
        [
            (concrete.concrete_rand, 1.0),
            (_paint_ceiling_rand, 3.0),
            (wood_planks.wood_planks_rand, 0.5),
            (tiles.tile_indoor_wall_rand, 0.5),
        ],
    )
    return func(rng_func, vector)


def _rug_wrinkles_overlay(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    material: pf.Material,
) -> pf.Material:
    wrinkles = wrinkles_rug_rand(rng, vector)
    return pf.Material(
        surface=material.surface,
        displacement=material.displacement + wrinkles,
    )


def fabric_floor_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.Material:
    (
        rng_uv,
        rng_base,
        rng_wrinkles_choice,
        rng_wrinkles,
        rng_wear_choice,
        rng_wear,
    ) = rng.spawn(6)
    vector = uv_maybe_rotate(rng_uv, vector)
    material = carpet.carpet_rand(rng_base, vector)
    wrinkles_func = pf.control.choice(
        rng_wrinkles_choice,
        [
            (lambda r, v, m: m, 1.0),
            (_rug_wrinkles_overlay, 2.0),
        ],
    )
    material = wrinkles_func(rng_wrinkles, vector, material)
    wear_func = pf.control.choice(
        rng_wear_choice,
        [
            (lambda r, v, m: m, 3.0),
            (scratches_overlay_rand, 1.0),
            (splats_overlay_rand, 1.0),
        ],
    )
    return wear_func(rng_wear, vector, material)


def rug_material_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.Material:
    return fabric_floor_rand(rng, vector)


def _mirror_splats_gradient(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector], material: pf.Material
) -> pf.Material:
    rng_reach, rng_splats = rng.spawn(2)
    uv = pf.nodes.shader.coord().uv
    reach = pf.random.log_uniform(rng_reach, 0.05, 0.5)
    return splats_overlay_rand(
        rng_splats,
        vector,
        material,
        allow_gradient=True,
        gradient_fac=uv.y,
        gradient_start=reach,
        gradient_end=0.0,
    )


def mirror_material_rand(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    rng_rough, rng_choice, rng_splats, rng_scratches = rng.spawn(4)
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
    splats_func = pf.control.choice(
        rng_choice,
        [
            (splats_overlay_rand, 4.0),
            (_mirror_splats_gradient, 1.0),
        ],
    )
    material = splats_func(rng_splats, vector, material)
    return scratches_overlay_rand(rng_scratches, vector, material)
