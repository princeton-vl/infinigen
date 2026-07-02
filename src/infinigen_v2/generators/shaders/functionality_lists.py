import math
from functools import partial

import procfunc as pf

from infinigen_v2.generators.shaders.composites import (
    bricks,
    fabric_patterned,
    paint_overlay,
    tiles,
    wood_planks,
)
from infinigen_v2.generators.shaders.composites.scratches_overlay import (
    scratches_overlay_distribution,
)
from infinigen_v2.generators.shaders.composites.splats_overlay import (
    splats_base_material_distribution,
    splats_overlay_distribution,
)
from infinigen_v2.generators.shaders.masks import cracks, splats
from infinigen_v2.generators.shaders.materials import (
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


def table_top_material_distribution(rng: pf.RNG, vec) -> pf.Material:
    rng_uv, rng_choice, rng_mat, rng_wear_choice, rng_wear = rng.spawn(5)
    vec = uv_maybe_rotate_90(rng_uv, vec)
    material_func = pf.control.choice(
        rng_choice,
        [
            (wood_grain.wood_grain_distribution, 1.0),
            (wood_planks.wood_planks_distribution, 1.5),
            (marble.marble_distribution, 1.0),
            (metal_brushed.metal_brushed_linear_distribution, 0.25),
            (metal_brushed.metal_brushed_radial_distribution, 0.25),
            (ceramic.ceramic_distribution, 1.0),
            (granite.granite_smooth_distribution, 1.0),
            (glass_colored.glass_colored_distribution, 0.5),
        ],
    )
    material = material_func(rng_mat, vec)
    wear = pf.control.choice(
        rng_wear_choice,
        [
            (lambda r, v, m: m, 3.0),
            (scratches_overlay_distribution, 1.0),
            (splats_overlay_distribution, 1.0),
            # (paint_overlay.cracked_paint_overlay_distribution, 0.5), # not realistic
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
    return splats.splats_mask_distribution(
        rng=rng_splat,
        vector=vector,
        allow_gradient=True,
        gradient_fac=uv.x,
        gradient_start=res[0],
        gradient_end=res[1],
    )


def glass_material_distribution(rng: pf.RNG, vec, glass_height=None) -> pf.Material:
    splats_vector = pf.nodes.shader.coord().object
    options = [
        (splats.splats_mask_distribution, 1.0),
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
    glass = glass_no_refraction.glass_no_refraction_distribution(
        rng_glass, vec, roughness=roughness
    )
    grime = splats_base_material_distribution(rng_grime, splats_vector)
    surface = pf.nodes.shader.mix_shader(factor=mask, a=glass.surface, b=grime.surface)
    return pf.Material(surface=surface)


def furniture_material_distribution(rng: pf.RNG, vec) -> pf.Material:
    rng_uv, rng_choice, rng_mat, rng_wear_choice, rng_wear = rng.spawn(5)
    vec = uv_maybe_rotate_90(rng_uv, vec)
    material_func = pf.control.choice(
        rng_choice,
        [
            (wood_grain.wood_grain_distribution, 1.0),
            (wood_planks.wood_planks_distribution, 1.0),
            (metal_brushed.metal_brushed_linear_distribution, 0.3),
            (plastic.plastic_grayscale_distribution, 0.5),
        ],
    )
    material = material_func(rng_mat, vec)
    wear = pf.control.choice(
        rng_wear_choice,
        [
            (lambda r, v, m: m, 3.0),
            (scratches_overlay_distribution, 1.0),
            (splats_overlay_distribution, 1.0),
            (paint_overlay.cracked_paint_overlay_distribution, 0.5),
        ],
    )
    return wear(rng_wear, vec, material)


def dark_scratches_overlay(rng: pf.RNG, vector, material: pf.Material) -> pf.Material:
    color = pf.color.hsv_color(
        hue=pf.random.uniform(rng, 0.0, 0.12),
        saturation=pf.random.clip_gaussian(rng, 0.15, 0.25, 0.0, 0.9),
        value=pf.random.uniform(rng, 0.0, 0.2),
    )
    scratch_shader = pf.nodes.shader.diffuse_bsdf(color=color, normal=(0.0, 0.0, 0.0))
    return scratches_overlay_distribution(
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
    color = fabric.fabric_color_distribution(rng_color, value=value)

    plain = partial(
        fabric.fabric_translucent_distribution,
        base_color=color,
        translucency=translucency,
    )
    opaque = partial(fabric.fabric_distribution, base_color=color)
    patterned = partial(
        fabric_patterned.fabric_patterned_translucent_distribution,
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
            (dark_scratches_overlay, 1.5),
            (splats_overlay_distribution, 2.0),
        ],
    )
    return wear_func(rng_wear, vec, material)


@pf.tracer.grammar
def paint_wall_distribution(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    displacement_pct = pf.random.uniform(rng, 0.0, 0.8)
    paint_value = pf.random.clip_gaussian(rng, 0.5, 0.4, 0.1, 0.9)
    color = paint.paint_color_distribution(rng, value=paint_value)
    return paint.paint_distribution(
        rng, vector, displacement_pct=displacement_pct, base_color=color
    )


def paint_flaked_distribution(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector]
) -> pf.Material:
    """Paint coat cracked and flaked over a light base (planks/concrete), revealing
    the surface beneath through the crack mask. Dedicated wall composite, kept off
    heavy bases (e.g. brick) so the combined shader stays under budget."""
    rng_paint, rng_choice, rng_base, rng_cracks, rng_overlay = rng.spawn(5)
    paint_mat = paint_wall_distribution(rng_paint, vector)
    base_material = pf.control.choice(
        rng_choice,
        [
            (wood_planks.wood_planks_distribution, 1.0),
            (concrete.concrete_distribution, 2.0),
        ],
    )
    base_material = base_material(rng_base, vector)
    mask = cracks.cracks_distribution(
        rng_cracks,
        vector,
        displacement_a=base_material.displacement,
        displacement_b=paint_mat.displacement,
        height_threshold=0.0,
    ).mask
    return paint_overlay.paint_overlay_distribution(
        rng_overlay, vector, material=base_material, paint=paint_mat, mask=mask
    )


def _layer_distribution(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector], material: pf.Material
) -> pf.Material:
    rng_choice, rng_layer = rng.spawn(2)
    layer = pf.control.choice(
        rng_choice,
        [
            (scratches_overlay_distribution, 1.0),
            (splats_overlay_distribution, 1.0),
        ],
    )
    return layer(rng_layer, vector, material)


@pf.tracer.grammar
def wall_material_distribution(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector]
) -> pf.Material:
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
            (paint_wall_distribution, 3.0),
            (wood_planks.wood_planks_distribution, 1.5),
            (paint_flaked_distribution, 1.0),
            (concrete.concrete_distribution, 1.0),
            (stone_smooth.stone_smooth_distribution, 0.5),
            (gravel_concrete.gravel_concrete_distribution, 0.5),
            (granite.granite_distribution, 0.5),
        ],
    )
    brick_tile = pf.control.choice(
        rng_brick,
        [
            (bricks.bricks_distribution, 2.0),
            (bricks.bricks_paint_distribution, 1.0),
            (bricks.bricks_pristine_distribution, 0.5),
            (tiles.tile_indoor_wall_distribution, 1.5),
        ],
    )
    func = pf.control.choice(
        rng_choice,
        [
            (lambda r, v: non_brick(r, v), 4.0),
            (lambda r, v: brick_tile(r, v), 2),
            (lambda r, v: _layer_distribution(r, v, non_brick(r, v)), 1.0),
        ],
    )
    return func(rng_func, vector)


@pf.tracer.grammar
def skirt_material_distribution(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector]
) -> pf.Material:
    rng_choice, rng_func = rng.spawn(2)
    func = pf.control.choice(
        rng_choice,
        [
            (paint.paint_distribution, 1.0),
            (wood_grain.wood_grain_distribution, 1.0),
        ],
    )
    return func(rng_func, vector)


@pf.tracer.grammar
def floor_material_distribution(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector]
) -> pf.Material:
    rng_uv, rng_layerable, rng_choice, rng_func = rng.spawn(4)
    vector = uv_maybe_rotate(rng_uv, vector)
    layerable = pf.control.choice(
        rng_layerable,
        [
            (concrete.concrete_distribution, 1.0),
            (wood_planks.wood_planks_distribution, 3.0),
            (carpet.carpet_distribution, 2.0),
        ],
    )
    func = pf.control.choice(
        rng_choice,
        [
            (lambda r, v: layerable(r, v), 2.0),
            (tiles.tile_indoor_ground_distribution, 2.0),
            (lambda r, v: _layer_distribution(r, v, layerable(r, v)), 2.0),
        ],
    )
    return func(rng_func, vector)


@pf.tracer.grammar
def ceiling_material_distribution(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector]
) -> pf.Material:
    rng_uv, rng_choice, rng_func = rng.spawn(3)
    vector = uv_maybe_rotate(rng_uv, vector)
    func = pf.control.choice(
        rng_choice,
        [
            (concrete.concrete_distribution, 1.0),
            (paint.paint_distribution, 3.0),
            (wood_planks.wood_planks_distribution, 0.5),
        ],
    )
    return func(rng_func, vector)
