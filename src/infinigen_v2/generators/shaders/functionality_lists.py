import procfunc as pf

from infinigen_v2.generators.shaders.composites import (
    bricks,
    paint_overlay,
    tiles,
    wood_planks,
)
from infinigen_v2.generators.shaders.composites.splats_overlay import (
    splats_overlay_distribution,
)
from infinigen_v2.generators.shaders.masks import cracks
from infinigen_v2.generators.shaders.materials import (
    carpet,
    ceramic,
    concrete,
    granite,
    marble,
    metal_brushed,
    paint,
    plastic,
    wood_grain,
)
from infinigen_v2.generators.shaders.util.coord import uv_maybe_rotate


def table_top_material_distribution(rng: pf.RNG, vec) -> pf.Material:
    vec = uv_maybe_rotate(rng, vec)
    material_func = pf.control.choice(
        rng,
        [
            (wood_grain.wood_grain_distribution, 1.0),
            (wood_planks.wood_planks_distribution, 1.5),
            (marble.marble_distribution, 1.0),
            (metal_brushed.metal_brushed_linear_distribution, 0.25),
            (metal_brushed.metal_brushed_radial_distribution, 0.25),
            (ceramic.ceramic_distribution, 1.0),
            (granite.granite_smooth_distribution, 1.0),
        ],
    )
    material = material_func(rng, vec)
    material = pf.control.choice(
        rng,
        [
            (lambda m: splats_overlay_distribution(rng, vec, m), 1.0),
            (lambda m: m, 3.0),
        ],
    )(material)
    return material


def furniture_material_distribution(rng: pf.RNG, vec) -> pf.Material:
    material_func = pf.control.choice(
        rng,
        [
            (wood_grain.wood_grain_distribution, 1.0),
            (wood_planks.wood_planks_distribution, 1.0),
            (metal_brushed.metal_brushed_linear_distribution, 0.3),
            (plastic.plastic_grayscale_distribution, 0.5),
        ],
    )
    return material_func(rng, vec)


@pf.tracer.grammar
def paint_wall_distribution(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    displacement_pct = pf.random.uniform(rng, 0.0, 0.8)
    paint_value = pf.random.clip_gaussian(rng, 0.5, 0.4, 0.1, 0.9)
    color = paint.paint_color_distribution(rng, value=paint_value)
    return paint.paint_distribution(
        rng, vector, displacement_pct=displacement_pct, base_color=color
    )


@pf.tracer.grammar
def paint_flaked_distribution(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector]
) -> pf.Material:
    paint_mat = paint_wall_distribution(rng, vector)

    base_material = pf.control.choice(
        rng,
        [
            (wood_planks.wood_planks_distribution, 1.0),
            (concrete.concrete_distribution, 2.0),
            # (tiles.tile_indoor_wall_distribution, 1.0), # svm out of stack space
        ],
    )
    base_material = base_material(rng, vector)

    mask = cracks.cracks_distribution(
        rng,
        vector,
        displacement_a=base_material.displacement,
        displacement_b=paint_mat.displacement,
        height_threshold=0.0,
    ).mask
    return paint_overlay.paint_overlay_distribution(
        rng, vector, material=base_material, paint=paint_mat, mask=mask
    )


@pf.tracer.grammar
def wall_material_distribution(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector]
) -> pf.Material:
    vector = uv_maybe_rotate(rng, vector)
    func = pf.control.choice(
        rng,
        [
            (paint_wall_distribution, 3.0),
            (bricks.bricks_distribution, 2.0),
            (bricks.bricks_paint_distribution, 1.0),
            (bricks.bricks_pristine_distribution, 0.5),
            (concrete.concrete_distribution, 0.5),
            (tiles.tile_indoor_wall_distribution, 1.5),
            (wood_planks.wood_planks_distribution, 1.5),
            (paint_flaked_distribution, 1.0),
        ],
    )
    return func(rng, vector)


@pf.tracer.grammar
def skirt_material_distribution(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector]
) -> pf.Material:
    func = pf.control.choice(
        rng,
        [
            (paint.paint_distribution, 1.0),
            (wood_grain.wood_grain_distribution, 1.0),
        ],
    )
    return func(rng, vector)


@pf.tracer.grammar
def floor_material_distribution(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector]
) -> pf.Material:
    vector = uv_maybe_rotate(rng, vector)
    func = pf.control.choice(
        rng,
        [
            (concrete.concrete_distribution, 1.0),
            (tiles.tile_indoor_ground_distribution, 3.0),
            (wood_planks.wood_planks_distribution, 3.0),
            (carpet.carpet_distribution, 2.0),
        ],
    )
    return func(rng, vector)


@pf.tracer.grammar
def ceiling_material_distribution(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector]
) -> pf.Material:
    vector = uv_maybe_rotate(rng, vector)
    func = pf.control.choice(
        rng,
        [
            (concrete.concrete_distribution, 1.0),
            (paint.paint_distribution, 3.0),
            (wood_planks.wood_planks_distribution, 0.5),
            (paint_flaked_distribution, 1.0),
        ],
    )
    return func(rng, vector)
