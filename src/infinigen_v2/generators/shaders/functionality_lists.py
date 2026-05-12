import procfunc as pf

from infinigen_v2.generators.shaders.composites import wood_planks
from infinigen_v2.generators.shaders.composites.splats_overlay import (
    splats_overlay_distribution,
)
from infinigen_v2.generators.shaders.materials import (
    ceramic,
    granite,
    marble,
    metal_brushed,
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
