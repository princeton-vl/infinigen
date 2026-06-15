import procfunc as pf
import procfunc.nodes.types as t

from infinigen_v2.generators.shaders.masks import splats
from infinigen_v2.generators.shaders.materials import (
    glass_no_refraction,
    metal_brushed,
)


def splats_base_material_distribution(
    rng: pf.RNG, vector: t.SocketOrVal[pf.Vector]
) -> pf.Material:
    del vector
    color = pf.color.hsv_color(
        hue=pf.random.uniform(rng, 0.0, 0.15),
        saturation=pf.random.uniform(rng, 0.0, 0.4),
        value=pf.random.uniform(rng, 0.5, 0.9),
    )
    principled = pf.nodes.shader.diffuse_bsdf(
        color=color,
        roughness=pf.random.uniform(rng, 0.7, 0.95),
    )
    return pf.Material(
        surface=principled,
    )


def splats_overlay_distribution(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    material: pf.Material,
    overlay_material: pf.Material | None = None,
    size: t.SocketOrVal[float] | None = None,
) -> pf.Material:
    if overlay_material is None:
        overlay_material = splats_base_material_distribution(rng=rng, vector=vector)

    mask = splats.splats_mask_distribution(rng=rng, vector=vector, size=size).mask
    shader = pf.nodes.shader.mix_shader(
        factor=mask, a=material.surface, b=overlay_material.surface
    )
    new_disp = overlay_material.displacement * mask.astype(dtype=pf.Vector)

    return pf.Material(
        surface=shader,
        displacement=material.displacement + new_disp,
    )


def metal_simple_distribution(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.Material:
    del vector
    color = pf.color.hsv_color(
        hue=0.0,
        saturation=0.0,
        value=pf.random.uniform(rng, 0.25, 0.6),
    )
    principled = pf.nodes.shader.principled_bsdf(
        base_color=color,
        roughness=pf.random.uniform(rng, 0.03, 0.25),
        metallic=1.0,
    )
    return pf.Material(
        surface=principled,
    )


def metal_splats_distribution(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.Material:
    base_metal_func = pf.control.choice(
        rng,
        [
            (metal_brushed.metal_brushed_linear_distribution, 1.0),
            (metal_brushed.metal_brushed_radial_distribution, 1.0),
            (metal_simple_distribution, 3.0),
        ],
    )
    material = base_metal_func(rng=rng, vector=vector)

    return splats_overlay_distribution(
        rng=rng,
        vector=vector,
        material=material,
    )


def glass_no_refraction_splats_distribution(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.Material:
    glass = glass_no_refraction.glass_no_refraction_distribution(rng=rng, vector=vector)
    return splats_overlay_distribution(
        rng=rng, vector=vector, material=glass, overlay_material=glass
    )
