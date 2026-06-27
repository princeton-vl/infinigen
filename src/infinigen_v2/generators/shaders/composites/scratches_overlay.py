import procfunc as pf
from procfunc.nodes import types as t

from infinigen_v2.generators.shaders.composites import wood_planks
from infinigen_v2.generators.shaders.masks import scratches
from infinigen_v2.generators.shaders.materials import metal_brushed, wood_grain


def _dark_scratch_shader_distribution(rng: pf.RNG) -> pf.ProcNode[pf.Shader]:
    color = pf.color.hsv_color(
        hue=pf.random.uniform(rng, 0.0, 0.12),
        saturation=pf.random.clip_gaussian(rng, 0.15, 0.25, 0.0, 0.9),
        value=pf.random.uniform(rng, 0.0, 0.2),
    )
    return pf.nodes.shader.diffuse_bsdf(color=color, normal=(0.0, 0.0, 0.0))


def _bright_scratch_shader_distribution(rng: pf.RNG) -> pf.ProcNode[pf.Shader]:
    color = pf.color.hsv_color(
        hue=pf.random.uniform(rng, 0.0, 0.12),
        saturation=pf.random.clip_gaussian(rng, 0.1, 0.15, 0.0, 0.5),
        value=pf.random.uniform(rng, 0.4, 0.9),
    )
    return pf.nodes.shader.anisotropic_bsdf(
        color=color,
        roughness=pf.random.uniform(rng, 0.3, 0.5),
        normal=(0.0, 0.0, 0.0),
    )


def scratch_shader_distribution(rng: pf.RNG) -> pf.ProcNode[pf.Shader]:
    func = pf.control.choice(
        rng,
        [
            (_dark_scratch_shader_distribution, 1.0),
            (_bright_scratch_shader_distribution, 1.0),
        ],
    )
    return func(rng)


def _scratches_one_layer(rng, vector, material, scratch_shader, scale):
    result = scratches.scratches_layer_distribution(
        rng, vector, material.surface, material.displacement, scratch_shader, scale
    )
    return pf.Material(surface=result.shader, displacement=result.displacement)


def _scratches_two_layers(rng, vector, material, scratch_shader, scale):
    rng_a, rng_b = rng.spawn(2)
    material = _scratches_one_layer(rng_a, vector, material, scratch_shader, scale)
    return _scratches_one_layer(rng_b, vector, material, scratch_shader, scale)


def _scratches_three_layers(rng, vector, material, scratch_shader, scale):
    rng_a, rng_b, rng_c = rng.spawn(3)
    material = _scratches_one_layer(rng_a, vector, material, scratch_shader, scale)
    material = _scratches_one_layer(rng_b, vector, material, scratch_shader, scale)
    return _scratches_one_layer(rng_c, vector, material, scratch_shader, scale)


def scratches_overlay_distribution(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    material: pf.Material,
    scratch_shader: t.SocketOrVal[pf.Shader] | None = None,
    scale: float | None = None,
) -> pf.Material:
    rng_shader, rng_layers = rng.spawn(2)

    if scratch_shader is None:
        scratch_shader = scratch_shader_distribution(rng_shader)

    func = pf.control.choice(
        rng_layers,
        [
            (_scratches_one_layer, 1.0),
            (_scratches_two_layers, 2.0),
            (_scratches_three_layers, 2.0),
        ],
    )
    return func(rng_layers, vector, material, scratch_shader, scale)


def scratched_wood_distribution(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.Material:
    rng_base, rng_overlay = rng.spawn(2)
    base_func = pf.control.choice(
        rng_base,
        [
            (wood_grain.wood_grain_distribution, 1.0),
            (wood_planks.wood_planks_distribution, 1.5),
        ],
    )
    material = base_func(rng_base, vector)
    return scratches_overlay_distribution(rng_overlay, vector, material)


def scratched_metal_distribution(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.Material:
    rng_base, rng_overlay = rng.spawn(2)
    base_func = pf.control.choice(
        rng_base,
        [
            (metal_brushed.metal_brushed_linear_distribution, 1.0),
            (metal_brushed.metal_brushed_radial_distribution, 1.0),
        ],
    )
    material = base_func(rng_base, vector)
    return scratches_overlay_distribution(rng_overlay, vector, material)
