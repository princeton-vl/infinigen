import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.base_materials import metal_brushed, wood_grain
from infinigen2.shaders.composites import wood_planks
from infinigen2.shaders.masks import scratches

__all__ = [
    "scratch_shader_rand",
    "scratched_metal_rand",
    "scratched_wood_rand",
    "scratches_brushed_preset",
    "scratches_deep_dirty_preset",
    "scratches_dense_preset",
    "scratches_light_varnish_preset",
    "scratches_overlay_rand",
    "scratches_shallow_preset",
]


def _dark_scratch_shader_rand(rng: pf.RNG) -> pf.ProcNode[pf.Shader]:
    color = pf.color.hsv_color(
        hue=pf.random.uniform(rng, 0.0, 0.12),
        saturation=pf.random.clip_gaussian(rng, 0.15, 0.25, 0.0, 0.9),
        value=pf.random.uniform(rng, 0.0, 0.2),
    )
    return pf.nodes.shader.diffuse_bsdf(color=color, normal=(0.0, 0.0, 0.0))


def _bright_scratch_shader_rand(rng: pf.RNG) -> pf.ProcNode[pf.Shader]:
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


def scratch_shader_rand(rng: pf.RNG) -> pf.ProcNode[pf.Shader]:
    rng_choice, rng_func = rng.spawn(2)
    func = pf.control.choice(
        rng_choice,
        [
            (_dark_scratch_shader_rand, 1.0),
            (_bright_scratch_shader_rand, 1.0),
        ],
    )
    return func(rng_func)


def _scratches_one_layer(rng, vector, material, scratch_shader, scale):
    result = scratches.scratches_layer_rand(
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


def scratches_overlay_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    material: pf.Material,
    scratch_shader: t.SocketOrVal[pf.Shader] | None = None,
    scale: float | None = None,
) -> pf.Material:
    rng_shader, rng_layers_choice, rng_layers_func = rng.spawn(3)

    if scratch_shader is None:
        scratch_shader = scratch_shader_rand(rng_shader)

    func = pf.control.choice(
        rng_layers_choice,
        [
            (_scratches_one_layer, 1.0),
            (_scratches_two_layers, 2.0),
            (_scratches_three_layers, 2.0),
        ],
    )
    return func(rng_layers_func, vector, material, scratch_shader, scale)


def scratched_wood_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.Material:
    rng_base_choice, rng_base_func, rng_overlay = rng.spawn(3)
    base_func = pf.control.choice(
        rng_base_choice,
        [
            (wood_grain.wood_grain_rand, 1.0),
            (wood_planks.wood_planks_rand, 1.5),
        ],
    )
    material = base_func(rng_base_func, vector)
    return scratches_overlay_rand(rng_overlay, vector, material)


def scratched_metal_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.Material:
    rng_base_choice, rng_base_func, rng_overlay = rng.spawn(3)
    base_func = pf.control.choice(
        rng_base_choice,
        [
            (metal_brushed.metal_brushed_linear_rand, 1.0),
            (metal_brushed.metal_brushed_radial_rand, 1.0),
        ],
    )
    material = base_func(rng_base_func, vector)
    return scratches_overlay_rand(rng_overlay, vector, material)


def _apply_scratch_mask(
    mask: pf.ProcNode[float],
    base: pf.Material,
    scratch_shader: pf.ProcNode[pf.Shader],
    depth: float,
) -> pf.Material:
    surface = pf.nodes.shader.mix_shader(factor=mask, a=base.surface, b=scratch_shader)
    if depth <= 0.0:
        return pf.Material(surface=surface, displacement=base.displacement)
    carve = pf.nodes.shader.displacement(
        height=mask, midlevel=0.0, scale=depth, normal=(0.0, 0.0, 0.0)
    )
    return pf.Material(surface=surface, displacement=base.displacement - carve)


def scratches_brushed_preset(vector: t.SocketOrVal[pf.Vector]) -> pf.Material:
    base = metal_brushed.metal_brushed(
        vector=vector,
        brush_type=3.0,
        brush_size=0.3,
        color=pf.Color((0.4035122, 0.4035122, 0.4035122)),
        roughness=0.38627362,
    )
    mask = scratches.scratches_brushed_mask_preset(vector)
    scratch_shader = pf.nodes.shader.anisotropic_bsdf(
        color=pf.Color((0.4975732, 0.4975732, 0.4975732)),
        roughness=0.4076923,
        normal=(0.0, 0.0, 0.0),
    )
    return _apply_scratch_mask(mask, base, scratch_shader, 0.0)


def scratches_dense_preset(vector: t.SocketOrVal[pf.Vector]) -> pf.Material:
    base = metal_brushed.metal_brushed(
        vector=vector,
        brush_type=3.0,
        brush_size=0.3,
        color=pf.Color((0.35180885, 0.35180885, 0.35180885)),
        color_variation=0.57465065,
        roughness=0.3,
    )
    mask = scratches.scratches_dense_mask_preset(vector)
    scratch_shader = pf.nodes.shader.anisotropic_bsdf(
        color=pf.Color((0.351809, 0.351809, 0.351809)),
        roughness=0.4,
        normal=(0.0, 0.0, 0.0),
    )
    return _apply_scratch_mask(mask, base, scratch_shader, 0.0)


def scratches_deep_dirty_preset(vector: t.SocketOrVal[pf.Vector]) -> pf.Material:
    base = wood_grain.wood_grain_brown_preset(vector)
    mask = scratches.scratches_deep_dirty_mask_preset(vector)
    scratch_shader = pf.nodes.shader.diffuse_bsdf(
        color=pf.Color((0.0059479414, 0.0059479414, 0.0059479414)),
        normal=(0.0, 0.0, 0.0),
    )
    return _apply_scratch_mask(mask, base, scratch_shader, 0.0005)


def scratches_light_varnish_preset(vector: t.SocketOrVal[pf.Vector]) -> pf.Material:
    base = wood_grain.wood_grain_varnished_preset(vector)
    mask = scratches.scratches_light_varnish_mask_preset(vector)
    scratch_shader = pf.nodes.shader.diffuse_bsdf(
        color=pf.Color((0.80003154, 0.4295859, 0.16547439)),
        normal=(0.0, 0.0, 0.0),
    )
    return _apply_scratch_mask(mask, base, scratch_shader, 0.0005)


def scratches_shallow_preset(vector: t.SocketOrVal[pf.Vector]) -> pf.Material:
    base = wood_grain.wood_grain_brown_preset(vector)
    mask = scratches.scratches_shallow_mask_preset(vector)
    return _apply_scratch_mask(mask, base, base.surface, 0.0003)
