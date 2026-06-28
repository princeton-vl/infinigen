import math
from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen_v2.generators.shaders.composites import (
    bricks,
    fabric_patterned,
    tiles,
    wood_planks,
)
from infinigen_v2.generators.shaders.materials import (
    brick_concrete,
    carpet,
    ceramic,
    concrete,
    fabric,
    glass_colored,
    granite,
    gravel_concrete,
    marble,
    metal_brushed,
    metal_hammered,
    paint,
    plastic,
    stone_smooth,
    terrazzo,
    wood_grain,
)
from infinigen_v2.generators.util.mesh import crease_sharp


@pf.tracer.grammar
def bsdf_simple_distribution(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.Material:
    del vector
    hue = pf.random.uniform(rng, 0.0, 1.0)
    saturation = pf.random.uniform(rng, 0.05, 0.95)
    value = pf.random.uniform(rng, 0.05, 0.95)
    roughness = pf.random.uniform(rng, 0.4, 0.97)
    metallic = pf.random.uniform(rng, 0.0, 1.0)
    surface = pf.nodes.shader.principled_bsdf(
        base_color=pf.color.hsv_color(hue=hue, saturation=saturation, value=value),
        roughness=roughness,
        metallic=metallic,
    )
    return pf.Material(surface=surface, displacement=None, volume=None)


@pf.tracer.grammar
def all_materials_distribution(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.Material:
    rng_choice, rng_func = rng.spawn(2)
    func = pf.control.choice(
        rng_choice,
        [
            (brick_concrete.brick_concrete_distribution, 1.0),
            (bricks.bricks_distribution, 1.0),
            (carpet.carpet_distribution, 1.0),
            (ceramic.ceramic_distribution, 1.0),
            (concrete.concrete_distribution, 1.0),
            (fabric_patterned.fabric_patterned_distribution, 1.0),
            (fabric.fabric_distribution, 1.0),
            (glass_colored.glass_colored_distribution, 1.0),
            (granite.granite_distribution, 1.0),
            (granite.granite_smooth_distribution, 1.0),
            (gravel_concrete.gravel_concrete_distribution, 1.0),
            (marble.marble_distribution, 1.0),
            (metal_brushed.metal_brushed_linear_distribution, 1.0),
            (metal_brushed.metal_brushed_radial_distribution, 1.0),
            (metal_hammered.metal_hammered_distribution, 1.0),
            (paint.paint_distribution, 1.0),
            (plastic.plastic_grayscale_distribution, 1.0),
            (plastic.plastic_opaque_distribution, 1.0),
            (plastic.plastic_translucent_distribution, 1.0),
            (stone_smooth.stone_smooth_distribution, 1.0),
            (terrazzo.terrazzo_distribution, 1.0),
            (tiles.tile_indoor_wall_distribution, 2.0),
            (wood_grain.wood_grain_distribution, 1.0),
            (wood_planks.wood_planks_distribution, 2.0),
        ],
    )
    return func(rng_func, vector)


def _end_fill_type_distribution(rng: pf.RNG) -> str:
    return pf.control.choice(rng, [("NGON", 3.0), ("NOTHING", 1.0)])


def _cone_distribution(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_cone(
        vertices=pf.random.randint(rng, 3, 33),
        radius2=pf.random.uniform(rng, 0.0, 0.9),
        depth=pf.random.uniform(rng, 0.5, 1.0),
        end_fill_type=_end_fill_type_distribution(rng),
    )


def _cylinder_distribution(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_cylinder(
        vertices=pf.random.randint(rng, 3, 33),
        depth=pf.random.uniform(rng, 0.2, 4.0),
        end_fill_type=_end_fill_type_distribution(rng),
    )


def _circle_distribution(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_circle(
        vertices=pf.random.randint(rng, 3, 33),
        fill_type=pf.control.choice(rng, [("NGON", 1.0), ("TRIFAN", 1.0)]),
    )


def _grid_distribution(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_grid(
        x_subdivisions=pf.random.randint(rng, 2, 12),
        y_subdivisions=pf.random.randint(rng, 2, 12),
    )


def _icosphere_distribution(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_icosphere(
        subdivisions=pf.random.randint(rng, 1, 4),
    )


def _torus_distribution(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_torus(
        major_radius=pf.random.uniform(rng, 0.5, 1.5),
        minor_radius=pf.random.uniform(rng, 0.05, 0.45),
        major_segments=pf.random.randint(rng, 3, 49),
        minor_segments=pf.random.randint(rng, 3, 13),
    )


def _cube_distribution(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_cube()


def _monkey_distribution(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_monkey()


def _plane_distribution(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_plane()


def _uv_sphere_distribution(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_uv_sphere(
        segments=pf.random.randint(rng, 3, 33),
        ring_count=pf.random.randint(rng, 3, 17),
    )


@pf.nodes.node_function
def _twist_warp(
    mesh: pf.ProcNode[pf.MeshObject],
    rate: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    position = pf.nodes.geo.input_position()
    xyz = pf.nodes.math.separate_xyz(position)
    rotated = pf.nodes.math.vector_rotate_axis_angle(
        vector=position,
        axis=(0.0, 0.0, 1.0),
        angle=xyz.z * rate,
    )
    return pf.nodes.geo.set_position(geometry=mesh, position=rotated)


@pf.nodes.node_function
def _taper_warp(
    mesh: pf.ProcNode[pf.MeshObject],
    z_min: t.SocketOrVal[float],
    z_max: t.SocketOrVal[float],
    scale_bottom: t.SocketOrVal[float],
    scale_top: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    position = pf.nodes.geo.input_position()
    xyz = pf.nodes.math.separate_xyz(position)
    factor = pf.nodes.math.map_range(
        value=xyz.z,
        from_min=z_min,
        from_max=z_max,
        to_min=scale_bottom,
        to_max=scale_top,
    )
    new_position = pf.nodes.math.combine_xyz(
        x=xyz.x * factor,
        y=xyz.y * factor,
        z=xyz.z,
    )
    return pf.nodes.geo.set_position(geometry=mesh, position=new_position)


@pf.nodes.node_function
def _noise_warp(
    mesh: pf.ProcNode[pf.MeshObject],
    scale: t.SocketOrVal[float],
    strength: t.SocketOrVal[float],
    phase: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    # unconnected noise vector samples position implicitly in geometry nodes
    noise = pf.nodes.texture.noise(
        vector=None, scale=scale, noise_dimensions="4D", w=phase
    )
    centered = pf.nodes.math.vector_subtract(noise.color, (0.5, 0.5, 0.5))
    offset = pf.nodes.math.vector_scale(vector=centered, scale=strength)
    return pf.nodes.geo.set_position(geometry=mesh, offset=offset)


def _base_primitive(rng: pf.RNG) -> pf.MeshObject:
    rng_choice, rng_func = rng.spawn(2)
    func = pf.control.choice(
        rng_choice,
        [
            (_circle_distribution, 0.2),
            (_cone_distribution, 1.5),
            (_cube_distribution, 3.0),
            (_cylinder_distribution, 2.0),
            (_grid_distribution, 0.2),
            # (_icosphere_distribution, 1.0),
            (_monkey_distribution, 0.2),
            (_plane_distribution, 0.2),
            (_torus_distribution, 1.0),
            (_uv_sphere_distribution, 1.0),
        ],
    )
    obj = func(rng_func)
    obj.item().name = func.__name__.removeprefix("_")
    return obj


class _EffectResult(NamedTuple):
    mesh: pf.MeshObject
    # render-time subdivision budget; effects producing dense meshes use fewer levels
    subsurf_levels: int


# Effects create their own primitive so each choice branch only mutates
# objects it created itself.


def _effect_none(rng: pf.RNG) -> _EffectResult:
    return _EffectResult(mesh=_base_primitive(rng), subsurf_levels=5)


def _effect_bevel(rng: pf.RNG) -> _EffectResult:
    obj = _base_primitive(rng)
    pf.ops.modifier.bevel(
        obj,
        width=pf.random.uniform(rng, 0.005, 0.03),
        segments=pf.random.randint(rng, 1, 5),
    )
    return _EffectResult(mesh=obj, subsurf_levels=4)


def _effect_wireframe(rng: pf.RNG) -> _EffectResult:
    rng_base, rng_choice = rng.spawn(2)
    obj = _base_primitive(rng_base)
    pf.ops.modifier.wireframe(
        obj,
        thickness=pf.random.uniform(rng, 0.02, 0.1),
        use_replace=pf.control.choice(rng_choice, [(True, 1.0), (False, 1.0)]),
    )
    return _EffectResult(mesh=obj, subsurf_levels=3)


def _effect_solidify(rng: pf.RNG) -> _EffectResult:
    obj = _base_primitive(rng)
    pf.ops.modifier.solidify(
        obj,
        thickness=pf.random.uniform(rng, 0.03, 0.1),
        offset=pf.random.uniform(rng, -1.0, 1.0),
    )
    return _EffectResult(mesh=obj, subsurf_levels=4)


def _effect_screw_ring(rng: pf.RNG) -> _EffectResult:
    obj = _base_primitive(rng)
    pf.ops.modifier.screw(
        obj,
        angle=pf.random.uniform(rng, math.pi / 2, 2 * math.pi),
    )
    return _EffectResult(mesh=obj, subsurf_levels=3)


def _effect_screw_spiral(rng: pf.RNG) -> _EffectResult:
    obj = _base_primitive(rng)
    pf.ops.modifier.screw(
        obj,
        angle=pf.random.uniform(rng, 0.2 * math.pi, 2 * math.pi),
        iterations=pf.random.randint(rng, 2, 5),
        screw_offset=pf.random.uniform(rng, 0.3, 1.2),
    )
    return _EffectResult(mesh=obj, subsurf_levels=2)


def _effect_decimate(rng: pf.RNG) -> _EffectResult:
    obj = _base_primitive(rng)
    pf.ops.mesh.subdivide(obj, number_cuts=4)
    pf.ops.modifier.decimate_collapse(
        obj,
        ratio=pf.random.uniform(rng, 0.02, 0.2),
    )
    return _EffectResult(mesh=obj, subsurf_levels=4)


def _effect_fractal_jitter(rng: pf.RNG) -> _EffectResult:
    obj = _base_primitive(rng)
    pf.ops.mesh.subdivide(
        obj,
        number_cuts=pf.random.randint(rng, 2, 5),
        fractal=pf.random.uniform(rng, 0.3, 1.5),
        seed=pf.random.randint(rng, 0, 100000),
    )
    return _EffectResult(mesh=obj, subsurf_levels=2)


def _effect_twist(rng: pf.RNG) -> _EffectResult:
    rng_base, rng_choice = rng.spawn(2)
    obj = _base_primitive(rng_base)
    pf.ops.mesh.subdivide(obj, number_cuts=4)
    sign = pf.control.choice(rng_choice, [(1.0, 1.0), (-1.0, 1.0)])
    warped = _twist_warp(obj, rate=sign * pf.random.uniform(rng, 0.4, 1.5))
    return _EffectResult(mesh=pf.nodes.to_mesh_object(warped), subsurf_levels=3)


def _effect_taper(rng: pf.RNG) -> _EffectResult:
    obj = _base_primitive(rng)
    pf.ops.mesh.subdivide(obj, number_cuts=4)
    warped = _taper_warp(
        obj,
        z_min=-1.0,
        z_max=1.0,
        scale_bottom=pf.random.uniform(rng, 0.2, 1.5),
        scale_top=pf.random.uniform(rng, 0.2, 1.5),
    )
    return _EffectResult(mesh=pf.nodes.to_mesh_object(warped), subsurf_levels=3)


def _effect_noise_warp(rng: pf.RNG) -> _EffectResult:
    obj = _base_primitive(rng)
    pf.ops.mesh.subdivide(obj, number_cuts=4)
    warped = _noise_warp(
        obj,
        scale=pf.random.uniform(rng, 0.4, 1.5),
        strength=pf.random.uniform(rng, 0.2, 0.8),
        phase=pf.random.uniform(rng, 0.0, 100.0),
    )
    return _EffectResult(mesh=pf.nodes.to_mesh_object(warped), subsurf_levels=2)


class PrimitivesResult(NamedTuple):
    mesh: pf.MeshObject


@pf.tracer.grammar
def primitives_distribution(
    rng: pf.RNG,
    target_size: float | None = None,
) -> PrimitivesResult:
    (
        rng_effect_choice,
        rng_effect,
        rng_crease,
        rng_aspect,
        rng_rot,
        rng_scale,
        rng_mat_choice,
        rng_mat,
    ) = rng.spawn(8)
    effect_func = pf.control.choice(
        rng_effect_choice,
        [
            (_effect_none, 2.0),
            (_effect_bevel, 1.0),
            (_effect_wireframe, 1.0),
            (_effect_solidify, 0.7),
            (_effect_screw_ring, 0.5),
            (_effect_screw_spiral, 0.5),
            (_effect_decimate, 0.7),
            (_effect_fractal_jitter, 0.7),
            (_effect_twist, 1.0),
            (_effect_taper, 1.0),
            (_effect_noise_warp, 1.0),
        ],
    )
    result = effect_func(rng_effect)
    obj = result.mesh

    creased = pf.control.choice(
        rng_crease,
        [
            (
                lambda obj: pf.nodes.to_mesh_object(
                    crease_sharp(obj, threshold_degrees=5)
                ),
                0.3,
            ),
            (lambda obj: obj, 0.7),
        ],
    )(obj)
    pf.ops.modifier.subdivide_surface(
        obj, levels=result.subsurf_levels, _skip_apply=True
    )

    aspect_x = pf.random.uniform(rng_aspect, 0.6, 1.6)
    aspect_y = pf.random.uniform(rng_aspect, 0.6, 1.6)
    aspect_z = pf.random.uniform(rng_aspect, 0.6, 1.6)
    s = 1.0
    if target_size is not None:
        dims = obj.item().dimensions
        current_max = max(dims.x * aspect_x, dims.y * aspect_y, dims.z * aspect_z)
        if current_max > 0:
            s = target_size / current_max
    pf.ops.object.set_transform(
        obj, scale=pf.Vector((s * aspect_x, s * aspect_y, s * aspect_z))
    )

    rotation = pf.Vector(
        (
            pf.random.uniform(rng_rot, 0, 2 * math.pi),
            pf.random.uniform(rng_rot, 0, 2 * math.pi),
            pf.random.uniform(rng_rot, 0, 2 * math.pi),
        )
    )
    scale = pf.Vector(
        (
            pf.random.clip_gaussian(rng_scale, 1.0, 0.1, 0.3, 3.0),
            pf.random.clip_gaussian(rng_scale, 1.0, 0.1, 0.3, 3.0),
            pf.random.clip_gaussian(rng_scale, 1.0, 0.1, 0.3, 3.0),
        )
    )

    vec = pf.nodes.shader.mapping(
        vector=pf.nodes.shader.coord().uv,
        rotation=rotation,
        scale=scale,
    )
    mat_func = pf.control.choice(
        rng_mat_choice,
        [(bsdf_simple_distribution, 1.0), (all_materials_distribution, 2.0)],
    )
    mat = mat_func(rng_mat, vec)
    pf.ops.object.set_material(
        obj,
        surface=getattr(mat, "surface", None),
        displacement=getattr(mat, "displacement", None),
    )
    return PrimitivesResult(mesh=obj)
