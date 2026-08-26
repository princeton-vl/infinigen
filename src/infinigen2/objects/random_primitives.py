# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import math
from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.dev import bsdf_simple_rand
from infinigen2.shaders.functionality_lists import all_materials_rand
from infinigen2.util.mesh import crease_by_angle

__all__ = [
    "EffectResult",
    "PrimitivesResult",
    "circle_rand",
    "cone_rand",
    "cube_rand",
    "cylinder_rand",
    "effect_bevel",
    "effect_decimate",
    "effect_fractal_jitter",
    "effect_noise_warp",
    "effect_none",
    "effect_screw_ring",
    "effect_screw_spiral",
    "effect_solidify",
    "effect_taper",
    "effect_twist",
    "effect_wireframe",
    "end_fill_type_rand",
    "grid_rand",
    "icosphere_rand",
    "monkey_rand",
    "plane_rand",
    "primitive_rand",
    "primitive_with_effect_rand",
    "torus_rand",
    "uv_sphere_rand",
]


def end_fill_type_rand(rng: pf.RNG) -> str:
    return pf.control.choice(rng, [("NGON", 3.0), ("NOTHING", 1.0)])


def cone_rand(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_cone(
        vertices=pf.random.randint(rng, 3, 33),
        radius2=pf.random.uniform(rng, 0.0, 0.9),
        depth=pf.random.uniform(rng, 0.5, 1.0),
        end_fill_type=end_fill_type_rand(rng),
    )


def cylinder_rand(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_cylinder(
        vertices=pf.random.randint(rng, 3, 33),
        depth=pf.random.uniform(rng, 0.2, 4.0),
        end_fill_type=end_fill_type_rand(rng),
    )


def circle_rand(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_circle(
        vertices=pf.random.randint(rng, 3, 33),
        fill_type=pf.control.choice(rng, [("NGON", 1.0), ("TRIFAN", 1.0)]),
    )


def grid_rand(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_grid(
        x_subdivisions=pf.random.randint(rng, 2, 12),
        y_subdivisions=pf.random.randint(rng, 2, 12),
    )


def icosphere_rand(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_icosphere(
        subdivisions=pf.random.randint(rng, 1, 4),
    )


def torus_rand(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_torus(
        major_radius=pf.random.uniform(rng, 0.5, 1.5),
        minor_radius=pf.random.uniform(rng, 0.05, 0.45),
        major_segments=pf.random.randint(rng, 3, 49),
        minor_segments=pf.random.randint(rng, 3, 13),
    )


def cube_rand(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_cube()


def monkey_rand(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_monkey()


def plane_rand(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_plane()


def uv_sphere_rand(rng: pf.RNG) -> pf.MeshObject:
    sphere = pf.nodes.geo.mesh_uv_sphere(
        segments=pf.random.randint(rng, 3, 33),
        rings=pf.random.randint(rng, 3, 17),
    )
    mesh = pf.nodes.geo.store_named_attribute(
        geometry=sphere.mesh,
        name="uv_map",
        value=sphere.uv_map,
        domain="CORNER",
        data_type="FLOAT2",
    )
    return pf.nodes.to_mesh_object(mesh)


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


def primitive_rand(rng: pf.RNG) -> pf.MeshObject:
    rng_choice, rng_func = rng.spawn(2)
    func = pf.control.choice(
        rng_choice,
        [
            (circle_rand, 0.2),
            (cone_rand, 1.5),
            (cube_rand, 3.0),
            (cylinder_rand, 2.0),
            (grid_rand, 0.2),
            # (icosphere_rand, 1.0),
            (monkey_rand, 0.2),
            (plane_rand, 0.2),
            (torus_rand, 1.0),
            (uv_sphere_rand, 1.0),
        ],
    )
    obj = func(rng_func)
    obj.item().name = func.__name__
    return obj


class EffectResult(NamedTuple):
    mesh: pf.MeshObject
    # render-time subdivision budget; effects producing dense meshes use fewer levels
    subsurf_levels: int


def effect_none(rng: pf.RNG, obj: pf.MeshObject) -> EffectResult:
    return EffectResult(mesh=obj, subsurf_levels=5)


def effect_bevel(rng: pf.RNG, obj: pf.MeshObject) -> EffectResult:
    pf.ops.modifier.bevel(
        obj,
        width=pf.random.uniform(rng, 0.005, 0.03),
        segments=pf.random.randint(rng, 1, 5),
    )
    return EffectResult(mesh=obj, subsurf_levels=4)


def effect_wireframe(rng: pf.RNG, obj: pf.MeshObject) -> EffectResult:
    rng_thickness, rng_choice = rng.spawn(2)
    pf.ops.modifier.wireframe(
        obj,
        thickness=pf.random.uniform(rng_thickness, 0.02, 0.1),
        use_replace=pf.control.choice(rng_choice, [(True, 1.0), (False, 1.0)]),
    )
    return EffectResult(mesh=obj, subsurf_levels=3)


def effect_solidify(rng: pf.RNG, obj: pf.MeshObject) -> EffectResult:
    pf.ops.modifier.solidify(
        obj,
        thickness=pf.random.uniform(rng, 0.03, 0.1),
        offset=pf.random.uniform(rng, -1.0, 1.0),
    )
    return EffectResult(mesh=obj, subsurf_levels=4)


def effect_screw_ring(rng: pf.RNG, obj: pf.MeshObject) -> EffectResult:
    pf.ops.modifier.screw(
        obj,
        angle=pf.random.uniform(rng, math.pi / 2, 2 * math.pi),
    )
    return EffectResult(mesh=obj, subsurf_levels=3)


def effect_screw_spiral(rng: pf.RNG, obj: pf.MeshObject) -> EffectResult:
    pf.ops.modifier.screw(
        obj,
        angle=pf.random.uniform(rng, 0.2 * math.pi, 2 * math.pi),
        iterations=pf.random.randint(rng, 2, 5),
        screw_offset=pf.random.uniform(rng, 0.3, 1.2),
    )
    return EffectResult(mesh=obj, subsurf_levels=2)


def effect_decimate(rng: pf.RNG, obj: pf.MeshObject) -> EffectResult:
    pf.ops.mesh.subdivide(obj, number_cuts=4)
    pf.ops.modifier.decimate_collapse(
        obj,
        ratio=pf.random.uniform(rng, 0.02, 0.2),
    )
    return EffectResult(mesh=obj, subsurf_levels=4)


def effect_fractal_jitter(rng: pf.RNG, obj: pf.MeshObject) -> EffectResult:
    pf.ops.mesh.subdivide(
        obj,
        number_cuts=pf.random.randint(rng, 2, 5),
        fractal=pf.random.uniform(rng, 0.3, 1.5),
        seed=pf.random.randint(rng, 0, 100000),
    )
    return EffectResult(mesh=obj, subsurf_levels=2)


def effect_twist(rng: pf.RNG, obj: pf.MeshObject) -> EffectResult:
    rng_rate, rng_choice = rng.spawn(2)
    pf.ops.mesh.subdivide(obj, number_cuts=4)
    sign = pf.control.choice(rng_choice, [(1.0, 1.0), (-1.0, 1.0)])
    warped = _twist_warp(obj, rate=sign * pf.random.uniform(rng_rate, 0.4, 1.5))
    return EffectResult(mesh=pf.nodes.to_mesh_object(warped), subsurf_levels=3)


def effect_taper(rng: pf.RNG, obj: pf.MeshObject) -> EffectResult:
    pf.ops.mesh.subdivide(obj, number_cuts=4)
    warped = _taper_warp(
        obj,
        z_min=-1.0,
        z_max=1.0,
        scale_bottom=pf.random.uniform(rng, 0.2, 1.5),
        scale_top=pf.random.uniform(rng, 0.2, 1.5),
    )
    return EffectResult(mesh=pf.nodes.to_mesh_object(warped), subsurf_levels=3)


def effect_noise_warp(rng: pf.RNG, obj: pf.MeshObject) -> EffectResult:
    pf.ops.mesh.subdivide(obj, number_cuts=4)
    warped = _noise_warp(
        obj,
        scale=pf.random.uniform(rng, 0.4, 1.5),
        strength=pf.random.uniform(rng, 0.2, 0.8),
        phase=pf.random.uniform(rng, 0.0, 100.0),
    )
    return EffectResult(mesh=pf.nodes.to_mesh_object(warped), subsurf_levels=2)


class PrimitivesResult(NamedTuple):
    mesh: pf.MeshObject


@pf.tracer.grammar
def primitive_with_effect_rand(
    rng: pf.RNG,
    target_size: float | None = None,
    max_subsurf_levels: int | None = None,
) -> PrimitivesResult:
    (
        rng_base,
        rng_effect_choice,
        rng_effect,
        rng_crease,
        rng_aspect,
        rng_rot,
        rng_scale,
        rng_mat_choice,
        rng_mat,
    ) = rng.spawn(9)

    effect_func = pf.control.choice(
        rng_effect_choice,
        [
            (effect_none, 2.0),
            (effect_bevel, 1.0),
            (effect_wireframe, 1.0),
            (effect_solidify, 0.7),
            (effect_screw_ring, 0.5),
            (effect_screw_spiral, 0.5),
            (effect_decimate, 0.7),
            (effect_fractal_jitter, 0.7),
            (effect_twist, 1.0),
            (effect_taper, 1.0),
            (effect_noise_warp, 1.0),
        ],
    )
    primitive = primitive_rand(rng_base)
    # blender dedupes the sampler's name with .00N; drop it before composing
    primitive_name = primitive.item().name.split(".")[0]
    result = effect_func(rng_effect, primitive)
    obj = result.mesh

    crease_threshold = pf.random.clip_gaussian(rng_crease, 40.0, 40.0, 0.0, 180.0)
    crease_softness = pf.random.clip_gaussian(rng_crease, 0.0, 20.0, 1.0, 60.0)
    obj = pf.nodes.to_mesh_object(
        crease_by_angle(
            obj,
            threshold_degrees=crease_threshold,
            softness_degrees=crease_softness,
        )
    )
    obj.item().name = f"{primitive_name}_{effect_func.__name__}"

    subsurf_levels = result.subsurf_levels
    if max_subsurf_levels is not None:
        subsurf_levels = min(subsurf_levels, max_subsurf_levels)
    pf.ops.modifier.subdivide_surface(obj, levels=subsurf_levels, _skip_apply=True)

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
        [(bsdf_simple_rand, 1.0), (all_materials_rand, 2.0)],
    )
    mat = mat_func(rng_mat, vec)
    pf.ops.object.set_material(
        obj,
        surface=getattr(mat, "surface", None),
        displacement=getattr(mat, "displacement", None),
    )
    return PrimitivesResult(mesh=obj)
