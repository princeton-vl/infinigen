import math
from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen_v2.generators.shaders.composites import tiles, wood_planks
from infinigen_v2.generators.shaders.materials import (
    brick_concrete,
    ceramic,
    concrete,
    fabric,
    glass_colored,
    granite,
    gravel_concrete,
    marble,
    metal_brushed,
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
    func = pf.control.choice(
        rng,
        [
            (tiles.tile_indoor_wall_distribution, 2.0),
            (wood_planks.wood_planks_distribution, 2.0),
            (marble.marble_distribution, 1.0),
            (granite.granite_distribution, 1.0),
            (granite.granite_smooth_distribution, 1.0),
            (ceramic.ceramic_distribution, 1.0),
            (wood_grain.wood_grain_distribution, 1.0),
            (metal_brushed.metal_brushed_linear_distribution, 1.0),
            (metal_brushed.metal_brushed_radial_distribution, 1.0),
            (glass_colored.glass_colored_distribution, 1.0),
            (paint.paint_distribution, 1.0),
            (plastic.plastic_opaque_distribution, 1.0),
            (plastic.plastic_grayscale_distribution, 1.0),
            (concrete.concrete_distribution, 1.0),
            (fabric.fabric_distribution, 1.0),
            (terrazzo.terrazzo_distribution, 1.0),
            (gravel_concrete.gravel_concrete_distribution, 1.0),
            (stone_smooth.stone_smooth_distribution, 1.0),
            (brick_concrete.brick_concrete_distribution, 1.0),
        ],
    )
    return func(rng, vector)


def _cone_distribution(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_cone(
        radius2=pf.random.uniform(rng, 0.0, 0.8),
        depth=pf.random.clip_gaussian(rng, 2.0, 1.0, 0.5, 4.0),
    )


def _cylinder_distribution(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_cylinder(
        depth=pf.random.clip_gaussian(rng, 1.0, 0.8, 0.2, 4.0),
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
    )


def _cube_distribution(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_cube()


def _monkey_distribution(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_monkey()


def _plane_distribution(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_plane()


def _uv_sphere_distribution(rng: pf.RNG) -> pf.MeshObject:
    return pf.ops.primitives.mesh_uv_sphere()


class PrimitivesResult(NamedTuple):
    mesh: pf.MeshObject


@pf.tracer.grammar
def primitives_distribution(
    rng: pf.RNG,
    target_size: float | None = None,
) -> PrimitivesResult:
    func = pf.control.choice(
        rng,
        [
            (_cone_distribution, 1.0),
            (_cube_distribution, 1.0),
            (_cylinder_distribution, 1.0),
            (_grid_distribution, 0.5),
            (_icosphere_distribution, 1.0),
            (_monkey_distribution, 1.0),
            (_plane_distribution, 0.5),
            (_torus_distribution, 1.0),
            (_uv_sphere_distribution, 1.0),
        ],
    )
    obj = func(rng)
    obj.item().name = func.__name__.removeprefix("_")

    obj = pf.control.choice(
        rng,
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
    pf.ops.modifier.subdivide_surface(obj, levels=5, _skip_apply=True)

    if target_size is not None:
        current_max = max(obj.item().dimensions)
        if current_max > 0:
            s = target_size / current_max
            pf.ops.object.set_transform(obj, scale=pf.Vector((s, s, s)))

    rotation = pf.Vector(
        (
            pf.random.uniform(rng, 0, 2 * math.pi),
            pf.random.uniform(rng, 0, 2 * math.pi),
            pf.random.uniform(rng, 0, 2 * math.pi),
        )
    )
    scale = pf.Vector(
        (
            pf.random.clip_gaussian(rng, 1.0, 0.1, 0.3, 3.0),
            pf.random.clip_gaussian(rng, 1.0, 0.1, 0.3, 3.0),
            pf.random.clip_gaussian(rng, 1.0, 0.1, 0.3, 3.0),
        )
    )

    vec = pf.nodes.shader.mapping(
        vector=pf.nodes.shader.coord().uv,
        rotation=rotation,
        scale=scale,
    )
    mat_func = pf.control.choice(
        rng, [(bsdf_simple_distribution, 1.0), (all_materials_distribution, 2.0)]
    )
    mat = mat_func(rng, vec)
    pf.ops.object.set_material(
        obj,
        surface=getattr(mat, "surface", None),
        displacement=getattr(mat, "displacement", None),
    )
    return PrimitivesResult(mesh=obj)
