# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma: original Infinigen v1 API binding for A.N.T. Landscape (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/terrain/assets/landtiles/ant_landscape.py)
# - Mingzhe Wang, Alexander Raistrick: refactor for Infinigen2

import logging
import math
from collections.abc import Callable
from typing import Any, NamedTuple, TypedDict

import bpy
import procfunc as pf
from procfunc.ops.addons import require_blender_addon
from procfunc.util import log
from typing_extensions import Unpack

from infinigen2.shaders.functionality_lists import terrain_material_rand

__all__ = [
    "LandscapeParameters",
    "LandscapeResult",
    "landscape",
    "landscape_canyon_rand",
    "landscape_cliff_rand",
    "landscape_generalized_rand",
    "landscape_mesa_rand",
    "landscape_mountain_rand",
    "landscape_rand",
    "landscape_rand_from_params",
    "landscape_river_rand",
    "landscape_volcano_rand",
]

logger = logging.getLogger(__name__)


class LandscapeParameters(TypedDict, total=False):
    noise_offset_x: float
    noise_offset_y: float
    noise_offset_z: float
    noise_size: float
    noise_size_x: float
    noise_size_y: float
    noise_type: str
    basis_type: str
    vl_basis_type: str
    distortion: float
    hard_noise: str
    noise_depth: int
    amplitude: float
    frequency: float
    dimension: float
    lacunarity: float
    offset: float
    gain: float
    marble_bias: str
    marble_sharp: str
    marble_shape: str
    height: float
    height_invert: bool
    height_offset: float
    fx_mixfactor: float
    fx_mix_mode: str
    fx_type: str
    fx_bias: str
    fx_turb: float
    fx_depth: int
    fx_amplitude: float
    fx_frequency: float
    fx_size: float
    fx_loc_x: float
    fx_loc_y: float
    fx_height: float
    fx_invert: bool
    fx_offset: float
    edge_falloff: str
    falloff_x: float
    falloff_y: float
    edge_level: float
    maximum: float
    minimum: float
    strata: float
    strata_type: str
    water_level: float


class LandscapeResult(NamedTuple):
    mesh: pf.MeshObject


def _remove_objects(objects: set[bpy.types.Object]) -> None:
    meshes = [obj.data for obj in objects if obj.type == "MESH"]
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _execute_landscape(**kwargs: Any) -> set[str]:
    return bpy.ops.mesh.landscape_add("EXEC_DEFAULT", **kwargs)


@pf.tracer.primitive
def landscape(
    material: pf.Material | None = None,
    **kwargs: Any,
) -> pf.MeshObject:
    """Create one mesh with the ANT Landscape operator."""
    require_blender_addon("antlandscape", allow_online=True)
    from bl_ext.blender_org import (  # keep-local
        antlandscape as ant_landscape_addon,
    )

    if kwargs.get("water_plane") is True:
        raise ValueError("landscape returns one mesh and cannot create a water plane")
    kwargs.setdefault("refresh", True)

    if not ant_landscape_addon.add_mesh_ant_landscape.AntAddLandscape.is_registered:
        ant_landscape_addon.register()

    before = set(bpy.data.objects)
    try:
        result = _execute_landscape(**kwargs)
    except Exception:
        _remove_objects(set(bpy.data.objects) - before)
        raise
    created = set(bpy.data.objects) - before
    if "FINISHED" not in result:
        _remove_objects(created)
        raise RuntimeError(f"mesh.landscape_add returned {result}")
    if len(created) != 1:
        _remove_objects(created)
        raise RuntimeError(
            f"mesh.landscape_add created {len(created)} objects; expected one"
        )

    obj = bpy.context.active_object
    if obj is None or obj not in created:
        _remove_objects(created)
        raise RuntimeError("mesh.landscape_add did not create a new active object")
    if obj.type != "MESH":
        _remove_objects(created)
        raise TypeError(f"mesh.landscape_add created {obj.type}, not MESH")
    if obj.mode != "OBJECT":
        mode_result = bpy.ops.object.mode_set(mode="OBJECT")
        if "FINISHED" not in mode_result:
            _remove_objects(created)
            raise RuntimeError("mesh.landscape_add could not enter Object mode")
    mesh = pf.MeshObject(obj)
    if material is not None:
        pf.ops.object.set_material(mesh, material=material)
    return mesh


def _scale_absolute_params(
    kwargs: dict[str, Any], scale_xy: float, scale_z: float
) -> None:
    """Rescale the length-valued operator params off the addon's native 2-unit box.

    ANT is unit-agnostic: its presets describe a shape on a 2x2 domain. Only these
    params carry length; everything else (strata, falloff exponents, and the whole
    fx layer, which is applied before the global height multiply) is dimensionless
    and must be left alone. The addon's own default -> default_large preset pair is
    exactly this rescale.
    """
    horizontal = {"noise_size": 1.0, "fx_size": 1.0}
    vertical = {
        "height": 0.5,
        "height_offset": 0.0,
        "edge_level": 0.0,
        "maximum": 1.0,
        "minimum": -1.0,
    }
    for defaults, scale in [(horizontal, scale_xy), (vertical, scale_z)]:
        for name, default in defaults.items():
            kwargs[name] = kwargs.get(name, default) * scale


def landscape_rand_from_params(
    rng: pf.RNG,
    dimensions: pf.Vector = pf.Vector((150, 150, 150)),
    mesh_resolution: float = 0.3,
    vert_group: str = "",
    material: pf.Material | None = None,
    **parameters: Unpack[LandscapeParameters],
) -> LandscapeResult:
    rng_landscape, rng_material = rng.spawn(2)
    if material is None:
        vector = pf.nodes.shader.coord().object
        material = terrain_material_rand(rng_material, vector)
    subdivision_x = log.clamp_with_log(
        math.ceil(dimensions.x / mesh_resolution),
        logger,
        "subdivision_x",
        max=2000,
    )
    subdivision_y = log.clamp_with_log(
        math.ceil(dimensions.y / mesh_resolution),
        logger,
        "subdivision_y",
        max=2000,
    )
    operator_kwargs: dict[str, Any] = {
        "land_material": "",
        "water_material": "",
        "texture_block": "",
        "at_cursor": False,
        "smooth_mesh": False,
        "tri_face": False,
        "sphere_mesh": False,
        "water_plane": False,
        "remove_double": False,
        "show_main_settings": False,
        "show_noise_settings": False,
        "show_displace_settings": False,
        "auto_refresh": False,
        "noise_offset_x": pf.random.uniform(rng_landscape, -100, 100),
        "noise_offset_y": pf.random.uniform(rng_landscape, -100, 100),
        "noise_offset_z": pf.random.uniform(rng_landscape, -100, 100),
        "refresh": True,
    }
    operator_kwargs.update(parameters)
    _scale_absolute_params(
        operator_kwargs,
        scale_xy=(dimensions.x + dimensions.y) / 4.0,
        scale_z=dimensions.z / 2.0,
    )
    operator_kwargs.update(
        random_seed=pf.random.randint(rng_landscape, 0, 1_000_000),
        mesh_size_x=dimensions.x,
        mesh_size_y=dimensions.y,
        subdivision_x=subdivision_x,
        subdivision_y=subdivision_y,
        ant_terrain_name=landscape.__name__,
        vert_group=vert_group,
    )
    mesh = landscape(material=material, **operator_kwargs)
    return LandscapeResult(mesh=mesh)


def landscape_generalized_rand(
    rng: pf.RNG,
    dimensions: pf.Vector = pf.Vector((150, 150, 150)),
    mesh_resolution: float = 0.3,
    material: pf.Material | None = None,
    **overrides: Unpack[LandscapeParameters],
) -> LandscapeResult:
    rng_landscape, rng_params = rng.spawn(2)
    params: LandscapeParameters = {
        "noise_size": pf.random.uniform(rng_params, 0.5, 1.5),
        "distortion": pf.random.uniform(rng_params, 0.5, 2.0),
        "noise_depth": pf.random.randint(rng_params, 7, 13),
        "amplitude": pf.random.uniform(rng_params, 0.4, 0.5),
        "frequency": pf.random.uniform(rng_params, 1.7, 2.0),
        "gain": pf.random.randint(rng_params, 1, 5),
        "height": pf.random.uniform(rng_params, 0.2, 1.8),
        "height_offset": pf.random.uniform(rng_params, -0.15, 0.2),
        "maximum": pf.random.uniform(rng_params, 0.25, 1.25),
        "minimum": pf.random.uniform(rng_params, -1.0, -0.2),
        "fx_frequency": pf.random.uniform(rng_params, 1.4, 1.9),
        "fx_amplitude": pf.random.uniform(rng_params, 0.38, 0.5),
        "fx_depth": pf.random.randint(rng_params, 0, 4),
        "fx_height": pf.random.uniform(rng_params, 0.25, 1.0),
        "fx_size": pf.random.uniform(rng_params, 1.0, 1.5),
        "fx_loc_x": pf.random.uniform(rng_params, -1.0, 3.0),
        "fx_loc_y": pf.random.uniform(rng_params, 0.0, 2.0),
        "fx_offset": pf.random.uniform(rng_params, 0.0, 0.06),
        "fx_turb": pf.random.uniform(rng_params, 0.0, 0.5),
        "falloff_x": pf.random.uniform(rng_params, 2.0, 40.0),
        "falloff_y": pf.random.uniform(rng_params, 2.0, 40.0),
        "edge_level": pf.random.uniform(rng_params, 0.0, 0.15),
        "strata": pf.random.uniform(rng_params, 1.0, 11.0),
    }
    params.update(overrides)
    return landscape_rand_from_params(
        rng_landscape,
        dimensions=dimensions,
        mesh_resolution=mesh_resolution,
        material=material,
        **params,
    )


def landscape_canyon_rand(
    rng: pf.RNG,
    dimensions: pf.Vector = pf.Vector((150, 150, 150)),
    mesh_resolution: float = 0.3,
    material: pf.Material | None = None,
) -> LandscapeResult:
    rng_landscape, rng_params = rng.spawn(2)
    return landscape_rand_from_params(
        rng_landscape,
        dimensions=dimensions,
        mesh_resolution=mesh_resolution,
        material=material,
        noise_offset_y=-0.25,
        noise_size_y=1.25,
        noise_size=1.5,
        noise_type="marble_noise",
        distortion=pf.random.normal(rng_params, 2, 0.05),
        hard_noise="1",
        noise_depth=12,
        marble_shape="4",
        height=0.6,
        fx_mix_mode="8",
        fx_type="20",
        fx_depth=3,
        fx_frequency=1.65,
        fx_size=1.5,
        fx_loc_x=3,
        fx_loc_y=2,
        fx_height=0.25,
        fx_offset=0.05,
        edge_falloff="2",
        edge_level=0.15,
        maximum=0.5,
        minimum=-0.2,
        strata_type="2",
        strata=pf.random.randint(rng_params, 6, 12),
    )


def landscape_cliff_rand(
    rng: pf.RNG,
    dimensions: pf.Vector = pf.Vector((150, 150, 150)),
    mesh_resolution: float = 0.3,
    material: pf.Material | None = None,
) -> LandscapeResult:
    rng_landscape, rng_params = rng.spawn(2)
    return landscape_rand_from_params(
        rng_landscape,
        dimensions=dimensions,
        mesh_resolution=mesh_resolution,
        material=material,
        noise_offset_y=-0.88,
        noise_offset_z=3.72529e-09,
        noise_size_x=2,
        noise_size_y=2,
        noise_type="marble_noise",
        basis_type="VORONOI_F2F1",
        distortion=pf.random.normal(rng_params, 0.5, 0.01),
        noise_depth=7,
        marble_shape="6",
        height=1.8,
        height_offset=-0.15,
        fx_height=0.5,
        edge_falloff="0",
        falloff_x=25,
        falloff_y=25,
        maximum=1.25,
        minimum=0,
        strata=11,
    )


def landscape_mesa_rand(
    rng: pf.RNG,
    dimensions: pf.Vector = pf.Vector((150, 150, 150)),
    mesh_resolution: float = 0.3,
    material: pf.Material | None = None,
) -> LandscapeResult:
    rng_landscape, rng_params = rng.spawn(2)
    return landscape_rand_from_params(
        rng_landscape,
        dimensions=dimensions,
        mesh_resolution=mesh_resolution,
        material=material,
        noise_size=pf.random.uniform(rng_params, 0.5, 1.0),
        noise_type="shattered_hterrain",
        basis_type="VORONOI_F1",
        vl_basis_type="VORONOI_F2F1",
        distortion=pf.random.normal(rng_params, 1.15, 0.01),
        hard_noise="1",
        amplitude=0.4,
        gain=4,
        height_offset=0.2,
        fx_frequency=pf.random.uniform(rng_params, 1.4, 1.6),
        fx_height=0.5,
        edge_falloff="3",
        falloff_x=3,
        falloff_y=3,
        maximum=0.25,
        minimum=0,
        strata=2.25,
        strata_type="2",
    )


def landscape_river_rand(
    rng: pf.RNG,
    dimensions: pf.Vector = pf.Vector((150, 150, 150)),
    mesh_resolution: float = 0.3,
    material: pf.Material | None = None,
) -> LandscapeResult:
    rng_landscape, rng_params = rng.spawn(2)
    return landscape_rand_from_params(
        rng_landscape,
        dimensions=dimensions,
        mesh_resolution=mesh_resolution,
        material=material,
        noise_type="marble_noise",
        marble_bias="2",
        marble_shape="7",
        height=0.2,
        fx_frequency=pf.random.uniform(rng_params, 1.4, 1.6),
        fx_height=0.5,
        edge_falloff="0",
        falloff_x=40,
        falloff_y=40,
        maximum=0.5,
        minimum=0,
        strata=pf.random.uniform(rng_params, 1, 1.5),
        strata_type="1",
    )


def landscape_volcano_rand(
    rng: pf.RNG,
    dimensions: pf.Vector = pf.Vector((150, 150, 150)),
    mesh_resolution: float = 0.3,
    material: pf.Material | None = None,
) -> LandscapeResult:
    rng_landscape, rng_params = rng.spawn(2)
    return landscape_rand_from_params(
        rng_landscape,
        dimensions=dimensions,
        mesh_resolution=mesh_resolution,
        material=material,
        noise_type="marble_noise",
        vl_basis_type="PERLIN_ORIGINAL",
        distortion=pf.random.normal(rng_params, 1.5, 0.01),
        frequency=pf.random.uniform(rng_params, 1.7, 1.9),
        gain=2,
        marble_bias="2",
        marble_sharp="3",
        marble_shape="1",
        height=0.6,
        fx_mix_mode="1",
        fx_type="14",
        fx_turb=0.5,
        fx_depth=2,
        fx_amplitude=0.38,
        fx_frequency=pf.random.uniform(rng_params, 1.4, 1.6),
        fx_size=1.15,
        fx_loc_x=-1,
        fx_loc_y=1,
        fx_height=0.5,
        fx_offset=0.06,
        edge_falloff="3",
        falloff_x=2,
        falloff_y=2,
        maximum=1,
        minimum=-1,
        strata=pf.random.randint(rng_params, 4, 6),
    )


def landscape_mountain_rand(
    rng: pf.RNG,
    dimensions: pf.Vector = pf.Vector((150, 150, 150)),
    mesh_resolution: float = 0.3,
    material: pf.Material | None = None,
) -> LandscapeResult:
    rng_landscape, rng_params = rng.spawn(2)
    return landscape_rand_from_params(
        rng_landscape,
        dimensions=dimensions,
        mesh_resolution=mesh_resolution,
        material=material,
        fx_height=1,
        edge_falloff="3",
        maximum=1,
        minimum=-1,
        strata=pf.random.randint(rng_params, 5, 10),
    )


def landscape_rand(
    rng: pf.RNG,
    dimensions: pf.Vector = pf.Vector((150, 150, 150)),
    mesh_resolution: float = 0.3,
    material: pf.Material | None = None,
) -> LandscapeResult:
    rng_choice, rng_landscape = rng.spawn(2)
    options: list[tuple[Callable[..., LandscapeResult], float]] = [
        (landscape_canyon_rand, 1.0),
        (landscape_cliff_rand, 1.0),
        (landscape_mesa_rand, 1.0),
        (landscape_river_rand, 1.0),
        (landscape_volcano_rand, 1.0),
        (landscape_mountain_rand, 1.0),
    ]
    function = pf.control.choice(rng_choice, options)
    return function(
        rng_landscape,
        dimensions=dimensions,
        mesh_resolution=mesh_resolution,
        material=material,
    )
