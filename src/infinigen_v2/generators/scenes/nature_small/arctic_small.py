import logging
from dataclasses import dataclass

import numpy as np
import procfunc as pf
from procfunc.util import log

from infinigen_v2.generators.lighting import sky_lighting
from infinigen_v2.generators.shaders import ice, water
from infinigen_v2.generators.terrain import ant_landscape

logger = logging.getLogger(__name__)


@pf.tracer.primitive
def generate_aliased_objects(
    obj: pf.MeshObject,
    positions: np.ndarray,
    rotations: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    suffix: str = "",
) -> list[pf.MeshObject]:
    n = len(positions)

    if rotations is None:
        rotations = np.zeros((n, 3), dtype=np.float32)
    if scales is None:
        scales = np.ones((n, 3), dtype=np.float32)

    aliases = []
    for i in range(n):
        alias_obj = pf.ops.object.alias(obj, suffix=f"{suffix}.{i}")
        alias_obj.location = positions[i]
        alias_obj.rotation_euler = rotations[i]
        alias_obj.scale = scales[i]

        aliases.append(alias_obj)

    return aliases


@pf.tracer.primitive
def generate_iceberg(
    heightmap: pf.MeshObject,
    material: pf.Material,
    mirror_cutoff_thresh: float = 0.065,
) -> pf.MeshObject:
    geo = pf.nodes.geo.object_info(heightmap).geometry

    bbox = pf.nodes.geo.bound_box(geo)
    zcoord = pf.nodes.math.separate_xyz(pf.nodes.geo.input_position()).z

    zplane = pf.nodes.math.vector_multiply_add(
        bbox.max, (0, 0, mirror_cutoff_thresh), bbox.min
    )
    zthresh = pf.nodes.math.separate_xyz(zplane).z
    selection = pf.nodes.func.greater_than(zcoord, zthresh)
    geo = pf.nodes.geo.separate_geometry(geo, selection, domain="FACE")

    toff = pf.nodes.math.vector_multiply(zplane, (0, 0, -1))
    geo = pf.nodes.geo.transform(
        geo, translation=toff, rotation=(0, 0, 0), scale=(1, 1, 1)
    )

    mirror = pf.nodes.geo.transform(
        geo, scale=(1, 1, -1), rotation=(0, 0, 0), translation=(0, 0, 0)
    )
    mirror = pf.nodes.geo.flip_faces(mirror)
    geo = pf.nodes.geo.join_geometry([geo, mirror])

    geo = pf.nodes.geo.set_material(geo, material)

    iceberg = pf.nodes.to_object(geo)
    return iceberg


@pf.tracer.primitive
def generate_water_bounds(
    scene_dim_meters: float,
    mesh_resolution: float,
    material: pf.Material,
) -> pf.MeshObject:
    n_subdivisions = log.clamp_with_log(
        int(scene_dim_meters / mesh_resolution),
        logger,
        "water_subdivisions",
        min=0,
        max=16000,
    )

    grid = pf.ops.primitives.grid(
        x_subdivisions=n_subdivisions,
        y_subdivisions=n_subdivisions,
        size=scene_dim_meters,
        location=(0, 0, 0),
    )

    pf.ops.object.set_material(grid, material)
    return grid


@dataclass
class IcebergParameters:
    subdivision_x: int = 100
    subdivision_y: int = 100
    mirror_cutoff_thresh: float = 0.065


def iceberg_distribution(rng: np.random.Generator) -> IcebergParameters:
    return IcebergParameters(
        subdivision_x=rng.integers(50, 200),
        subdivision_y=rng.integers(50, 200),
        mirror_cutoff_thresh=rng.uniform(0.03, 0.1),
    )


def sample_iceberg(
    rng: np.random.Generator,
    subdivision_x: int | None = None,
    subdivision_y: int | None = None,
    mirror_cutoff_thresh: float | None = None,
) -> pf.MeshObject:
    if subdivision_x is None or subdivision_y is None:
        params = iceberg_distribution(rng)
        subdivision_x = subdivision_x or params.subdivision_x
        subdivision_y = subdivision_y or params.subdivision_y
        mirror_cutoff_thresh = mirror_cutoff_thresh or params.mirror_cutoff_thresh

    heightmap = ant_landscape.sample_mesa_object(
        rng, subdivision_x=subdivision_x, subdivision_y=subdivision_y
    )
    material = ice.sample_ice_material(rng)

    return generate_iceberg(
        heightmap=heightmap,
        material=material,
        mirror_cutoff_thresh=mirror_cutoff_thresh,
    )


def sample_water_bounds(
    rng: np.random.Generator,
    scene_dim_meters: float,
    mesh_resolution: float,
) -> pf.MeshObject:
    material = water.sample_water_material(rng)
    return generate_water_bounds(
        scene_dim_meters=scene_dim_meters,
        mesh_resolution=mesh_resolution,
        material=material,
    )


@dataclass
class ArcticSmallSceneParameters:
    seabed_zpos: float = -15.0
    scene_dim_meters: float = 50.0
    mesh_resolution: float = 0.05
    num_large_icebergs: int = 3
    num_floating_pieces: int = 15


def arctic_small_scene_distribution(
    rng: np.random.Generator,
) -> ArcticSmallSceneParameters:
    return ArcticSmallSceneParameters(
        seabed_zpos=rng.uniform(-25.0, -5.0),
        scene_dim_meters=50.0,
        mesh_resolution=0.05,
        num_large_icebergs=rng.integers(1, 5),
        num_floating_pieces=rng.integers(5, 30),
    )


def sample_arctic_small_scene(
    rng: np.random.Generator,
    **overrides,
) -> pf.Collection:
    params = arctic_small_scene_distribution(rng)

    # Override any provided parameters
    for key, value in overrides.items():
        if hasattr(params, key):
            setattr(params, key, value)

    # Calculate terrain subdivisions for the iceberg object
    tile_dim = params.scene_dim_meters / 3
    tile_subdivisions = log.clamp_with_log(
        int(tile_dim / params.mesh_resolution),
        logger,
        "tile_subdivisions",
        min=0,
        max=16000,
    )
    iceberg_obj = sample_iceberg(
        rng, subdivision_x=tile_subdivisions, subdivision_y=tile_subdivisions
    )

    # Calculate seabed subdivisions for the large landscape
    seabed_subdivisions = log.clamp_with_log(
        int(params.scene_dim_meters / params.mesh_resolution),
        logger,
        "seabed_subdivisions",
        min=0,
        max=4096,
    )
    seabed_obj = ant_landscape.sample_ant_landscape_object(
        rng, subdivision_x=seabed_subdivisions, subdivision_y=seabed_subdivisions
    )

    water_bounds_obj = sample_water_bounds(
        rng,
        scene_dim_meters=params.scene_dim_meters + 50,
        mesh_resolution=params.mesh_resolution,
    )

    sky_lighting.sample_sky_with_sun_lamp(rng)

    # Generate internal aliasing parameters
    large_base_xy = rng.uniform(15, 30)
    large_base_z_ratio = rng.uniform(0.7, 2)
    large_scale_variation = rng.uniform(0.1, 0.3)
    large_size_variation = rng.uniform(0.1, 0.5)
    large_submersion = rng.uniform(0, 0.1)

    floating_base_xy = rng.uniform(5, 15)
    floating_base_z = rng.uniform(1, 3)
    floating_scale_variation = rng.uniform(0.1, 0.3)
    floating_size_variation = rng.uniform(0.1, 0.5)
    floating_submersion = rng.uniform(0, 0.1)

    # Set up seabed position and scale
    seabed_obj.location.z = params.seabed_zpos
    seabed_obj.scale = [params.scene_dim_meters / 2, params.scene_dim_meters / 2, 1.0]

    # Generate large icebergs
    base_xy = large_base_xy
    base_z = base_xy * large_base_z_ratio

    large_scales = (
        np.array([base_xy, base_xy, base_z])[None]
        * rng.uniform(
            1 - large_scale_variation,
            1 + large_scale_variation,
            size=(params.num_large_icebergs, 1),
        )
        * rng.uniform(
            1 - large_size_variation,
            1 + large_size_variation,
            size=(params.num_large_icebergs, 3),
        )
    )
    large_positions = rng.uniform(
        [-params.scene_dim_meters / 2, -params.scene_dim_meters / 2, 0],
        [params.scene_dim_meters / 2, params.scene_dim_meters / 2, 0],
        size=(params.num_large_icebergs, 3),
    )
    large_positions[:, -1] -= large_scales[:, -1] * rng.uniform(
        0, large_submersion, params.num_large_icebergs
    )
    large_rotations = rng.uniform(
        0, np.array([0, 0, 2 * np.pi])[None], size=(params.num_large_icebergs, 3)
    )
    aliased_large = generate_aliased_objects(
        obj=iceberg_obj,
        positions=large_positions,
        rotations=large_rotations,
        scales=large_scales,
        suffix="large_iceberg",
    )

    # Generate floating pieces
    base_xy = floating_base_xy
    base_z = floating_base_z
    floating_scales = (
        np.array([base_xy, base_xy, base_z])[None]
        * rng.uniform(
            1 - floating_scale_variation,
            1 + floating_scale_variation,
            size=(params.num_floating_pieces, 1),
        )
        * rng.uniform(
            1 - floating_size_variation,
            1 + floating_size_variation,
            size=(params.num_floating_pieces, 3),
        )
    )

    floating_positions = rng.uniform(
        [-params.scene_dim_meters / 2, -params.scene_dim_meters / 2, -0.05],
        [params.scene_dim_meters / 2, params.scene_dim_meters / 2, 0],
        size=(params.num_floating_pieces, 3),
    )
    floating_positions[:, -1] -= floating_scales[:, -1] * rng.uniform(
        0, floating_submersion, params.num_floating_pieces
    )
    floating_rotations = rng.uniform(
        0, np.array([0, 0, 2 * np.pi])[None], size=(params.num_floating_pieces, 3)
    )
    aliased_floating = generate_aliased_objects(
        obj=iceberg_obj,
        positions=floating_positions,
        scales=floating_scales,
        rotations=floating_rotations,
        suffix="floating_iceberg",
    )

    return pf.Collection(
        [seabed_obj, aliased_large, aliased_floating, water_bounds_obj]
    )
