# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging

import numpy as np
import procfunc as pf
from procfunc import types as t

from infinigen2.cameras import camera_with_distance_framing_objects
from infinigen2.lighting import sky_lighting
from infinigen2.objects.dev import banana
from infinigen2.scenes.demo_util import (
    DevSceneResult,
    demo_cube,
    grid_plane,
    hardcoded_camera,
    scale_reference,
)
from infinigen2.shaders.dev import developer_grid

__all__ = [
    "material_banana",
    "material_cube",
    "material_monkey",
    "material_plane_horizontal_uv",
    "material_plane_orthographic",
    "material_plane_uv",
    "material_sphere",
    "material_torus_uv",
]

logger = logging.getLogger(__name__)


def _demo_sky() -> pf.World:
    return sky_lighting.nishita_sky(
        sun_rotation_deg=260, sun_elevation_deg=30
    ).environment


@pf.tracer.grammar
def material_sphere(
    rng: pf.RNG,
    material: pf.Material | None = None,
    environment: pf.World | None = None,
    subdivisions: int = 4,
    radius: float = 0.75,
) -> DevSceneResult:
    sphere = pf.ops.primitives.mesh_uv_sphere(
        radius=radius,
        location=t.Vector((0, 0, radius)),
    )
    pf.ops.modifier.subdivide_surface(sphere, levels=subdivisions, _skip_apply=True)

    if material is None:
        material = developer_grid(vector=pf.nodes.shader.coord().generated)

    pf.ops.object.set_material(sphere, material=material)

    if environment is None:
        environment = _demo_sky()

    cam = hardcoded_camera(base_location=sphere.item().location, dist_mult=0.7)
    plane = grid_plane()
    ref = scale_reference(location=t.Vector((0.38, 0.76, -0.05)))

    return DevSceneResult(
        environment=environment, all_objects=[sphere, plane, ref], cameras=[cam]
    )


@pf.tracer.grammar
def material_torus_uv(
    rng: pf.RNG,
    material: pf.Material | None = None,
    environment: pf.World | None = None,
    major_radius: float = 0.5,
    minor_radius: float = 0.25,
) -> DevSceneResult:
    obj = pf.ops.primitives.mesh_torus(
        major_segments=256,
        minor_segments=128,
        minor_radius=minor_radius,
        major_radius=major_radius,
        rotation=np.deg2rad(np.array((-40, -25, 140))),
    )

    diameters = 2 * np.pi * np.array([major_radius, minor_radius])
    uvs = pf.ops.attr.uv_coords(obj) * diameters.reshape(1, 2)
    pf.ops.attr.write_uv_coords(obj, uvs)

    pf.ops.mesh.transform_apply(obj)
    pf.ops.modifier.subdivide_surface(obj, levels=3, _skip_apply=True)
    obj.item().location.z = obj.item().dimensions.z / 2

    if material is None:
        logger.warning("No material provided; using a default material.")
        material = developer_grid(vector=pf.nodes.shader.coord().uv)

    pf.ops.object.set_material(obj, material=material)

    if environment is None:
        environment = _demo_sky()

    cam = hardcoded_camera(base_location=obj.item().location, dist_mult=0.65)
    plane = grid_plane()
    ref = scale_reference(location=t.Vector((0.38, 0.76, -0.05)))

    return DevSceneResult(
        environment=environment, all_objects=[obj, plane, ref], cameras=[cam]
    )


@pf.tracer.grammar
def material_plane_uv(
    rng: pf.RNG,
    material: pf.Material | None = None,
    environment: pf.World | None = None,
) -> DevSceneResult:
    size = 2
    obj = pf.ops.primitives.mesh_plane(
        location=t.Vector((0, 0, 1)),
        size=size,
        rotation=t.Euler((np.pi / 2, 0, 0)),
    )

    uvs = pf.ops.attr.uv_coords(obj)
    pf.ops.attr.write_uv_coords(obj, uvs * size)

    pf.ops.mesh.subdivide(obj, number_cuts=100)
    pf.ops.modifier.subdivide_surface(obj, levels=3, _skip_apply=True)

    if material is None:
        logger.warning("No material provided; using a default material.")
        material = developer_grid(vector=pf.nodes.shader.coord().uv)

    pf.ops.object.set_material(obj, material=material)
    cam = hardcoded_camera(
        base_location=obj.item().location,
        dist_mult=0.8,
        yaw_offset_deg=-3,
    )
    pf.ops.mesh.transform_apply(obj)

    plane = grid_plane()
    if environment is None:
        environment = _demo_sky()
    ref = scale_reference(location=t.Vector((0.38, 0.76, -0.05)))

    return DevSceneResult(
        environment=environment, all_objects=[obj, plane, ref], cameras=[cam]
    )


@pf.tracer.grammar
def material_plane_horizontal_uv(
    rng: pf.RNG,
    material: pf.Material | None = None,
    environment: pf.World | None = None,
) -> DevSceneResult:
    size = 1
    obj = pf.ops.primitives.mesh_plane(
        location=t.Vector((0, 0, 0.02)),
        size=size,
        rotation=t.Euler((0, 0, 0)),
    )

    uvs = pf.ops.attr.uv_coords(obj)
    pf.ops.attr.write_uv_coords(obj, uvs * size)

    pf.ops.mesh.subdivide(obj, number_cuts=100)
    pf.ops.modifier.subdivide_surface(obj, levels=3)

    if material is None:
        logger.warning("No material provided; using a default material.")
        material = developer_grid(vector=pf.nodes.shader.coord().uv)

    pf.ops.object.set_material(obj, material=material)
    plane = grid_plane()
    ref = scale_reference(location=t.Vector((0.65, 0.0, -0.05)))

    cam = camera_with_distance_framing_objects(
        [obj], t.Vector((-0.7, -1.0, 1.2)), margin_pct=-0.425
    )

    if environment is None:
        environment = _demo_sky()

    return DevSceneResult(
        environment=environment, all_objects=[obj, plane, ref], cameras=[cam]
    )


@pf.tracer.grammar
def material_monkey(
    rng: pf.RNG,
    material: pf.Material | None = None,
    environment: pf.World | None = None,
) -> DevSceneResult:
    s = 0.3
    obj = pf.ops.primitives.mesh_monkey(
        size=s,
        location=t.Vector((0, 0, s / 2)),
        rotation=t.Euler((0, 0, np.pi * 0.1)),
    )
    pf.ops.modifier.subdivide_surface(obj, levels=5, _skip_apply=True)

    if material is None:
        logger.warning("No material provided; using a default material.")
        material = developer_grid(vector=pf.nodes.shader.coord().generated)
    pf.ops.object.set_material(obj, material=material)

    cam = hardcoded_camera(base_location=obj.item().location, dist_mult=0.15)
    plane = grid_plane()
    if environment is None:
        environment = _demo_sky()
    ref = pf.ops.primitives.mesh_cylinder(
        radius=0.1, depth=0.02, location=t.Vector((0.1, 0.1, 0.01)), vertices=128
    )

    return DevSceneResult(
        environment=environment, all_objects=[obj, plane, ref], cameras=[cam]
    )


def _orthographic_camera_top_down(size: float, height: float = 5.0) -> pf.CameraObject:
    cam = pf.ops.primitives.perspective_camera()
    cam.item().data.type = "ORTHO"
    cam.item().data.ortho_scale = size
    cam.item().location = t.Vector((0.0, 0.0, height))
    cam.item().rotation_euler = t.Euler((0.0, 0.0, 0.0))
    return cam


@pf.tracer.grammar
def material_plane_orthographic(
    rng: pf.RNG,
    material: pf.Material | None = None,
    environment: pf.World | None = None,
    size: float = 1.333,
) -> DevSceneResult:
    """Orthographic top-down plane scene for pixel-perfect GT validation.

    Every pixel sees the same flat plane face: object index is uniform
    and all surface normals point straight up (+Z world space).
    """
    obj = pf.ops.primitives.mesh_plane(
        location=t.Vector((0, 0, 0)),
        size=size,
    )

    if material is None:
        logger.warning("No material provided; using a default material.")
        material = developer_grid(vector=pf.nodes.shader.coord().generated)

    pf.ops.object.set_material(obj, material=material)

    cam = _orthographic_camera_top_down(size=size)
    if environment is None:
        environment = sky_lighting.nishita_sky().environment

    return DevSceneResult(environment=environment, all_objects=[obj], cameras=[cam])


@pf.tracer.grammar
def material_cube(
    rng: pf.RNG,
    material: pf.Material | None = None,
    environment: pf.World | None = None,
) -> DevSceneResult:
    s = 0.5
    obj = demo_cube(size=s)

    if material is None:
        logger.warning("No material provided; using a default material.")
        material = developer_grid(vector=pf.nodes.shader.coord().generated)
    pf.ops.object.set_material(obj, material=material)

    horiz = t.Vector((5, -4, 0))
    cam_elevation_deg = 45
    cam_dir = t.Vector((5, -4, horiz.length * np.tan(np.deg2rad(cam_elevation_deg))))
    cam = camera_with_distance_framing_objects([obj], cam_dir, margin_pct=0.05)
    plane = grid_plane()
    if environment is None:
        environment = _demo_sky()
    angle = np.pi * 0.15
    half = s / 2
    corner_x = half * (np.cos(angle) - np.sin(angle))
    corner_y = half * (np.sin(angle) + np.cos(angle))
    ref = pf.ops.primitives.mesh_cylinder(
        radius=0.1,
        depth=0.02,
        location=t.Vector((corner_x, corner_y, 0.01)),
        vertices=128,
    )

    return DevSceneResult(
        environment=environment, all_objects=[obj, plane, ref], cameras=[cam]
    )


@pf.tracer.grammar
def material_banana(
    rng: pf.RNG,
    material: pf.Material | None = None,
    environment: pf.World | None = None,
) -> DevSceneResult:
    obj = banana()
    obj.item().location.z = obj.item().dimensions.z / 2
    pf.ops.modifier.subdivide_surface(obj, levels=3, _skip_apply=True)

    if material is None:
        logger.warning("No material provided; using a default material.")
        material = developer_grid(vector=pf.nodes.shader.coord().generated)
    pf.ops.object.set_material(obj, material=material)

    cam = hardcoded_camera(
        base_location=obj.item().location,
        dist_mult=0.06,
    )
    plane = grid_plane()
    if environment is None:
        environment = _demo_sky()
    ref = scale_reference(location=t.Vector((0.38, 0.76, -0.05)))
    return DevSceneResult(
        environment=environment, all_objects=[obj, plane, ref], cameras=[cam]
    )
