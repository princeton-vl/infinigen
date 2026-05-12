import logging
from typing import NamedTuple

import numpy as np
import procfunc as pf
from procfunc import types as t

from infinigen_v2.generators.cameras import camera_with_distance_framing_object
from infinigen_v2.generators.lighting import sky_lighting
from infinigen_v2.generators.objects.dev import banana
from infinigen_v2.generators.shaders.dev import developer_grid

logger = logging.getLogger(__name__)


class DevSceneResult(NamedTuple):
    all_objects: list
    cameras: list
    lights: list


@pf.tracer.generator
def scale_reference(
    location: np.ndarray,
    radius: float = 0.3,
) -> pf.MeshObject:
    height = 1.65
    location = np.array(location) + np.array((0, 0, height / 2 - 0.05))
    res = pf.ops.primitives.mesh_cylinder(
        radius=radius,
        depth=height,
        location=location,
    )
    pf.ops.mesh.subdivide(res, number_cuts=1)
    pf.ops.modifier.subdivide_surface(res, levels=3, _skip_apply=True)
    return res


@pf.tracer.generator
def hardcoded_camera(
    base_location: t.Vector,
    dist_mult: float = 1,
    elevation_deg: float = 19,
    altitude: float = 2.2,
    yaw_offset_deg: float = 0,
) -> pf.CameraObject:
    res = pf.ops.primitives.perspective_camera()

    obj = res.item()

    obj.location = base_location + t.Vector((5, -4, altitude)) * dist_mult
    obj.keyframe_insert("location", frame=0)
    obj.location.x += 1
    obj.keyframe_insert("location", frame=10)
    obj.rotation_euler = np.deg2rad(
        np.array([90 - elevation_deg, 0, 52 + yaw_offset_deg])
    )
    obj.keyframe_insert("rotation_euler")

    return res


def grid_plane() -> pf.MeshObject:
    plane = pf.ops.primitives.mesh_plane(location=t.Vector((0, 0, 0)), size=8)
    material = developer_grid(vector=pf.nodes.shader.coord().generated)
    pf.ops.object.set_material(plane, material=material)
    return plane


def demo_cube(size: float = 1.0) -> pf.MeshObject:
    obj = pf.ops.primitives.mesh_cube(
        size=size,
        location=t.Vector((0, 0, size / 2)),
        rotation=t.Euler((0, 0, np.pi * 0.15)),
    )
    pf.ops.modifier.bevel(obj, width=0.06 * size, segments=2)
    pf.ops.modifier.subdivide_surface(obj, levels=6, _skip_apply=True)
    return obj


@pf.tracer.grammar
def object_demo(
    rng: pf.RNG,
    obj: pf.MeshObject | None = None,
    camera: pf.CameraObject | None = None,
    lighting: pf.Object | None = None,
) -> DevSceneResult:
    if obj is None:
        logger.warning("No object provided; using a default object.")
        obj = demo_cube()
    bbox = pf.ops.attr.bbox_min_max(obj, global_coords=True)
    obj.item().location.z -= bbox[0][-1]

    if camera is None:
        camera = camera_with_distance_framing_object(
            obj, t.Vector((1, 1, 0.4)), margin_pct=0.1
        )

    if lighting is None:
        lighting = sky_lighting.nishita_sky(
            sun_rotation_deg=200,
            sun_elevation_deg=30,
        )
    background = grid_plane()

    ref_rad = 0.3
    dims = np.array(obj.item().dimensions)
    pos = (-dims[0] / 2 - ref_rad - 0.1, 0, 0)
    scale_ref = scale_reference(location=pos, radius=ref_rad)

    return DevSceneResult(
        lights=[lighting], all_objects=[obj, background, scale_ref], cameras=[camera]
    )


@pf.tracer.grammar
def material_sphere(
    rng: pf.RNG,
    material: pf.Material | None = None,
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

    lighting = sky_lighting.nishita_sky(
        sun_rotation_deg=260,
        sun_elevation_deg=30,
    )

    cam = hardcoded_camera(base_location=sphere.item().location, dist_mult=0.7)
    plane = grid_plane()
    ref = scale_reference(location=t.Vector((0.38, 0.76, -0.05)))

    return DevSceneResult(
        lights=[lighting], all_objects=[sphere, plane, ref], cameras=[cam]
    )


@pf.tracer.grammar
def material_torus_uv(
    rng: pf.RNG,
    material: pf.Material | None = None,
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

    lighting = sky_lighting.nishita_sky(
        sun_rotation_deg=260,
        sun_elevation_deg=30,
    )

    cam = hardcoded_camera(base_location=obj.item().location, dist_mult=0.65)
    plane = grid_plane()
    ref = scale_reference(location=t.Vector((0.38, 0.76, -0.05)))

    return DevSceneResult(
        lights=[lighting], all_objects=[obj, plane, ref], cameras=[cam]
    )


@pf.tracer.grammar
def material_plane_uv(
    rng: pf.RNG,
    material: pf.Material | None = None,
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
    lighting = sky_lighting.nishita_sky(
        sun_rotation_deg=260,
        sun_elevation_deg=30,
    )
    ref = scale_reference(location=t.Vector((0.38, 0.76, -0.05)))

    return DevSceneResult(
        lights=[lighting], all_objects=[obj, plane, ref], cameras=[cam]
    )


@pf.tracer.grammar
def material_plane_horizontal_uv(
    rng: pf.RNG,
    material: pf.Material | None = None,
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

    cam = camera_with_distance_framing_object(
        obj, t.Vector((-0.7, -1.0, 1.2)), margin_pct=-0.425
    )

    lighting = sky_lighting.nishita_sky(
        sun_rotation_deg=260,
        sun_elevation_deg=30,
    )

    return DevSceneResult(
        lights=[lighting], all_objects=[obj, plane, ref], cameras=[cam]
    )


@pf.tracer.grammar
def material_monkey(
    rng: pf.RNG,
    material: pf.Material | None = None,
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
    lighting = sky_lighting.nishita_sky(
        sun_rotation_deg=260,
        sun_elevation_deg=30,
    )
    ref = pf.ops.primitives.mesh_cylinder(
        radius=0.1, depth=0.02, location=t.Vector((0.1, 0.1, 0.01)), vertices=128
    )

    return DevSceneResult(
        lights=[lighting], all_objects=[obj, plane, ref], cameras=[cam]
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
    lighting = sky_lighting.nishita_sky()

    return DevSceneResult(lights=[lighting], all_objects=[obj], cameras=[cam])


@pf.tracer.grammar
def material_cube(
    rng: pf.RNG,
    material: pf.Material | None = None,
) -> DevSceneResult:
    s = 0.5
    obj = demo_cube(size=s)

    if material is None:
        logger.warning("No material provided; using a default material.")
        material = developer_grid(vector=pf.nodes.shader.coord().generated)
    pf.ops.object.set_material(obj, material=material)

    cam = hardcoded_camera(
        base_location=obj.item().location + t.Vector((0, 0, 0.47)),
        dist_mult=0.32,
        elevation_deg=30,
    )
    plane = grid_plane()
    lighting = sky_lighting.nishita_sky(
        sun_rotation_deg=260,
        sun_elevation_deg=30,
    )
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
        lights=[lighting], all_objects=[obj, plane, ref], cameras=[cam]
    )


@pf.tracer.grammar
def material_banana(
    rng: pf.RNG,
    material: pf.Material | None = None,
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
    lighting = sky_lighting.nishita_sky(
        sun_rotation_deg=260,
        sun_elevation_deg=30,
    )
    ref = scale_reference(location=t.Vector((0.38, 0.76, -0.05)))
    return DevSceneResult(
        lights=[lighting], all_objects=[obj, plane, ref], cameras=[cam]
    )
