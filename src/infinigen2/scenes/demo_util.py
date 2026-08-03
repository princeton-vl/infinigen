# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from typing import NamedTuple

import numpy as np
import procfunc as pf
from procfunc import types as t

from infinigen2.shaders.dev import developer_grid

__all__ = [
    "DevSceneResult",
    "demo_cube",
    "grid_plane",
    "hardcoded_camera",
    "scale_reference",
]


class DevSceneResult(NamedTuple):
    all_objects: list
    cameras: list
    lights: list = ()
    environment: pf.World | None = None
    curves: list = ()


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
