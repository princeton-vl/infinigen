# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Beining Han: original Infinigen v1 nodegroup (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/shelves/simple_desk.py)
# - Alexander Raistrick: transpile to procfunc/v2

from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.functionality_lists import (
    furniture_material_rand,
    table_top_material_rand,
)

__all__ = [
    "DeskResult",
    "desk",
    "desk_rand",
]


class DeskResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def _tagged_cube(
    size: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode:
    cube = pf.nodes.geo.mesh_cube(size)
    return cube.mesh


@pf.nodes.node_function
def _table_top(
    depth: t.SocketOrVal[float],
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    thickness: t.SocketOrVal[float],
) -> pf.ProcNode:
    tagged_cube_size_z = thickness + 0.0
    tagged_cube_size = pf.nodes.math.combine_xyz(x=width, y=depth, z=tagged_cube_size_z)
    tagged_cube_result = _tagged_cube(size=tagged_cube_size)

    translation_z_0 = tagged_cube_size_z * 0.5
    translation = pf.nodes.math.combine_xyz(z=height - translation_z_0)

    transform = pf.nodes.geo.transform(
        geometry=tagged_cube_result,
        translation=translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    return transform


@pf.nodes.node_function
def _table_legs(
    thickness: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    radius: t.SocketOrVal[float],
    width: t.SocketOrVal[float],
    depth: t.SocketOrVal[float],
    dist: t.SocketOrVal[float],
) -> pf.ProcNode:
    cylinder_depth = height - thickness
    cylinder = pf.nodes.geo.mesh_cylinder(
        vertices=32, radius=radius, depth=cylinder_depth
    )

    transform_b = dist + 0.0
    transform_1_translation_x = (width * 0.5) - transform_b
    transform_translation_x = transform_1_translation_x * -1.0
    transform_2_translation_y = (0.5 * depth) - transform_b
    transform_translation_y = transform_2_translation_y * -1.0
    transform_translation_z = cylinder_depth * 0.5
    transform_translation = pf.nodes.math.combine_xyz(
        x=transform_translation_x,
        y=transform_translation_y,
        z=transform_translation_z,
    )
    transform = pf.nodes.geo.transform(
        geometry=cylinder.mesh,
        translation=transform_translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    transform_1_translation = pf.nodes.math.combine_xyz(
        x=transform_1_translation_x,
        y=transform_translation_y,
        z=transform_translation_z,
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=cylinder.mesh,
        translation=transform_1_translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    transform_2_translation = pf.nodes.math.combine_xyz(
        x=transform_translation_x,
        y=transform_2_translation_y,
        z=transform_translation_z,
    )
    transform_2 = pf.nodes.geo.transform(
        geometry=cylinder.mesh,
        translation=transform_2_translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    transform_3_translation = pf.nodes.math.combine_xyz(
        x=transform_1_translation_x,
        y=transform_2_translation_y,
        z=transform_translation_z,
    )
    transform_3 = pf.nodes.geo.transform(
        geometry=cylinder.mesh,
        translation=transform_3_translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    join = pf.nodes.geo.join_geometry(
        [transform, transform_1, transform_2, transform_3]
    )

    realize_instances = pf.nodes.geo.realize_instances(join)
    return realize_instances


@pf.nodes.node_function
def _desk_geometry(
    dimensions: t.SocketOrVal[pf.Vector],
    thickness: t.SocketOrVal[float],
    leg_radius: t.SocketOrVal[float],
    leg_dist: t.SocketOrVal[float],
    top_material: t.SocketOrVal[pf.Material],
    leg_material: t.SocketOrVal[pf.Material],
) -> pf.ProcNode:
    depth, width, height = dimensions.x, dimensions.y, dimensions.z

    top = _table_top(depth=depth, width=width, height=height, thickness=thickness)
    top_with_mat = pf.nodes.geo.set_material(geometry=top, material=top_material)

    legs = _table_legs(
        thickness=thickness,
        height=height,
        radius=leg_radius,
        width=width,
        depth=depth,
        dist=leg_dist,
    )
    legs_with_mat = pf.nodes.geo.set_material(geometry=legs, material=leg_material)

    joined = pf.nodes.geo.join_geometry([top_with_mat, legs_with_mat])
    realized = pf.nodes.geo.realize_instances(joined)
    triangulated = pf.nodes.geo.triangulate(realized)
    rotated = pf.nodes.geo.transform(
        geometry=triangulated,
        rotation=(0.0, 0.0, 1.5708),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    return rotated


def desk(
    dimensions: pf.Vector | None = None,
    thickness: float = 0.02,
    leg_radius: float = 0.0175,
    leg_dist: float = 0.0525,
    top_material: pf.Material | None = None,
    leg_material: pf.Material | None = None,
) -> DeskResult:
    if dimensions is None:
        dimensions = pf.Vector((0.575, 1.0, 0.715))
    if top_material is None:
        top_material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    if leg_material is None:
        leg_material = pf.Material(surface=pf.nodes.shader.principled_bsdf())

    geo = _desk_geometry(
        dimensions=dimensions,
        thickness=thickness,
        leg_radius=leg_radius,
        leg_dist=leg_dist,
        top_material=top_material,
        leg_material=leg_material,
    )
    return DeskResult(mesh=pf.nodes.to_mesh_object(geo))


def desk_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    top_material: pf.Material | None = None,
    leg_material: pf.Material | None = None,
) -> DeskResult:
    if dimensions is None:
        depth = pf.random.uniform(rng, 0.45, 0.7)
        width = pf.random.uniform(rng, 0.7, 1.3)
        height = pf.random.uniform(rng, 0.6, 0.83)
        dimensions = pf.Vector((depth, width, height))

    thickness = pf.random.uniform(rng, 0.01, 0.03)
    leg_radius = pf.random.uniform(rng, 0.01, 0.025)
    leg_dist = pf.random.uniform(rng, 0.035, 0.07)

    vec = pf.nodes.shader.geometry().position
    if top_material is None:
        top_material = table_top_material_rand(rng, vec)
    if leg_material is None:
        leg_material = furniture_material_rand(rng, vec)

    geo = _desk_geometry(
        dimensions=dimensions,
        thickness=thickness,
        leg_radius=leg_radius,
        leg_dist=leg_dist,
        top_material=top_material,
        leg_material=leg_material,
    )
    return DeskResult(mesh=pf.nodes.to_mesh_object(geo))


if __name__ == "__main__":
    table_legs_result = _table_legs()
    table_top_result = _table_top()
