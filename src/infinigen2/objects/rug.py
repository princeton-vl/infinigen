# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen rug (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/elements/rug.py)
# - Alexander Raistrick: refactor for Infinigen2

from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.functionality_lists import rug_material_rand
from infinigen2.util import mesh

__all__ = [
    "RugResult",
    "rug",
    "rug_rand",
]

_RUG_PLANAR_LOOPS = 30
_RUG_SUBDIV_LEVELS = 3


class RugResult(NamedTuple):
    mesh: pf.MeshObject


def _round_corners(
    geometry: pf.ProcNode,
    half_length: t.SocketOrVal[float],
    half_width: t.SocketOrVal[float],
    fillet_radius: t.SocketOrVal[float],
) -> pf.ProcNode:
    position = pf.nodes.math.separate_xyz(pf.nodes.geo.input_position())
    du = pf.nodes.math.maximum(
        pf.nodes.math.absolute(position.x) - (half_length - fillet_radius), 0.0
    )
    dv = pf.nodes.math.maximum(
        pf.nodes.math.absolute(position.y) - (half_width - fillet_radius), 0.0
    )
    radial = pf.nodes.math.sqrt(du * du + dv * dv)
    square_to_circle = pf.nodes.math.maximum(du, dv) / pf.nodes.math.maximum(
        radial, 1e-9
    )
    shift_x = pf.nodes.math.sign(position.x) * du * (square_to_circle - 1.0)
    shift_y = pf.nodes.math.sign(position.y) * dv * (square_to_circle - 1.0)
    return pf.nodes.geo.set_position(
        geometry=geometry,
        offset=pf.nodes.math.combine_xyz(x=shift_x, y=shift_y),
    )


@pf.nodes.node_function
def _rug_geometry(
    width: t.SocketOrVal[float],
    length: t.SocketOrVal[float],
    fillet_radius: t.SocketOrVal[float],
    thickness: t.SocketOrVal[float],
    material: t.SocketOrVal[pf.Material],
) -> t.ProcNode[pf.MeshObject]:
    size = pf.nodes.math.combine_xyz(x=length, y=width, z=thickness)
    edge_radius = thickness * 0.4
    support_loop_offset = pf.nodes.math.combine_xyz(
        x=edge_radius, y=edge_radius, z=edge_radius
    )
    box = mesh.box_with_support_loops(
        size=size,
        vertices_x=_RUG_PLANAR_LOOPS + 4,
        vertices_y=_RUG_PLANAR_LOOPS + 4,
        vertices_z=4,
        support_loop_offset=support_loop_offset,
    )
    without_bottom = pf.nodes.geo.delete_geometry(
        geometry=box,
        selection=pf.nodes.geo.input_normal().z < -0.9,
        domain="FACE",
    )
    rounded = _round_corners(
        without_bottom,
        half_length=length * 0.5,
        half_width=width * 0.5,
        fillet_radius=fillet_radius,
    )
    rounded = mesh.metric_box_uv(rounded)
    raised = pf.nodes.geo.transform(
        geometry=rounded,
        translation=pf.nodes.math.combine_xyz(z=thickness * 0.5),
    )
    geo = pf.nodes.geo.set_material(geometry=raised, material=material)
    geo = pf.nodes.geo.set_shade_smooth(geometry=geo, shade_smooth=True)
    return geo


@pf.tracer.generator
def rug(
    width: float = 2.5,
    length: float = 3.125,
    fillet_radius: float = 0.625,
    thickness: float = 0.015,
    material: pf.Material | None = None,
) -> RugResult:
    if material is None:
        material = pf.Material(surface=pf.nodes.shader.principled_bsdf())

    geo = _rug_geometry(
        width=width,
        length=length,
        fillet_radius=fillet_radius,
        thickness=thickness,
        material=material,
    )
    obj = pf.nodes.to_mesh_object(geo)
    return RugResult(mesh=obj)


def rug_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
) -> RugResult:
    rng_dims, rng_corner, rng_radius, rng_material = rng.spawn(4)
    if dimensions is None:
        width = pf.random.clip_gaussian(rng_dims, 2.5, 0.8, 1.5, 4.0)
        length = width * pf.random.uniform(rng_dims, 1.0, 1.5)
        thickness = pf.random.uniform(rng_dims, 0.01, 0.02)
        dimensions = (length, width, thickness)
    min_dim = min(dimensions[1], dimensions[0])
    rounded_radius = pf.random.uniform(rng_radius, 0.0, min_dim / 2)
    fillet_radius = pf.control.choice(rng_corner, [(0.0, 1.0), (rounded_radius, 2.0)])

    if material is None:
        vec = pf.nodes.shader.geometry().position
        material = rug_material_rand(rng_material, vec)

    obj = rug(
        width=dimensions[1],
        length=dimensions[0],
        fillet_radius=fillet_radius,
        thickness=dimensions[2],
        material=material,
    ).mesh
    pf.ops.modifier.subdivide_surface(obj, levels=_RUG_SUBDIV_LEVELS, _skip_apply=True)
    return RugResult(mesh=obj)
