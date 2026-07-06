# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen v1 nodegroup (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/elements/rug.py)
# - Alexander Raistrick: transpile to procfunc/v2

from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.functionality_lists import rug_material_rand

__all__ = [
    "RugResult",
    "rug",
    "rug_rand",
]


class RugResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def _rug_geometry(
    width: t.SocketOrVal[float],
    length: t.SocketOrVal[float],
    fillet_radius: t.SocketOrVal[float],
    thickness: t.SocketOrVal[float],
    material: t.SocketOrVal[pf.Material],
) -> t.ProcNode[pf.MeshObject]:
    quad = pf.nodes.geo.curve_quadrilateral(width=length, height=width)

    filleted = pf.nodes.geo.fillet_curve_poly(
        curve=quad,
        radius=fillet_radius,
        limit_radius=True,
        count=16,
    )

    filled = pf.nodes.geo.fill_curve(curve=filleted)

    extruded = pf.nodes.geo.extrude_mesh(
        mesh=filled,
        offset_scale=thickness,
    )

    geo = pf.nodes.geo.set_material(geometry=extruded.mesh, material=material)
    geo = pf.nodes.geo.set_shade_smooth(geometry=geo, shade_smooth=False)
    return geo


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
    return RugResult(mesh=pf.nodes.to_mesh_object(geo))


def rug_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
) -> RugResult:
    if dimensions is None:
        width = pf.random.clip_gaussian(rng, 2.5, 0.8, 1.5, 4.0)
        length = width * pf.random.uniform(rng, 1.0, 1.5)
        thickness = pf.random.uniform(rng, 0.01, 0.02)
        dimensions = (length, width, thickness)
    length, width, thickness = dimensions
    min_dim = min(width, length)
    fillet_radius = pf.random.uniform(rng, 0.0, min_dim / 2)

    vec = pf.nodes.shader.geometry().position
    mat_shader = rug_material_rand(rng, vec)
    mat = mat_shader

    geo = _rug_geometry(
        width=width,
        length=length,
        fillet_radius=fillet_radius,
        thickness=thickness,
        material=mat,
    )

    obj = pf.nodes.to_mesh_object(geo)
    return RugResult(mesh=obj)
