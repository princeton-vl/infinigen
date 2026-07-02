# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen v1 nodegroup (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/elements/rug.py)
# - Alexander Raistrick: transpile to procfunc/v2

from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.composites.fabric_patterned import (
    fabric_patterned_rand,
)
from infinigen2.shaders.materials import carpet
from infinigen2.shaders.materials.fabric import fabric_rand

__all__ = [
    "RugResult",
    "rug_geometry",
    "rug_material_rand",
    "rug_rand",
]


class RugResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def rug_geometry(
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


def rug_material_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.Material:
    rng_choice, rng_func = rng.spawn(2)
    func = pf.control.choice(
        rng_choice,
        [
            (fabric_patterned_rand, 3.0),
            (fabric_rand, 1.0),
            (lambda rng, vector, **_: carpet.carpet_rand(rng, vector), 2.0),
        ],
    )
    return func(rng_func, vector)


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

    geo = rug_geometry(
        width=width,
        length=length,
        fillet_radius=fillet_radius,
        thickness=thickness,
        material=mat,
    )

    obj = pf.nodes.to_mesh_object(geo)
    return RugResult(mesh=obj)
