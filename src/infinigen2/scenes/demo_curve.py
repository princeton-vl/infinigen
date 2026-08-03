# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.curves.skirting_board_profile import (
    skirting_profile_rand,
    trim_profile_rand,
)
from infinigen2.scenes.demo_object import object_demo
from infinigen2.scenes.demo_util import DevSceneResult
from infinigen2.shaders.functionality_lists import skirt_material_rand
from infinigen2.util.curve import curve_to_mesh_with_uv, fillet_mask

__all__ = [
    "curve_demo",
    "trim_demo",
]

logger = logging.getLogger(__name__)


@pf.nodes.node_function
def _demo_base_curve(
    size: t.SocketOrVal[float],
    fillet_large: t.SocketOrVal[float],
    fillet_small: t.SocketOrVal[float],
) -> pf.ProcNode[pf.CurveObject]:
    square = pf.nodes.geo.curve_quadrilateral(width=size, height=size)

    # per-corner fillet radius: corner 0 large, corner 1 small, the rest sharp
    input_index = pf.nodes.geo.input_index()
    radius = pf.nodes.func.switch(
        switch=pf.nodes.func.equal(a=input_index, b=0),
        a=pf.nodes.func.switch(
            switch=pf.nodes.func.equal(a=input_index, b=1),
            a=0.0,
            b=fillet_small,
        ),
        b=fillet_large,
    )
    selection = pf.nodes.func.greater_than(a=radius, b=1e-06)
    curved = fillet_mask(
        geometry=square,
        selection=selection,
        fillet_vertices=12,
        radius=radius,
    ).curve

    return pf.nodes.geo.resample_curve_length(curved, length=0.02)


@pf.tracer.grammar
def curve_demo(
    rng: pf.RNG,
    profile: pf.CurveObject | None = None,
    material: pf.Material | None = None,
    size: float = 1.5,
    fillet_large: float = 0.75,
    fillet_small: float = 0.1,
) -> DevSceneResult:
    """Emit a representative base path curve into the pipeline state, and render it
    swept with a default profile so it stands alone as a scene. Downstream curve
    generators receive the path via their `curve` / `base_curve` parameter."""
    rng_profile, rng_mat, rng_scene = rng.spawn(3)

    base_node = _demo_base_curve(
        size=size, fillet_large=fillet_large, fillet_small=fillet_small
    )
    base_curve = pf.nodes.to_curve_object(base_node)

    if profile is None:
        logger.warning("No profile provided; using a default skirting profile.")
        profile = skirting_profile_rand(rng_profile)
    if material is None:
        material = skirt_material_rand(rng_mat, pf.nodes.shader.coord().uv)

    base_geo = pf.nodes.geo.object_info(base_curve).geometry
    profile_geo = pf.nodes.geo.object_info(profile).geometry

    swept = curve_to_mesh_with_uv(base_geo, profile_geo, fill_caps=True).mesh
    swept = pf.nodes.geo.flip_faces(swept)
    swept = pf.nodes.to_mesh_object(swept)

    pf.ops.object.set_material(
        swept,
        surface=material.surface,
        displacement=material.displacement,
    )

    scene = object_demo(rng_scene, all_objects=[swept])
    return scene._replace(curves=[base_curve])


@pf.tracer.grammar
def trim_demo(
    rng: pf.RNG,
    size: float = 1.5,
    fillet_large: float = 0.75,
    fillet_small: float = 0.1,
) -> DevSceneResult:
    rng_profile, rng_demo = rng.spawn(2)
    profile = trim_profile_rand(rng_profile)
    return curve_demo(
        rng_demo,
        profile=profile,
        size=size,
        fillet_large=fillet_large,
        fillet_small=fillet_small,
    )
