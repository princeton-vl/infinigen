# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.curves.skirting_board_profile import skirting_profile_rand
from infinigen2.scenes.asset_demo import DevSceneResult, object_demo
from infinigen2.shaders.functionality_lists import skirt_material_rand
from infinigen2.util.curve import curve_to_mesh_with_uv, fillet_mask

__all__ = [
    "curve_demo",
]


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
    size: float = 3.0,
    fillet_large: float = 1.5,
    fillet_small: float = 0.2,
) -> DevSceneResult:
    rng_profile, rng_mat, rng_scene = rng.spawn(3)

    base = _demo_base_curve(
        size=size, fillet_large=fillet_large, fillet_small=fillet_small
    )
    profile = pf.nodes.geo.object_info(skirting_profile_rand(rng_profile)).geometry

    skirt = curve_to_mesh_with_uv(base, profile, fill_caps=True).mesh
    skirt = pf.nodes.geo.flip_faces(skirt)
    skirt = pf.nodes.to_mesh_object(skirt)

    material = skirt_material_rand(rng_mat, pf.nodes.shader.coord().uv)
    pf.ops.object.set_material(
        skirt,
        surface=material.surface,
        displacement=material.displacement,
    )

    return object_demo(rng_scene, all_objects=[skirt])
