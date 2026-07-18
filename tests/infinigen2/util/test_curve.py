# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import bpy
import numpy as np
import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.util.curve import curve_to_mesh_with_uv


@pf.nodes.node_function
def _cyclic_profile_tube(circle_res: t.SocketOrVal[int]) -> pf.ProcNode[pf.MeshObject]:
    line = pf.nodes.geo.resample_curve_count(
        curve=pf.nodes.geo.curve_line(start=(0, 0, 0.0), end=(0, 0, 0.2)), count=4
    )
    radius = pf.nodes.math.map_range(
        value=pf.nodes.geo.spline_parameter().factor, to_max=0.185, to_min=0.16
    )
    rail = pf.nodes.geo.set_curve_radius(curve=line, radius=radius)
    return curve_to_mesh_with_uv(
        curve=rail, profile=pf.nodes.geo.curve_circle(circle_res)
    ).mesh


@pf.nodes.node_function
def _cyclic_rail_wall() -> pf.ProcNode[pf.MeshObject]:
    rail = pf.nodes.geo.curve_circle(resolution=8, radius=2.0)
    profile = pf.nodes.geo.curve_line(start=(0, 0, 0), end=(0, 0, 1.0))
    return curve_to_mesh_with_uv(curve=rail, profile=profile).mesh


def _corner_uvs(obj):
    me = obj.item().evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    uv = me.uv_layers["UVMap"].data
    return me, np.array([(uv[i].uv[0], uv[i].uv[1]) for i in range(len(uv))])


def _max_face_span(me, arr, axis):
    spans = []
    for p in me.polygons:
        vals = [arr[c][axis] for c in p.loop_indices]
        spans.append(max(vals) - min(vals))
    return max(spans)


def test_cyclic_profile_seam_continuous():
    # Low-res cyclic profile must not sawtooth: wrap face continues, no UV.y snap to 0
    obj = pf.nodes.to_mesh_object(_cyclic_profile_tube(circle_res=16))
    me, arr = _corner_uvs(obj)
    perimeter = arr[:, 1].max()
    # V is metric: the widest station's real circumference is 2*pi*0.185, not the unit 2*pi
    assert abs(perimeter - 2 * np.pi * 0.185) < 0.02
    single_segment = perimeter / 16
    assert _max_face_span(me, arr, 1) < 2 * single_segment
    assert not np.isnan(arr).any()


def test_cyclic_rail_seam_still_handled():
    # Original rail-seam handling stays intact; line profile adds no false UV.y wrap
    obj = pf.nodes.to_mesh_object(_cyclic_rail_wall())
    me, arr = _corner_uvs(obj)
    perimeter = arr[:, 0].max()
    assert perimeter > 12.0
    assert _max_face_span(me, arr, 0) < perimeter / 4
    # unset rail radius falls back to 1.0 so V stays metric (profile length 1.0), not collapsed
    assert 0.99 < arr[:, 1].max() <= 1.0 + 1e-4
    assert not np.isnan(arr).any()
