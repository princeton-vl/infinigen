# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

__all__ = [
    "CurveToMeshWithUvResult",
    "FilletMaskResult",
    "corner_ring_edges",
    "curve_to_mesh_with_uv",
    "curve_to_noncyclic",
    "fillet_mask",
    "outline_corner",
]


@pf.nodes.node_function
def outline_corner(corner_degrees: t.SocketOrVal[float]) -> pf.ProcNode[bool]:
    position = pf.nodes.geo.input_position()
    prev_index = pf.nodes.geo.offset_point_in_curve(offset=-1).point_index
    next_index = pf.nodes.geo.offset_point_in_curve(offset=1).point_index
    prev_position = pf.nodes.geo.field_at_index(value=position, index=prev_index)
    next_position = pf.nodes.geo.field_at_index(value=position, index=next_index)
    incoming = pf.nodes.math.vector_normalize(position - prev_position)
    outgoing = pf.nodes.math.vector_normalize(next_position - position)
    turn_cos = pf.nodes.math.vector_dot_product(a=incoming, b=outgoing)
    corner_cos = pf.nodes.math.cos(pf.nodes.math.deg_to_rad(corner_degrees))
    return pf.nodes.func.less_than(a=turn_cos, b=corner_cos)


@pf.nodes.node_function
def corner_ring_edges(
    ring: t.SocketOrVal[float],
    is_corner: t.SocketOrVal[float],
) -> pf.ProcNode[bool]:
    ends = pf.nodes.geo.input_mesh_edge_vertices()
    ring_1 = pf.nodes.geo.field_at_index(value=ring, index=ends.vertex_index_1)
    ring_2 = pf.nodes.geo.field_at_index(value=ring, index=ends.vertex_index_2)
    corner_1 = pf.nodes.geo.field_at_index(value=is_corner, index=ends.vertex_index_1)
    corner_2 = pf.nodes.geo.field_at_index(value=is_corner, index=ends.vertex_index_2)
    same_ring = pf.nodes.math.absolute(ring_1 - ring_2) < 1e-4
    both_corner = pf.nodes.math.minimum(corner_1, corner_2) > 0.999
    return pf.nodes.func.boolean_and(same_ring, both_corner)


class CurveToMeshWithUvResult(NamedTuple):
    mesh: pf.ProcNode[pf.MeshObject]
    vector: pf.ProcNode[pf.Vector]


@pf.nodes.node_function
def _seam_continuous_length(
    is_forward_corner: t.SocketOrVal[bool],
    point_index: t.SocketOrVal[int],
    point_length: t.SocketOrVal[float],
    curve: pf.ProcNode[pf.CurveObject],
) -> pf.ProcNode[float]:
    # Wrap face's forward corner reuses point 0 (length 0); use full length so UV doesn't snap to 0
    is_seam_start = pf.nodes.func.equal(a=0, b=point_index)
    do_replace = pf.nodes.func.boolean_and(a=is_forward_corner, b=is_seam_start)
    full_length = pf.nodes.geo.curve_length(curve)
    return pf.nodes.func.switch(switch=do_replace, a=point_length, b=full_length)


@pf.nodes.node_function
def curve_to_mesh_with_uv(
    curve: pf.ProcNode[pf.CurveObject],
    profile: pf.ProcNode[pf.CurveObject],
    fill_caps: t.SocketOrVal[bool] = False,
    swap_uv: t.SocketOrVal[bool] = False,
) -> CurveToMeshWithUvResult:
    input_index = pf.nodes.geo.input_index()
    index_on_curve = pf.nodes.geo.capture_attribute(geometry=curve, index=input_index)

    spline_parameter = pf.nodes.geo.spline_parameter()
    # unset curves report radius 0 but curve_to_mesh sweeps them at 1.0; keep V metric
    input_radius = pf.nodes.geo.input_radius()
    is_unset = pf.nodes.func.less_than(a=input_radius, b=1e-6)
    effective_radius = pf.nodes.func.switch(switch=is_unset, a=input_radius, b=1.0)
    length_on_curve = pf.nodes.geo.capture_attribute(
        geometry=index_on_curve.geometry,
        length=spline_parameter.length,
        radius=effective_radius,
    )

    profile_input_index = pf.nodes.geo.input_index()
    index_on_profile = pf.nodes.geo.capture_attribute(
        geometry=profile, index=profile_input_index
    )

    spline_parameter_1 = pf.nodes.geo.spline_parameter()
    length_on_profile = pf.nodes.geo.capture_attribute(
        geometry=index_on_profile.geometry,
        length=spline_parameter_1.length,
    )

    to_mesh = pf.nodes.geo.curve_to_mesh(
        curve=length_on_curve.geometry,
        profile_curve=length_on_profile.geometry,
        fill_caps=fill_caps,
    )

    corner_in_mesh = pf.nodes.geo.input_index()
    face_of_corner = pf.nodes.geo.face_of_corner(corner_in_mesh)

    # Quad corners {0,1,2,3}: rail-forward is {2,3}, profile-forward is {1,2}
    is_rail_forward = pf.nodes.func.greater_than(a=face_of_corner.index_in_face, b=1)
    is_profile_forward = pf.nodes.func.boolean_or(
        a=pf.nodes.func.equal(a=face_of_corner.index_in_face, b=1),
        b=pf.nodes.func.equal(a=face_of_corner.index_in_face, b=2),
    )

    u = _seam_continuous_length(
        is_rail_forward,
        index_on_curve.index,
        length_on_curve.length,
        index_on_curve.geometry,
    )
    v = _seam_continuous_length(
        is_profile_forward,
        index_on_profile.index,
        length_on_profile.length,
        index_on_profile.geometry,
    )
    v_metric = v * length_on_curve.radius
    out_x = pf.nodes.func.switch(switch=swap_uv, a=u, b=v_metric)
    out_y = pf.nodes.func.switch(switch=swap_uv, a=v_metric, b=u)
    vector = pf.nodes.math.combine_xyz(x=out_x, y=out_y)

    mesh_with_uv = pf.nodes.geo.store_named_attribute(
        domain="CORNER",
        geometry=to_mesh,
        name="UVMap",
        value=vector,
        data_type="FLOAT2",
    )
    face_count = pf.nodes.geo.attribute_domain_size(to_mesh).face_count
    first_cap = face_count.astype(dtype=float) - 2.0
    is_cap = pf.nodes.func.boolean_and(
        a=fill_caps,
        b=pf.nodes.func.greater_than(
            a=face_of_corner.face_index.astype(dtype=float),
            b=first_cap - 1.0,
        ),
    )
    position = pf.nodes.geo.input_position()
    normal = pf.nodes.math.vector_absolute(pf.nodes.geo.input_normal())
    cap_facing_x = pf.nodes.math.combine_xyz(x=position.y, y=position.z)
    cap_facing_y = pf.nodes.math.combine_xyz(x=position.x, y=position.z)
    cap_facing_z = pf.nodes.math.combine_xyz(x=position.x, y=position.y)
    cap_uv = pf.nodes.func.switch(
        switch=normal.y > normal.x,
        a=cap_facing_x,
        b=cap_facing_y,
    )
    cap_uv = pf.nodes.func.switch(
        switch=normal.z > pf.nodes.math.maximum(normal.x, normal.y),
        a=cap_uv,
        b=cap_facing_z,
    )
    mesh_with_uv = pf.nodes.geo.store_named_attribute(
        domain="CORNER",
        geometry=mesh_with_uv,
        name="UVMap",
        selection=is_cap,
        value=cap_uv,
        data_type="FLOAT2",
    )
    return CurveToMeshWithUvResult(
        mesh=mesh_with_uv,
        vector=vector,
    )


class FilletMaskResult(NamedTuple):
    curve: pf.ProcNode[pf.CurveObject]
    is_fillet_point: pf.ProcNode[bool]
    is_flat_point: pf.ProcNode[bool]
    orig_point_idx: pf.ProcNode[int]


@pf.nodes.node_function
def fillet_mask(
    geometry: pf.ProcNode[pf.CurveObject],
    selection: t.SocketOrVal[bool],
    fillet_vertices: t.SocketOrVal[int],
    radius: t.SocketOrVal[float],
) -> FilletMaskResult:
    # Code generated by procfunc v0.16.0

    capture_attribute = pf.nodes.geo.capture_attribute(
        geometry=geometry, value=selection
    )

    fillet_no_rad = pf.nodes.geo.fillet_curve_poly(
        curve=capture_attribute.geometry,
        radius=1e-06,
        limit_radius=True,
        count=fillet_vertices,
    )
    fillet_with_rad = pf.nodes.geo.fillet_curve_poly(
        curve=capture_attribute.geometry,
        radius=radius + 1e-05,
        limit_radius=True,
        count=fillet_vertices,
    )

    input_position = pf.nodes.geo.input_position()
    input_index = pf.nodes.geo.input_index()

    query_fillet_pos = pf.nodes.geo.sample_index(
        geometry=fillet_with_rad,
        index=input_index,
        value=input_position,
    )

    selective_apply_fillet = pf.nodes.geo.set_position(
        geometry=fillet_no_rad,
        selection=capture_attribute.value,
        position=query_fillet_pos,
    )

    input_index_1 = pf.nodes.geo.input_index()

    is_flat_point_is_fillet_edge_a_b = fillet_vertices.astype(dtype=float) + 1.0
    is_flat_point_is_fillet_edge_a = pf.nodes.math.floor_mod(
        a=input_index_1.astype(dtype=float),
        b=is_flat_point_is_fillet_edge_a_b,
    )
    is_flat_point_is_fillet_edge = pf.nodes.func.equal(
        a=is_flat_point_is_fillet_edge_a, b=1.0, epsilon=0.01
    )

    capture_attribute_1 = pf.nodes.geo.capture_attribute(
        geometry=selective_apply_fillet,
        is_fillet_edge=is_flat_point_is_fillet_edge,
    )

    curve_value = pf.nodes.math.floor(
        input_index_1.astype(dtype=float) / is_flat_point_is_fillet_edge_a_b
    )

    capture_attribute_2 = pf.nodes.geo.capture_attribute(
        geometry=capture_attribute_1.geometry,
        value=curve_value.astype(dtype=int),
    )

    is_fillet_point = pf.nodes.func.boolean_not(capture_attribute_1.is_fillet_edge)
    return FilletMaskResult(
        curve=capture_attribute_2.geometry,
        is_fillet_point=is_fillet_point,
        is_flat_point=capture_attribute_1.is_fillet_edge,
        orig_point_idx=capture_attribute_2.value,
    )


@pf.nodes.node_function
def curve_to_noncyclic(
    input: pf.ProcNode[pf.CurveObject],
) -> pf.ProcNode[pf.CurveObject]:
    # Code generated by procfunc v0.16.0

    curve_line = pf.nodes.geo.curve_line(start=(0, 0, 0), end=(0, 0, 1))

    attribute_domain_size = pf.nodes.geo.attribute_domain_size(
        geometry=input, component="CURVE"
    )

    count = attribute_domain_size.point_count.astype(dtype=float) + 1.0

    resample_curve_count = pf.nodes.geo.resample_curve_count(
        curve=curve_line,
        count=count.astype(dtype=int),
    )

    input_position = pf.nodes.geo.input_position()
    input_index = pf.nodes.geo.input_index()

    sample_index_index = pf.nodes.math.floor_mod(
        a=input_index.astype(dtype=float),
        b=attribute_domain_size.point_count.astype(dtype=float),
    )
    sample_index = pf.nodes.geo.sample_index(
        geometry=input,
        index=sample_index_index.astype(dtype=int),
        value=input_position,
    )

    set_position = pf.nodes.geo.set_position(
        geometry=resample_curve_count, position=sample_index
    )
    return set_position
