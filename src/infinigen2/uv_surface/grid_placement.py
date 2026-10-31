# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import math
from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

__all__ = [
    "FacesForInstanceGridBboxesResult",
    "GridFromSpacingResult",
    "GridWithIndicesResult",
    "NormedUvToBoundsUvResult",
    "SubgridResult",
    "faces_for_instance_grid_bboxes",
    "grid_between",
    "grid_from_spacing",
    "grid_with_indices",
    "normed_uv_to_bounds_uv",
    "place_instances_on_uv_grid",
    "subgrid",
]


class SubgridResult(NamedTuple):
    is_boundary: pf.ProcNode[bool]
    is_boundary_x: pf.ProcNode[bool]
    is_boundary_y: pf.ProcNode[bool]
    subgrid_verts_x: pf.ProcNode[int]
    subgrid_verts_y: pf.ProcNode[int]
    subgrid_index_x: pf.ProcNode[int]
    subgrid_index_y: pf.ProcNode[int]


@pf.nodes.node_function
def subgrid(
    x_index: t.SocketOrVal[int],
    y_index: t.SocketOrVal[int],
    n_verts_x: t.SocketOrVal[int],
    n_verts_y: t.SocketOrVal[int],
    margin_verts_x: t.SocketOrVal[int],
    margin_verts_y: t.SocketOrVal[int],
) -> SubgridResult:
    is_boundary_x_a = pf.nodes.func.less_than(a=x_index, b=margin_verts_x)
    is_boundary_x_b_b = n_verts_x.astype(dtype=float) - margin_verts_x.astype(
        dtype=float
    )
    is_boundary_x_b = pf.nodes.func.greater_equal(
        a=x_index, b=is_boundary_x_b_b.astype(dtype=int)
    )
    is_boundary_x = pf.nodes.func.boolean_or(a=is_boundary_x_a, b=is_boundary_x_b)
    is_boundary_y_a = pf.nodes.func.less_than(a=y_index, b=margin_verts_y)
    is_boundary_y_b_b = n_verts_y.astype(dtype=float) - margin_verts_y.astype(
        dtype=float
    )
    is_boundary_y_b = pf.nodes.func.greater_equal(
        a=y_index, b=is_boundary_y_b_b.astype(dtype=int)
    )
    is_boundary_y = pf.nodes.func.boolean_or(a=is_boundary_y_a, b=is_boundary_y_b)
    is_boundary = pf.nodes.func.boolean_or(a=is_boundary_x, b=is_boundary_y)

    subgrid_verts_x_1 = pf.nodes.math.multiply_add(
        a=margin_verts_x.astype(dtype=float),
        b=-2.0,
        addend=n_verts_x.astype(dtype=float),
    )
    subgrid_verts_x = subgrid_verts_x_1.astype(dtype=int)
    subgrid_verts_y_1 = pf.nodes.math.multiply_add(
        a=margin_verts_y.astype(dtype=float),
        b=-2.0,
        addend=n_verts_y.astype(dtype=float),
    )
    subgrid_verts_y = subgrid_verts_y_1.astype(dtype=int)
    subgrid_index_x = x_index.astype(dtype=float) - margin_verts_x.astype(dtype=float)
    subgrid_index_y = y_index.astype(dtype=float) - margin_verts_y.astype(dtype=float)
    return SubgridResult(
        is_boundary=is_boundary,
        is_boundary_x=is_boundary_x,
        is_boundary_y=is_boundary_y,
        subgrid_verts_x=subgrid_verts_x,
        subgrid_verts_y=subgrid_verts_y,
        subgrid_index_x=subgrid_index_x,
        subgrid_index_y=subgrid_index_y,
    )


class GridWithIndicesResult(NamedTuple):
    mesh: pf.ProcNode[pf.MeshObject]
    index_x: pf.ProcNode[int]
    index_y: pf.ProcNode[int]
    uv_integer: pf.ProcNode[pf.Vector]
    uv_factor: pf.ProcNode[pf.Vector]


@pf.nodes.node_function
def grid_with_indices(
    vertices_x: t.SocketOrVal[int],
    vertices_y: t.SocketOrVal[int],
) -> GridWithIndicesResult:
    curve_line = pf.nodes.geo.curve_line(end=(1.0, 0.0, 0.0), start=(0, 0, 0))

    resample_curve_count = pf.nodes.geo.resample_curve_count(
        curve=curve_line, count=vertices_x
    )

    spline_parameter = pf.nodes.geo.spline_parameter()

    capture_attribute = pf.nodes.geo.capture_attribute(
        geometry=resample_curve_count,
        factor=spline_parameter.factor,
    )

    input_index = pf.nodes.geo.input_index()

    capture_attribute_1 = pf.nodes.geo.capture_attribute(
        geometry=capture_attribute.geometry,
        index=input_index,
    )

    curve_line_1 = pf.nodes.geo.curve_line(end=(-1.0, 0.0, 0.0), start=(0, 0, 0))

    resample_curve_count_1 = pf.nodes.geo.resample_curve_count(
        curve=curve_line_1, count=vertices_y
    )

    capture_attribute_2 = pf.nodes.geo.capture_attribute(
        geometry=resample_curve_count_1,
        factor=spline_parameter.factor,
    )

    input_index_1 = pf.nodes.geo.input_index()

    capture_attribute_3 = pf.nodes.geo.capture_attribute(
        geometry=capture_attribute_2.geometry,
        index=input_index_1,
    )

    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=capture_attribute_1.geometry,
        profile_curve=capture_attribute_3.geometry,
    )

    uv_integer = pf.nodes.math.combine_xyz(
        x=capture_attribute_1.index.astype(dtype=float),
        y=capture_attribute_3.index.astype(dtype=float),
    )
    uv_factor = pf.nodes.math.combine_xyz(
        x=capture_attribute.factor, y=capture_attribute_2.factor
    )
    return GridWithIndicesResult(
        mesh=curve_to,
        index_x=capture_attribute_1.index,
        index_y=capture_attribute_3.index,
        uv_integer=uv_integer,
        uv_factor=uv_factor,
    )


class NormedUvToBoundsUvResult(NamedTuple):
    uv_out: pf.ProcNode[pf.Vector]
    lower: pf.ProcNode[pf.Vector]
    upper: pf.ProcNode[pf.Vector]


@pf.nodes.node_function
def normed_uv_to_bounds_uv(
    geometry: pf.ProcNode[pf.MeshObject],
    target_uv: t.SocketOrVal[pf.Vector],
    query_uv: t.SocketOrVal[pf.Vector],
    margin_low: t.SocketOrVal[pf.Vector],
    margin_high: t.SocketOrVal[pf.Vector],
) -> NormedUvToBoundsUvResult:
    attribute_statistic = pf.nodes.geo.attribute_statistic(
        geometry=geometry, attribute=target_uv
    )

    lower = attribute_statistic.min + margin_low
    upper = attribute_statistic.max - margin_high

    uv_out = pf.nodes.math.map_range(
        value=query_uv,
        from_min=(0.0, 0.0, 0.0),
        from_max=(1.0, 1.0, 1.0),
        to_min=lower,
        to_max=upper,
    )

    return NormedUvToBoundsUvResult(uv_out=uv_out, lower=lower, upper=upper)


class GridFromSpacingResult(NamedTuple):
    grid_mesh: pf.ProcNode[pf.MeshObject]
    query_uv: pf.ProcNode[pf.Vector]
    index_x: pf.ProcNode[int]
    index_y: pf.ProcNode[int]


def _footprint_uv_bounds(
    instance: pf.ProcNode[pf.MeshObject],
    rotation_offset: t.SocketOrVal[pf.Vector],
) -> tuple[t.SocketOrVal[pf.Vector], t.SocketOrVal[pf.Vector]]:
    """(U, V) footprint bounds of an instance placed on the surface.

    Instances are authored in the wall frame (X out of wall, Y along wall = U,
    Z up = V); after the per-asset rotation_offset the footprint the niche grid
    sees is the reoriented Y/Z extent, packed into the x,y of each result.
    """
    placed = pf.nodes.geo.transform(geometry=instance, rotation=rotation_offset)
    bb = pf.nodes.geo.bound_box(placed)
    bb_min = pf.nodes.math.combine_xyz(x=bb.min.y, y=bb.min.z)
    bb_max = pf.nodes.math.combine_xyz(x=bb.max.y, y=bb.max.z)
    return bb_min, bb_max


def _uv_is_on_surface(
    surface: pf.ProcNode[pf.MeshObject],
    uv_field: t.SocketOrVal[pf.Vector],
    query_uv: t.SocketOrVal[pf.Vector],
    offset_u: t.SocketOrVal[float],
    offset_v: t.SocketOrVal[float],
) -> pf.ProcNode[bool]:
    return pf.nodes.geo.sample_uv_surface(
        mesh=surface,
        value=pf.nodes.geo.input_position(),
        sample_uv=query_uv + pf.nodes.math.combine_xyz(x=offset_u, y=offset_v),
        uv_map=uv_field,
    ).is_valid


@pf.nodes.node_function
def _instance_footprint_is_valid(
    surface: pf.ProcNode[pf.MeshObject],
    uv_field: t.SocketOrVal[pf.Vector],
    query_uv: t.SocketOrVal[pf.Vector],
    instance: pf.ProcNode[pf.MeshObject],
    rotation_offset: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode[bool]:
    lo, hi = _footprint_uv_bounds(instance, rotation_offset)
    lower_left = _uv_is_on_surface(surface, uv_field, query_uv, lo.x, lo.y)
    lower_right = _uv_is_on_surface(surface, uv_field, query_uv, hi.x, lo.y)
    upper_left = _uv_is_on_surface(surface, uv_field, query_uv, lo.x, hi.y)
    upper_right = _uv_is_on_surface(surface, uv_field, query_uv, hi.x, hi.y)
    lower = pf.nodes.func.boolean_and(a=lower_left, b=lower_right)
    upper = pf.nodes.func.boolean_and(a=upper_left, b=upper_right)
    return pf.nodes.func.boolean_and(a=lower, b=upper)


@pf.nodes.node_function
def grid_from_spacing(
    uv_surface: pf.ProcNode[pf.MeshObject],
    target_uv: t.SocketOrVal[pf.Vector],
    instance: pf.ProcNode[pf.MeshObject],
    spacing: t.SocketOrVal[pf.Vector],
    margin_low: t.SocketOrVal[pf.Vector],
    margin_high: t.SocketOrVal[pf.Vector],
    x_instances_max: t.SocketOrVal[int] = 1000,
    y_instances_max: t.SocketOrVal[int] = 1000,
    rotation_offset: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
) -> GridFromSpacingResult:
    surface_minmax = pf.nodes.geo.attribute_statistic(
        geometry=uv_surface, attribute=target_uv
    )
    uv_range_dims = surface_minmax.max - surface_minmax.min

    bb_min, bb_max = _footprint_uv_bounds(instance, rotation_offset)

    min_center_uv = margin_low - bb_min
    max_center_uv = margin_high + bb_max

    uv_range_margined = (uv_range_dims - min_center_uv) - max_center_uv

    dims_with_spacing = (bb_max - bb_min) + spacing

    vertices_vec = pf.nodes.math.vector_ceil(uv_range_margined / dims_with_spacing)

    clip_b = pf.nodes.math.combine_xyz(
        x=x_instances_max.astype(dtype=float),
        y=y_instances_max.astype(dtype=float),
    )
    vertices_vec = pf.nodes.math.vector_minimum(a=vertices_vec, b=clip_b)

    grid_with_indices_result = grid_with_indices(
        vertices_x=vertices_vec.x.astype(dtype=int),
        vertices_y=vertices_vec.y.astype(dtype=int),
    )

    min_final = surface_minmax.min + min_center_uv
    max_final = surface_minmax.max - max_center_uv

    query_uv = pf.nodes.math.map_range(
        value=grid_with_indices_result.uv_factor,
        from_min=(0.0, 0.0, 0.0),
        from_max=(1.0, 1.0, 1.0),
        to_min=min_final,
        to_max=max_final,
    )

    return GridFromSpacingResult(
        grid_mesh=grid_with_indices_result.mesh,
        query_uv=query_uv,
        index_x=grid_with_indices_result.index_x,
        index_y=grid_with_indices_result.index_y,
    )


@pf.nodes.node_function
def grid_between(
    first_uv: t.SocketOrVal[pf.Vector],
    last_uv: t.SocketOrVal[pf.Vector],
    count: t.SocketOrVal[int],
) -> GridFromSpacingResult:
    """`count` query points evenly spaced in UV from `first_uv` to `last_uv`."""
    grid = grid_with_indices(vertices_x=count, vertices_y=1)
    query_uv = pf.nodes.math.map_range(
        value=grid.uv_factor,
        from_min=(0.0, 0.0, 0.0),
        from_max=(1.0, 1.0, 1.0),
        to_min=first_uv,
        to_max=last_uv,
    )
    return GridFromSpacingResult(
        grid_mesh=grid.mesh,
        query_uv=query_uv,
        index_x=grid.index_x,
        index_y=grid.index_y,
    )


class FacesForInstanceGridBboxesResult(NamedTuple):
    mesh: pf.ProcNode[pf.MeshObject]
    is_instance_face: pf.ProcNode[bool]
    inset: pf.ProcNode[pf.Vector]  # per point: expanded mouth -> exact footprint


class _InstanceGridLayoutResult(NamedTuple):
    mesh: pf.ProcNode[pf.MeshObject]
    uv_factor: pf.ProcNode[pf.Vector]
    subgrid_index_x: pf.ProcNode[int]
    subgrid_index_y: pf.ProcNode[int]
    is_boundary_x: pf.ProcNode[bool]
    is_boundary_y: pf.ProcNode[bool]
    instance_uv: pf.ProcNode[pf.Vector]
    bb_min: t.SocketOrVal[pf.Vector]
    bb_max: t.SocketOrVal[pf.Vector]


class _InstanceFaceCoordinatesResult(NamedTuple):
    x: pf.ProcNode[int]
    y: pf.ProcNode[int]


@pf.nodes.node_function
def _instance_grid_layout(
    instance: pf.ProcNode[pf.MeshObject],
    query_grid: pf.ProcNode[pf.MeshObject],
    instance_uvs: t.SocketOrVal[pf.Vector],
    grid_index_x: t.SocketOrVal[int],
    grid_index_y: t.SocketOrVal[int],
    verts_per_instance_x: t.SocketOrVal[int],
    verts_per_instance_y: t.SocketOrVal[int],
    margin_verts_x: t.SocketOrVal[int],
    margin_verts_y: t.SocketOrVal[int],
    rotation_offset: t.SocketOrVal[pf.Vector],
) -> _InstanceGridLayoutResult:
    index_x_stat = pf.nodes.geo.attribute_statistic(
        geometry=query_grid,
        attribute=grid_index_x.astype(dtype=float),
    )
    n_instances_x = index_x_stat.max + 1.0
    fillmesh_verts_x = pf.nodes.math.multiply_add(
        a=n_instances_x,
        b=verts_per_instance_x.astype(dtype=float),
        addend=margin_verts_x.astype(dtype=float) * 2.0,
    )

    index_y_stat = pf.nodes.geo.attribute_statistic(
        geometry=query_grid,
        attribute=grid_index_y.astype(dtype=float),
    )
    n_instances_y = index_y_stat.max + 1.0
    fillmesh_verts_y = pf.nodes.math.multiply_add(
        a=n_instances_y,
        b=verts_per_instance_y.astype(dtype=float),
        addend=margin_verts_y.astype(dtype=float) * 2.0,
    )

    grid_result = grid_with_indices(
        vertices_x=fillmesh_verts_x.astype(dtype=int),
        vertices_y=fillmesh_verts_y.astype(dtype=int),
    )
    subgrid_result = subgrid(
        x_index=grid_result.index_x,
        y_index=grid_result.index_y,
        n_verts_x=fillmesh_verts_x.astype(dtype=int),
        n_verts_y=fillmesh_verts_y.astype(dtype=int),
        margin_verts_x=margin_verts_x,
        margin_verts_y=margin_verts_y,
    )

    instance_x = pf.nodes.math.floor(
        subgrid_result.subgrid_index_x.astype(dtype=float)
        / verts_per_instance_x.astype(dtype=float)
    )
    instance_x = pf.nodes.math.clamp(value=instance_x, max=index_x_stat.max)
    instance_y = pf.nodes.math.floor(
        subgrid_result.subgrid_index_y.astype(dtype=float)
        / verts_per_instance_y.astype(dtype=float)
    )
    instance_y = pf.nodes.math.clamp(value=instance_y, max=index_y_stat.max)
    instance_id = pf.nodes.math.multiply_add(
        a=instance_x,
        b=n_instances_y,
        addend=instance_y,
    )
    instance_uv = pf.nodes.geo.sample_index(
        geometry=query_grid,
        index=instance_id.astype(dtype=int),
        value=instance_uvs,
    )
    bb_min, bb_max = _footprint_uv_bounds(instance, rotation_offset)
    return _InstanceGridLayoutResult(
        mesh=grid_result.mesh,
        uv_factor=grid_result.uv_factor,
        subgrid_index_x=subgrid_result.subgrid_index_x,
        subgrid_index_y=subgrid_result.subgrid_index_y,
        is_boundary_x=subgrid_result.is_boundary_x,
        is_boundary_y=subgrid_result.is_boundary_y,
        instance_uv=instance_uv,
        bb_min=bb_min,
        bb_max=bb_max,
    )


@pf.nodes.node_function
def _instance_outline_factor(
    subgrid_index_x: t.SocketOrVal[int],
    subgrid_index_y: t.SocketOrVal[int],
    verts_per_instance_x: t.SocketOrVal[int],
    verts_per_instance_y: t.SocketOrVal[int],
    footprint_height: t.SocketOrVal[float],
    top_profile_height: t.SocketOrVal[float],
    bottom_profile_height: t.SocketOrVal[float],
) -> pf.ProcNode[pf.Vector]:
    """Normalized footprint position of each instance vertex.

    Columns are spaced uniformly in angle around ellipse quarters spanning the
    full footprint width, with the given profile heights in metres; zero heights
    with two columns give the plain bbox corners.
    """
    column = pf.nodes.math.floor_mod(
        a=subgrid_index_x.astype(dtype=float),
        b=verts_per_instance_x.astype(dtype=float),
    )
    angle = column / (verts_per_instance_x.astype(dtype=float) - 1.0) * math.pi
    x_factor = (1.0 - pf.nodes.math.cos(angle)) * 0.5
    profile_inset = 1.0 - pf.nodes.math.sin(angle)
    height = pf.nodes.math.maximum(footprint_height, 1e-9)
    bottom = bottom_profile_height * profile_inset / height
    top = 1.0 - top_profile_height * profile_inset / height
    row = pf.nodes.math.floor_mod(
        a=subgrid_index_y.astype(dtype=float),
        b=verts_per_instance_y.astype(dtype=float),
    )
    row_factor = row / (verts_per_instance_y.astype(dtype=float) - 1.0)
    y_factor = pf.nodes.math.mix(a=bottom, b=top, factor=row_factor)
    return pf.nodes.math.combine_xyz(x=x_factor, y=y_factor)


@pf.nodes.node_function
def _instance_face_coordinates(
    grid_mesh: pf.ProcNode[pf.MeshObject],
    subgrid_index_x: pf.ProcNode[int],
    subgrid_index_y: pf.ProcNode[int],
) -> _InstanceFaceCoordinatesResult:
    corner = pf.nodes.geo.corners_of_face()
    vertex = pf.nodes.geo.vertex_of_corner(corner.corner_index)
    x = pf.nodes.geo.sample_index(
        geometry=grid_mesh,
        index=vertex,
        value=subgrid_index_x,
    )
    y = pf.nodes.geo.sample_index(
        geometry=grid_mesh,
        index=vertex,
        value=subgrid_index_y,
    )
    return _InstanceFaceCoordinatesResult(x=x, y=y)


@pf.nodes.node_function
def _faces_from_instance_grid(
    target_surface: pf.ProcNode[pf.MeshObject],
    target_uv: t.SocketOrVal[pf.Vector],
    grid_mesh: pf.ProcNode[pf.MeshObject],
    outer_vertex_uv: t.SocketOrVal[pf.Vector],
    inner_vertex_uv: t.SocketOrVal[pf.Vector],
    is_instance_face: t.SocketOrVal[bool],
) -> FacesForInstanceGridBboxesResult:
    surf_stat = pf.nodes.geo.attribute_statistic(
        geometry=target_surface,
        attribute=target_uv,
    )
    # clamp the query a hair inside the surface UV bounds; a query past the edge hits no
    # face -> value (0,0,0), which merge welds into one stray origin vert
    uv_lo = surf_stat.min + pf.Vector((0.001, 0.001, 0.0))
    uv_hi = surf_stat.max - pf.Vector((0.001, 0.001, 0.0))
    clamped_outer_uv = pf.nodes.math.vector_minimum(
        a=pf.nodes.math.vector_maximum(a=outer_vertex_uv, b=uv_lo), b=uv_hi
    )
    clamped_inner_uv = pf.nodes.math.vector_minimum(
        a=pf.nodes.math.vector_maximum(a=inner_vertex_uv, b=uv_lo), b=uv_hi
    )
    outer_position = pf.nodes.geo.sample_uv_surface(
        mesh=target_surface,
        value=pf.nodes.geo.input_position(),
        sample_uv=clamped_outer_uv,
        uv_map=target_uv,
    )
    inner_position = pf.nodes.geo.sample_uv_surface(
        mesh=target_surface,
        value=pf.nodes.geo.input_position(),
        sample_uv=clamped_inner_uv,
        uv_map=target_uv,
    )
    positioned = pf.nodes.geo.set_position(
        geometry=grid_mesh,
        position=outer_position.value,
    )
    with_inset = pf.nodes.geo.capture_attribute(
        geometry=positioned,
        domain="POINT",
        inset=inner_position.value - outer_position.value,
    )
    captured = pf.nodes.geo.capture_attribute(
        geometry=with_inset.geometry,
        domain="FACE",
        boolean=is_instance_face,
    )
    # store the sampled surface UVs back as UVMap so the wall stays textured
    mesh_with_uv = pf.nodes.geo.store_named_attribute(
        geometry=captured.geometry,
        name="UVMap",
        value=clamped_outer_uv,
        domain="CORNER",
        data_type="FLOAT2",
    )
    # clamped footprint rows coincide with the surface boundary row
    welded = pf.nodes.geo.merge_by_distance(mesh_with_uv, distance=0.001)
    return FacesForInstanceGridBboxesResult(
        mesh=welded,
        is_instance_face=captured.boolean,
        inset=with_inset.inset,
    )


@pf.nodes.node_function
def faces_for_instance_grid_bboxes(
    target_surface: pf.ProcNode[pf.MeshObject],
    target_uv: t.SocketOrVal[pf.Vector],
    instance: pf.ProcNode[pf.MeshObject],
    query_grid: pf.ProcNode[pf.MeshObject],
    instance_uvs: t.SocketOrVal[pf.Vector],
    grid_index_x: t.SocketOrVal[int],
    grid_index_y: t.SocketOrVal[int],
    verts_per_instance_x: t.SocketOrVal[int] = 2,
    verts_per_instance_y: t.SocketOrVal[int] = 2,
    margin_verts_x: t.SocketOrVal[int] = 1,
    margin_verts_y: t.SocketOrVal[int] = 1,
    face_expand_margin: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    rotation_offset: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    top_profile_height: t.SocketOrVal[float] = 0.0,
    bottom_profile_height: t.SocketOrVal[float] = 0.0,
) -> FacesForInstanceGridBboxesResult:
    """Remesh a surface into a grid with one hole face-patch per instance footprint.

    Nonzero profile heights (metres) round the top/bottom of each footprint into
    ellipse quarters; use more than two `verts_per_instance_x` columns for them.
    `face_expand_margin` grows the mesh outline; `inset` maps it back per point.
    """
    layout = _instance_grid_layout(
        instance=instance,
        query_grid=query_grid,
        instance_uvs=instance_uvs,
        grid_index_x=grid_index_x,
        grid_index_y=grid_index_y,
        verts_per_instance_x=verts_per_instance_x,
        verts_per_instance_y=verts_per_instance_y,
        margin_verts_x=margin_verts_x,
        margin_verts_y=margin_verts_y,
        rotation_offset=rotation_offset,
    )
    footprint_size = pf.nodes.math.separate_xyz(layout.bb_max - layout.bb_min)
    factor = _instance_outline_factor(
        subgrid_index_x=layout.subgrid_index_x,
        subgrid_index_y=layout.subgrid_index_y,
        verts_per_instance_x=verts_per_instance_x,
        verts_per_instance_y=verts_per_instance_y,
        footprint_height=footprint_size.y,
        top_profile_height=top_profile_height,
        bottom_profile_height=bottom_profile_height,
    )
    outer_instance_uv = pf.nodes.math.map_range(
        value=factor,
        from_min=(0.0, 0.0, 0.0),
        from_max=(1.0, 1.0, 1.0),
        to_min=layout.bb_min - face_expand_margin + layout.instance_uv,
        to_max=layout.bb_max + face_expand_margin + layout.instance_uv,
    )
    inset_direction = factor * -2.0 + pf.Vector((1.0, 1.0, 1.0))
    inner_instance_uv = outer_instance_uv + face_expand_margin * inset_direction

    boundary_uv = normed_uv_to_bounds_uv(
        geometry=target_surface,
        target_uv=target_uv,
        query_uv=layout.uv_factor,
        margin_low=(0.0, 0.0, 0.0),
        margin_high=(0.0, 0.0, -0.1),
    )
    boundary_factor = pf.nodes.math.combine_xyz(
        x=layout.is_boundary_x.astype(dtype=float),
        y=layout.is_boundary_y.astype(dtype=float),
    )
    outer_vertex_uv = pf.nodes.math.map_range(
        value=boundary_factor,
        from_min=(0.0, 0.0, 0.0),
        from_max=(1.0, 1.0, 1.0),
        to_min=outer_instance_uv,
        to_max=boundary_uv.uv_out,
    )
    inner_vertex_uv = pf.nodes.math.map_range(
        value=boundary_factor,
        from_min=(0.0, 0.0, 0.0),
        from_max=(1.0, 1.0, 1.0),
        to_min=inner_instance_uv,
        to_max=boundary_uv.uv_out,
    )

    face = _instance_face_coordinates(
        grid_mesh=layout.mesh,
        subgrid_index_x=layout.subgrid_index_x,
        subgrid_index_y=layout.subgrid_index_y,
    )
    face_x = pf.nodes.math.floor_mod(
        a=face.x.astype(dtype=float),
        b=verts_per_instance_x.astype(dtype=float),
    )
    is_instance_x = pf.nodes.func.less_than(
        a=face_x, b=verts_per_instance_x.astype(dtype=float) - 1.0
    )
    face_y = pf.nodes.math.floor_mod(
        a=face.y.astype(dtype=float),
        b=verts_per_instance_y.astype(dtype=float),
    )
    is_instance_y = pf.nodes.func.less_than(
        a=face_y, b=verts_per_instance_y.astype(dtype=float) - 1.0
    )
    is_instance_face = pf.nodes.func.boolean_and(a=is_instance_x, b=is_instance_y)
    return _faces_from_instance_grid(
        target_surface=target_surface,
        target_uv=target_uv,
        grid_mesh=layout.mesh,
        outer_vertex_uv=outer_vertex_uv,
        inner_vertex_uv=inner_vertex_uv,
        is_instance_face=is_instance_face,
    )


@pf.nodes.node_function
def place_instances_on_uv_grid(
    surface: pf.ProcNode[pf.MeshObject],
    uv_field: t.SocketOrVal[pf.Vector],
    grid_mesh: pf.ProcNode[pf.MeshObject],
    query_uv: t.SocketOrVal[pf.Vector],
    instance: pf.ProcNode[pf.MeshObject],
    secondary_axis_vector: t.SocketOrVal[pf.Vector] = (0, 0, 1),
    rotation_offset: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    normal_offset: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[t.Instances]:
    position_sample = pf.nodes.geo.sample_uv_surface(
        mesh=surface,
        value=pf.nodes.geo.input_position(),
        sample_uv=query_uv,
        uv_map=uv_field,
    )
    normal = pf.nodes.geo.sample_uv_surface(
        mesh=surface,
        value=pf.nodes.geo.input_normal(),
        sample_uv=query_uv,
        uv_map=uv_field,
    ).value

    footprint_valid = _instance_footprint_is_valid(
        surface=surface,
        uv_field=uv_field,
        query_uv=query_uv,
        instance=instance,
        rotation_offset=rotation_offset,
    )
    sample_valid = pf.nodes.func.boolean_and(
        a=position_sample.is_valid,
        b=footprint_valid,
    )

    seated = position_sample.value + normal * normal_offset
    grid_positioned = pf.nodes.geo.set_position(geometry=grid_mesh, position=seated)

    # local +X -> normal (out), +Z -> secondary reference (world up on walls),
    # +Y -> along surface; rotation_offset reorients in the instance's own frame
    rotation = pf.nodes.func.axes_to_rotation(
        primary_axis_vector=normal,
        secondary_axis_vector=secondary_axis_vector,
        primary_axis="X",
        secondary_axis="Z",
    )
    rotation = pf.nodes.func.rotate_rotation(
        rotation=rotation, rotate_by=rotation_offset, rotation_space="LOCAL"
    )

    return pf.nodes.geo.instance_on_points(
        points=grid_positioned,
        instance=instance,
        rotation=rotation,
        selection=sample_valid,
    )
