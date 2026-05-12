from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t


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
    curve_line = pf.nodes.geo.curve_line(end=(1.0, 0.0, 0.0))

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

    curve_line_1 = pf.nodes.geo.curve_line(end=(-1.0, 0.0, 0.0))

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

    uv_integer = pf.nodes.func.combine_xyz(
        x=capture_attribute_1.index.astype(dtype=float),
        y=capture_attribute_3.index.astype(dtype=float),
    )
    uv_factor = pf.nodes.func.combine_xyz(
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

    uv_out = pf.nodes.func.map_range(
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
) -> GridFromSpacingResult:
    surface_minmax = pf.nodes.geo.attribute_statistic(
        geometry=uv_surface, attribute=target_uv
    )
    uv_range_dims = surface_minmax.max - surface_minmax.min

    bound_box = pf.nodes.geo.bound_box(instance)

    min_center_uv = margin_low - bound_box.min
    max_center_uv = margin_high + bound_box.max

    uv_range_margined = (uv_range_dims - min_center_uv) - max_center_uv

    dims_with_spacing = (bound_box.max - bound_box.min) + spacing

    vertices_vec = pf.nodes.math.vector_ceil(uv_range_margined / dims_with_spacing)

    clip_b = pf.nodes.func.combine_xyz(
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

    query_uv = pf.nodes.func.map_range(
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


class FacesForInstanceGridBboxesResult(NamedTuple):
    mesh: pf.ProcNode[pf.MeshObject]
    is_instance_face: pf.ProcNode[bool]


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
) -> FacesForInstanceGridBboxesResult:
    n_verts_x = pf.nodes.geo.attribute_statistic(
        geometry=query_grid,
        attribute=grid_index_x.astype(dtype=float),
    )

    fillmesh_verts_x = pf.nodes.math.multiply_add(
        a=n_verts_x.max + 1.0,
        b=verts_per_instance_x.astype(dtype=float),
        addend=margin_verts_x.astype(dtype=float) * 2.0,
    )

    n_verts_y = pf.nodes.geo.attribute_statistic(
        geometry=query_grid,
        attribute=grid_index_y.astype(dtype=float),
    )

    n_instances_y = n_verts_y.max + 1.0
    fillmesh_verts_y = pf.nodes.math.multiply_add(
        a=n_instances_y,
        b=verts_per_instance_y.astype(dtype=float),
        addend=margin_verts_y.astype(dtype=float) * 2.0,
    )

    grid_result = grid_with_indices(
        vertices_x=fillmesh_verts_x.astype(dtype=int),
        vertices_y=fillmesh_verts_y.astype(dtype=int),
    )

    input_position = pf.nodes.geo.input_position()

    subgrid_result = subgrid(
        x_index=grid_result.index_x,
        y_index=grid_result.index_y,
        n_verts_x=fillmesh_verts_x.astype(dtype=int),
        n_verts_y=fillmesh_verts_y.astype(dtype=int),
        margin_verts_x=margin_verts_x,
        margin_verts_y=margin_verts_y,
    )

    mix_value = pf.nodes.func.combine_xyz(
        x=subgrid_result.is_boundary_x.astype(dtype=float),
        y=subgrid_result.is_boundary_y.astype(dtype=float),
    )

    corner_idx_x_a = pf.nodes.math.floor_mod(
        a=subgrid_result.subgrid_index_x.astype(dtype=float),
        b=verts_per_instance_x.astype(dtype=float),
    )
    corner_idx_x = pf.nodes.func.equal(a=corner_idx_x_a, b=1.0, epsilon=0.001)

    corner_idx_y_a = pf.nodes.math.floor_mod(
        a=subgrid_result.subgrid_index_y.astype(dtype=float),
        b=verts_per_instance_y.astype(dtype=float),
    )
    corner_idx_y = pf.nodes.func.equal(a=corner_idx_y_a, b=1.0, epsilon=0.001)

    take_which_corner_value = pf.nodes.func.combine_xyz(
        x=corner_idx_x.astype(dtype=float),
        y=corner_idx_y.astype(dtype=float),
    )

    bound_box = pf.nodes.geo.bound_box(instance)

    subgrid_instance_x = pf.nodes.math.floor(
        subgrid_result.subgrid_index_x.astype(dtype=float)
        / verts_per_instance_x.astype(dtype=float)
    )
    instance_a = pf.nodes.math.clamp(value=subgrid_instance_x, max=n_verts_x.max)

    subgrid_instance_y = pf.nodes.math.floor(
        subgrid_result.subgrid_index_y.astype(dtype=float)
        / verts_per_instance_y.astype(dtype=float)
    )
    instance_addend = pf.nodes.math.clamp(value=subgrid_instance_y, max=n_verts_y.max)

    instance_id = pf.nodes.math.multiply_add(
        a=instance_a, b=n_instances_y, addend=instance_addend
    )
    instance_uv = pf.nodes.geo.sample_index(
        geometry=query_grid,
        index=instance_id.astype(dtype=int),
        value=instance_uvs,
    )

    take_which_corner = pf.nodes.func.map_range(
        value=take_which_corner_value,
        from_min=(0.0, 0.0, 0.0),
        from_max=(1.0, 1.0, 1.0),
        to_min=bound_box.min + instance_uv,
        to_max=bound_box.max + instance_uv,
    )

    normed_uv = normed_uv_to_bounds_uv(
        geometry=target_surface,
        target_uv=target_uv,
        query_uv=grid_result.uv_factor,
        margin_low=(0.0, 0.0, 0.0),
        margin_high=(0.0, 0.0, -0.1),
    )

    mix_for_corners_vs_boundaries = pf.nodes.func.map_range(
        value=mix_value,
        from_min=(0.0, 0.0, 0.0),
        from_max=(1.0, 1.0, 1.0),
        to_min=take_which_corner,
        to_max=normed_uv.uv_out,
    )

    sample_uv_surface = pf.nodes.geo.sample_uv_surface(
        mesh=target_surface,
        value=input_position,
        sample_uv=mix_for_corners_vs_boundaries,
        uv_map=target_uv,
    )

    set_position = pf.nodes.geo.set_position(
        geometry=grid_result.mesh,
        position=sample_uv_surface.value,
    )

    corners_of_face = pf.nodes.geo.corners_of_face()
    vertex_of_corner = pf.nodes.geo.vertex_of_corner(corners_of_face.corner_index)

    sample_index = pf.nodes.geo.sample_index(
        geometry=grid_result.mesh,
        index=vertex_of_corner,
        value=subgrid_result.subgrid_index_x,
    )
    capture_a_1 = pf.nodes.math.floor_mod(
        a=sample_index.astype(dtype=float),
        b=verts_per_instance_x.astype(dtype=float),
    )
    is_instance_x = pf.nodes.func.equal(a=capture_a_1, b=0.0, epsilon=0.001)

    sample_index_1 = pf.nodes.geo.sample_index(
        geometry=grid_result.mesh,
        index=vertex_of_corner,
        value=subgrid_result.subgrid_index_y,
    )
    capture_a_0 = pf.nodes.math.floor_mod(
        a=sample_index_1.astype(dtype=float),
        b=verts_per_instance_y.astype(dtype=float),
    )
    is_instance_y = pf.nodes.func.equal(a=capture_a_0, b=0.0, epsilon=0.001)

    is_instance_face = pf.nodes.func.boolean_and(a=is_instance_x, b=is_instance_y)

    capture_attribute = pf.nodes.geo.capture_attribute(
        geometry=set_position,
        domain="FACE",
        boolean=is_instance_face,
    )

    return FacesForInstanceGridBboxesResult(
        mesh=capture_attribute.geometry,
        is_instance_face=capture_attribute.boolean,
    )


@pf.nodes.node_function
def place_instances_on_uv_grid(
    surface: pf.ProcNode[pf.MeshObject],
    uv_field: t.SocketOrVal[pf.Vector],
    grid_mesh: pf.ProcNode[pf.MeshObject],
    query_uv: t.SocketOrVal[pf.Vector],
    instance: pf.ProcNode[pf.MeshObject],
) -> pf.ProcNode[t.Instances]:
    input_position = pf.nodes.geo.input_position()
    sample_uv = pf.nodes.geo.sample_uv_surface(
        mesh=surface,
        value=input_position,
        sample_uv=query_uv,
        uv_map=uv_field,
    )
    grid_positioned = pf.nodes.geo.set_position(
        geometry=grid_mesh, position=sample_uv.value
    )

    input_normal = pf.nodes.geo.input_normal()
    sample_normal = pf.nodes.geo.sample_uv_surface(
        mesh=surface,
        value=input_normal,
        sample_uv=query_uv,
        uv_map=uv_field,
    )
    rotation = pf.nodes.func.axes_to_rotation(
        primary_axis_vector=sample_normal.value,
        secondary_axis_vector=(0, 0, 1),
        primary_axis="Z",
        secondary_axis="Y",
    )

    return pf.nodes.geo.instance_on_points(
        points=grid_positioned,
        instance=instance,
        rotation=rotation,
    )
