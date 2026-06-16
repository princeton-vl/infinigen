from typing import NamedTuple

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t


class ExtrudeSeamlessResult(NamedTuple):
    mesh: pf.ProcNode[pf.MeshObject]
    top: pf.ProcNode[bool]
    side: pf.ProcNode[bool]


@pf.nodes.node_function
def extrude_mesh_seamless_uvs(
    mesh: pf.ProcNode[pf.MeshObject],
    selection: t.SocketOrVal[bool],
    offset_scale: t.SocketOrVal[float],
    uv_winding_sign: t.SocketOrVal[float] = 1.0,
) -> ExtrudeSeamlessResult:
    """Extrude faces and continue source UVs onto the new side faces seamlessly.

    Each side corner gets uv0 plus a perpendicular offset proportional to its
    extrusion depth. The boundary edge driving that offset is found per corner via
    its own next-corner edge (the one with both ends on the depth-0 ring), then
    broadcast across the side face with accumulate_field.
    """

    uv_name = "UVMap"
    pos0 = pf.nodes.geo.input_position()
    uv0 = pf.nodes.geo.input_named_attribute(
        uv_name, data_type="FLOAT_VECTOR"
    ).attribute
    cap = pf.nodes.geo.capture_attribute(mesh, domain="CORNER", p0=pos0, uv0=uv0)

    ext = pf.nodes.geo.extrude_mesh(
        mesh=cap.geometry,
        selection=selection,
        offset_scale=offset_scale,
        individual=False,
    )

    # depth is 0 on the boundary ring, extrusion depth on the moved ring
    pos = pf.nodes.geo.input_position()
    depth = pf.nodes.math.vector_length(pos - cap.p0)
    idx = pf.nodes.geo.input_index()

    nc = pf.nodes.geo.offset_corner_in_face(corner_index=idx, offset=1)
    depth_n = pf.nodes.geo.field_at_index(value=depth, index=nc, domain="CORNER")
    uv0_n = pf.nodes.geo.field_at_index(value=cap.uv0, index=nc, domain="CORNER")
    p0_n = pf.nodes.geo.field_at_index(value=cap.p0, index=nc, domain="CORNER")
    on_edge = (
        pf.nodes.func.less_than(a=depth, b=1e-4).astype(dtype=float)
        * pf.nodes.func.less_than(a=depth_n, b=1e-4).astype(dtype=float)
    ).astype(dtype=pf.Vector)

    face = pf.nodes.geo.face_of_corner(corner_index=idx).face_index
    e_uv = pf.nodes.geo.accumulate_field(
        value=(uv0_n - cap.uv0) * on_edge, group_id=face, domain="CORNER"
    ).total
    e_3d = pf.nodes.geo.accumulate_field(
        value=(p0_n - cap.p0) * on_edge, group_id=face, domain="CORNER"
    ).total

    scale = pf.nodes.math.vector_length(e_uv) / pf.nodes.math.maximum(
        pf.nodes.math.vector_length(e_3d), 1e-9
    )
    exyz = pf.nodes.math.separate_xyz(e_uv)
    rot90 = pf.nodes.math.combine_xyz(x=exyz.y, y=exyz.x * -1.0)
    perp = pf.nodes.math.vector_normalize(rot90) * (uv_winding_sign * -1.0).astype(
        dtype=pf.Vector
    )

    uv_new = cap.uv0 + perp * (scale * depth).astype(dtype=pf.Vector)

    out = pf.nodes.geo.store_named_attribute(
        geometry=ext.mesh,
        name=uv_name,
        selection=ext.side,
        value=uv_new.astype(dtype=pf.Vector),
        domain="CORNER",
        data_type="FLOAT2",
    )
    return ExtrudeSeamlessResult(mesh=out, top=ext.top, side=ext.side)


def uv_winding_sign(obj: pf.MeshObject) -> float:
    """Sign of the first face's UV winding (+1/-1), for un-mirrored continuation."""
    me = obj.item().data
    if not len(me.uv_layers) or not len(me.polygons):
        return 1.0
    uv = pf.ops.attr.uv_coords(obj)
    start = int(pf.ops.attr.loop_starts(obj)[0])
    total = int(pf.ops.attr.polygon_loop_totals(obj).astype(int)[0])
    p = uv[start : start + total]
    shoelace = np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(
        p[:, 1], np.roll(p[:, 0], -1)
    )
    return 1.0 if shoelace >= 0 else -1.0


class ExtrudeChamferResult(NamedTuple):
    mesh: pf.ProcNode[pf.MeshObject]
    top: pf.ProcNode[bool]  # the deep (full-depth) face cap
    chamfer: pf.ProcNode[bool]  # the angled lip ring at the mouth
    side: pf.ProcNode[bool]  # the deep straight sill walls


@pf.nodes.node_function
def extrude_inwards_with_chamfer(
    mesh: pf.ProcNode[pf.MeshObject],
    selection: t.SocketOrVal[bool],
    offset_scale: t.SocketOrVal[float],
    chamfer: t.SocketOrVal[float] = 0.006,
    uv_winding_sign: t.SocketOrVal[float] = 1.0,
) -> ExtrudeChamferResult:
    """Extrude a flat face selection inward in two steps, leaving a chamfered lip.

    First a short extrude insets the rim by `chamfer` along each in-plane axis and
    in depth by `chamfer`, forming an angled bevel at the mouth; then the resulting
    top face is extruded the remaining depth straight in (with seamless UVs on the
    deep walls). `offset_scale` is the signed full inward depth (negative = into the
    surface); the chamfer consumes `chamfer` of that depth. The in-plane axes are the surface's own (horizontal
    along-wall, up-wall) basis derived from the flat-face normal, so the inset stays
    correct on walls of any world orientation. The flat normal and face center are
    captured before the extrude (where every face is coplanar) to avoid the rim
    vertex normals/centroids being skewed by the freshly created bevel faces.
    """
    depth_sign = pf.nodes.math.sign(offset_scale)
    chamfer_depth = pf.nodes.math.multiply(a=chamfer, b=depth_sign)

    flat = pf.nodes.geo.capture_attribute(
        domain="FACE",
        geometry=mesh,
        surf_n=pf.nodes.geo.input_normal(),
        center=pf.nodes.geo.input_position(),
    )
    extrude_dir = pf.nodes.math.vector_normalize(flat.surf_n)
    use_x = pf.nodes.func.greater_than(
        a=pf.nodes.math.absolute(pf.nodes.math.separate_xyz(extrude_dir).z), b=0.9
    ).astype(dtype=float)
    ref = pf.nodes.math.vector_scale(
        vector=(0.0, 0.0, 1.0), scale=1.0 - use_x
    ) + pf.nodes.math.vector_scale(vector=(1.0, 0.0, 0.0), scale=use_x)
    axis_u = pf.nodes.math.vector_normalize(
        pf.nodes.math.vector_cross_product(extrude_dir, ref)
    )
    axis_v = pf.nodes.math.vector_cross_product(extrude_dir, axis_u)
    to_center = flat.center - pf.nodes.geo.input_position()
    inset_u = pf.nodes.math.sign(pf.nodes.math.vector_dot_product(to_center, axis_u))
    inset_v = pf.nodes.math.sign(pf.nodes.math.vector_dot_product(to_center, axis_v))
    chamfer_offset = (
        pf.nodes.math.vector_scale(
            vector=axis_u, scale=pf.nodes.math.multiply(a=inset_u, b=chamfer)
        )
        + pf.nodes.math.vector_scale(
            vector=axis_v, scale=pf.nodes.math.multiply(a=inset_v, b=chamfer)
        )
        + pf.nodes.math.vector_scale(vector=extrude_dir, scale=chamfer_depth)
    )
    lip = pf.nodes.geo.extrude_mesh(
        mesh=flat.geometry,
        selection=selection,
        offset_scale=0.0,
        individual=False,
        mode="FACES",
    )
    lip_in = pf.nodes.geo.set_position(
        geometry=lip.mesh, selection=lip.top, offset=chamfer_offset
    )
    remaining = pf.nodes.math.subtract(offset_scale, chamfer_depth)
    deep = extrude_mesh_seamless_uvs(
        mesh=lip_in,
        selection=lip.top,
        offset_scale=remaining,
        uv_winding_sign=uv_winding_sign,
    )
    return ExtrudeChamferResult(
        mesh=deep.mesh, top=deep.top, chamfer=lip.side, side=deep.side
    )


class WallCutoutResult(NamedTuple):
    wall: pf.ProcNode[pf.MeshObject]  # holed front wall
    sill: pf.ProcNode[pf.MeshObject]  # reveal/jamb tunnels
    lightblocker: pf.ProcNode[pf.MeshObject]  # opaque backing


@pf.nodes.node_function
def wall_cutout_split(
    geometry: pf.ProcNode[pf.MeshObject],
    thickness: t.SocketOrVal[float],
    blocker_thickness: t.SocketOrVal[float] = 0.1,
    uv_winding_sign: t.SocketOrVal[float] = 1.0,
    delete_facecap: t.SocketOrVal[bool] = True,
    chamfer: t.SocketOrVal[float] = 0.006,
) -> WallCutoutResult:
    """Split a flat `cutout_sel`-tagged surface into holed wall, sill tunnels, and lightblocker backing.

    The mouth of each hole gets a `chamfer`-wide angled lip, kept on the wall (it
    reads as a bevel of the wall surface around the opening); the sill is just the
    deep reveal tunnel. The footprint must be pre-expanded by `chamfer` (see
    `face_expand_margin`) so the inner opening lands at the originally-intended size.
    """
    sel = pf.nodes.geo.input_named_attribute(
        name="cutout_sel", data_type=pf.NodeDataType.BOOLEAN
    ).attribute

    lip = extrude_inwards_with_chamfer(
        mesh=geometry,
        selection=sel,
        offset_scale=pf.nodes.math.multiply(a=thickness, b=-1.0),
        chamfer=chamfer,
        uv_winding_sign=uv_winding_sign,
    )
    tagged = pf.nodes.geo.store_named_attribute(
        geometry=lip.mesh,
        name="is_sill",
        value=lip.side,
        domain="FACE",
        data_type="BOOLEAN",
    )
    drop_cap = pf.nodes.func.boolean_and(a=lip.top, b=delete_facecap)
    niche = pf.nodes.geo.delete_geometry(
        tagged, selection=drop_cap, domain="FACE", mode="ALL"
    )
    # thicken the niche into the lightblocker backing
    blocker_offset = pf.nodes.math.multiply(a=blocker_thickness, b=-1.0)
    lightblocker = pf.nodes.geo.extrude_mesh(
        niche, offset_scale=blocker_offset, individual=False, mode="FACES"
    ).mesh
    # split sills off; remainder is the holed wall
    is_sill = pf.nodes.geo.input_named_attribute(
        name="is_sill", data_type=pf.NodeDataType.BOOLEAN
    ).attribute
    sill_sep = pf.nodes.geo.separate_geometry(niche, selection=is_sill, domain="FACE")
    return WallCutoutResult(
        wall=sill_sep.inverted, sill=sill_sep.selection, lightblocker=lightblocker
    )


class GridFromCurvesResult(NamedTuple):
    mesh: pf.ProcNode[pf.MeshObject]
    index_x: pf.ProcNode[int]
    index_y: pf.ProcNode[int]
    side_edge_index: pf.ProcNode[int]


@pf.nodes.node_function
def grid_from_curves(
    dimension_x: t.SocketOrVal[float],
    dimension_y: t.SocketOrVal[float],
    vertices_x: t.SocketOrVal[int],
    vertices_y: t.SocketOrVal[int],
) -> GridFromCurvesResult:
    # Code generated by procfunc v0.12.0

    curve_line_end = pf.nodes.math.combine_xyz(dimension_x)
    curve_line = pf.nodes.geo.curve_line(end=curve_line_end, start=(0, 0, 0))

    input_index = pf.nodes.geo.input_index()

    capture_attribute = pf.nodes.geo.capture_attribute(
        geometry=curve_line, index=input_index
    )

    resample_curve_count = pf.nodes.geo.resample_curve_count(
        curve=capture_attribute.geometry,
        count=vertices_x,
    )

    input_index_1 = pf.nodes.geo.input_index()

    capture_attribute_1 = pf.nodes.geo.capture_attribute(
        geometry=resample_curve_count, index=input_index_1
    )

    curve_line_1_end = pf.nodes.math.vector_scale(
        vector=(-1.0, 0.0, 0.0), scale=dimension_y
    )
    curve_line_1 = pf.nodes.geo.curve_line(end=curve_line_1_end, start=(0, 0, 0))

    resample_curve_count_1 = pf.nodes.geo.resample_curve_count(
        curve=curve_line_1, count=vertices_y
    )

    capture_attribute_2 = pf.nodes.geo.capture_attribute(
        geometry=resample_curve_count_1, index=input_index_1
    )

    curve_to = pf.nodes.geo.curve_to_mesh(
        curve=capture_attribute_1.geometry,
        profile_curve=capture_attribute_2.geometry,
    )

    flip_faces = pf.nodes.geo.flip_faces(curve_to)

    mesh_result = pf.nodes.func.equal(a=capture_attribute.index, b=0)

    capture_attribute_3 = pf.nodes.geo.capture_attribute(
        domain="EDGE", geometry=flip_faces, result=mesh_result
    )

    side_edge_index = capture_attribute_3.result.astype(dtype=int)
    return GridFromCurvesResult(
        mesh=capture_attribute_3.geometry,
        index_x=capture_attribute_1.index,
        index_y=capture_attribute_2.index,
        side_edge_index=side_edge_index,
    )


def store_metric_box_uvs(mesh: pf.ProcNode) -> pf.ProcNode:
    """Box-projected UVs in world units (1 UV unit = 1 meter), as 'uv_map'.
    Inlines into the calling node_function; corner-domain normals evaluate to the
    exact owning-face normal, which picks the projection axis per face."""
    axis = pf.nodes.math.separate_xyz(
        pf.nodes.math.vector_absolute(pf.nodes.geo.input_normal())
    )
    p = pf.nodes.math.separate_xyz(pf.nodes.geo.input_position())
    uv = pf.nodes.math.combine_xyz(
        axis.x * p.y + axis.y * p.x + axis.z * p.x,
        axis.x * p.z + axis.y * p.z + axis.z * p.y,
    )
    return pf.nodes.geo.store_named_attribute(
        geometry=mesh, name="uv_map", value=uv, domain="CORNER"
    )


class CubeWithVertexIndicesResult(NamedTuple):
    mesh: pf.ProcNode[pf.MeshObject]
    index_x: pf.ProcNode[int]
    index_y: pf.ProcNode[int]
    index_z: pf.ProcNode[int]


@pf.nodes.node_function
def cube_with_vertex_indices(
    size: t.SocketOrVal[pf.Vector] = (1, 1, 1),
    vertices_x: t.SocketOrVal[int] = 2,
    vertices_y: t.SocketOrVal[int] = 2,
    vertices_z: t.SocketOrVal[int] = 2,
) -> CubeWithVertexIndicesResult:
    cube = pf.nodes.geo.mesh_cube(
        size=size,
        vertices_x=vertices_x,
        vertices_y=vertices_y,
        vertices_z=vertices_z,
    )

    index_max = pf.nodes.math.combine_xyz(
        pf.nodes.math.subtract(vertices_x, 1),
        pf.nodes.math.subtract(vertices_y, 1),
        pf.nodes.math.subtract(vertices_z, 1),
    )
    index_float = pf.nodes.math.map_range(
        clamp=False,
        value=pf.nodes.geo.input_position(),
        from_min=pf.nodes.math.vector_scale(vector=size, scale=-0.5),
        from_max=pf.nodes.math.vector_scale(vector=size, scale=0.5),
        to_min=(0, 0, 0),
        to_max=index_max,
    )
    index_xyz = pf.nodes.math.separate_xyz(index_float)

    capture = pf.nodes.geo.capture_attribute(
        geometry=cube.mesh,
        index_x=pf.nodes.math.round(index_xyz.x).astype(dtype=int),
        index_y=pf.nodes.math.round(index_xyz.y).astype(dtype=int),
        index_z=pf.nodes.math.round(index_xyz.z).astype(dtype=int),
    )
    return CubeWithVertexIndicesResult(
        mesh=store_metric_box_uvs(capture.geometry),
        index_x=capture.index_x,
        index_y=capture.index_y,
        index_z=capture.index_z,
    )


@pf.nodes.node_function
def corner_box(
    size: t.SocketOrVal[pf.Vector] = (1, 1, 1),
    loops_x: t.SocketOrVal[int] = 0,
    loops_y: t.SocketOrVal[int] = 0,
    loops_z: t.SocketOrVal[int] = 0,
    support_loop_offset: t.SocketOrVal[pf.Vector] = (0.05, 0.05, 0.05),
) -> CubeWithVertexIndicesResult:
    cube = cube_with_vertex_indices(
        size=size,
        vertices_x=pf.nodes.math.add(loops_x, 4),
        vertices_y=pf.nodes.math.add(loops_y, 4),
        vertices_z=pf.nodes.math.add(loops_z, 4),
    )
    index = pf.nodes.math.combine_xyz(cube.index_x, cube.index_y, cube.index_z)

    half = pf.nodes.math.vector_scale(vector=size, scale=0.5)
    neg_half = pf.nodes.math.vector_scale(vector=size, scale=-0.5)
    ones = (1.0, 1.0, 1.0)

    even_index_max = pf.nodes.math.combine_xyz(
        pf.nodes.math.add(loops_x, 2),
        pf.nodes.math.add(loops_y, 2),
        pf.nodes.math.add(loops_z, 2),
    )
    even = pf.nodes.math.map_range(
        clamp=False,
        value=index,
        from_min=ones,
        from_max=even_index_max,
        to_min=neg_half,
        to_max=half,
    )
    clamped = pf.nodes.math.vector_maximum(
        pf.nodes.math.vector_minimum(even, half - support_loop_offset),
        neg_half + support_loop_offset,
    )

    mask_first = pf.nodes.math.vector_subtract(
        ones, pf.nodes.math.vector_minimum(index, ones)
    )
    index_last = pf.nodes.math.vector_add(even_index_max, ones)
    mask_last = pf.nodes.math.vector_subtract(
        ones,
        pf.nodes.math.vector_minimum(
            pf.nodes.math.vector_subtract(index_last, index), ones
        ),
    )
    corner_correction = pf.nodes.math.vector_multiply(
        mask_last - mask_first, support_loop_offset
    )

    repositioned = pf.nodes.geo.set_position(
        geometry=cube.mesh, position=clamped + corner_correction
    )
    return CubeWithVertexIndicesResult(
        mesh=store_metric_box_uvs(repositioned),
        index_x=cube.index_x,
        index_y=cube.index_y,
        index_z=cube.index_z,
    )


class InsetFacesResult(NamedTuple):
    geometry: pf.ProcNode[pf.MeshObject]
    center: pf.ProcNode[bool]
    side: pf.ProcNode[bool]


@pf.nodes.node_function
def inset_faces(
    mesh: pf.ProcNode[pf.MeshObject],
    selection: t.SocketOrVal[bool],
    distance: t.SocketOrVal[float],
) -> InsetFacesResult:
    # Code generated by procfunc v0.12.0

    extrude = pf.nodes.geo.extrude_mesh(
        mesh=mesh, selection=selection, offset_scale=0.0
    )

    input_position = pf.nodes.geo.input_position()

    capture_attribute = pf.nodes.geo.capture_attribute(
        domain="FACE",
        geometry=extrude.mesh,
        position=input_position,
    )

    geometry_offset_vector = pf.nodes.math.vector_normalize(
        capture_attribute.position - input_position
    )
    geometry_offset = pf.nodes.math.vector_scale(
        vector=geometry_offset_vector, scale=distance
    )

    set_position = pf.nodes.geo.set_position(
        geometry=capture_attribute.geometry,
        selection=extrude.top,
        offset=geometry_offset,
    )
    return InsetFacesResult(
        geometry=set_position,
        center=extrude.top,
        side=extrude.side,
    )


@pf.nodes.node_function
def grid_from_corners(
    point_1: t.SocketOrVal[pf.Vector],
    point_2: t.SocketOrVal[pf.Vector],
    vertices_x: t.SocketOrVal[int],
    vertices_y: t.SocketOrVal[int],
) -> pf.ProcNode:
    # Code generated by procfunc v0.11.0

    grid = pf.nodes.geo.mesh_grid(vertices_x=vertices_x, vertices_y=vertices_y)

    input_position = pf.nodes.geo.input_position()

    result_0_position = pf.nodes.math.map_range(
        clamp=False,
        value=input_position,
        from_min=(-0.5, -0.5, -0.5),
        from_max=(0.5, 0.5, 0.5),
        to_min=point_1,
        to_max=point_2,
    )

    set_position = pf.nodes.geo.set_position(
        geometry=grid.mesh, position=result_0_position
    )
    return set_position


@pf.nodes.node_function
def crease_sharp(
    mesh: pf.ProcNode[pf.MeshObject],
    threshold_degrees: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    # Code generated by procfunc v0.12.0

    input_edge_angle = pf.nodes.geo.input_mesh_edge_angle()

    radians = pf.nodes.math.deg_to_rad(threshold_degrees)
    mask = pf.nodes.func.greater_than(a=input_edge_angle.unsigned_angle, b=radians)
    store_named_attribute = pf.nodes.geo.store_named_attribute(
        domain="EDGE",
        geometry=mesh,
        name="crease_edge",
        value=mask.astype(dtype=float),
    )
    return store_named_attribute
