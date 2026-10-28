# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from typing import NamedTuple

import bpy
import numpy as np
import procfunc as pf
from procfunc.nodes import types as t
from procfunc.nodes.util.bpy_node_info import NodeDataType

from infinigen2.util.curve import curve_to_mesh_with_uv

__all__ = [
    "ExtrudeSeamlessResult",
    "LoftingResult",
    "WallCutoutResult",
    "box",
    "box_with_support_loops",
    "center_footprint",
    "crease_all_edges",
    "crease_by_angle",
    "crease_sharp",
    "evaluated_world_bbox",
    "extrude_mesh_seamless_uvs",
    "extrude_mesh_seamless_uvs_along",
    "face_selection_boundary_curve",
    "fill_between_curves",
    "grid_from_corners",
    "lofting",
    "mesh_from_corners",
    "mesh_from_quads",
    "metric_box_uv",
    "metric_cylinder_uv",
    "quad_cap",
    "quad_cylinder",
    "quad_disc",
    "refine_lattice_cage",
    "reversed_faces",
    "split_gaps",
    "uv_winding_sign",
    "wall_cutout_split",
]


def evaluated_world_bbox(obj: pf.MeshObject) -> tuple[np.ndarray, np.ndarray]:
    item = obj.item()
    evaluated = item.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    count = len(mesh.vertices)
    matrix = np.asarray(item.matrix_world)
    if count == 0:
        local = np.asarray(item.bound_box)
    else:
        local = np.empty(count * 3)
        mesh.vertices.foreach_get("co", local)
        local = local.reshape(-1, 3)
    evaluated.to_mesh_clear()
    world = (matrix[:3, :3] @ local.T).T + matrix[:3, 3]
    return world.min(axis=0), world.max(axis=0)


def quad_cap(
    profile: pf.ProcNode[pf.CurveObject], insets: int = 1, scale: float = 0.55
) -> pf.ProcNode[pf.MeshObject]:
    """Fill a closed profile with concentric quad rings around a centre face.

    A cap filled as one bare ngon shares its vertices with the rim, so a rim
    crease drags on the whole face; a single inset ring isolates the two and the
    subdivided cap comes out flat. Ring vertices inherit the profile's angular
    spacing, which is what keeps the subdivided silhouette round.
    """
    cap = pf.nodes.geo.fill_curve(profile, mode="NGONS")
    for _ in range(insets):
        extruded = pf.nodes.geo.extrude_mesh(
            cap, offset_scale=0.0, individual=False, mode="FACES"
        )
        cap = pf.nodes.geo.scale_elements(
            extruded.mesh, scale=scale, selection=extruded.top
        )
    position = pf.nodes.geo.input_position()
    uv = pf.nodes.math.combine_xyz(x=position.x, y=position.y)
    return pf.nodes.geo.store_named_attribute(
        geometry=cap, name="UVMap", value=uv, domain="CORNER", data_type="FLOAT2"
    )


def quad_disc(
    radius: t.SocketOrVal[float] = 1.0,
    resolution: t.SocketOrVal[int] = 20,
    insets: int = 1,
) -> pf.ProcNode[pf.MeshObject]:
    """Disc of concentric quad rings, uniformly spaced around the rim."""
    circle = pf.nodes.geo.curve_circle(radius=radius, resolution=resolution)
    return quad_cap(circle, insets=insets)


def quad_cylinder(
    radius: t.SocketOrVal[float] = 1.0,
    depth: t.SocketOrVal[float] = 1.0,
    resolution: t.SocketOrVal[int] = 20,
    insets: int = 1,
) -> pf.ProcNode[pf.MeshObject]:
    """Cylinder capped by quad rings, centred on the origin like mesh_cylinder.

    mesh_cylinder caps each end with a bare ngon whose vertices are the creased
    rim itself, which tilts the subdivided face; this sweeps the same circle for
    the wall and caps it with quad_cap instead.
    """
    profile = pf.nodes.geo.curve_circle(radius=radius, resolution=resolution)
    line = pf.nodes.geo.curve_line(
        start=(0.0, 0.0, 0.0), end=pf.nodes.math.combine_xyz(z=depth)
    )
    walls = curve_to_mesh_with_uv(curve=line, profile=profile, fill_caps=False).mesh
    cap = quad_cap(profile, insets=insets)
    cap_start = pf.nodes.geo.flip_faces(cap)
    cap_end = pf.nodes.geo.transform(
        geometry=cap, translation=pf.nodes.math.combine_xyz(z=depth)
    )
    solid = pf.nodes.geo.join_geometry([walls, cap_start, cap_end])
    centred = pf.nodes.geo.transform(
        geometry=solid, translation=pf.nodes.math.combine_xyz(z=depth * -0.5)
    )
    return pf.nodes.geo.merge_by_distance(centred, distance=1e-6)


def metric_box_uv(geometry: pf.ProcNode) -> pf.ProcNode:
    """Store metric box-projection UVs computed from the current positions.

    Per face, picks the two in-plane position axes (U/V in meters) and puts V on
    the axis with the larger board extent so woodgrain runs along the board's
    long dimension, then writes them as a CORNER "UVMap".
    """
    bbox = pf.nodes.geo.bound_box(geometry)
    size = bbox.max - bbox.min
    position = pf.nodes.geo.input_position()
    abs_normal = pf.nodes.math.vector_absolute(pf.nodes.geo.input_normal())
    uv_facing_x = pf.nodes.func.switch(
        switch=size.z > size.y,
        a=pf.nodes.math.combine_xyz(x=position.z, y=position.y),
        b=pf.nodes.math.combine_xyz(x=position.y, y=position.z),
    )
    uv_facing_y = pf.nodes.func.switch(
        switch=size.z > size.x,
        a=pf.nodes.math.combine_xyz(x=position.z, y=position.x),
        b=pf.nodes.math.combine_xyz(x=position.x, y=position.z),
    )
    uv_facing_z = pf.nodes.func.switch(
        switch=size.y > size.x,
        a=pf.nodes.math.combine_xyz(x=position.y, y=position.x),
        b=pf.nodes.math.combine_xyz(x=position.x, y=position.y),
    )
    uv_side = pf.nodes.func.switch(
        switch=abs_normal.y > 0.5, a=uv_facing_x, b=uv_facing_y
    )
    uv = pf.nodes.func.switch(switch=abs_normal.z > 0.5, a=uv_side, b=uv_facing_z)
    return pf.nodes.geo.store_named_attribute(
        geometry=geometry,
        name="UVMap",
        value=uv,
        domain="CORNER",
        data_type="FLOAT2",
    )


@pf.tracer.primitive(mutates=["obj"])
def metric_cylinder_uv(obj: pf.MeshObject) -> None:
    """Project sides at the widest circumference, with metric Z and planar caps."""
    positions = pf.ops.attr.vertex_positions(obj)
    radius = np.linalg.norm(positions[:, :2], axis=1).max()
    pf.ops.uv.cylinder_project(obj, direction="ALIGN_TO_OBJECT", correct_aspect=False)
    uv = pf.ops.attr.uv_coords(obj)
    mesh = obj.item().data
    corner_positions = positions[[loop.vertex_index for loop in mesh.loops]]
    uv[:, 0] *= 2 * np.pi * radius
    uv[:, 1] = corner_positions[:, 2]
    caps = np.abs(pf.ops.attr.polygon_normals(obj)[:, 2]) > 0.9999
    cap_corners = np.repeat(caps, [face.loop_total for face in mesh.polygons])
    uv[cap_corners] = corner_positions[cap_corners, :2]
    pf.ops.attr.write_uv_coords(obj, uv)


class _CylinderSideResult(NamedTuple):
    geometry: pf.ProcNode[pf.MeshObject]
    top: pf.ProcNode[pf.MeshObject]
    side: pf.ProcNode[pf.MeshObject]
    bottom: pf.ProcNode[pf.MeshObject]


@pf.nodes.node_function
def _cylinder_side(
    u_resolution: t.SocketOrVal[int] = 32, v_resolution: t.SocketOrVal[int] = 0
) -> _CylinderSideResult:
    cylinder_side_segments = v_resolution.astype(dtype=float) - 1.0
    cylinder = pf.nodes.geo.mesh_cylinder(
        vertices=u_resolution, side_segments=cylinder_side_segments.astype(dtype=int)
    )
    store_named_attribute = pf.nodes.geo.store_named_attribute(
        geometry=cylinder.mesh,
        name="UVMap",
        value=cylinder.uv_map,
        domain="CORNER",
        data_type="FLOAT2",
    )
    return _CylinderSideResult(
        store_named_attribute, cylinder.top, cylinder.side, cylinder.bottom
    )


@pf.nodes.node_function
def _flip_index(
    v_resolution: t.SocketOrVal[int] = 0, u_resolution: t.SocketOrVal[int] = 0
) -> pf.ProcNode[float]:
    input_index = pf.nodes.geo.input_index()
    result_0_a_0 = input_index.astype(dtype=float) % v_resolution.astype(dtype=float)
    result_0_b = pf.nodes.math.floor(
        input_index.astype(dtype=float) / v_resolution.astype(dtype=float)
    )
    add = result_0_a_0 * u_resolution.astype(dtype=float) + result_0_b
    return add


class LoftingResult(NamedTuple):
    geometry: pf.ProcNode[pf.CurveObject]
    top: pf.ProcNode[bool]
    side: pf.ProcNode[bool]
    bottom: pf.ProcNode[bool]


@pf.nodes.node_function
def lofting(
    profile_curves: t.ProcNode[pf.CurveObject],
    u_resolution: t.SocketOrVal[int] = 32,
    v_resolution: t.SocketOrVal[int] = 32,
    use_nurb: t.SocketOrVal[bool] = False,
) -> LoftingResult:
    """Loft a tube surface through a stack of cross-section profile curves.

    profile_curves is a single geometry holding N separate profile splines, one
    per cross-section ordered along the loft (spline 0 = bottom rim ... spline
    N-1 = top rim). Each profile is resampled to u_resolution points and the N
    profiles are interpolated (Catmull-Rom, or NURBS if use_nurb) into
    v_resolution rows, yielding a u_resolution x v_resolution cylinder-topology
    mesh. Returns the lofted geometry plus the cylinder's top/side/bottom masks.
    """
    cylinder_side_result = _cylinder_side(
        u_resolution=u_resolution, v_resolution=v_resolution
    )
    input_index = pf.nodes.geo.input_index()
    field_on_domain = pf.nodes.geo.field_on_domain(
        value=input_index, domain="CURVE", data_type=NodeDataType.INT
    )
    curve_line = pf.nodes.geo.curve_line(start=(0.0, 0.0, 0.0), end=(0.0, 0.0, 1.0))
    attribute_domain_size = pf.nodes.geo.attribute_domain_size(
        geometry=profile_curves, component="CURVE"
    )
    resample_curve_count = pf.nodes.geo.resample_curve_count(
        curve=curve_line, count=attribute_domain_size.spline_count
    )
    instance_on_points = pf.nodes.geo.instance_on_points(
        points=profile_curves,
        instance=resample_curve_count,
        selection=field_on_domain == 0,
    )
    realize_instances = pf.nodes.geo.realize_instances(instance_on_points)
    input_position = pf.nodes.geo.input_position()
    flip_index_result = _flip_index(
        v_resolution=attribute_domain_size.spline_count, u_resolution=u_resolution
    )
    sample_index = pf.nodes.geo.sample_index(
        geometry=profile_curves,
        index=flip_index_result.astype(dtype=int),
        value=input_position,
        data_type=NodeDataType.FLOAT_VECTOR,
    )
    set_position_1 = pf.nodes.geo.set_position(
        geometry=realize_instances, position=sample_index, offset=(0.0, 0.0, 0.0)
    )
    curve_spline_type = pf.nodes.geo.curve_spline_type(
        curve=set_position_1, spline_type="CATMULL_ROM"
    )
    curve_spline_type_1 = pf.nodes.geo.curve_spline_type(
        curve=set_position_1, spline_type="NURBS"
    )
    resample_curve = pf.nodes.func.switch(
        switch=use_nurb,
        a=curve_spline_type,
        b=curve_spline_type_1,
        data_type=NodeDataType.GEOMETRY,
    )
    resample_curve_count_1 = pf.nodes.geo.resample_curve_count(
        curve=resample_curve, count=v_resolution
    )
    input_position_1 = pf.nodes.geo.input_position()
    flip_index_result_1 = _flip_index(
        v_resolution=u_resolution, u_resolution=v_resolution
    )
    sample_index_1 = pf.nodes.geo.sample_index(
        geometry=resample_curve_count_1,
        index=flip_index_result_1.astype(dtype=int),
        value=input_position_1,
        data_type=NodeDataType.FLOAT_VECTOR,
    )
    set_position = pf.nodes.geo.set_position(
        geometry=cylinder_side_result.geometry,
        position=sample_index_1,
        offset=(0.0, 0.0, 0.0),
    )
    return LoftingResult(
        set_position,
        cylinder_side_result.top,
        cylinder_side_result.side,
        cylinder_side_result.bottom,
    )


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
    """Extrude faces along their normals and continue source UVs seamlessly.

    Each side corner gets uv0 plus a perpendicular offset proportional to its
    extrusion depth. The boundary edge driving that offset is found per corner via
    its own next-corner edge (the one with both ends on the depth-0 ring), then
    broadcast across the side face with accumulate_field.
    """

    return extrude_mesh_seamless_uvs_along(
        mesh=mesh,
        selection=selection,
        offset_scale=offset_scale,
        offset=pf.nodes.geo.input_normal(),
        uv_winding_sign=uv_winding_sign,
    )


@pf.nodes.node_function
def extrude_mesh_seamless_uvs_along(
    mesh: pf.ProcNode[pf.MeshObject],
    selection: t.SocketOrVal[bool],
    offset_scale: t.SocketOrVal[float],
    offset: t.SocketOrVal[pf.Vector],
    uv_winding_sign: t.SocketOrVal[float] = 1.0,
) -> ExtrudeSeamlessResult:
    """Continue source UVs while extruding along an explicit offset."""
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
        offset=offset,
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


def center_footprint(obj: pf.MeshObject) -> None:
    """Shift geometry so its bounding box is centered on x/y, leaving z alone.

    Storage carcasses are built into the positive octant while tables are built
    centered on their footprint; placement code sets a location expecting the
    latter, so a carcass used as a table has to be recentered first.
    """
    bbox = pf.ops.attr.bbox_min_max(obj, global_coords=False)
    lo, hi = bbox[0], bbox[1]
    pf.ops.mesh.move(obj, value=(-(lo[0] + hi[0]) * 0.5, -(lo[1] + hi[1]) * 0.5, 0.0))


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


class _ExtrudeChamferResult(NamedTuple):
    mesh: pf.ProcNode[pf.MeshObject]
    top: pf.ProcNode[bool]  # the deep (full-depth) face cap
    chamfer: pf.ProcNode[bool]  # the angled lip ring at the mouth
    side: pf.ProcNode[bool]  # the deep straight sill walls


@pf.nodes.node_function
def _extrude_inwards_with_chamfer(
    mesh: pf.ProcNode[pf.MeshObject],
    selection: t.SocketOrVal[bool],
    offset_scale: t.SocketOrVal[float],
    chamfer: t.SocketOrVal[float] = 0.006,
    uv_winding_sign: t.SocketOrVal[float] = 1.0,
) -> _ExtrudeChamferResult:
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
    return _ExtrudeChamferResult(
        mesh=deep.mesh, top=deep.top, chamfer=lip.side, side=deep.side
    )


@pf.nodes.node_function
def face_selection_boundary_curve(
    mesh: pf.ProcNode[pf.MeshObject],
    selection: t.SocketOrVal[bool],
) -> pf.ProcNode[pf.CurveObject]:
    """Extract the edge loops separating selected from unselected faces as curves.

    Curve normals are set to the inward surface direction (-mesh normal), matching
    the skirting sweep convention: a profile authored in -X/-Y swept along the
    result protrudes out the front of the surface.
    """
    cap_p = pf.nodes.geo.capture_attribute(
        geometry=mesh, domain="POINT", inward=-pf.nodes.geo.input_normal()
    )
    # capture as float on FACE so the EDGE read averages adjacent faces (bools OR)
    cap_f = pf.nodes.geo.capture_attribute(
        geometry=cap_p.geometry, domain="FACE", sel=selection.astype(dtype=float)
    )
    cap_e = pf.nodes.geo.capture_attribute(
        geometry=cap_f.geometry, domain="EDGE", sel=cap_f.sel
    )
    boundary = pf.nodes.func.boolean_and(a=cap_e.sel > 0.01, b=cap_e.sel < 0.99)
    curve = pf.nodes.geo.mesh_to_curve(cap_e.geometry, selection=boundary)
    return pf.nodes.geo.set_curve_normal(curve, normal=cap_p.inward, mode="FREE")


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

    lip = _extrude_inwards_with_chamfer(
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


class _GridFromCurvesResult(NamedTuple):
    mesh: pf.ProcNode[pf.MeshObject]
    index_x: pf.ProcNode[int]
    index_y: pf.ProcNode[int]
    side_edge_index: pf.ProcNode[int]


@pf.nodes.node_function
def _grid_from_curves(
    dimension_x: t.SocketOrVal[float],
    dimension_y: t.SocketOrVal[float],
    vertices_x: t.SocketOrVal[int],
    vertices_y: t.SocketOrVal[int],
) -> _GridFromCurvesResult:
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
    return _GridFromCurvesResult(
        mesh=capture_attribute_3.geometry,
        index_x=capture_attribute_1.index,
        index_y=capture_attribute_2.index,
        side_edge_index=side_edge_index,
    )


def _store_metric_box_uvs(mesh: pf.ProcNode) -> pf.ProcNode:
    """Box-projected UVs in world units (1 UV unit = 1 meter), as 'UVMap'.
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
        geometry=mesh,
        name="UVMap",
        value=uv,
        domain="CORNER",
        data_type="FLOAT2",
    )


def box(
    size: t.SocketOrVal[pf.Vector] = (1, 1, 1),
    location: t.SocketOrVal[pf.Vector] = (0, 0, 0),
    anchor: t.SocketOrVal[pf.Vector] = (0.5, 0.5, 0.5),
    vertices_x: t.SocketOrVal[int] = 2,
    vertices_y: t.SocketOrVal[int] = 2,
    vertices_z: t.SocketOrVal[int] = 2,
    crease: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[pf.MeshObject]:
    """Create a box with ``location`` at its normalized ``anchor`` point."""
    cube = pf.nodes.geo.mesh_cube(
        size=size,
        vertices_x=vertices_x,
        vertices_y=vertices_y,
        vertices_z=vertices_z,
    )
    center_factor = pf.nodes.math.map_range(
        value=anchor,
        from_min=(0.0, 0.0, 0.0),
        from_max=(1.0, 1.0, 1.0),
        to_min=(0.5, 0.5, 0.5),
        to_max=(-0.5, -0.5, -0.5),
    )
    center = pf.nodes.math.vector_multiply_add(a=center_factor, b=size, addend=location)
    placed = pf.nodes.geo.transform(geometry=cube.mesh, translation=center)
    creased = pf.nodes.geo.store_named_attribute(
        geometry=placed, domain="EDGE", name="crease_edge", value=crease
    )
    return _store_metric_box_uvs(creased)


@pf.nodes.node_function
def box_with_support_loops(
    size: t.SocketOrVal[pf.Vector] = (1, 1, 1),
    location: t.SocketOrVal[pf.Vector] = (0, 0, 0),
    anchor: t.SocketOrVal[pf.Vector] = (0.5, 0.5, 0.5),
    vertices_x: t.SocketOrVal[int] = 4,
    vertices_y: t.SocketOrVal[int] = 4,
    vertices_z: t.SocketOrVal[int] = 4,
    support_loop_offset: t.SocketOrVal[pf.Vector] = (0.05, 0.05, 0.05),
    crease: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[pf.MeshObject]:
    """Create a support-loop box with the same placement contract as ``box``."""
    cube = pf.nodes.geo.mesh_cube(
        size=size,
        vertices_x=vertices_x,
        vertices_y=vertices_y,
        vertices_z=vertices_z,
    )

    vertices_x_float = vertices_x.astype(dtype=float)
    vertices_y_float = vertices_y.astype(dtype=float)
    vertices_z_float = vertices_z.astype(dtype=float)
    index_max = pf.nodes.math.combine_xyz(
        pf.nodes.math.subtract(vertices_x_float, 1.0),
        pf.nodes.math.subtract(vertices_y_float, 1.0),
        pf.nodes.math.subtract(vertices_z_float, 1.0),
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
        index=pf.nodes.math.combine_xyz(
            pf.nodes.math.round(index_xyz.x),
            pf.nodes.math.round(index_xyz.y),
            pf.nodes.math.round(index_xyz.z),
        ),
    )

    index = capture.index
    half = pf.nodes.math.vector_scale(vector=size, scale=0.5)
    neg_half = pf.nodes.math.vector_scale(vector=size, scale=-0.5)
    ones = (1.0, 1.0, 1.0)
    interior_index_max = pf.nodes.math.combine_xyz(
        pf.nodes.math.subtract(vertices_x_float, 2.0),
        pf.nodes.math.subtract(vertices_y_float, 2.0),
        pf.nodes.math.subtract(vertices_z_float, 2.0),
    )
    # spread between support loops; full-size spread then clamp stacks loops on them
    even = pf.nodes.math.map_range(
        clamp=False,
        value=index,
        from_min=ones,
        from_max=interior_index_max,
        to_min=neg_half + support_loop_offset,
        to_max=half - support_loop_offset,
    )
    clamped = pf.nodes.math.vector_maximum(
        pf.nodes.math.vector_minimum(even, half - support_loop_offset),
        neg_half + support_loop_offset,
    )
    mask_first = pf.nodes.math.vector_subtract(
        ones, pf.nodes.math.vector_minimum(index, ones)
    )
    index_last = pf.nodes.math.vector_add(interior_index_max, ones)
    mask_last = pf.nodes.math.vector_subtract(
        ones,
        pf.nodes.math.vector_minimum(
            pf.nodes.math.vector_subtract(index_last, index), ones
        ),
    )
    corrected = clamped + pf.nodes.math.vector_multiply(
        mask_last - mask_first, support_loop_offset
    )
    position = pf.nodes.math.separate_xyz(pf.nodes.geo.input_position())
    corrected_xyz = pf.nodes.math.separate_xyz(corrected)
    offset = pf.nodes.math.separate_xyz(support_loop_offset)
    supported_position = pf.nodes.math.combine_xyz(
        x=pf.nodes.func.switch(switch=offset.x > 0.0, a=position.x, b=corrected_xyz.x),
        y=pf.nodes.func.switch(switch=offset.y > 0.0, a=position.y, b=corrected_xyz.y),
        z=pf.nodes.func.switch(switch=offset.z > 0.0, a=position.z, b=corrected_xyz.z),
    )
    supported = pf.nodes.geo.set_position(
        geometry=capture.geometry, position=supported_position
    )
    center_factor = pf.nodes.math.map_range(
        value=anchor,
        from_min=(0.0, 0.0, 0.0),
        from_max=(1.0, 1.0, 1.0),
        to_min=(0.5, 0.5, 0.5),
        to_max=(-0.5, -0.5, -0.5),
    )
    center = pf.nodes.math.vector_multiply_add(a=center_factor, b=size, addend=location)
    placed = pf.nodes.geo.transform(
        geometry=supported,
        translation=center,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    creased = pf.nodes.geo.store_named_attribute(
        geometry=placed, domain="EDGE", name="crease_edge", value=crease
    )
    return _store_metric_box_uvs(creased)


@pf.nodes.node_function
def fill_between_curves(
    curve_left: t.SocketOrVal[pf.CurveObject],
    curve_right: t.SocketOrVal[pf.CurveObject],
    n_points: t.SocketOrVal[int] = 8,
    n_rows: t.SocketOrVal[int] = 2,
) -> pf.ProcNode[pf.MeshObject]:
    """Ruled surface bridging two curves: resample both to n_points, then lerp
    an n_rows x n_points grid between the sample positions. n_rows > 2 keeps
    faces small and near-planar so downstream bevel/warp modifiers behave.

    Stores metric UVs with U along the curves and V between them."""
    grid = pf.nodes.geo.mesh_grid(vertices_x=n_rows, vertices_y=n_points)
    left = pf.nodes.geo.resample_curve_count(curve=curve_left, count=n_points)
    right = pf.nodes.geo.resample_curve_count(curve=curve_right, count=n_points)
    left = pf.nodes.geo.capture_attribute(
        geometry=left, length=pf.nodes.geo.spline_parameter().length
    )
    index = pf.nodes.geo.input_index()
    index_f = index.astype(dtype=float)
    n_points_f = n_points.astype(dtype=float)
    row = pf.nodes.math.floor(index_f / n_points_f)
    along = index_f - row * n_points_f
    factor = row / (n_rows.astype(dtype=float) - 1.0)
    position = pf.nodes.geo.input_position()
    left_pos = pf.nodes.geo.sample_index(
        geometry=left.geometry,
        index=along.astype(dtype=int),
        value=position,
        data_type=NodeDataType.FLOAT_VECTOR,
    )
    right_pos = pf.nodes.geo.sample_index(
        geometry=right,
        index=along.astype(dtype=int),
        value=position,
        data_type=NodeDataType.FLOAT_VECTOR,
    )
    lerped = left_pos + (right_pos - left_pos) * factor
    u = pf.nodes.geo.sample_index(
        geometry=left.geometry,
        index=along.astype(dtype=int),
        value=left.length,
    )
    v = factor * pf.nodes.math.vector_length(right_pos - left_pos)
    uv = pf.nodes.geo.capture_attribute(
        geometry=grid.mesh,
        uv=pf.nodes.math.combine_xyz(x=u, y=v),
    )
    positioned = pf.nodes.geo.set_position(
        geometry=uv.geometry, position=lerped, offset=(0.0, 0.0, 0.0)
    )
    return pf.nodes.geo.store_named_attribute(
        geometry=positioned,
        name="UVMap",
        value=uv.uv,
        domain="CORNER",
        data_type="FLOAT2",
    )


class _InsetFacesResult(NamedTuple):
    geometry: pf.ProcNode[pf.MeshObject]
    center: pf.ProcNode[bool]
    side: pf.ProcNode[bool]


@pf.nodes.node_function
def _inset_faces(
    mesh: pf.ProcNode[pf.MeshObject],
    selection: t.SocketOrVal[bool],
    distance: t.SocketOrVal[float],
) -> _InsetFacesResult:
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
    return _InsetFacesResult(
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

    position = pf.nodes.math.map_range(
        clamp=False,
        value=input_position,
        from_min=(-0.5, -0.5, -0.5),
        from_max=(0.5, 0.5, 0.5),
        to_min=point_1,
        to_max=point_2,
    )

    set_position = pf.nodes.geo.set_position(geometry=grid.mesh, position=position)
    return set_position


def split_gaps(lines: np.ndarray, max_cell: float) -> np.ndarray:
    """Sorted `lines` plus evenly spaced extras so no gap exceeds `max_cell`."""
    gaps = np.diff(lines)
    counts = np.maximum(1, np.ceil(gaps / max_cell - 1e-6)).astype(int)
    gap_of_line = np.repeat(np.arange(len(gaps)), counts)
    step = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    inner = lines[gap_of_line] + step * (gaps / counts)[gap_of_line]
    return np.concatenate([inner, lines[-1:]])


def mesh_from_corners(
    corners: np.ndarray, sizes: np.ndarray, uvs: np.ndarray
) -> pf.MeshObject:
    """Shared-vertex mesh from flat (N, 3) face corners, per-face corner counts and (N, 2) UVs."""
    _, first, inverse = np.unique(
        np.round(corners, 6), axis=0, return_index=True, return_inverse=True
    )
    data = bpy.data.meshes.new("mesh_from_corners")
    data.vertices.add(len(first))
    data.vertices.foreach_set("co", corners[first].astype(np.float32).ravel())
    data.loops.add(len(corners))
    data.loops.foreach_set("vertex_index", inverse.reshape(-1).astype(np.int32))
    data.polygons.add(len(sizes))
    starts = np.cumsum(sizes) - sizes
    data.polygons.foreach_set("loop_start", starts.astype(np.int32))
    data.update(calc_edges=True)
    obj = bpy.data.objects.new("mesh_from_corners", data)
    bpy.context.scene.collection.objects.link(obj)
    mesh = pf.MeshObject(obj)
    pf.ops.attr.write_attribute(mesh, uvs, "UVMap", domain="CORNER")
    return mesh


def mesh_from_quads(corners: np.ndarray, uvs: np.ndarray) -> pf.MeshObject:
    sizes = np.full(len(corners), 4)
    return mesh_from_corners(corners.reshape(-1, 3), sizes, uvs.reshape(-1, 2))


def reversed_faces(sizes: np.ndarray) -> np.ndarray:
    """Corner permutation that reverses the winding of every face in a flat layout."""
    starts = np.repeat(np.cumsum(sizes) - sizes, sizes)
    offset = np.arange(sizes.sum()) - starts
    return starts + np.repeat(sizes, sizes) - 1 - offset


def _quad_uv_area(quad_uv: np.ndarray) -> np.ndarray:
    u = quad_uv[..., 0]
    v = quad_uv[..., 1]
    return 0.5 * np.sum(
        u * np.roll(v, -1, axis=-1) - np.roll(u, -1, axis=-1) * v, axis=-1
    )


def _cells_per_quad(
    lines: np.ndarray, low: np.ndarray, high: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    first = np.searchsorted(lines, low - 1e-5)
    last = np.searchsorted(lines, high + 1e-5) - 1
    return first, last - first


def refine_lattice_cage(obj: pf.MeshObject, max_cell: float) -> pf.MeshObject:
    """Split oversized cells of a quad cage whose faces are axis-aligned UV rectangles.

    New lines go only inside UV gaps wider than `max_cell` metres, so small faces
    (chamfers, reveals) stay untouched and no T-junctions appear. Returns `obj`
    unchanged if any face is not a UV rectangle or nothing needs splitting.
    """
    if np.any(pf.ops.attr.loop_totals(obj) != 4):
        return obj
    loops = pf.ops.attr.loop_starts(obj)[:, None] + np.arange(4)
    quad_uv = pf.ops.attr.uv_coords(obj)[loops]
    quad_xyz = pf.ops.attr.vertex_positions(obj)[
        pf.ops.attr.loop_vertex_indices(obj)[loops]
    ]
    low = quad_uv.min(axis=1)
    high = quad_uv.max(axis=1)
    area = _quad_uv_area(quad_uv)
    if not np.allclose(np.prod(high - low, axis=1), np.abs(area), atol=1e-6):
        return obj

    u_lines = np.unique(np.round(quad_uv[..., 0], 5))
    v_lines = np.unique(np.round(quad_uv[..., 1], 5))
    us = split_gaps(u_lines, max_cell)
    vs = split_gaps(v_lines, max_cell)
    if len(us) == len(u_lines) and len(vs) == len(v_lines):
        return obj

    # affine uv -> xyz per quad, exact for flat axis-aligned UV rectangles
    design = np.concatenate([quad_uv, np.ones((len(quad_uv), 4, 1))], axis=2)
    uv_to_xyz = np.linalg.pinv(design) @ quad_xyz

    u_first, u_count = _cells_per_quad(us, low[:, 0], high[:, 0])
    v_first, v_count = _cells_per_quad(vs, low[:, 1], high[:, 1])
    counts = u_count * v_count
    quad = np.repeat(np.arange(len(counts)), counts)
    step = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    iu = u_first[quad] + step // v_count[quad]
    iv = v_first[quad] + step % v_count[quad]
    corner_u = np.stack([us[iu], us[iu + 1], us[iu + 1], us[iu]], axis=1)
    corner_v = np.stack([vs[iv], vs[iv], vs[iv + 1], vs[iv + 1]], axis=1)
    cell_uv = np.stack([corner_u, corner_v], axis=2)
    flipped = area[quad] < 0
    cell_uv[flipped] = cell_uv[flipped, ::-1]

    cell_design = np.concatenate([cell_uv, np.ones((len(cell_uv), 4, 1))], axis=2)
    cell_xyz = cell_design @ uv_to_xyz[quad]
    return mesh_from_quads(cell_xyz, cell_uv)


def crease_all_edges(obj: pf.MeshObject) -> None:
    pf.ops.attr.write_attribute(obj, 1.0, "crease_edge", domain="EDGE", overwrite=True)


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


@pf.nodes.node_function
def crease_by_angle(
    mesh: pf.ProcNode[pf.MeshObject],
    threshold_degrees: t.SocketOrVal[float],
    softness_degrees: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    # Soft crease_sharp: crease ramps 0..1 over threshold +/- softness_degrees.
    angle = pf.nodes.geo.input_mesh_edge_angle().unsigned_angle
    lo = pf.nodes.math.deg_to_rad(threshold_degrees - softness_degrees)
    hi = pf.nodes.math.deg_to_rad(threshold_degrees + softness_degrees)
    crease = pf.nodes.math.map_range(
        value=angle,
        from_min=lo,
        from_max=hi,
        to_min=0.0,
        to_max=1.0,
        clamp=True,
        interpolation_type="SMOOTHSTEP",
    )
    return pf.nodes.geo.store_named_attribute(
        domain="EDGE",
        geometry=mesh,
        name="crease_edge",
        value=crease,
    )
