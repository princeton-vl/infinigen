# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from typing import NamedTuple

import numpy as np
import procfunc as pf
from mathutils import Euler
from procfunc.nodes import types as t

from infinigen2.curves.skirting_board_profile import trim_profile_rand
from infinigen2.objects import storage, wall_art, window
from infinigen2.objects.door import (
    door_composite_rand,
    door_double_rand,
    door_glass_from_profile_rand,
)
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import keep_non_colliding
from infinigen2.scenes.placement.distribute import (
    duplicates,
    propagate_modifiers_to_instances,
)
from infinigen2.scenes.room.wall_base import (
    ROOM_SUBSURF_LEVELS,
    WallResult,
    extrude_for_thickness,
    finish_wall_plane,
    fit_grid_margins,
    plain_wall,
    plane_to_posed_canonical_mesh,
    resolve_wall_inputs,
    seat_upright_cabinet,
    upright_cabinet_footprint,
    wall_plain_rand,
    wall_storage_width_rand,
    wall_uv_dimensions,
)
from infinigen2.shaders.functionality_lists import skirt_material_rand
from infinigen2.util import mesh as mesh_util
from infinigen2.util.curve import curve_to_mesh_with_uv
from infinigen2.uv_surface import grid_placement

__all__ = [
    "CutoutResult",
    "arrange_window_portals",
    "cutout_spaced_instances",
    "cutout_trim_rand",
    "wall_cubby_rand",
    "wall_doors_rand",
    "wall_full_window_rand",
    "wall_painting_grid_rand",
    "wall_storage_shelf_rand",
    "wall_windows_rand",
    "window_spaced_rand",
]

logger = logging.getLogger(__name__)


def _subdivide_rounded_cutout(
    mesh: pf.ProcNode[pf.MeshObject],
    threshold_degrees: float,
    sharp_edges: t.SocketOrVal[bool] = False,
    levels: int = ROOM_SUBSURF_LEVELS,
) -> pf.MeshObject:
    """Crease folds and corners only, so arch edge chains subdivide into curves."""
    creased = mesh_util.crease_sharp(mesh, threshold_degrees=threshold_degrees)
    is_boundary = pf.nodes.func.equal(a=pf.nodes.geo.input_mesh_edge_neighbors(), b=1)
    creased = pf.nodes.geo.store_named_attribute(
        geometry=creased,
        name="crease_edge",
        value=1.0,
        domain="EDGE",
        selection=pf.nodes.func.boolean_or(is_boundary, sharp_edges),
    )
    obj = pf.nodes.to_mesh_object(creased)
    pf.ops.modifier.subdivide_surface(
        obj,
        levels=levels,
        boundary_smooth="PRESERVE_CORNERS",
        _skip_apply=True,
    )
    return obj


@pf.nodes.node_function
def _outline_corner(corner_degrees: t.SocketOrVal[float]) -> pf.ProcNode[bool]:
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
def _split_long_segments(
    curve: pf.ProcNode[pf.CurveObject], max_length: t.SocketOrVal[float]
) -> pf.ProcNode[pf.CurveObject]:
    position = pf.nodes.geo.input_position()
    next_index = pf.nodes.geo.offset_point_in_curve(offset=1).point_index
    next_position = pf.nodes.geo.field_at_index(value=position, index=next_index)
    length = pf.nodes.math.vector_distance(position, next_position)
    cuts = pf.nodes.math.maximum(pf.nodes.math.ceil(length / max_length) - 1.0, 0.0)
    return pf.nodes.geo.subdivide_curve(curve, cuts=cuts.astype(dtype=int))


@pf.nodes.node_function
def _corner_ring_edges(
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


@pf.nodes.node_function
def _smooth_outline_curve(
    curve: pf.ProcNode[pf.CurveObject],
    corner_degrees: t.SocketOrVal[float],
) -> pf.ProcNode[pf.CurveObject]:
    """Bezier through the outline points, keeping turns above `corner_degrees` sharp."""
    marked = pf.nodes.geo.capture_attribute(
        geometry=curve, is_corner=_outline_corner(corner_degrees)
    )
    bezier = pf.nodes.geo.curve_spline_type(marked.geometry, spline_type="BEZIER")
    smooth = pf.nodes.geo.curve_set_handles(bezier, handle_type="AUTO")
    return pf.nodes.geo.curve_set_handles(
        smooth, selection=marked.is_corner, handle_type="VECTOR"
    )


class CutoutResult(NamedTuple):
    geom: pf.MeshObject  # posed holed surface
    sill: pf.MeshObject | None  # reveal/jamb tunnels
    lightblocker: pf.MeshObject | None  # opaque backing
    aliases: list[pf.MeshObject]  # placed instance aliases
    trim_edges: pf.CurveObject | None  # mouth outline loops, posed like the surface


@pf.nodes.node_function
def _inset_cutout_split(
    geometry: pf.ProcNode[pf.MeshObject],
    selection: t.SocketOrVal[bool],
    inset: t.SocketOrVal[pf.Vector],
    thickness: t.SocketOrVal[float],
    blocker_thickness: t.SocketOrVal[float] = 0.1,
    uv_winding_sign: t.SocketOrVal[float] = 1.0,
    delete_facecap: t.SocketOrVal[bool] = True,
    chamfer: t.SocketOrVal[float] = 0.006,
) -> mesh_util.WallCutoutResult:
    """`mesh_util.wall_cutout_split`, but each mouth vertex moves by its own `inset`.

    `selection` and `inset` must come from `faces_for_instance_grid_bboxes` on
    `geometry`, so rounded mouths keep a uniform chamfer lip.
    """
    flat = pf.nodes.geo.capture_attribute(
        domain="FACE", geometry=geometry, surf_n=pf.nodes.geo.input_normal()
    )
    extrude_dir = pf.nodes.math.vector_normalize(flat.surf_n)
    lip = pf.nodes.geo.extrude_mesh(
        mesh=flat.geometry,
        selection=selection,
        offset_scale=0.0,
        individual=False,
        mode="FACES",
    )
    chamfer_offset = inset + pf.nodes.math.vector_scale(
        vector=extrude_dir, scale=pf.nodes.math.multiply(a=chamfer, b=-1.0)
    )
    lip_in = pf.nodes.geo.set_position(
        geometry=lip.mesh, selection=lip.top, offset=chamfer_offset
    )
    deep = mesh_util.extrude_mesh_seamless_uvs(
        mesh=lip_in,
        selection=lip.top,
        offset_scale=pf.nodes.math.subtract(chamfer, thickness),
        uv_winding_sign=uv_winding_sign,
    )
    tagged = pf.nodes.geo.store_named_attribute(
        geometry=deep.mesh,
        name="is_sill",
        value=deep.side,
        domain="FACE",
        data_type="BOOLEAN",
    )
    drop_cap = pf.nodes.func.boolean_and(a=deep.top, b=delete_facecap)
    niche = pf.nodes.geo.delete_geometry(
        tagged, selection=drop_cap, domain="FACE", mode="ALL"
    )
    blocker_offset = pf.nodes.math.multiply(a=blocker_thickness, b=-1.0)
    lightblocker = pf.nodes.geo.extrude_mesh(
        niche, offset_scale=blocker_offset, individual=False, mode="FACES"
    )
    is_sill = pf.nodes.geo.input_named_attribute(
        name="is_sill", data_type=pf.NodeDataType.BOOLEAN
    )
    sill_sep = pf.nodes.geo.separate_geometry(
        niche, selection=is_sill.attribute, domain="FACE"
    )
    return mesh_util.WallCutoutResult(
        wall=sill_sep.inverted,
        sill=sill_sep.selection,
        lightblocker=lightblocker.mesh,
    )


def cutout_spaced_instances(
    surface: pf.MeshObject,
    instance: pf.MeshObject,
    surface_material: pf.Material,
    spacing: pf.Vector,
    margin_low: pf.Vector,
    margin_high: pf.Vector,
    x_instances_max: int = 1000,
    y_instances_max: int = 1,
    canonical_up_axis: str = "Z",
    instance_secondary_axis: tuple[float, float, float] = (0, 0, 1),
    wall_thickness: float = 0.1,
    recess: bool = True,
    recess_pct: float = 1.0,
    keep_facecap: bool = False,
    rotation_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    footprint: pf.MeshObject | None = None,
    chamfer: float = 0.006,
    standoff: float = 0.0,
    top_profile_height: float = 0.0,
    bottom_profile_height: float = 0.0,
) -> CutoutResult:
    """Cut instance-footprint niches in a surface and place instance aliases over them.

    Instance template must be x-centered (flip-invariant w.r.t. UV sign).
    `trim_edges` is the sharp mouth outline of every cutout on the wall surface,
    posed like the wall, with curve normals pre-set for sweeping a trim profile.
    Nonzero profile heights (metres) round the top/bottom of each footprint bbox
    into ellipse quarters spanning its full width.
    """
    uv_meters = pf.nodes.geo.input_named_attribute(
        name="UVMap", data_type=pf.NodeDataType.FLOAT_VECTOR
    ).attribute

    if footprint is None:
        footprint = instance

    cutout_chamfer = chamfer if recess else 0.0
    is_rectangular = top_profile_height <= 0.0 and bottom_profile_height <= 0.0
    columns = 2 if is_rectangular else 9

    grid_res = grid_placement.grid_from_spacing(
        uv_surface=surface,
        target_uv=uv_meters,
        instance=footprint,
        spacing=spacing,
        margin_low=margin_low,
        margin_high=margin_high,
        x_instances_max=x_instances_max,
        y_instances_max=y_instances_max,
        rotation_offset=rotation_offset,
    )

    faces_res = grid_placement.faces_for_instance_grid_bboxes(
        target_surface=surface,
        target_uv=uv_meters,
        instance=footprint,
        query_grid=grid_res.grid_mesh,
        instance_uvs=grid_res.query_uv,
        grid_index_x=grid_res.index_x,
        grid_index_y=grid_res.index_y,
        verts_per_instance_x=columns,
        verts_per_instance_y=2,
        margin_verts_x=1,
        margin_verts_y=1,
        face_expand_margin=pf.Vector((cutout_chamfer, cutout_chamfer, 0.0)),
        rotation_offset=rotation_offset,
        top_profile_height=top_profile_height,
        bottom_profile_height=bottom_profile_height,
    )

    trim_curve = mesh_util.face_selection_boundary_curve(
        mesh=faces_res.mesh, selection=faces_res.is_instance_face
    )
    if not is_rectangular:
        trim_curve = _smooth_outline_curve(trim_curve, corner_degrees=45.0)
    trim_edges = pf.nodes.to_curve_object(trim_curve)
    pf.ops.object.set_transform(
        trim_edges, surface.item().location, surface.item().rotation_euler
    )
    trim_edges.item().name = "cutout_trim_edges"

    sill: pf.MeshObject | None = None
    lightblocker: pf.MeshObject | None = None
    if recess:
        # split into holed wall, sill tunnels, and lightblocker backing
        split = _inset_cutout_split(
            faces_res.mesh,
            selection=faces_res.is_instance_face,
            inset=faces_res.inset,
            thickness=wall_thickness,
            uv_winding_sign=mesh_util.uv_winding_sign(surface),
            delete_facecap=not keep_facecap,
            chamfer=cutout_chamfer,
        )

        if is_rectangular:
            sill = pf.nodes.to_mesh_object(split.sill)
            mesh_util.crease_all_edges(sill)
            pf.ops.modifier.subdivide_surface(
                sill, ROOM_SUBSURF_LEVELS, _skip_apply=True
            )
        else:
            sill = _subdivide_rounded_cutout(split.sill, threshold_degrees=60.0)
        pf.ops.object.set_transform(
            sill, surface.item().location, surface.item().rotation_euler
        )
        pf.ops.object.set_material(
            sill,
            surface=surface_material.surface,
            displacement=surface_material.displacement,
        )
        sill.item().name = "room_wall_sill"

        lightblocker = pf.nodes.to_mesh_object(split.lightblocker)
        pf.ops.object.set_transform(
            lightblocker, surface.item().location, surface.item().rotation_euler
        )
        lightblocker.item().name = "room_wall_back"

        cut = split.wall
    else:
        # flat hole: drop the footprint faces, leaving a sharp-edged opening
        cut = pf.nodes.geo.separate_geometry(
            faces_res.mesh, selection=faces_res.is_instance_face, domain="FACE"
        ).inverted
    # weld coincident verts so boundary slivers don't break canonicalization
    geom = pf.nodes.geo.merge_by_distance(cut, distance=0.001)
    if is_rectangular:
        geom = finish_wall_plane(pf.nodes.to_mesh_object(geom), surface_material)
    else:
        geom = _subdivide_rounded_cutout(geom, threshold_degrees=35.0)
        pf.ops.object.set_material(
            geom,
            surface=surface_material.surface,
            displacement=surface_material.displacement,
        )
    pf.ops.object.set_transform(
        geom, surface.item().location, surface.item().rotation_euler
    )

    recess_depth = wall_thickness * recess_pct if recess else 0.0
    instances = grid_placement.place_instances_on_uv_grid(
        surface=surface,
        uv_field=uv_meters,
        grid_mesh=grid_res.grid_mesh,
        query_uv=grid_res.query_uv,
        instance=instance,
        secondary_axis_vector=instance_secondary_axis,
        rotation_offset=rotation_offset,
        normal_offset=standoff - recess_depth,
    )
    aliases = pf.nodes.to_aliases(instances)
    propagate_modifiers_to_instances([instance], aliases)

    geom = plane_to_posed_canonical_mesh(geom, up_axis=canonical_up_axis)
    return CutoutResult(geom, sill, lightblocker, aliases, trim_edges)


def cutout_trim_rand(
    rng: pf.RNG,
    trim_edges: pf.CurveObject,
    material: pf.Material,
    profile_curve: pf.CurveObject | None = None,
) -> pf.MeshObject:
    """Sweep a symmetric trim profile around cutout mouth outlines (window casing)."""
    if profile_curve is None:
        height = pf.random.uniform(rng, 0.05, 0.12)
        width = height * pf.random.uniform(rng, 0.2, 0.5)
        profile_curve = trim_profile_rand(rng, width=width, height=height)
    curve_geo = pf.nodes.geo.object_info(trim_edges).geometry
    curve_geo = _split_long_segments(curve_geo, max_length=0.025)
    marked = pf.nodes.geo.capture_attribute(
        geometry=curve_geo,
        ring=pf.nodes.geo.input_index().astype(dtype=float),
        is_corner=_outline_corner(45.0).astype(dtype=float),
    )
    profile_geo = pf.nodes.geo.object_info(profile_curve).geometry
    trim = curve_to_mesh_with_uv(marked.geometry, profile_geo).mesh
    trim = pf.nodes.geo.flip_faces(trim)
    corner_edges = _corner_ring_edges(ring=marked.ring, is_corner=marked.is_corner)
    obj = _subdivide_rounded_cutout(
        trim, threshold_degrees=35.0, sharp_edges=corner_edges, levels=1
    )
    pf.ops.object.set_transform(
        obj, trim_edges.item().location, trim_edges.item().rotation_euler
    )
    pf.ops.object.set_material(
        obj, surface=material.surface, displacement=material.displacement
    )
    return obj


def arrange_window_portals(
    window_aliases: list[pf.MeshObject],
    window_obj: pf.MeshObject,
    window_portal: pf.LightObject,
) -> list[pf.LightObject]:
    relative_location = (
        window_obj.item().matrix_world.inverted() @ window_portal.item().location
    )
    relative_rot_quat = (
        window_portal.item().rotation_euler.to_quaternion()
        @ window_obj.item().rotation_euler.to_quaternion().inverted()
    )
    light_locations = np.array(
        [obj.item().matrix_world @ relative_location for obj in window_aliases]
    )
    light_rotations = np.array(
        [
            (
                Euler(obj.item().rotation_euler).to_quaternion() @ relative_rot_quat
            ).to_euler()
            for obj in window_aliases
        ]
    )
    return duplicates(window_portal, light_locations, light_rotations)


def _resolve_window_inputs(
    rng: pf.RNG,
    wall: pf.MeshObject,
    window_obj: pf.MeshObject | None,
    window_portal: pf.LightObject | None,
    top_profile_height: float,
    bottom_profile_height: float,
    window_spacing: float | None,
    window_bottom: float | None,
) -> tuple[
    pf.RNG,
    pf.MeshObject,
    pf.LightObject | None,
    float,
    float,
    float,
    float,
]:
    if (
        window_obj is not None
        and window_spacing is not None
        and window_bottom is not None
    ):
        return (
            rng,
            window_obj,
            window_portal,
            top_profile_height,
            bottom_profile_height,
            window_spacing,
            window_bottom,
        )

    rng_defaults, rng_feature, rng_window = rng.spawn(3)
    wall_width, wall_height = wall_uv_dimensions(wall)
    if window_obj is None:
        width = max(1.0, min(2.0, 0.5 * wall_width))
        height = max(1.0, min(2.0, 0.7 * wall_height))
        dimensions = window.window_dimensions_rand(
            rng_defaults, width=width, height=height
        )
        window_result = window.window_composite_rand(rng_window, dimensions=dimensions)
        window_obj = window_result.mesh
        window_portal = window_result.light
        top_profile_height = window_result.profile.top_profile_height
        bottom_profile_height = window_result.profile.bottom_profile_height
        wall_offset = pf.Vector((0.0, dimensions.y * -0.5, 0.0))
        pf.ops.object.set_transform(window_obj, location=wall_offset)
        pf.ops.mesh.transform_apply(window_obj)
        if window_portal is not None:
            pf.ops.object.set_transform(
                window_portal,
                location=window_portal.item().location + wall_offset,
            )

    _depth, width, height = window_obj.item().dimensions
    if window_spacing is None:
        window_spacing = pf.random.uniform(rng_defaults, 0.1, 0.25) * width
    if window_bottom is None:
        wmin, _ = pf.ops.attr.bbox_min_max(window_obj)
        free_height = max(0.0, wall_height - height)
        bottom_fraction = pf.random.uniform(rng_defaults, 0.35, 0.65)
        window_bottom = free_height * bottom_fraction - wmin[2]
    return (
        rng_feature,
        window_obj,
        window_portal,
        top_profile_height,
        bottom_profile_height,
        window_spacing,
        window_bottom,
    )


@pf.tracer.grammar
def window_spaced_rand(
    rng: pf.RNG,
    wall: pf.MeshObject,
    window_obj: pf.MeshObject,
    wall_material: pf.Material,
    spacing: float,
    window_bottom: float,
    wall_thickness: float = 0.05,
    top_profile_height: float = 0.0,
    bottom_profile_height: float = 0.0,
    reveal_depth: float | None = None,
) -> "CutoutResult":
    width = window_obj.item().dimensions.y
    wmin, _ = pf.ops.attr.bbox_min_max(window_obj)

    wall_uv_width, _ = wall_uv_dimensions(wall)

    slack = max(0.0, wall_uv_width - width - 0.01)
    max_margin = min(0.3 * wall_uv_width, 2.0 * width, slack)
    edge_margin = pf.random.uniform(rng, min(0.1, max_margin), max_margin)
    margin_split = pf.random.clip_gaussian(rng, 0.5, 0.3, 0.05, 0.95)
    margin_low_x = edge_margin * margin_split
    margin_high_x = edge_margin * (1.0 - margin_split)
    margin_bottom = window_bottom + wmin[2]

    if width + margin_low_x + margin_high_x > wall_uv_width:
        logger.warning(
            "window: %.2fm wall too narrow for a %.2fm window + margins %.2f/%.2f; "
            "falling back to plain wall",
            wall_uv_width,
            width,
            margin_low_x,
            margin_high_x,
        )
        wall, wall_thick = plain_wall(wall, wall_material, wall_thickness)
        return CutoutResult(wall, None, wall_thick, [], None)

    if reveal_depth is None:
        reveal_depth = pf.random.uniform(rng, 0.1, 0.7)
    recess_pct = 0.9 + 0.1 * pf.random.uniform(rng, 0.0, 1.0)

    res = cutout_spaced_instances(
        surface=wall,
        instance=window_obj,
        surface_material=wall_material,
        spacing=pf.Vector((spacing, 0, 0)),
        margin_low=pf.Vector((margin_low_x, margin_bottom, 0)),
        margin_high=pf.Vector((margin_high_x, 0, 0)),
        wall_thickness=reveal_depth,
        recess_pct=recess_pct,
        chamfer=pf.random.clip_gaussian(rng, 0.006, 0.002, 0.004, 0.010),
        top_profile_height=top_profile_height,
        bottom_profile_height=bottom_profile_height,
    )
    if not res.aliases:
        logger.warning(
            "window: grid fit 0 windows on %.2fm wall (window %.2f, spacing %.2f)",
            wall_uv_width,
            width,
            spacing,
        )

    return res


@pf.tracer.grammar
def wall_windows_rand(
    rng: pf.RNG,
    wall: pf.MeshObject | None = None,
    wall_material: pf.Material | None = None,
    window_obj: pf.MeshObject | None = None,
    window_portal: pf.LightObject | None = None,
    top_profile_height: float = 0.0,
    bottom_profile_height: float = 0.0,
    window_spacing: float | None = None,
    window_bottom: float | None = None,
    wall_thickness: float = 0.05,
    reveal_depth: float | None = None,
) -> WallResult:
    rng, wall, wall_material = resolve_wall_inputs(rng, wall, wall_material)
    (
        rng,
        window_obj,
        window_portal,
        top_profile_height,
        bottom_profile_height,
        window_spacing,
        window_bottom,
    ) = _resolve_window_inputs(
        rng,
        wall,
        window_obj,
        window_portal,
        top_profile_height,
        bottom_profile_height,
        window_spacing,
        window_bottom,
    )
    res = window_spaced_rand(
        rng,
        wall,
        window_obj,
        wall_material,
        window_spacing,
        window_bottom,
        wall_thickness,
        top_profile_height=top_profile_height,
        bottom_profile_height=bottom_profile_height,
        reveal_depth=reveal_depth,
    )
    portals = []
    if window_portal is not None and res.aliases:
        portals = arrange_window_portals(res.aliases, window_obj, window_portal)

    trims = []
    if res.trim_edges is not None and res.aliases:
        rng_mat, rng_choice, rng_trim = rng.spawn(3)

        def _with_trim() -> list[pf.MeshObject]:
            vec = pf.nodes.shader.coord().uv
            trim_mat = skirt_material_rand(rng_mat, vec)
            return [cutout_trim_rand(rng_trim, res.trim_edges, trim_mat)]

        def _no_trim() -> list[pf.MeshObject]:
            return []

        trim_option = pf.control.choice(
            rng_choice, [(_with_trim, 0.4), (_no_trim, 0.6)]
        )
        trims = trim_option()

    backs = [res.lightblocker] if res.lightblocker is not None else []
    sills = [res.sill] if res.sill is not None else []
    return WallResult(
        all_objects=[res.geom, *backs, *sills, *res.aliases, *trims],
        wall_planes=[res.geom],
        backs=backs,
        sills=sills,
        storage_containers=sills,
        supports=[],
        lights=portals,
        decorations={"window": res.aliases, "window_trim": trims},
        storages=sills,
    )


@pf.tracer.grammar
def wall_painting_grid_rand(
    rng: pf.RNG,
    wall: pf.MeshObject | None = None,
    wall_material: pf.Material | None = None,
    wall_thickness: float = 0.05,
    colliders: ccol.CollisionSet | None = None,
) -> WallResult:
    rng, wall, wall_material = resolve_wall_inputs(rng, wall, wall_material)
    wall_width, wall_height = wall_uv_dimensions(wall)

    # margins first, then size each painting against the remaining wall region
    top_gap = max(0.12, 0.10 * wall_height)
    side_margin = wall_width * pf.random.uniform(rng, 0.075, 0.25) * 2.0
    avail_w = wall_width - side_margin
    # keep paintings out of the bottom 30% of the wall: they live in the top region
    bottom_min = 0.30 * wall_height
    avail_v = wall_height - bottom_min - top_gap

    art_height = pf.random.uniform(rng, 0.5, max(0.5, 0.9 * avail_v))
    art_width = pf.random.uniform(rng, 0.5, max(0.5, 0.9 * avail_w))
    art_depth = pf.random.uniform(rng, 0.03, 0.06)
    art = wall_art.wall_art_rand(
        rng, dimensions=pf.Vector((art_depth, art_width, art_height))
    ).mesh

    spacing_x = pf.random.uniform(rng, 0.1, 0.5)
    spacing_y = pf.random.uniform(rng, 0.1, 0.5)
    margin_split = pf.random.uniform(rng, 0.375, 0.625)

    # stack 1..3 rows in the top region, biased upward: the slack is pushed below
    # the block (never above), so the bottom edge stays >= 30% of the wall height
    max_rows = max(1, int((avail_v + spacing_y) / (art_height + spacing_y)))
    n_rows = min(pf.random.randint(rng, 1, 4), max_rows)
    block_h = n_rows * art_height + (n_rows - 1) * spacing_y
    v_slack = max(0.0, avail_v - block_h)
    up_bias = pf.random.uniform(rng, 0.5, 1.0)
    margin_bottom = bottom_min + v_slack * up_bias
    margin_top = top_gap + v_slack * (1.0 - up_bias)

    uv_meters = pf.nodes.geo.input_named_attribute(
        name="UVMap", data_type=pf.NodeDataType.FLOAT_VECTOR
    ).attribute
    grid_res = grid_placement.grid_from_spacing(
        uv_surface=wall,
        target_uv=uv_meters,
        instance=art,
        spacing=pf.Vector((spacing_x, spacing_y, 0)),
        margin_low=pf.Vector((side_margin * margin_split, margin_bottom, 0)),
        margin_high=pf.Vector((side_margin * (1 - margin_split), margin_top, 0)),
        y_instances_max=n_rows,
    )
    instances = grid_placement.place_instances_on_uv_grid(
        surface=wall,
        uv_field=uv_meters,
        grid_mesh=grid_res.grid_mesh,
        query_uv=grid_res.query_uv,
        instance=art,
    )
    painting_aliases = pf.nodes.to_aliases(instances)
    propagate_modifiers_to_instances([art], painting_aliases)
    if not painting_aliases:
        logger.warning(
            "painting: grid fit 0 paintings on %.2fx%.2fm wall "
            "(art %.2fx%.2f, spacing %.2f/%.2f, %d rows)",
            wall_width,
            wall_height,
            art_width,
            art_height,
            spacing_x,
            spacing_y,
            n_rows,
        )
    if colliders is not None:
        painting_aliases, _ = keep_non_colliding(
            painting_aliases, colliders, key=lambda obj: obj
        )

    wall, wall_thick = plain_wall(wall, wall_material, wall_thickness)
    return WallResult(
        all_objects=[wall, wall_thick, *painting_aliases],
        wall_planes=[wall],
        backs=[wall_thick],
        sills=[],
        storage_containers=[],
        supports=[],
        lights=[],
        decorations={"painting": painting_aliases},
    )


@pf.tracer.grammar
def wall_storage_shelf_rand(
    rng: pf.RNG,
    wall: pf.MeshObject | None = None,
    wall_material: pf.Material | None = None,
    wall_thickness: float = 0.05,
) -> WallResult:
    rng, wall, wall_material = resolve_wall_inputs(rng, wall, wall_material)
    wall_width, wall_height = wall_uv_dimensions(wall)

    depth = pf.random.uniform(rng, 0.3, 0.61)

    # half-depth edge margins keep corner cabinets clear
    min_margin = depth * 0.5

    # bimodal cabinet height: short band vs tall band
    def _short_band() -> float:
        return wall_height * pf.random.uniform(rng, 0.20, 0.50)

    def _tall_band() -> float:
        return wall_height * pf.random.uniform(rng, 0.60, 0.98)

    height = pf.control.choice(rng, [(_short_band, 1.0), (_tall_band, 1.0)])()

    width = wall_storage_width_rand(rng, wall_width, min_margin)
    spacing_x = pf.random.uniform(rng, 0.1, 0.5)

    margin_split = pf.random.uniform(rng, 0.375, 0.625)
    margin_low_x, margin_high_x, _ = fit_grid_margins(
        wall_width, width, spacing_x, min_margin, margin_split
    )
    recess_frac = pf.random.uniform(rng, 0.0, 1.0)
    hole_depth = max(0.02, recess_frac**0.5 * depth)

    cab = storage.storage_cell_shelf_rand(
        rng,
        dimensions=pf.Vector((depth, width, height)),
        back_width=0.0,
    ).mesh
    cab = seat_upright_cabinet(cab, width, height, back_depth=depth)
    footprint = upright_cabinet_footprint(width, height)

    geom, sill, lightblocker, cabinet_aliases, _trim_edges = cutout_spaced_instances(
        surface=wall,
        instance=cab,
        surface_material=wall_material,
        spacing=pf.Vector((spacing_x, 0, 0)),
        margin_low=pf.Vector((margin_low_x, 0.0, 0)),
        margin_high=pf.Vector((margin_high_x, 0.0, 0)),
        wall_thickness=hole_depth,
        recess_pct=0.0,
        keep_facecap=True,
        footprint=footprint,
        chamfer=pf.random.clip_gaussian(rng, 0.006, 0.002, 0.004, 0.010),
        standoff=pf.random.uniform(rng, 0.02, 0.05),
    )
    if not cabinet_aliases:
        logger.warning(
            "wall_storage: grid fit 0 cabinets on %.2fm wall "
            "(cabinet width %.2f, spacing_x %.2f, margins %.2f/%.2f)",
            wall_width,
            width,
            spacing_x,
            margin_low_x,
            margin_high_x,
        )

    backs = [lightblocker] if lightblocker is not None else []
    sills = [sill] if sill is not None else []
    return WallResult(
        all_objects=[geom, *backs, *sills, *cabinet_aliases],
        wall_planes=[geom],
        backs=backs,
        sills=sills,
        storage_containers=cabinet_aliases,
        supports=cabinet_aliases,
        lights=[],
        decorations={"wall_storage": cabinet_aliases},
        storages=cabinet_aliases,
    )


@pf.tracer.grammar
def wall_cubby_rand(
    rng: pf.RNG,
    wall: pf.MeshObject | None = None,
    wall_material: pf.Material | None = None,
    wall_thickness: float = 0.05,
) -> WallResult:
    rng, wall, wall_material = resolve_wall_inputs(rng, wall, wall_material)
    wall_width, wall_height = wall_uv_dimensions(wall)

    depth = pf.random.uniform(rng, 0.3, 0.61)
    min_margin = depth * 0.5
    width = wall_storage_width_rand(rng, wall_width, min_margin)

    bottom = pf.random.uniform(rng, 0.25, 0.45) * wall_height
    top = (0.02 + 0.18 * pf.random.uniform(rng, 0.0, 1.0) ** 2) * wall_height
    band = max(0.1, wall_height - bottom - top)

    # mode at the 0.6 m minimum (stacks fit) with a tail up to full-band alcoves
    hi = max(0.6, min(band, 0.8 * wall_height))
    height = pf.random.clip_gaussian(rng, 0.6, 0.5, 0.6, hi)
    height = min(height, band, 0.8 * wall_height)
    spacing_x = pf.random.uniform(rng, 0.1, 0.5)
    spacing_y = pf.random.uniform(rng, 0.05, 0.6) * height

    margin_split = pf.random.uniform(rng, 0.375, 0.625)
    margin_low_x, margin_high_x, _ = fit_grid_margins(
        wall_width, width, spacing_x, min_margin, margin_split
    )
    hole_depth = depth

    cab = storage.storage_cell_shelf_rand(
        rng,
        dimensions=pf.Vector((depth, width, height)),
        back_width=0.0,
    ).mesh
    cab = seat_upright_cabinet(cab, width, height, back_depth=hole_depth)
    footprint = upright_cabinet_footprint(width, height)

    geom, sill, lightblocker, cabinet_aliases, _trim_edges = cutout_spaced_instances(
        surface=wall,
        instance=cab,
        surface_material=wall_material,
        spacing=pf.Vector((spacing_x, spacing_y, 0)),
        margin_low=pf.Vector((margin_low_x, bottom, 0)),
        margin_high=pf.Vector((margin_high_x, top, 0)),
        x_instances_max=3,
        y_instances_max=3,
        wall_thickness=hole_depth,
        recess_pct=0.0,
        keep_facecap=True,
        footprint=footprint,
        chamfer=pf.random.clip_gaussian(rng, 0.006, 0.002, 0.004, 0.010),
    )
    if not cabinet_aliases:
        logger.warning(
            "wall_cubby: grid fit 0 cubbies on %.2fx%.2fm wall "
            "(cubby %.2fx%.2f, spacing %.2f/%.2f)",
            wall_width,
            wall_height,
            width,
            height,
            spacing_x,
            spacing_y,
        )

    backs = [lightblocker] if lightblocker is not None else []

    def _keep_cabinets() -> WallResult:
        return WallResult(
            all_objects=[geom, *backs, *cabinet_aliases],
            wall_planes=[geom],
            backs=backs,
            sills=[],
            storage_containers=cabinet_aliases,
            supports=cabinet_aliases,
            lights=[],
            decorations={"wall_cubby": cabinet_aliases},
            storages=cabinet_aliases,
        )

    def _drop_cabinets() -> WallResult:
        sills = [sill] if sill is not None else []
        return WallResult(
            all_objects=[geom, *backs, *sills],
            wall_planes=[geom],
            backs=backs,
            sills=sills,
            storage_containers=sills,
            supports=[],
            lights=[],
            decorations={},
            storages=sills,
        )

    return pf.control.choice(rng, [(_keep_cabinets, 1.0), (_drop_cabinets, 2.0)])()


@pf.tracer.grammar
def wall_doors_rand(
    rng: pf.RNG,
    wall: pf.MeshObject | None = None,
    wall_material: pf.Material | None = None,
    wall_thickness: float = 0.05,
) -> WallResult:
    rng, wall, wall_material = resolve_wall_inputs(rng, wall, wall_material)
    wall_width, wall_height = wall_uv_dimensions(wall)

    door_width = pf.random.uniform(rng, 0.85, 1.2)
    door_height = min(pf.random.uniform(rng, 2.0, 2.2), wall_height * 0.9)
    door_thickness = pf.random.clip_gaussian(rng, 0.0318, 0.0127, 0.0254, 0.0762)
    door_func = pf.control.choice(
        rng, [(door_composite_rand, 2.0), (door_double_rand, 1.0)]
    )

    spacing_x = pf.random.uniform(rng, 2.0, 6.0)
    max_margin = wall_width - door_width - 0.2
    if max_margin < 0.1:
        logger.warning(
            "door: %.2fm wall too narrow for a %.2fm door; falling back to plain wall",
            wall_width,
            door_width,
        )
        return wall_plain_rand(rng, wall, wall_material, wall_thickness)
    edge_margin = pf.random.uniform(rng, 0.1, max_margin)
    margin_split = pf.random.uniform(rng, 0.0, 1.0)
    margin_low_x = edge_margin * margin_split
    margin_high_x = edge_margin * (1.0 - margin_split)

    reveal_depth = pf.random.uniform(rng, 0.1, 0.3)
    recess_pct = 0.9 + 0.1 * pf.random.uniform(rng, 0.0, 1.0)

    dimensions = pf.Vector((door_thickness, door_width, door_height))

    def cut_door(door: pf.MeshObject, top_profile_height: float) -> CutoutResult:
        # centre the slab along the wall (Y); an off-centre along-wall origin lands it
        # beside its hole. Use door_width, not the bbox, so the handle bump does not bias it
        pf.ops.object.set_transform(door, location=(0.0, -door_width * 0.5, 0.0))
        pf.ops.mesh.transform_apply(door)
        # lift the door 1cm off the wall's bottom edge (V is the .y margin) so its
        # chamfered cutout stays inside the wall UV; below that sample_uv_surface
        # returns origin verts
        floor_lift = 0.01
        return cutout_spaced_instances(
            surface=wall,
            instance=door,
            surface_material=wall_material,
            spacing=pf.Vector((spacing_x, 0, 0)),
            margin_low=pf.Vector((margin_low_x, floor_lift, 0)),
            margin_high=pf.Vector((margin_high_x, 0.0, 0)),
            x_instances_max=2,
            wall_thickness=reveal_depth,
            recess_pct=recess_pct,
            top_profile_height=top_profile_height,
        )

    def rectangular_door(rng: pf.RNG) -> CutoutResult:
        return cut_door(door_func(rng, dimensions=dimensions).mesh, 0.0)

    def arched_door(rng: pf.RNG) -> CutoutResult:
        rng_profile, rng_door = rng.spawn(2)
        profile = window.top_rounded_profile_rand(rng_profile, dimensions)
        door = door_glass_from_profile_rand(rng_door, profile)
        return cut_door(door.mesh, profile.top_profile_height)

    rng_shape_choice, _ = rng.spawn(2)
    door_shape_func = pf.control.choice(
        rng_shape_choice, [(rectangular_door, 0.9), (arched_door, 0.1)]
    )
    geom, sill, lightblocker, door_aliases, _trim_edges = door_shape_func(rng)
    if not door_aliases:
        logger.warning(
            "door: grid fit 0 doors on %.2fm wall (door %.2f, spacing %.2f, "
            "margins %.2f/%.2f)",
            wall_width,
            door_width,
            spacing_x,
            margin_low_x,
            margin_high_x,
        )

    # a door reveal reaches the floor and is never a placeable sill, so group it with backs
    reveal = [sill] if sill is not None else []
    backs = ([lightblocker] if lightblocker is not None else []) + reveal
    door_clearances = [
        _door_affordance_collider(alias, door_width, wall_height, index)
        for index, alias in enumerate(door_aliases)
    ]
    return WallResult(
        all_objects=[geom, *backs, *door_aliases],
        wall_planes=[geom],
        backs=backs,
        sills=[],
        storage_containers=[],
        supports=[],
        lights=[],
        decorations={"door": door_aliases},
        colliders=ccol.collision_set(door_aliases + door_clearances),
    )


def _door_affordance_collider(
    door: pf.MeshObject,
    width: float,
    height: float,
    index: int,
) -> pf.MeshObject:
    minimum, maximum = pf.ops.attr.bbox_min_max(door, global_coords=False)
    local_center = pf.Vector((minimum + maximum) / 2)
    center = door.item().matrix_world @ local_center
    tangent = door.item().matrix_world.to_3x3() @ pf.Vector((0.0, 1.0, 0.0))
    angle = float(np.arctan2(tangent.y, tangent.x) - np.pi / 2)
    collider = pf.ops.primitives.mesh_cube(size=1.0)
    pf.ops.mesh.transform(collider, scale=(2 * width, width, height))
    pf.ops.object.set_transform(
        collider,
        location=(center.x, center.y, height / 2),
        rotation_euler=(0.0, 0.0, angle),
    )
    collider.item().name = f"door_affordance_{index:02d}"
    collider.item().hide_render = True
    collider.item().display_type = "WIRE"
    return collider


@pf.tracer.grammar
def wall_full_window_rand(
    rng: pf.RNG,
    wall: pf.MeshObject | None = None,
    wall_material: pf.Material | None = None,
    wall_thickness: float = 0.05,
) -> WallResult:
    rng, wall, wall_material = resolve_wall_inputs(rng, wall, wall_material)
    wall_width, wall_height = wall_uv_dimensions(wall)

    eps = 0.02
    gap_x = 0.10 * wall_width
    gap_y = 0.10 * wall_height
    slack_x = max(
        wall_width * (1.0 - pf.random.clip_gaussian(rng, 0.85, 0.1, 0.5, 0.95)),
        2.0 * gap_x,
    )
    slack_y = max(
        wall_height * (1.0 - pf.random.clip_gaussian(rng, 0.85, 0.1, 0.5, 0.95)),
        2.0 * gap_y,
    )
    target_w = wall_width - slack_x - eps
    target_h = wall_height - slack_y - eps
    if target_w <= 0.0 or target_h <= 0.0:
        logger.warning(
            "full_window: %.2fx%.2fm wall too small for a window; "
            "falling back to plain wall",
            wall_width,
            wall_height,
        )
        return wall_plain_rand(rng, wall, wall_material, wall_thickness)

    win_dims = window.window_dimensions_rand(rng, width=target_w, height=target_h)
    win_result = window.window_rectangular_composite_rand(rng, dimensions=win_dims)
    win_obj = win_result.mesh
    wall_offset = pf.Vector((0.0, win_dims.y * -0.5, 0.0))
    pf.ops.object.set_transform(win_obj, location=wall_offset)
    pf.ops.mesh.transform_apply(win_obj)
    if win_result.light is not None:
        pf.ops.object.set_transform(
            win_result.light,
            location=win_result.light.item().location + wall_offset,
        )

    inner_x = slack_x - 2.0 * gap_x
    inner_y = slack_y - 2.0 * gap_y
    split_x = pf.random.uniform(rng, 0.0, 1.0)
    split_y = pf.random.uniform(rng, 0.0, 1.0)

    geom, _sill, _lightblocker, win_aliases, _trim_edges = cutout_spaced_instances(
        surface=wall,
        instance=win_obj,
        surface_material=wall_material,
        spacing=pf.Vector((0, 0, 0)),
        margin_low=pf.Vector((gap_x + inner_x * split_x, gap_y + inner_y * split_y, 0)),
        margin_high=pf.Vector(
            (gap_x + inner_x * (1.0 - split_x), gap_y + inner_y * (1.0 - split_y), 0)
        ),
        x_instances_max=1,
        y_instances_max=1,
        recess=False,
    )
    if not win_aliases:
        logger.warning(
            "full_window: 0 windows fit on %.2fx%.2fm wall (target %.2fx%.2f)",
            wall_width,
            wall_height,
            target_w,
            target_h,
        )

    wall_back = extrude_for_thickness(geom, wall_thickness)
    pf.ops.object.set_transform(
        wall_back, geom.item().location, geom.item().rotation_euler
    )
    wall_back.item().name = "room_wall_back"

    portals = []
    if win_result.light is not None and win_aliases:
        portals = arrange_window_portals(win_aliases, win_obj, win_result.light)
    return WallResult(
        all_objects=[geom, wall_back, *win_aliases],
        wall_planes=[geom],
        backs=[wall_back],
        sills=[],
        storage_containers=[],
        supports=[],
        lights=portals,
        decorations={"window": win_aliases},
    )
