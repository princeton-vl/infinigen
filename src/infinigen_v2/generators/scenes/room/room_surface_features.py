import logging
from typing import NamedTuple

import numpy as np
import procfunc as pf
from mathutils import Euler, Matrix

from infinigen_v2.generators.curves.skirting_board_profile import (
    skirting_profile_distribution,
)
from infinigen_v2.generators.objects import lamp, slot_cabinet, wall_art, window
from infinigen_v2.generators.objects.ceiling_light import ceiling_light_distribution
from infinigen_v2.generators.scenes import collision_collection as ccol
from infinigen_v2.generators.scenes.placement_utils import (
    duplicates,
    keep_non_colliding,
)
from infinigen_v2.generators.scenes.room.room_shape import RoomShapeResult
from infinigen_v2.generators.shaders.functionality_lists import (
    ceiling_material_distribution,
    floor_material_distribution,
    furniture_material_distribution,
    skirt_material_distribution,
    wall_material_distribution,
)
from infinigen_v2.generators.util import mesh as mesh_util
from infinigen_v2.generators.util.curve import curve_to_mesh_with_uv
from infinigen_v2.generators.uv_surface import grid_placement

logger = logging.getLogger(__name__)


class WallFeatureResult(NamedTuple):
    wall_geom: list[pf.MeshObject]  # wall surfaces (≥1; split options return multiple)
    backs: list[pf.MeshObject]  # structural lightblocker meshes (wall backs)
    sills: list[pf.MeshObject]  # cutout reveal/sill meshes
    storage: list[pf.MeshObject]  # shelf/storage surfaces for downstream filling
    lights: list[pf.LightObject]  # window portals or other attached lights
    decorations: dict[
        str, list[pf.MeshObject]
    ]  # window/painting/shelf/door instances by type


def extrude_for_thickness(obj: pf.MeshObject, thickness: float) -> pf.MeshObject:
    """Solidify a flat surface into a back-thickness slab (the lightblocker body)."""
    geo = pf.nodes.geo.object_info(obj).geometry
    extruded = mesh_util.extrude_mesh_seamless_uvs(
        mesh=geo,
        selection=True,
        offset_scale=-thickness,
        uv_winding_sign=mesh_util.uv_winding_sign(obj),
    )
    result = pf.nodes.to_mesh_object(extruded.mesh)
    result.item().name = obj.item().name + "_thickened"
    return result


def plane_to_posed_canonical_mesh(
    obj: pf.MeshObject,
    up_axis: str = "Z",
) -> pf.MeshObject:
    """Bake a planar mesh's location/rotation into the object transform, appearance unchanged."""
    # area-weighted normal (sliver faces have garbage normals)
    bbox_min, bbox_max = pf.ops.attr.bbox_min_max(obj, global_coords=False)
    normals = pf.ops.attr.polygon_normals(obj)
    areas = pf.ops.attr.polygon_areas(obj)

    pos = pf.Vector((np.array(bbox_min) + np.array(bbox_max)) / 2)
    normal = pf.Vector((normals * areas[:, None]).sum(axis=0)).normalized()

    # normal -> +X, keeping up_axis upright
    rot = normal.to_track_quat("X", up_axis).inverted()

    pf.ops.object.set_transform(obj, location=-pos)
    pf.ops.mesh.transform_apply(obj)
    pf.ops.object.set_transform(obj, rotation_euler=rot.to_euler())
    pf.ops.mesh.transform_apply(obj)

    pf.ops.object.set_transform(obj, location=pos)
    pf.ops.object.set_transform(obj, rotation_euler=rot.inverted().to_euler())

    return obj


def finish_cutout_mesh(
    obj: pf.MeshObject,
    surface: pf.MeshObject,
    material: pf.Material,
) -> pf.MeshObject:
    """Pose a cut surface piece on its source, material it, and crease-subdivide for displacement."""
    pf.ops.object.set_transform(
        obj, surface.item().location, surface.item().rotation_euler
    )
    pf.ops.object.set_material(
        obj, surface=material.surface, displacement=material.displacement
    )
    pf.ops.attr.write_attribute(obj, 1.0, "crease_edge", domain="EDGE")
    pf.ops.modifier.subdivide_surface(obj, levels=8, _skip_apply=True)
    return obj


def arrange_window_portals(
    walls_window_aliases: list[pf.MeshObject],
    window_obj: pf.MeshObject,
    window_portal: pf.LightObject,
) -> list[pf.LightObject]:
    relative_rot_quat = (
        window_portal.item().rotation_euler.to_quaternion()
        @ window_obj.item().rotation_euler.to_quaternion().inverted()
    )
    light_locations = np.array([obj.item().location for obj in walls_window_aliases])
    light_rotations = np.array(
        [
            (
                Euler(obj.item().rotation_euler).to_quaternion() @ relative_rot_quat
            ).to_euler()
            for obj in walls_window_aliases
        ]
    )
    return duplicates(window_portal, light_locations, light_rotations)


@pf.tracer.grammar
def window_spaced_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    window_obj: pf.MeshObject,
    wall_material: pf.Material,
    spacing: float,
    window_bottom: float,
    wall_thickness: float = 0.05,
) -> tuple[
    pf.MeshObject, pf.MeshObject | None, pf.MeshObject | None, list[pf.MeshObject]
]:
    width = window_obj.item().dimensions.x
    wmin, _ = pf.ops.attr.bbox_min_max(window_obj)

    wall_uvs = pf.ops.attr.uv_coords(wall)
    wall_uv_width = wall_uvs[:, 0].max() - wall_uvs[:, 0].min()

    max_margin = min(0.3 * wall_uv_width, 2.0 * width)
    edge_margin = pf.random.uniform(rng, 0.1, max_margin) if max_margin > 0.1 else 0.1
    margin_split = pf.random.clip_gaussian(rng, 0.5, 0.3, 0.05, 0.95)
    margin_low_x = edge_margin * margin_split
    margin_high_x = edge_margin * (1.0 - margin_split)
    margin_bottom = window_bottom + wmin[1]

    if width + margin_low_x + margin_high_x > wall_uv_width:
        logger.warning(
            "window: %.2fm wall too narrow for a %.2fm window + margins %.2f/%.2f; "
            "falling back to plain wall",
            wall_uv_width,
            width,
            margin_low_x,
            margin_high_x,
        )
        wall, wall_thick = _plain_wall(wall, wall_material, wall_thickness)
        return wall, None, wall_thick, []

    reveal_depth = pf.random.uniform(rng, 0.1, 0.7)
    recess_pct = 0.9 + 0.1 * pf.random.uniform(rng, 0.0, 1.0)

    geom, sill, lightblocker, window_aliases = cutout_spaced_instances(
        surface=wall,
        instance=window_obj,
        surface_material=wall_material,
        spacing=pf.Vector((spacing, 0, 0)),
        margin_low=pf.Vector((margin_low_x, margin_bottom, 0)),
        margin_high=pf.Vector((margin_high_x, 0, 0)),
        wall_thickness=reveal_depth,
        recess_pct=recess_pct,
        chamfer=pf.random.clip_gaussian(rng, 0.006, 0.002, 0.004, 0.010),
    )
    if not window_aliases:
        logger.warning(
            "window: grid fit 0 windows on %.2fm wall (window %.2f, spacing %.2f)",
            wall_uv_width,
            width,
            spacing,
        )

    return geom, sill, lightblocker, window_aliases


def _plain_wall(
    wall: pf.MeshObject, wall_material: pf.Material, wall_thickness: float
) -> tuple[pf.MeshObject, pf.MeshObject]:
    """Materialise, thicken and canonicalise a bare wall (no cutouts)."""
    wall_thick = extrude_for_thickness(wall, wall_thickness)
    wall_thick.item().name = "room_wall_back"
    pf.ops.object.set_material(
        wall,
        surface=wall_material.surface,
        displacement=wall_material.displacement,
    )
    pf.ops.modifier.subdivide_surface(wall, levels=8, _skip_apply=True)
    wall = plane_to_posed_canonical_mesh(wall)
    return wall, wall_thick


@pf.tracer.grammar
def wall_plain_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    wall_material: pf.Material,
    wall_thickness: float = 0.05,
) -> WallFeatureResult:
    wall, wall_thick = _plain_wall(wall, wall_material, wall_thickness)
    return WallFeatureResult(
        wall_geom=[wall],
        backs=[wall_thick],
        sills=[],
        storage=[],
        lights=[],
        decorations={},
    )


@pf.tracer.grammar
def wall_windows_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    wall_material: pf.Material,
    window_obj: pf.MeshObject,
    window_portal: pf.LightObject | None,
    window_spacing: float,
    window_bottom: float,
    wall_thickness: float = 0.05,
) -> WallFeatureResult:
    geom, sill, lightblocker, window_aliases = window_spaced_distribution(
        rng,
        wall,
        window_obj,
        wall_material,
        window_spacing,
        window_bottom,
        wall_thickness,
    )
    portals = []
    if window_portal is not None and window_aliases:
        portals = arrange_window_portals(window_aliases, window_obj, window_portal)
    return WallFeatureResult(
        wall_geom=[geom],
        backs=[lightblocker] if lightblocker is not None else [],
        sills=[sill] if sill is not None else [],
        storage=[],
        lights=portals,
        decorations={"window": window_aliases},
    )


@pf.tracer.grammar
def wall_window_extrusion_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    wall_material: pf.Material,
    window_obj: pf.MeshObject,
    window_portal: pf.LightObject | None,
    window_spacing: float,
    window_bottom: float,
    wall_thickness: float = 0.05,
) -> WallFeatureResult:
    raise NotImplementedError


@pf.tracer.grammar
def wall_vertical_split_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    wall_material: pf.Material,
    wall_material_alt: pf.Material,
    wall_thickness: float = 0.05,
) -> WallFeatureResult:
    raise NotImplementedError


@pf.tracer.grammar
def wall_horizontal_split_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    wall_material: pf.Material,
    wall_material_alt: pf.Material,
    wall_thickness: float = 0.05,
) -> WallFeatureResult:
    raise NotImplementedError


@pf.tracer.grammar
def wall_painting_grid_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    wall_material: pf.Material,
    wall_thickness: float = 0.05,
) -> WallFeatureResult:
    wall_uvs = pf.ops.attr.uv_coords(wall)
    wall_width = wall_uvs[:, 0].max() - wall_uvs[:, 0].min()
    wall_height = wall_uvs[:, 1].max() - wall_uvs[:, 1].min()

    # margins first, then size each painting dimension from 0.5m up to 80% of the
    # wall extent minus its margins
    top_gap = max(0.12, 0.10 * wall_height)
    side_margin = wall_width * pf.random.uniform(rng, 0.075, 0.25) * 2.0
    avail_w = wall_width - side_margin
    # keep paintings out of the bottom 30% of the wall: they live in the top region
    bottom_min = 0.30 * wall_height
    avail_v = wall_height - bottom_min - top_gap
    art_height = pf.random.uniform(rng, 0.5, 0.8 * avail_v)
    # width caps at 2x height so paintings never become thin horizontal bars;
    # portrait (taller than wide) is unconstrained
    art_width = pf.random.uniform(rng, 0.5, min(0.8 * avail_w, 2.0 * art_height))
    art_depth = pf.random.uniform(rng, 0.03, 0.06)
    art = wall_art.wall_art_distribution(
        rng, dimensions=pf.Vector((art_depth, art_width, art_height))
    ).mesh

    # reorient into the wall instance frame (x along wall, y up, z out)
    rot = Matrix(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))).to_euler()
    pf.ops.object.set_transform(art, rotation_euler=rot)
    pf.ops.mesh.transform_apply(art)

    wall_thick = extrude_for_thickness(wall, wall_thickness)
    wall_thick.item().name = "room_wall_back"

    # small gaps (down to 0.1m) let paintings cluster into tight grids
    spacing_x = pf.random.uniform(rng, 0.1, 1.5 * art_width)
    spacing_y = pf.random.uniform(rng, 0.1, 1.5 * art_height)
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

    geom, _sill, _lightblocker, painting_aliases = cutout_spaced_instances(
        surface=wall,
        instance=art,
        surface_material=wall_material,
        spacing=pf.Vector((spacing_x, spacing_y, 0)),
        margin_low=pf.Vector((side_margin * margin_split, margin_bottom, 0)),
        margin_high=pf.Vector((side_margin * (1 - margin_split), margin_top, 0)),
        y_instances_max=n_rows,
        recess=False,
    )
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

    return WallFeatureResult(
        wall_geom=[geom],
        backs=[wall_thick],
        sills=[],
        storage=[],
        lights=[],
        decorations={"painting": painting_aliases},
    )


@pf.tracer.grammar
def wall_board_shelf_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    wall_material: pf.Material,
    wall_thickness: float = 0.05,
) -> WallFeatureResult:
    wall_uvs = pf.ops.attr.uv_coords(wall)
    wall_width = wall_uvs[:, 0].max() - wall_uvs[:, 0].min()
    wall_height = wall_uvs[:, 1].max() - wall_uvs[:, 1].min()

    shelf_depth = pf.random.uniform(rng, 0.225, 0.45)
    shelf_thickness = pf.random.uniform(rng, 0.02, 0.05)
    spacing_x = pf.random.uniform(rng, 0.03, 0.25)
    spacing_y = pf.random.uniform(rng, 0.5, 0.65)

    # shelf width spans 0.7m up to the full available wall, skewed toward narrow
    # (squared) so multi-column runs are common while a single full-width shelf
    # stays possible; the column count emerges from the width+spacing+margin fit
    # (like the window grid), never chosen up front
    side_margin = wall_width * pf.random.uniform(rng, 0.04, 0.25) * 2.0
    available_x = wall_width - side_margin
    shelf_width = (
        0.7 + (max(0.7, available_x) - 0.7) * pf.random.uniform(rng, 0.0, 1.0) ** 2
    )

    cube = pf.nodes.geo.mesh_cube(size=(shelf_width, shelf_depth, shelf_thickness))
    slab_geo = pf.nodes.geo.transform(
        cube.mesh, translation=(0.0, shelf_depth * 0.5, 0.0)
    )
    slab_geo = slot_cabinet.metric_box_uv(slab_geo)
    slab = pf.nodes.to_mesh_object(slab_geo)

    footprint = pf.nodes.to_mesh_object(
        pf.nodes.geo.mesh_cube(size=(shelf_width, shelf_thickness, 0.0)).mesh
    )
    vec = pf.nodes.shader.coord().uv
    shelf_material = furniture_material_distribution(rng, vec)
    pf.ops.object.set_material(
        slab,
        surface=shelf_material.surface,
        displacement=shelf_material.displacement,
    )

    top_frac = pf.random.uniform(rng, 0.12, 0.25)
    bot_frac = pf.random.uniform(rng, top_frac, 0.50)
    margin_top = wall_height * top_frac
    margin_bottom = wall_height * bot_frac
    margin_split = pf.random.uniform(rng, 0.375, 0.625)

    uv_meters = pf.nodes.geo.input_named_attribute(
        name="UVMap", data_type=pf.NodeDataType.FLOAT_VECTOR
    ).attribute

    grid_res = grid_placement.grid_from_spacing(
        uv_surface=wall,
        target_uv=uv_meters,
        instance=footprint,
        spacing=pf.Vector((spacing_x, spacing_y, 0)),
        margin_low=pf.Vector((side_margin * margin_split, margin_bottom, 0)),
        margin_high=pf.Vector((side_margin * (1 - margin_split), margin_top, 0)),
        y_instances_max=pf.random.randint(rng, 4, 12),
    )
    instances = grid_placement.place_instances_on_uv_grid(
        surface=wall,
        uv_field=uv_meters,
        grid_mesh=grid_res.grid_mesh,
        query_uv=grid_res.query_uv,
        instance=slab,
        up_align=True,
    )
    shelf_aliases = pf.nodes.to_aliases(instances)
    if not shelf_aliases:
        logger.warning(
            "wall_board_shelf: no shelves fit on %.2fx%.2fm wall (width %.2f, "
            "spacing_y %.2f, margins t/b %.2f/%.2f)",
            wall_width,
            wall_height,
            shelf_width,
            spacing_y,
            margin_top,
            margin_bottom,
        )

    wall_thick = extrude_for_thickness(wall, wall_thickness)
    wall_thick.item().name = "room_wall_back"
    pf.ops.object.set_material(
        wall,
        surface=wall_material.surface,
        displacement=wall_material.displacement,
    )
    pf.ops.modifier.subdivide_surface(wall, levels=8, _skip_apply=True)
    wall = plane_to_posed_canonical_mesh(wall)

    return WallFeatureResult(
        wall_geom=[wall],
        backs=[wall_thick],
        sills=[],
        storage=shelf_aliases,
        lights=[],
        decorations={"wall_board_shelf": shelf_aliases},
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
    up_align: bool = False,
    rotation_offset: float = 0.0,
    footprint: pf.MeshObject | None = None,
    chamfer: float = 0.006,
    standoff: float = 0.0,
) -> tuple[
    pf.MeshObject, pf.MeshObject | None, pf.MeshObject | None, list[pf.MeshObject]
]:
    """Cut instance-footprint niches in a surface and place instance aliases over them.

    Returns (posed surface, sill or None, lightblocker or None, aliases).
    Instance template must be x-centered (flip-invariant w.r.t. UV sign).
    """
    uv_meters = pf.nodes.geo.input_named_attribute(
        name="UVMap", data_type=pf.NodeDataType.FLOAT_VECTOR
    ).attribute

    if footprint is None:
        footprint = instance

    cutout_chamfer = chamfer if recess else 0.0

    grid_res = grid_placement.grid_from_spacing(
        uv_surface=surface,
        target_uv=uv_meters,
        instance=footprint,
        spacing=spacing,
        margin_low=margin_low,
        margin_high=margin_high,
        x_instances_max=x_instances_max,
        y_instances_max=y_instances_max,
    )

    faces_res = grid_placement.faces_for_instance_grid_bboxes(
        target_surface=surface,
        target_uv=uv_meters,
        instance=footprint,
        query_grid=grid_res.grid_mesh,
        instance_uvs=grid_res.query_uv,
        grid_index_x=grid_res.index_x,
        grid_index_y=grid_res.index_y,
        verts_per_instance_x=2,
        verts_per_instance_y=2,
        margin_verts_x=1,
        margin_verts_y=1,
        face_expand_margin=pf.Vector((cutout_chamfer, cutout_chamfer, 0.0)),
    )

    sill: pf.MeshObject | None = None
    lightblocker: pf.MeshObject | None = None
    if recess:
        # split into holed wall, sill tunnels, and lightblocker backing
        tagged = pf.nodes.geo.store_named_attribute(
            geometry=faces_res.mesh,
            name="cutout_sel",
            value=faces_res.is_instance_face,
            domain="FACE",
            data_type="BOOLEAN",
        )
        split = mesh_util.wall_cutout_split(
            tagged,
            thickness=wall_thickness,
            uv_winding_sign=mesh_util.uv_winding_sign(surface),
            delete_facecap=not keep_facecap,
            chamfer=cutout_chamfer,
        )

        sill = finish_cutout_mesh(
            pf.nodes.to_mesh_object(split.sill), surface, surface_material
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
    geom = finish_cutout_mesh(pf.nodes.to_mesh_object(geom), surface, surface_material)

    instances = grid_placement.place_instances_on_uv_grid(
        surface=surface,
        uv_field=uv_meters,
        grid_mesh=grid_res.grid_mesh,
        query_uv=grid_res.query_uv,
        instance=instance,
        secondary_axis_vector=instance_secondary_axis,
        up_align=up_align,
        rotation_offset=rotation_offset,
    )
    aliases = pf.nodes.to_aliases(instances)

    into_wall_local = (
        pf.Vector((0.0, -1.0, 0.0)) if up_align else pf.Vector((0.0, 0.0, -1.0))
    )
    # up_align seats the cabinet with local +X out of the wall; non-up_align uses -into_wall
    out_of_wall_local = pf.Vector((1.0, 0.0, 0.0)) if up_align else -into_wall_local
    recess_depth = wall_thickness * recess_pct if recess else 0.0
    if recess_depth or standoff:
        for alias in aliases:
            item = alias.item()
            rot = item.rotation_euler.to_matrix()
            offset = (rot @ into_wall_local) * recess_depth + (
                rot @ out_of_wall_local
            ) * standoff
            item.location = item.location + offset

    geom = plane_to_posed_canonical_mesh(geom, up_axis=canonical_up_axis)
    return geom, sill, lightblocker, aliases


def upright_cabinet_footprint(width: float, height: float) -> pf.MeshObject:
    """Flat (U=width, V=height) rect for niche-grid sizing of an up_aligned cabinet."""
    return pf.nodes.to_mesh_object(
        pf.nodes.geo.mesh_cube(size=(width, height, 0.0)).mesh
    )


def seat_upright_cabinet(
    cab: pf.MeshObject, width: float, height: float, back_depth: float
) -> pf.MeshObject:
    """Center an upright cabinet (X=depth out, Y=width, Z=height) for up_align placement.

    Y/Z centered on the niche; X shifted so the back sits back_depth behind the surface
    (opening at depth-back_depth). up_align maps local X->+normal, Y->along-wall, Z->up.
    """
    bmin, bmax = pf.ops.attr.bbox_min_max(cab)
    pf.ops.object.set_transform(
        cab,
        location=(
            -bmin[0] - back_depth,
            -(bmin[1] + bmax[1]) / 2,
            -(bmin[2] + bmax[2]) / 2,
        ),
    )
    pf.ops.mesh.transform_apply(cab)
    return cab


def fit_grid_margins(
    extent: float,
    item_size: float,
    spacing: float,
    min_margin: float,
    split: float,
    n_cap: int = 1000,
) -> tuple[float, float, int]:
    """Fit item count across extent and push slack into edge margins (low, high, n)."""
    usable = extent - 2 * min_margin
    n = 1
    if usable > item_size:
        n = max(1, int((usable - item_size) / (item_size + spacing)) + 1)
    n = min(n, n_cap)
    used = n * item_size + (n - 1) * spacing
    slack = max(0.02, extent - used - 2 * min_margin) - 0.02
    return min_margin + slack * split, min_margin + slack * (1 - split), n


@pf.tracer.grammar
def wall_storage_shelf_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    wall_material: pf.Material,
    wall_thickness: float = 0.05,
) -> WallFeatureResult:
    wall_uvs = pf.ops.attr.uv_coords(wall)
    wall_width = wall_uvs[:, 0].max() - wall_uvs[:, 0].min()
    wall_height = wall_uvs[:, 1].max() - wall_uvs[:, 1].min()

    depth = pf.random.uniform(rng, 0.3, 0.61)

    # half-depth edge margins keep corner cabinets clear; guard tiny walls
    min_margin = depth * 0.5
    usable_width = max(0.8, wall_width - 2 * min_margin)

    # bimodal cabinet height: short band vs tall band
    def _short_band() -> float:
        return wall_height * pf.random.uniform(rng, 0.20, 0.50)

    def _tall_band() -> float:
        return wall_height * pf.random.uniform(rng, 0.60, 0.98)

    height = pf.control.choice(rng, [(_short_band, 1.0), (_tall_band, 1.0)])()

    max_width = usable_width * 0.98
    width = pf.random.uniform(rng, min(1.5, max_width), max_width)
    spacing_x = pf.random.uniform(rng, 0.1, 0.5)

    margin_split = pf.random.uniform(rng, 0.375, 0.625)
    margin_low_x, margin_high_x, _ = fit_grid_margins(
        wall_width, width, spacing_x, min_margin, margin_split
    )
    recess_frac = pf.random.uniform(rng, 0.0, 1.0)
    hole_depth = max(0.02, recess_frac**0.5 * depth)

    cab = slot_cabinet.slot_cabinet_distribution(
        rng,
        dimensions=pf.Vector((depth, width, height)),
        back_width=0.0,
    ).mesh
    cab = seat_upright_cabinet(cab, width, height, back_depth=depth)
    footprint = upright_cabinet_footprint(width, height)

    geom, sill, lightblocker, cabinet_aliases = cutout_spaced_instances(
        surface=wall,
        instance=cab,
        surface_material=wall_material,
        spacing=pf.Vector((spacing_x, 0, 0)),
        margin_low=pf.Vector((margin_low_x, 0.0, 0)),
        margin_high=pf.Vector((margin_high_x, 0.0, 0)),
        wall_thickness=hole_depth,
        recess_pct=0.0,
        keep_facecap=True,
        up_align=True,
        rotation_offset=np.pi / 2,
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

    return WallFeatureResult(
        wall_geom=[geom],
        backs=[lightblocker] if lightblocker is not None else [],
        sills=[sill] if sill is not None else [],
        storage=cabinet_aliases,
        lights=[],
        decorations={"wall_storage": cabinet_aliases},
    )


@pf.tracer.grammar
def wall_cubby_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    wall_material: pf.Material,
    wall_thickness: float = 0.05,
) -> WallFeatureResult:
    wall_uvs = pf.ops.attr.uv_coords(wall)
    wall_width = wall_uvs[:, 0].max() - wall_uvs[:, 0].min()
    wall_height = wall_uvs[:, 1].max() - wall_uvs[:, 1].min()

    depth = pf.random.uniform(rng, 0.3, 0.61)
    min_margin = depth * 0.5

    bottom = pf.random.uniform(rng, 0.25, 0.45) * wall_height
    top = (0.02 + 0.18 * pf.random.uniform(rng, 0.0, 1.0) ** 2) * wall_height
    band = max(0.1, wall_height - bottom - top)

    # mode at the 0.6 m minimum (stacks fit) with a tail up to full-band alcoves
    hi = max(0.6, min(band, 0.8 * wall_height))
    height = pf.random.clip_gaussian(rng, 0.6, 0.5, 0.6, hi)
    height = min(height, band, 0.8 * wall_height)
    aspect = pf.random.clip_gaussian(rng, 4.0, 1.0, 0.2, 5.0)
    width = min(height * aspect, 0.8 * wall_width)

    spacing_x = pf.random.uniform(rng, 0.1, 0.5)
    spacing_y = pf.random.uniform(rng, 0.05, 0.6) * height

    margin_split = pf.random.uniform(rng, 0.375, 0.625)
    margin_low_x, margin_high_x, _ = fit_grid_margins(
        wall_width, width, spacing_x, min_margin, margin_split
    )
    hole_depth = depth

    cab = slot_cabinet.slot_cabinet_distribution(
        rng,
        dimensions=pf.Vector((depth, width, height)),
        back_width=0.0,
    ).mesh
    cab = seat_upright_cabinet(cab, width, height, back_depth=hole_depth)
    footprint = upright_cabinet_footprint(width, height)

    geom, sill, lightblocker, cabinet_aliases = cutout_spaced_instances(
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
        up_align=True,
        rotation_offset=np.pi / 2,
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

    def _keep_cabinets() -> WallFeatureResult:
        return WallFeatureResult(
            wall_geom=[geom],
            backs=[lightblocker] if lightblocker is not None else [],
            sills=[],
            storage=cabinet_aliases,
            lights=[],
            decorations={"wall_cubby": cabinet_aliases},
        )

    def _drop_cabinets() -> WallFeatureResult:
        return WallFeatureResult(
            wall_geom=[geom],
            backs=[lightblocker] if lightblocker is not None else [],
            sills=[sill] if sill is not None else [],
            storage=[],
            lights=[],
            decorations={},
        )

    return pf.control.choice(rng, [(_keep_cabinets, 1.0), (_drop_cabinets, 2.0)])()


@pf.tracer.grammar
def wall_storage_flush_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    wall_material: pf.Material,
    wall_thickness: float = 0.05,
) -> WallFeatureResult:
    wall_uvs = pf.ops.attr.uv_coords(wall)
    wall_width = wall_uvs[:, 0].max() - wall_uvs[:, 0].min()
    wall_height = wall_uvs[:, 1].max() - wall_uvs[:, 1].min()

    depth = pf.random.uniform(rng, 0.3, 0.61)
    min_margin = depth * 0.5
    usable_width = max(0.8, wall_width - 2 * min_margin)

    def _short_band() -> float:
        return wall_height * pf.random.uniform(rng, 0.20, 0.50)

    def _tall_band() -> float:
        return wall_height * pf.random.uniform(rng, 0.60, 0.98)

    height = pf.control.choice(rng, [(_short_band, 1.0), (_tall_band, 1.0)])()

    max_width = usable_width * 0.98
    width = pf.random.uniform(rng, min(1.5, max_width), max_width)
    spacing_x = pf.random.uniform(rng, 0.1, 0.5)

    margin_split = pf.random.uniform(rng, 0.375, 0.625)
    margin_low_x, margin_high_x, _ = fit_grid_margins(
        wall_width, width, spacing_x, min_margin, margin_split
    )

    cab = slot_cabinet.slot_cabinet_distribution(
        rng,
        dimensions=pf.Vector((depth, width, height)),
    ).mesh
    cab = seat_upright_cabinet(cab, width, height, back_depth=0.0)
    footprint = upright_cabinet_footprint(width, height)

    uv_meters = pf.nodes.geo.input_named_attribute(
        name="UVMap", data_type=pf.NodeDataType.FLOAT_VECTOR
    ).attribute
    grid_res = grid_placement.grid_from_spacing(
        uv_surface=wall,
        target_uv=uv_meters,
        instance=footprint,
        spacing=pf.Vector((spacing_x, 0, 0)),
        margin_low=pf.Vector((margin_low_x, 0.0, 0)),
        margin_high=pf.Vector((margin_high_x, 0.0, 0)),
        y_instances_max=1,
    )
    instances = grid_placement.place_instances_on_uv_grid(
        surface=wall,
        uv_field=uv_meters,
        grid_mesh=grid_res.grid_mesh,
        query_uv=grid_res.query_uv,
        instance=cab,
        up_align=True,
        rotation_offset=np.pi / 2,
    )
    aliases = pf.nodes.to_aliases(instances)
    if not aliases:
        logger.warning(
            "wall_storage_flush: no cabinets fit on %.2fm-wide wall "
            "(cabinet width %.2f, spacing_x %.2f, margins %.2f/%.2f)",
            wall_width,
            width,
            spacing_x,
            margin_low_x,
            margin_high_x,
        )

    # up_align seats the cabinet with local +X out of the wall
    out_of_wall_local = pf.Vector((1.0, 0.0, 0.0))
    standoff = pf.random.uniform(rng, 0.02, 0.05)
    for alias in aliases:
        item = alias.item()
        out_of_wall = item.rotation_euler.to_matrix() @ out_of_wall_local
        item.location = item.location + out_of_wall * standoff

    wall, wall_thick = _plain_wall(wall, wall_material, wall_thickness)
    return WallFeatureResult(
        wall_geom=[wall],
        backs=[wall_thick],
        sills=[],
        storage=aliases,
        lights=[],
        decorations={"wall_storage": aliases},
    )


@pf.tracer.grammar
def wall_doors_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    wall_material: pf.Material,
    wall_thickness: float = 0.05,
) -> WallFeatureResult:
    wall_uvs = pf.ops.attr.uv_coords(wall)
    wall_width = wall_uvs[:, 0].max() - wall_uvs[:, 0].min()
    wall_height = wall_uvs[:, 1].max() - wall_uvs[:, 1].min()

    door_width = pf.random.uniform(rng, 0.85, 1.2)
    door_height = min(pf.random.uniform(rng, 2.0, 2.2), wall_height * 0.9)
    door_thickness = pf.random.uniform(rng, 0.04, 0.07)

    spacing_x = pf.random.uniform(rng, 2.0, 6.0)
    max_margin = max(0.01, wall_width - door_width - 0.2)
    edge_margin = pf.random.uniform(rng, 0.1, max_margin)
    margin_split = pf.random.uniform(rng, 0.0, 1.0)
    margin_low_x = edge_margin * margin_split
    margin_high_x = edge_margin * (1.0 - margin_split)

    wall_thick = extrude_for_thickness(wall, wall_thickness)
    wall_thick.item().name = "room_wall_back"

    # door slab, x-centered, bottom at y=0
    cube = pf.nodes.geo.mesh_cube(size=(door_width, door_height, door_thickness))
    slab = pf.nodes.geo.transform(cube.mesh, translation=(0.0, door_height * 0.5, 0.0))
    slab = slot_cabinet.metric_box_uv(slab)
    door = pf.nodes.to_mesh_object(slab)
    vec = pf.nodes.shader.coord().uv
    door_material = furniture_material_distribution(rng, vec)
    pf.ops.object.set_material(
        door,
        surface=door_material.surface,
        displacement=door_material.displacement,
    )

    geom, _sill, _lightblocker, door_aliases = cutout_spaced_instances(
        surface=wall,
        instance=door,
        surface_material=wall_material,
        spacing=pf.Vector((spacing_x, 0, 0)),
        margin_low=pf.Vector((margin_low_x, 0.0, 0)),
        margin_high=pf.Vector((margin_high_x, 0.0, 0)),
        x_instances_max=2,
        recess=False,
    )
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

    return WallFeatureResult(
        wall_geom=[geom],
        backs=[wall_thick],
        sills=[],
        storage=[],
        lights=[],
        decorations={"door": door_aliases},
    )


@pf.tracer.grammar
def wall_full_window_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    wall_material: pf.Material,
    wall_thickness: float = 0.05,
) -> WallFeatureResult:
    wall_uvs = pf.ops.attr.uv_coords(wall)
    wall_width = wall_uvs[:, 0].max() - wall_uvs[:, 0].min()
    wall_height = wall_uvs[:, 1].max() - wall_uvs[:, 1].min()

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

    win_dims = window.window_dimensions_distribution(
        rng, width=target_w, height=target_h
    )
    win_result = window.window_distribution(rng, dimensions=win_dims)
    win_obj = win_result.mesh
    pf.ops.object.set_transform(
        win_obj,
        scale=(
            target_w / win_obj.item().dimensions.x,
            target_h / win_obj.item().dimensions.y,
            1.0,
        ),
    )
    pf.ops.mesh.transform_apply(win_obj)

    inner_x = slack_x - 2.0 * gap_x
    inner_y = slack_y - 2.0 * gap_y
    split_x = pf.random.uniform(rng, 0.0, 1.0)
    split_y = pf.random.uniform(rng, 0.0, 1.0)

    geom, _sill, _lightblocker, win_aliases = cutout_spaced_instances(
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
    return WallFeatureResult(
        wall_geom=[geom],
        backs=[wall_back],
        sills=[],
        storage=[],
        lights=portals,
        decorations={"window": win_aliases},
    )


@pf.tracer.grammar
def wall_decoration_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    window_obj: pf.MeshObject,
    window_portal: pf.LightObject | None,
    wall_material: pf.Material,
    wall_material_alt: pf.Material,
    window_spacing: float | None = None,
    window_bottom: float | None = None,
    wall_thickness: float = 0.05,
) -> WallFeatureResult:
    def plain(rng, wall, wall_material):
        return wall_plain_distribution(
            rng, wall, wall_material, wall_thickness=wall_thickness
        )

    def windows(rng, wall, wall_material):
        return wall_windows_distribution(
            rng,
            wall,
            wall_material,
            window_obj=window_obj,
            window_portal=window_portal,
            window_spacing=window_spacing,
            window_bottom=window_bottom,
            wall_thickness=wall_thickness,
        )

    def window_extrusion(rng, wall, wall_material):
        return wall_window_extrusion_distribution(
            rng,
            wall,
            wall_material,
            window_obj=window_obj,
            window_portal=window_portal,
            window_spacing=window_spacing,
            window_bottom=window_bottom,
            wall_thickness=wall_thickness,
        )

    def vertical_split(rng, wall, wall_material):
        return wall_vertical_split_distribution(
            rng,
            wall,
            wall_material,
            wall_material_alt=wall_material_alt,
            wall_thickness=wall_thickness,
        )

    def horizontal_split(rng, wall, wall_material):
        return wall_horizontal_split_distribution(
            rng,
            wall,
            wall_material,
            wall_material_alt=wall_material_alt,
            wall_thickness=wall_thickness,
        )

    def painting_grid(rng, wall, wall_material):
        return wall_painting_grid_distribution(
            rng, wall, wall_material, wall_thickness=wall_thickness
        )

    def board_shelf(rng, wall, wall_material):
        return wall_board_shelf_distribution(
            rng, wall, wall_material, wall_thickness=wall_thickness
        )

    def storage_shelf(rng, wall, wall_material):
        return wall_storage_shelf_distribution(
            rng, wall, wall_material, wall_thickness=wall_thickness
        )

    def storage_flush(rng, wall, wall_material):
        return wall_storage_flush_distribution(
            rng, wall, wall_material, wall_thickness=wall_thickness
        )

    def cubby(rng, wall, wall_material):
        return wall_cubby_distribution(
            rng, wall, wall_material, wall_thickness=wall_thickness
        )

    def doors(rng, wall, wall_material):
        return wall_doors_distribution(
            rng, wall, wall_material, wall_thickness=wall_thickness
        )

    def full_wall_window(rng, wall, wall_material):
        return wall_full_window_distribution(
            rng, wall, wall_material, wall_thickness=wall_thickness
        )

    rng_choice, rng_feature = rng.spawn(2)
    option = pf.control.choice(
        rng_choice,
        [
            (plain, 0.5),
            (windows, 3.0),
            # (window_extrusion, 0.0),
            # (vertical_split, 0.0),
            # (horizontal_split, 0.0),
            (painting_grid, 1.0),
            (board_shelf, 1.0),
            # (storage_shelf, 0.0),
            (storage_flush, 1.0),
            (cubby, 0.5),
            (doors, 0.2),
            (full_wall_window, 1.0),
        ],
    )
    return option(rng=rng_feature, wall=wall, wall_material=wall_material)


class WallFeaturesResult(NamedTuple):
    wall_planes: list[pf.MeshObject]
    backs: list[pf.MeshObject]  # structural wall-back lightblocker meshes
    sills: list[pf.MeshObject]  # cutout reveal/sill meshes
    storage: list[pf.MeshObject]
    lights: list[pf.LightObject]  # window portals
    decorations: dict[
        str, list[pf.MeshObject]
    ]  # window/painting/shelf/door instances by type


class CeilingFeaturesResult(NamedTuple):
    floor: pf.MeshObject
    ceiling: pf.MeshObject
    backs: list[pf.MeshObject]  # ceiling-back lightblocker meshes
    sills: list[pf.MeshObject]  # skylight/bar reveal meshes
    light_meshes: list[pf.MeshObject]  # lamp/skylight/bar housing meshes
    lights: list[pf.LightObject]


@pf.tracer.grammar
@pf.tracer.grammar
def skirting_on_walls_distribution(
    rng: pf.RNG,
    walls: list[pf.MeshObject],
    material: pf.Material,
    profile_curve: pf.CurveObject | None = None,
) -> list[pf.MeshObject]:
    if profile_curve is None:
        profile_curve = skirting_profile_distribution(rng)
    profile_curve_geo = pf.nodes.geo.object_info(profile_curve).geometry

    # follow the bottom/top boundary edges of the wall meshes
    wall_geos = [
        pf.nodes.geo.transform(
            pf.nodes.geo.object_info(w).geometry,
            translation=tuple(w.item().location),
            rotation=tuple(w.item().rotation_euler),
        )
        for w in walls
    ]
    joined = pf.nodes.geo.join_geometry(wall_geos)
    # weld coincident verts so boundary slivers collapse
    joined = pf.nodes.geo.merge_by_distance(joined, distance=0.005)

    position_z = pf.nodes.geo.input_position().z
    z_stat = pf.nodes.geo.attribute_statistic(geometry=joined, attribute=position_z)
    # tight band: only edges flat at the extremes (avoids corner-junction spikes)
    near_bottom = position_z < z_stat.min + 0.005
    near_top = position_z > z_stat.max - 0.005

    # small fillet rounds corner junctions to avoid miter spikes
    floor_curve_node = pf.nodes.geo.mesh_to_curve(joined, selection=near_bottom)
    floor_curve_node = pf.nodes.geo.fillet_curve_poly(
        floor_curve_node, radius=0.02, limit_radius=True, count=2
    )

    ceiling_curve_node = pf.nodes.geo.mesh_to_curve(joined, selection=near_top)
    ceiling_curve_node = pf.nodes.geo.fillet_curve_poly(
        ceiling_curve_node, radius=0.02, limit_radius=True, count=2
    )
    ceiling_curve_node = pf.nodes.geo.reverse_curve(ceiling_curve_node)
    ceiling_curve_node = pf.nodes.geo.set_curve_tilt(ceiling_curve_node, tilt=np.pi)

    geoms = pf.control.choice(
        rng,
        [
            ([floor_curve_node], 3.0),
            ([ceiling_curve_node], 1.0),
            ([floor_curve_node, ceiling_curve_node], 2.0),
        ],
    )
    path_curves = pf.nodes.geo.join_geometry(geoms)

    skirt = curve_to_mesh_with_uv(path_curves, profile_curve_geo).mesh
    skirt = pf.nodes.geo.flip_faces(skirt)
    skirt = pf.nodes.to_mesh_object(skirt)

    pf.ops.object.set_material(
        skirt,
        surface=material.surface,
        displacement=material.displacement,
    )

    return [skirt]


def ceiling_shade_lamp_template(rng: pf.RNG, energy: float) -> lamp.LampResult:
    bot_radius = pf.random.clip_gaussian(rng, 0.35, 0.1, 0.1, 0.6)
    top_radius = bot_radius * pf.random.clip_gaussian(rng, 1.1, 0.1, 0.9, 1.5)
    height = pf.random.clip_gaussian(rng, 0.3, 0.15, 0.2, 0.6)
    result = lamp.lamp_distribution(
        rng,
        energy=energy,
        head_top_radius=top_radius,
        head_bot_radius=bot_radius,
        height=height,
    )
    result.mesh.item().rotation_euler = (np.pi, 0, 0)
    # flip the light through the same pi-about-x as the shade
    lx, ly, lz = result.light.item().location
    result.light.item().parent = None
    result.light.item().location = (lx, -ly, -lz)
    return result


@pf.tracer.grammar
def ceiling_light_placement_distribution(
    rng: pf.RNG,
    ceiling: pf.MeshObject,
    dimensions: pf.Vector,
) -> tuple[list[pf.MeshObject], list[pf.LightObject]]:
    approx_room_area = dimensions.x * dimensions.y
    w_per_area = pf.random.clip_gaussian(rng, 0.9, 2.5, 0.3, 12.0)
    total_energy = approx_room_area * w_per_area

    spacing_x = pf.random.uniform(rng, 1.5, 2.5)
    spacing_y = pf.random.uniform(rng, 1.5, 2.5)
    margin_x = pf.random.uniform(rng, 0.4, 1.5)
    margin_y = pf.random.uniform(rng, 0.4, 1.5)

    n_estimate_x = max(1, int((dimensions.x - 2 * margin_x) // spacing_x) + 1)
    n_estimate_y = max(1, int((dimensions.y - 2 * margin_y) // spacing_y) + 1)
    per_energy = total_energy / (n_estimate_x * n_estimate_y)

    template_fn = pf.control.choice(
        rng,
        [
            (ceiling_light_distribution, 2.5),
            (ceiling_shade_lamp_template, 1.0),
        ],
    )
    lamp_template = template_fn(rng, energy=per_energy)
    lamp_template.mesh.item().name = template_fn.__name__
    mesh_template = lamp_template.mesh

    # light offset relative to template origin
    light_offset = None
    if lamp_template.light is not None:
        light_offset = np.array(
            lamp_template.light.item().location - mesh_template.item().location
        )

    # bake inverse of the grid's Ry(pi) so lamps keep their authored orientation
    instance_quat = Euler((0.0, np.pi, 0.0)).to_quaternion()
    template_quat = Euler(mesh_template.item().rotation_euler).to_quaternion()
    pf.ops.object.set_transform(
        mesh_template,
        rotation_euler=(instance_quat.inverted() @ template_quat).to_euler(),
    )
    pf.ops.mesh.transform_apply(mesh_template)
    bmin, bmax = pf.ops.attr.bbox_min_max(mesh_template)
    center = (np.array(bmin) + np.array(bmax)) / 2
    pf.ops.object.set_transform(mesh_template, location=(-center[0], -center[1], 0.0))
    pf.ops.mesh.transform_apply(mesh_template)

    lamp_w = np.array(bmax) - np.array(bmin)
    gap_x = max(0.1, spacing_x - lamp_w[0])
    gap_y = max(0.1, spacing_y - lamp_w[1])

    # 2d grid placement, no cutting
    pf.ops.uv.cube_project(ceiling, uv_name="UVMap")
    uv_meters = pf.nodes.geo.input_named_attribute(
        name="UVMap", data_type=pf.NodeDataType.FLOAT_VECTOR
    ).attribute
    grid_res = grid_placement.grid_from_spacing(
        uv_surface=ceiling,
        target_uv=uv_meters,
        instance=mesh_template,
        spacing=pf.Vector((gap_x, gap_y, 0)),
        margin_low=pf.Vector((margin_x, margin_y, 0)),
        margin_high=pf.Vector((margin_x, margin_y, 0)),
        x_instances_max=n_estimate_x,
        y_instances_max=n_estimate_y,
    )
    instances = grid_placement.place_instances_on_uv_grid(
        surface=ceiling,
        uv_field=uv_meters,
        grid_mesh=grid_res.grid_mesh,
        query_uv=grid_res.query_uv,
        instance=mesh_template,
        secondary_axis_vector=(0, 1, 0),
    )
    meshes = pf.nodes.to_aliases(instances)

    if lamp_template.light is None or not meshes or light_offset is None:
        return meshes, []

    light_locations = np.array([m.item().location for m in meshes]) + light_offset
    lights = duplicates(lamp_template.light, light_locations)
    # rescale to the actual placed count to hit total_energy
    for light in lights:
        light.item().data.energy = total_energy / len(lights)
    lights = pf.control.choice(rng, [(lights, 3.0), ([], 1.0)])
    return meshes, lights


@pf.tracer.grammar
def wall_feature_distribution(
    rng: pf.RNG,
    shape: RoomShapeResult,
    wall_thickness: float = 0.1,
) -> WallFeaturesResult:
    vec_wall = pf.nodes.shader.coord().uv

    rng_materials, rng_window, rng_walls = rng.spawn(3)
    rng_mat_1, rng_mat_2 = rng_materials.spawn(2)
    wall_material_1 = wall_material_distribution(rng_mat_1, vec_wall)
    wall_material_2 = wall_material_distribution(rng_mat_2, vec_wall)

    wall_back = extrude_for_thickness(shape.walls, wall_thickness)
    wall_back.item().name = "room_wall_back"

    pf.ops.object.set_material(
        shape.walls,
        surface=wall_material_1.surface,
        displacement=wall_material_1.displacement,
    )
    pf.ops.modifier.subdivide_surface(shape.walls, levels=8, _skip_apply=True)

    edge_gap_pct = 0.10
    edge_gap = edge_gap_pct * shape.dimensions.z
    usable_height = shape.dimensions.z - 2.0 * edge_gap
    window_height_pct = pf.random.clip_gaussian(
        rng_window, 0.75, 0.2, 0.6, 1.0 - 2.0 * edge_gap_pct
    )
    window_height = shape.dimensions.z * window_height_pct
    # squared uniform biases width narrow
    max_window_width = max(2.0, max(shape.dimensions.x, shape.dimensions.y) - 0.5)
    window_width = (
        1.0 + (max_window_width - 1.0) * pf.random.uniform(rng_window, 0.0, 1.0) ** 2
    )
    window_dimensions = window.window_dimensions_distribution(
        rng_window, width=window_width, height=window_height
    )
    window_result = window.window_distribution(rng_window, dimensions=window_dimensions)
    window_obj = window_result.mesh
    window_portal = window_result.light

    pf.ops.object.set_transform(
        window_obj, scale=(1.0, window_height / window_obj.item().dimensions.y, 1.0)
    )
    pf.ops.mesh.transform_apply(window_obj)

    _width, _height, _depth = window_obj.item().dimensions
    wmin, _wmax = pf.ops.attr.bbox_min_max(window_obj)
    free_height = usable_height - _height
    window_bottom_pct = pf.random.clip_gaussian(rng_window, 0.7, 0.15, 0.35, 0.85)
    window_bottom = edge_gap + free_height * window_bottom_pct - wmin[1]
    window_spacing = pf.random.uniform(rng_window, 0.1, 0.25) * _width

    wall_planes = []
    backs = [wall_back]
    sills = []
    storage = []
    lights = []
    decorations: dict[str, list[pf.MeshObject]] = {}

    # cull decorations against those on other walls (seed empty; walls are coincident)
    colliders = ccol.collision_set([])
    for wall, rng_wall in zip(
        shape.flat_walls, rng_walls.spawn(len(shape.flat_walls)), strict=True
    ):
        rng_wall_mat, rng_wall_dec = rng_wall.spawn(2)
        mat = pf.control.choice(
            rng_wall_mat,
            [(wall_material_1, 3), (wall_material_2, 1)],
        )
        result = wall_decoration_distribution(
            rng_wall_dec,
            wall,
            window_obj,
            window_portal,
            wall_material=mat,
            wall_material_alt=wall_material_2,
            window_spacing=window_spacing,
            window_bottom=window_bottom,
            wall_thickness=wall_thickness,
        )
        flat_decorations = [o for objs in result.decorations.values() for o in objs]
        kept, colliders = keep_non_colliding(
            flat_decorations, colliders, key=lambda o: o
        )
        dropped = set(id(o) for o in flat_decorations) - set(id(o) for o in kept)
        wall_planes.extend(result.wall_geom)
        backs.extend(result.backs)
        sills.extend(result.sills)
        storage.extend(o for o in result.storage if id(o) not in dropped)
        lights.extend(result.lights)
        for kind, objs in result.decorations.items():
            decorations.setdefault(kind, []).extend(
                o for o in objs if id(o) not in dropped
            )

    for kind, objs in sorted(decorations.items()):
        logger.info(f"Created {len(objs)} wall {kind} objects")
    logger.info(f"Created {len(storage)} wall storage surfaces")

    return WallFeaturesResult(
        wall_planes=wall_planes,
        backs=backs,
        sills=sills,
        storage=storage,
        lights=lights,
        decorations=decorations,
    )


@pf.tracer.grammar
def ceiling_skylights_distribution(
    rng: pf.RNG,
    ceiling: pf.MeshObject,
    ceiling_material: pf.Material,
) -> tuple[
    pf.MeshObject,
    list[pf.MeshObject],
    list[pf.MeshObject],
    list[pf.MeshObject],
    list[pf.LightObject],
]:
    # metric planar UVs for the cutout grid
    pf.ops.uv.cube_project(ceiling, uv_name="UVMap")

    ceiling_uvs = pf.ops.attr.uv_coords(ceiling)
    extent_x = ceiling_uvs[:, 0].max() - ceiling_uvs[:, 0].min()
    extent_y = ceiling_uvs[:, 1].max() - ceiling_uvs[:, 1].min()

    # each side must fit the smaller extent (either orientation), plus headroom
    floor_margin = 0.1
    max_side = max(0.45, min(extent_x, extent_y) - 2 * floor_margin - 0.1)

    # short side + aspect, then random long axis
    skylight_short = min(pf.random.uniform(rng, 0.45, 1.1), max_side)
    skylight_aspect = pf.random.uniform(rng, 1.0, 10.0)
    skylight_long = min(skylight_short * skylight_aspect, 3.5, max_side)
    skylight_width, skylight_length = pf.control.choice(
        rng,
        [
            ((skylight_short, skylight_long), 1.0),
            ((skylight_long, skylight_short), 1.0),
        ],
    )
    window_dimensions = window.window_dimensions_distribution(
        rng, width=skylight_width, height=skylight_length
    )
    # skylights never get curtains
    window_result = window.window_distribution(
        rng,
        dimensions=window_dimensions,
        curtain=pf.ops.primitives.mesh_single_vertex(),
    )
    win = window_result.mesh

    # lay the window flat into the ceiling, facing down, origin-centered
    pf.ops.object.set_transform(win, rotation_euler=(np.pi, 0.0, 0.0))
    pf.ops.mesh.transform_apply(win)
    bmin, bmax = pf.ops.attr.bbox_min_max(win)
    center = (np.array(bmin) + np.array(bmax)) / 2
    pf.ops.object.set_transform(win, location=(-center[0], -center[1], 0.0))
    pf.ops.mesh.transform_apply(win)
    bmin, bmax = pf.ops.attr.bbox_min_max(win)
    win_dims = np.array(bmax) - np.array(bmin)

    spacing = pf.random.uniform(rng, 0.4, 1.8)
    margin_frac = pf.random.uniform(rng, 0.0, 1.0)

    # fit per-axis counts; min_margin within [floor, (extent-win)/2]
    extents = (extent_x, extent_y)
    margins = []
    for axis, n_cap in ((0, 4), (1, 3)):
        extent = extents[axis]
        max_margin = max(floor_margin, (extent - win_dims[axis]) / 2)
        min_margin = floor_margin + (max_margin - floor_margin) * margin_frac
        split = pf.random.uniform(rng, 0.4, 0.6)
        low, high, _ = fit_grid_margins(
            extent, win_dims[axis], spacing, min_margin, split, n_cap
        )
        margins.append((low, high))

    reveal_depth = pf.random.uniform(rng, 0.1, 1.0)
    recess_pct = pf.random.uniform(rng, 0.0, 1.0)

    geom, sill, lightblocker, skylight_aliases = cutout_spaced_instances(
        surface=ceiling,
        instance=win,
        surface_material=ceiling_material,
        spacing=pf.Vector((spacing, spacing, 0)),
        margin_low=pf.Vector((margins[0][0], margins[1][0], 0)),
        margin_high=pf.Vector((margins[0][1], margins[1][1], 0)),
        x_instances_max=4,
        y_instances_max=3,
        canonical_up_axis="Y",
        instance_secondary_axis=(0, 1, 0),
        wall_thickness=reveal_depth,
        recess_pct=recess_pct,
        chamfer=pf.random.clip_gaussian(rng, 0.006, 0.002, 0.004, 0.010),
    )

    portals: list[pf.LightObject] = []
    if window_result.light is not None and skylight_aliases:
        portals = arrange_window_portals(skylight_aliases, win, window_result.light)

    backs = [lightblocker] if lightblocker is not None else []
    sills = [sill] if sill is not None else []
    return geom, backs, sills, skylight_aliases, portals


@pf.tracer.grammar
def ceiling_light_bars_distribution(
    rng: pf.RNG,
    ceiling: pf.MeshObject,
    ceiling_material: pf.Material,
    dimensions: pf.Vector,
) -> tuple[
    pf.MeshObject,
    list[pf.MeshObject],
    list[pf.MeshObject],
    list[pf.MeshObject],
    list[pf.LightObject],
]:
    # metric planar UVs for the cutout grid
    pf.ops.uv.cube_project(ceiling, uv_name="UVMap")

    ceiling_uvs = pf.ops.attr.uv_coords(ceiling)
    extent_x = ceiling_uvs[:, 0].max() - ceiling_uvs[:, 0].min()
    extent_y = ceiling_uvs[:, 1].max() - ceiling_uvs[:, 1].min()

    # bars run long along x, thin along y, rows stacked across y
    bar_length = min(pf.random.uniform(rng, 1.5, 4.0), extent_x - 0.4)
    bar_width = pf.random.uniform(rng, 0.02, 1.0)
    bar_depth = pf.random.uniform(rng, 0.03, 0.06)

    # thin box housing, x/y-centered (symmetric -> flip-invariant)
    cube = pf.nodes.geo.mesh_cube(size=(bar_length, bar_width, bar_depth))
    housing = pf.nodes.to_mesh_object(cube.mesh)
    pf.ops.uv.cube_project(housing, uv_name="UVMap")
    pf.ops.object.set_material(
        housing,
        surface=ceiling_material.surface,
        displacement=ceiling_material.displacement,
    )

    spacing = pf.random.uniform(rng, 0.4, 2.4)
    min_margin_x = pf.random.uniform(rng, 0.1, 0.3)
    min_margin_y = pf.random.uniform(rng, 0.3, 1.2)

    # rows across y, single centered span along x
    split_y = pf.random.uniform(rng, 0.4, 0.6)
    margin_y_low, margin_y_high, _ = fit_grid_margins(
        extent_y, bar_width, spacing, min_margin_y, split_y, n_cap=4
    )
    split_x = pf.random.uniform(rng, 0.4, 0.6)
    margin_x_low, margin_x_high, _ = fit_grid_margins(
        extent_x, bar_length, spacing, min_margin_x, split_x, n_cap=1
    )

    reveal_depth = pf.random.uniform(rng, 0.1, 1.0)
    recess_pct = pf.random.uniform(rng, 0.0, 1.0)

    geom, sill, lightblocker, bar_aliases = cutout_spaced_instances(
        surface=ceiling,
        instance=housing,
        surface_material=ceiling_material,
        spacing=pf.Vector((spacing, spacing, 0)),
        margin_low=pf.Vector((margin_x_low, margin_y_low, 0)),
        margin_high=pf.Vector((margin_x_high, margin_y_high, 0)),
        x_instances_max=1,
        y_instances_max=4,
        canonical_up_axis="Y",
        instance_secondary_axis=(0, 1, 0),
        wall_thickness=reveal_depth,
        recess_pct=recess_pct,
        chamfer=pf.random.clip_gaussian(rng, 0.006, 0.002, 0.004, 0.010),
    )

    bar_ceiling_locs = [np.array(alias.item().location) for alias in bar_aliases]

    approx_room_area = dimensions.x * dimensions.y
    w_per_area = pf.random.clip_gaussian(rng, 1, 1, 0.3, 12.0)
    per_energy = approx_room_area * w_per_area / max(1, len(bar_aliases))

    # shared blackbody temperature, indoor range
    temperature = pf.random.clip_gaussian(rng, 4500, 1000, 2000, 8000)

    # one area lamp per bar, just below the ceiling, facing down
    lights: list[pf.LightObject] = []
    for ceiling_loc in bar_ceiling_locs:
        lamp_energy = per_energy * pf.random.uniform(rng, 0.75, 1.25)
        light = pf.ops.primitives.light.area_lamp(
            shape="RECTANGLE",
            size_x=bar_length,
            size_y=bar_width,
            energy=lamp_energy,
        )
        blackbody = pf.nodes.color.blackbody(temperature=temperature)
        emission = pf.nodes.shader.emission(color=blackbody, strength=lamp_energy)
        pf.nodes.to_light(light, surface=emission)
        light.item().location = (
            ceiling_loc[0],
            ceiling_loc[1],
            ceiling_loc[2] - 0.03,
        )
        lights.append(light)

    lights = pf.control.choice(rng, [(lights, 7.0), ([], 1.0)])
    backs = [lightblocker] if lightblocker is not None else []
    sills = [sill] if sill is not None else []
    return geom, backs, sills, bar_aliases, lights


@pf.tracer.grammar
def ceiling_feature_distribution(
    rng: pf.RNG,
    shape: RoomShapeResult,
    wall_thickness: float = 0.1,
) -> CeilingFeaturesResult:
    vec_pos = pf.nodes.shader.geometry().position

    rng_floor_mat, rng_ceiling_mat, rng_choice, rng_feature = rng.spawn(4)

    floor_mat = floor_material_distribution(rng_floor_mat, vec_pos)
    pf.ops.object.set_material(
        shape.floor,
        surface=floor_mat.surface,
        displacement=floor_mat.displacement,
    )
    pf.ops.modifier.subdivide_surface(shape.floor, levels=8, _skip_apply=True)

    ceiling_mat = ceiling_material_distribution(rng_ceiling_mat, vec_pos)
    pf.ops.object.set_material(
        shape.ceiling,
        surface=ceiling_mat.surface,
        displacement=ceiling_mat.displacement,
    )
    pf.ops.modifier.subdivide_surface(shape.ceiling, levels=8, _skip_apply=True)

    def ceiling_plain_with_lights(rng: pf.RNG):
        ceiling_back = extrude_for_thickness(shape.ceiling, wall_thickness)
        ceiling_back.item().name = "room_wall_back"
        lamp_meshes, lamp_lights = ceiling_light_placement_distribution(
            rng, shape.ceiling, dimensions=shape.dimensions
        )
        return shape.ceiling, [ceiling_back], [], lamp_meshes, lamp_lights

    def ceiling_skylights(rng: pf.RNG):
        return ceiling_skylights_distribution(rng, shape.ceiling, ceiling_mat)

    def ceiling_light_bars(rng: pf.RNG):
        return ceiling_light_bars_distribution(
            rng,
            shape.ceiling,
            ceiling_mat,
            dimensions=shape.dimensions,
        )

    option = pf.control.choice(
        rng_choice,
        [
            (ceiling_plain_with_lights, 3.0),
            (ceiling_skylights, 1.0),
            (ceiling_light_bars, 1.0),
        ],
    )
    ceiling_geom, backs, sills, light_meshes, ceiling_lights = option(rng_feature)

    return CeilingFeaturesResult(
        floor=shape.floor,
        ceiling=ceiling_geom,
        backs=backs,
        sills=sills,
        light_meshes=light_meshes,
        lights=ceiling_lights,
    )


@pf.tracer.grammar
def skirting_distribution(
    rng: pf.RNG,
    walls: list[pf.MeshObject],
) -> list[pf.MeshObject]:
    vec_wall = pf.nodes.shader.coord().uv
    rng_mat, rng_choice, rng_skirt = rng.spawn(3)
    skirt_mat = skirt_material_distribution(rng_mat, vec_wall)
    skirt_option = pf.control.choice(
        rng_choice,
        [(skirting_on_walls_distribution, 0.85), (lambda *_, **__: [], 0.15)],
    )
    return skirt_option(
        rng_skirt,
        walls=walls,
        material=skirt_mat,
    )
