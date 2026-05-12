from typing import NamedTuple

import numpy as np
import procfunc as pf
from mathutils import Euler

from infinigen_v2.generators.curves.skirting_board_profile import (
    skirting_profile_distribution,
)
from infinigen_v2.generators.objects import lamp, window
from infinigen_v2.generators.objects.ceiling_light import ceiling_light_distribution
from infinigen_v2.generators.scenes.placement_utils import (
    aliases,
    compute_grid_locations,
    duplicates,
)
from infinigen_v2.generators.scenes.room.room_shape import RoomShapeResult
from infinigen_v2.generators.shaders.composites import (
    bricks,
    paint_overlay,
    tiles,
    wood_planks,
)
from infinigen_v2.generators.shaders.masks import cracks
from infinigen_v2.generators.shaders.materials import (
    carpet,
    concrete,
    paint,
    wood_grain,
)
from infinigen_v2.generators.shaders.util.coord import uv_maybe_rotate
from infinigen_v2.generators.util.curve import curve_to_mesh_with_uv
from infinigen_v2.generators.uv_surface import grid_placement


class WallFeatureResult(NamedTuple):
    wall_geom: list[pf.MeshObject]  # wall surfaces (≥1; split options return multiple)
    extras: list[pf.MeshObject]  # wall backs, window instances, paintings, etc.
    storage: list[pf.MeshObject]  # shelf/storage surfaces for downstream filling
    lights: list[pf.LightObject]  # window portals or other attached lights


def extrude_for_thickness(obj: pf.MeshObject, thickness: float) -> pf.MeshObject:
    geo = pf.nodes.geo.object_info(obj).geometry
    geo = pf.nodes.geo.transform(geo, translation=(0, 0, thickness * 0.5))

    # scale to give margin above/below the wall to avoid light leakage
    extruded = pf.nodes.geo.extrude_mesh(geo, offset_scale=-thickness, individual=False)

    result = pf.nodes.to_mesh_object(extruded.mesh)
    result.item().name = obj.item().name + "_thickened"
    return result


def plane_to_posed_canonical_mesh(
    obj: pf.MeshObject,
    up_axis: str = "Z",
) -> pf.MeshObject:
    """
    Instead of location,rotation=0,0,0, convert to meaningful location/rotation but with mesh appearance unchanged

    Assumes planar geometry, i.e. all polygons have the ~same normal
    """
    # Calculate bbox center and mean normal
    bbox_min, bbox_max = pf.ops.attr.bbox_min_max(obj, global_coords=False)
    normals = pf.ops.attr.polygon_normals(obj)

    pos = pf.Vector((np.array(bbox_min) + np.array(bbox_max)) / 2)
    normal = pf.Vector(np.mean(normals, axis=0)).normalized()

    # Create rotation: normal -> +X, constrained to keep up_axis upright
    rot = normal.to_track_quat("X", up_axis).inverted()

    # Apply translation first, then rotation, to move geometry to canonical local space
    pf.ops.object.set_transform(obj, location=-pos)
    pf.ops.mesh.transform_apply(obj)
    pf.ops.object.set_transform(obj, rotation_euler=rot.to_euler())
    pf.ops.mesh.transform_apply(obj)

    # Set final object transform to place it back in world
    pf.ops.object.set_transform(obj, location=pos)
    pf.ops.object.set_transform(obj, rotation_euler=rot.inverted().to_euler())

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
    window_height: float,
    wall_thickness: float = 0.05,
) -> tuple[pf.MeshObject, list[pf.MeshObject], list[pf.MeshObject]]:
    width = window_obj.item().dimensions.x
    wmin, _ = pf.ops.attr.bbox_min_max(window_obj)

    wall_uvs = pf.ops.attr.uv_coords(wall)
    wall_uv_width = wall_uvs[:, 0].max() - wall_uvs[:, 0].min()

    max_margin = min(0.3 * wall_uv_width, 2.0 * width)
    edge_margin = pf.random.uniform(rng, 0.1, max_margin) if max_margin > 0.1 else 0.1
    margin_split = pf.random.clip_gaussian(rng, 0.5, 0.3, 0.05, 0.95)
    margin_low_x = edge_margin * margin_split
    margin_high_x = edge_margin * (1.0 - margin_split)
    margin_bottom = window_height + wmin[1]

    uv_meters = pf.nodes.geo.input_named_attribute(
        name="UVMap", data_type=pf.NodeDataType.FLOAT_VECTOR
    ).attribute

    grid_res = grid_placement.grid_from_spacing(
        uv_surface=wall,
        target_uv=uv_meters,
        instance=window_obj,
        spacing=pf.Vector((spacing, 0, 0)),
        margin_low=pf.Vector((margin_low_x, margin_bottom, 0)),
        margin_high=pf.Vector((margin_high_x, 0, 0)),
        x_instances_max=1000,
        y_instances_max=1,
    )

    faces_res = grid_placement.faces_for_instance_grid_bboxes(
        target_surface=wall,
        target_uv=uv_meters,
        instance=window_obj,
        query_grid=grid_res.grid_mesh,
        instance_uvs=grid_res.query_uv,
        grid_index_x=grid_res.index_x,
        grid_index_y=grid_res.index_y,
        verts_per_instance_x=2,
        verts_per_instance_y=2,
        margin_verts_x=1,
        margin_verts_y=1,
    )

    geom = pf.nodes.geo.separate_geometry(
        faces_res.mesh, selection=faces_res.is_instance_face, domain="FACE"
    ).inverted
    geom = pf.nodes.to_mesh_object(geom)
    pf.ops.object.set_transform(geom, wall.item().location, wall.item().rotation_euler)

    pf.ops.object.set_material(
        geom,
        surface=wall_material.surface,
        displacement=wall_material.displacement,
    )

    wall_thick = extrude_for_thickness(geom, wall_thickness)

    pf.ops.attr.write_attribute(geom, 1.0, "crease_edge", domain="EDGE")
    pf.ops.modifier.subdivide_surface(geom, levels=9, _skip_apply=True)

    instances = grid_placement.place_instances_on_uv_grid(
        surface=wall,
        uv_field=uv_meters,
        grid_mesh=grid_res.grid_mesh,
        query_uv=grid_res.query_uv,
        instance=window_obj,
    )
    window_aliases = pf.nodes.to_aliases(instances)

    wall.item().hide_viewport = True
    wall.item().hide_render = True

    return geom, [wall_thick], window_aliases


@pf.tracer.grammar
def wall_plain_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    wall_material: pf.Material,
    wall_thickness: float = 0.05,
) -> WallFeatureResult:
    wall_thick = extrude_for_thickness(wall, wall_thickness)
    pf.ops.object.set_material(
        wall,
        surface=wall_material.surface,
        displacement=wall_material.displacement,
    )
    pf.ops.modifier.subdivide_surface(wall, levels=10, _skip_apply=True)
    wall = plane_to_posed_canonical_mesh(wall)
    return WallFeatureResult(
        wall_geom=[wall], extras=[wall_thick], storage=[], lights=[]
    )


@pf.tracer.grammar
def wall_windows_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    wall_material: pf.Material,
    window_obj: pf.MeshObject,
    window_portal: pf.LightObject | None,
    window_spacing: float,
    window_height: float,
    wall_thickness: float = 0.05,
) -> WallFeatureResult:
    geom, extras, window_aliases = window_spaced_distribution(
        rng,
        wall,
        window_obj,
        wall_material,
        window_spacing,
        window_height,
        wall_thickness,
    )
    portals = []
    if window_portal is not None and window_aliases:
        portals = arrange_window_portals(window_aliases, window_obj, window_portal)
    geom = plane_to_posed_canonical_mesh(geom)
    return WallFeatureResult(
        wall_geom=[geom], extras=extras + window_aliases, storage=[], lights=portals
    )


@pf.tracer.grammar
def wall_window_extrusion_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    wall_material: pf.Material,
    window_obj: pf.MeshObject,
    window_portal: pf.LightObject | None,
    window_spacing: float,
    window_height: float,
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
    raise NotImplementedError


@pf.tracer.grammar
def wall_shelf_grid_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    wall_material: pf.Material,
    wall_thickness: float = 0.05,
) -> WallFeatureResult:
    raise NotImplementedError


@pf.tracer.grammar
def wall_storage_shelf_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    wall_material: pf.Material,
    wall_thickness: float = 0.05,
) -> WallFeatureResult:
    raise NotImplementedError


@pf.tracer.grammar
def wall_decoration_distribution(
    rng: pf.RNG,
    wall: pf.MeshObject,
    window_obj: pf.MeshObject,
    window_portal: pf.LightObject | None,
    wall_material: pf.Material,
    wall_material_alt: pf.Material,
    window_spacing: float | None = None,
    window_height: float | None = None,
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
            window_height=window_height,
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
            window_height=window_height,
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

    def shelf_grid(rng, wall, wall_material):
        return wall_shelf_grid_distribution(
            rng, wall, wall_material, wall_thickness=wall_thickness
        )

    def storage_shelf(rng, wall, wall_material):
        return wall_storage_shelf_distribution(
            rng, wall, wall_material, wall_thickness=wall_thickness
        )

    option = pf.control.choice(
        rng,
        [
            (plain, 1.0),
            (windows, 2.0),
            (window_extrusion, 0.0),
            (vertical_split, 0.0),
            (horizontal_split, 0.0),
            (painting_grid, 0.0),
            (shelf_grid, 0.0),
            (storage_shelf, 0.0),
        ],
    )
    return option(rng=rng, wall=wall, wall_material=wall_material)


class WallFeaturesResult(NamedTuple):
    wall_planes: list[pf.MeshObject]
    extras: list[pf.MeshObject]  # wall-back thickness mesh + window instances
    storage: list[pf.MeshObject]
    lights: list[pf.LightObject]  # window portals


class CeilingFeaturesResult(NamedTuple):
    floor: pf.MeshObject
    ceiling: pf.MeshObject
    extras: list[pf.MeshObject]  # ceiling-back thickness mesh + ceiling light meshes
    lights: list[pf.LightObject]


@pf.tracer.grammar
def paint_wall_distribution(rng: pf.RNG, vector: pf.ProcNode[pf.Vector]) -> pf.Material:
    displacement_pct = pf.random.uniform(rng, 0.0, 0.8)
    paint_value = pf.random.clip_gaussian(rng, 0.5, 0.4, 0.1, 0.9)
    color = paint.paint_color_distribution(rng, value=paint_value)
    return paint.paint_distribution(
        rng, vector, displacement_pct=displacement_pct, base_color=color
    )


@pf.tracer.grammar
def paint_flaked_distribution(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector]
) -> pf.Material:
    paint_mat = paint_wall_distribution(rng, vector)

    base_material = pf.control.choice(
        rng,
        [
            (wood_planks.wood_planks_distribution, 1.0),
            (concrete.concrete_distribution, 2.0),
            # (tiles.tile_indoor_wall_distribution, 1.0), # svm out of stack space
        ],
    )
    base_material = base_material(rng, vector)

    mask = cracks.cracks_distribution(
        rng,
        vector,
        displacement_a=base_material.displacement,
        displacement_b=paint_mat.displacement,
        height_threshold=0.0,
    )
    return paint_overlay.paint_overlay_distribution(
        rng, vector, material=base_material, paint=paint_mat, mask=mask
    )


@pf.tracer.grammar
def wall_material_distribution(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector]
) -> pf.Material:
    vector = uv_maybe_rotate(rng, vector)
    func = pf.control.choice(
        rng,
        [
            (paint_wall_distribution, 3.0),
            (bricks.bricks_distribution, 2.0),
            (bricks.bricks_paint_distribution, 1.0),
            (bricks.bricks_pristine_distribution, 0.5),
            (concrete.concrete_distribution, 0.5),
            (tiles.tile_indoor_wall_distribution, 1.5),
            (wood_planks.wood_planks_distribution, 1.5),
            (paint_flaked_distribution, 1.0),
        ],
    )
    return func(rng, vector)


@pf.tracer.grammar
def skirt_material_distribution(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector]
) -> pf.Material:
    func = pf.control.choice(
        rng,
        [
            (paint.paint_distribution, 1.0),
            (wood_grain.wood_grain_distribution, 1.0),
        ],
    )
    return func(rng, vector)


@pf.tracer.grammar
def floor_material_distribution(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector]
) -> pf.Material:
    vector = uv_maybe_rotate(rng, vector)
    func = pf.control.choice(
        rng,
        [
            (concrete.concrete_distribution, 1.0),
            (tiles.tile_indoor_ground_distribution, 3.0),
            (wood_planks.wood_planks_distribution, 3.0),
            (carpet.carpet_distribution, 2.0),
        ],
    )
    return func(rng, vector)


@pf.tracer.grammar
def ceiling_material_distribution(
    rng: pf.RNG, vector: pf.ProcNode[pf.Vector]
) -> pf.Material:
    vector = uv_maybe_rotate(rng, vector)
    func = pf.control.choice(
        rng,
        [
            (concrete.concrete_distribution, 1.0),
            (paint.paint_distribution, 3.0),
            (wood_planks.wood_planks_distribution, 0.5),
            (paint_flaked_distribution, 1.0),
        ],
    )
    return func(rng, vector)


@pf.tracer.grammar
def skirting_on_walls_distribution(
    rng: pf.RNG,
    floor_curve: pf.CurveObject,
    room_height: float,
    material: pf.Material,
    profile_curve: pf.CurveObject | None = None,
) -> list[pf.MeshObject]:
    assert isinstance(floor_curve, pf.CurveObject), floor_curve

    if profile_curve is None:
        profile_curve = skirting_profile_distribution(rng)
    profile_curve_geo = pf.nodes.geo.object_info(profile_curve).geometry

    floor_curve_node = pf.nodes.geo.object_info(floor_curve).geometry

    ceiling_curve_node = pf.nodes.geo.transform(
        geometry=floor_curve_node,
        translation=(0, 0, room_height),
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
    result.light.item().parent = None
    result.light.item().location = (0.0, 0.0, -height)
    return result


@pf.tracer.grammar
def ceiling_light_placement_distribution(
    rng: pf.RNG,
    dimensions: pf.Vector,
) -> tuple[list[pf.MeshObject], list[pf.LightObject]]:
    approx_room_area = dimensions.x * dimensions.y
    w_per_area = pf.random.clip_gaussian(rng, 0.9, 0.4, 0.2, 2.0)
    total_energy = approx_room_area * w_per_area

    spacing = pf.Vector(
        (
            pf.random.uniform(rng, 1.5, 2.5),
            pf.random.uniform(rng, 1.5, 2.5),
            1.0,
        )
    )
    room_margins = pf.Vector(
        (
            pf.random.uniform(rng, 0.05, 0.15) * dimensions.x,
            pf.random.uniform(rng, 0.05, 0.15) * dimensions.y,
            0.0,
        )
    )
    box_min = pf.Vector((room_margins.x, room_margins.y, dimensions.z + 0.02))
    box_max = pf.Vector(
        (
            dimensions.x - room_margins.x,
            dimensions.y - room_margins.y,
            dimensions.z + 0.02,
        )
    )

    locations = compute_grid_locations(box_min, box_max, spacing)
    per_energy = total_energy / len(locations)

    template_fn = pf.control.choice(
        rng,
        [
            (ceiling_light_distribution, 1.0),
            (ceiling_shade_lamp_template, 1.0),
        ],
    )
    lamp_template = template_fn(rng, energy=per_energy)
    lamp_template.mesh.item().name = template_fn.__name__

    meshes = aliases(lamp_template.mesh, locations)

    if lamp_template.light is None:
        return meshes, []

    offset = np.array(
        lamp_template.light.item().location - lamp_template.mesh.item().location
    )
    light_locations = locations + offset
    lights = duplicates(lamp_template.light, light_locations)
    return meshes, lights


@pf.tracer.grammar
def wall_feature_distribution(
    rng: pf.RNG,
    shape: RoomShapeResult,
    wall_thickness: float = 0.1,
) -> WallFeaturesResult:
    vec_wall = pf.nodes.shader.uv_map(uv_map="UVMap")

    wall_material_1 = wall_material_distribution(rng, vec_wall)
    wall_material_2 = wall_material_distribution(rng, vec_wall)

    wall_back = extrude_for_thickness(shape.walls, wall_thickness)

    pf.ops.object.set_material(
        shape.walls,
        surface=wall_material_1.surface,
        displacement=wall_material_1.displacement,
    )
    pf.ops.modifier.subdivide_surface(shape.walls, levels=9, _skip_apply=True)

    window_height_pct = pf.random.clip_gaussian(rng, 0.75, 0.2, 0.6, 0.9)
    window_height = shape.dimensions.z * window_height_pct
    window_width = pf.random.uniform(rng, 1.0, 2.0)
    window_dimensions = window.window_dimensions_distribution(
        rng, width=window_width, height=window_height
    )
    window_result = window.window_distribution(rng, dimensions=window_dimensions)
    window_obj = window_result.mesh
    window_portal = window_result.light

    _width, _height, _depth = window_obj.item().dimensions
    wmin, _wmax = pf.ops.attr.bbox_min_max(window_obj)
    margin_total = shape.dimensions.z - _height
    window_bottom_pct = pf.random.clip_gaussian(rng, 0.7, 0.15, 0.35, 0.85)
    window_bottom = window_bottom_pct * margin_total - wmin[1]
    window_spacing = pf.random.uniform(rng, 0.1, 0.25) * _width

    wall_planes = []
    extras = [wall_back]
    storage = []
    lights = []
    for wall in shape.flat_walls:
        mat = pf.control.choice(
            rng,
            [(wall_material_1, 3), (wall_material_2, 1)],
        )
        result = wall_decoration_distribution(
            rng,
            wall,
            window_obj,
            window_portal,
            wall_material=mat,
            wall_material_alt=wall_material_2,
            window_spacing=window_spacing,
            window_height=window_bottom,
            wall_thickness=wall_thickness,
        )
        wall_planes.extend(result.wall_geom)
        extras.extend(result.extras)
        storage.extend(result.storage)
        lights.extend(result.lights)

    return WallFeaturesResult(
        wall_planes=wall_planes,
        extras=extras,
        storage=storage,
        lights=lights,
    )


@pf.tracer.grammar
def ceiling_feature_distribution(
    rng: pf.RNG,
    shape: RoomShapeResult,
    wall_thickness: float = 0.1,
) -> CeilingFeaturesResult:
    vec_pos = pf.nodes.shader.geometry().position

    floor_mat = floor_material_distribution(rng, vec_pos)
    pf.ops.object.set_material(
        shape.floor,
        surface=floor_mat.surface,
        displacement=floor_mat.displacement,
    )
    pf.ops.modifier.subdivide_surface(shape.floor, levels=9, _skip_apply=True)

    ceiling_mat = ceiling_material_distribution(rng, vec_pos)
    pf.ops.object.set_material(
        shape.ceiling,
        surface=ceiling_mat.surface,
        displacement=ceiling_mat.displacement,
    )
    pf.ops.modifier.subdivide_surface(shape.ceiling, levels=9, _skip_apply=True)

    ceiling_back = extrude_for_thickness(shape.ceiling, wall_thickness)

    ceiling_meshes, ceiling_lights = ceiling_light_placement_distribution(
        rng, dimensions=shape.dimensions
    )

    return CeilingFeaturesResult(
        floor=shape.floor,
        ceiling=shape.ceiling,
        extras=[ceiling_back] + ceiling_meshes,
        lights=ceiling_lights,
    )


@pf.tracer.grammar
def skirting_distribution(
    rng: pf.RNG,
    shape: RoomShapeResult,
) -> list[pf.MeshObject]:
    vec_wall = pf.nodes.shader.uv_map(uv_map="UVMap")
    skirt_mat = skirt_material_distribution(rng, vec_wall)
    skirt_option = pf.control.choice(
        rng,
        [(skirting_on_walls_distribution, 0.85), (lambda *_, **__: [], 0.15)],
    )
    return skirt_option(
        rng,
        floor_curve=pf.nodes.to_curve_object(shape.edge_curve.curve),
        room_height=shape.dimensions.z,
        material=skirt_mat,
    )
