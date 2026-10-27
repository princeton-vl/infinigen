# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from typing import NamedTuple

import numpy as np
import procfunc as pf

from infinigen2.objects import lamp, window
from infinigen2.objects.ceiling_light import light_ceiling_rand
from infinigen2.scenes.placement.distribute import (
    duplicates,
    propagate_modifiers_to_instances,
)
from infinigen2.scenes.room.room_shape import RoomShapeResult
from infinigen2.scenes.room.wall_base import (
    ROOM_SUBSURF_LEVELS,
    extrude_for_thickness,
    fit_grid_margins,
    name_objects,
)
from infinigen2.scenes.room.wall_cutouts import (
    arrange_window_portals,
    cutout_spaced_instances,
)
from infinigen2.shaders.functionality_lists import (
    ceiling_material_rand,
    floor_material_rand,
)
from infinigen2.util import mesh as mesh_util
from infinigen2.util.scene_cleanup import delete_objects
from infinigen2.uv_surface import grid_placement

logger = logging.getLogger(__name__)

__all__ = [
    "CeilingFeaturesResult",
    "CeilingGridResult",
    "ceiling_feature_rand",
    "ceiling_lamp_grid_rand",
    "ceiling_lamp_lights",
    "ceiling_light_bar_grid_rand",
    "ceiling_light_bar_lights_rand",
    "ceiling_light_bars_rand",
    "ceiling_light_placement_rand",
    "ceiling_skylight_grid_rand",
    "ceiling_skylights_rand",
]


class CeilingFeaturesResult(NamedTuple):
    floor: pf.MeshObject
    ceiling: pf.MeshObject
    backs: list[pf.MeshObject]  # ceiling-back lightblocker meshes
    sills: list[pf.MeshObject]  # skylight/bar reveal meshes
    light_meshes: list[pf.MeshObject]  # lamp/skylight/bar housing meshes
    lights: list[pf.LightObject]


class CeilingGridResult(NamedTuple):
    instance: pf.MeshObject
    light: pf.LightObject | None  # positioned relative to each placed instance
    light_offset: tuple[float, float, float]
    footprint: tuple[float, float]
    spacing: tuple[float, float]
    margin_low: tuple[float, float]
    margin_high: tuple[float, float]
    counts: tuple[int, int]
    rotation_offset: tuple[float, float, float]
    reveal_depth: float
    recess_pct: float
    chamfer: float


def _uv_extent(ceiling: pf.MeshObject) -> tuple[float, float]:
    pf.ops.uv.cube_project(ceiling, uv_name="UVMap")
    ceiling_uvs = pf.ops.attr.uv_coords(ceiling)
    extent = ceiling_uvs.max(axis=0) - ceiling_uvs.min(axis=0)
    return float(extent[0]), float(extent[1])


@pf.tracer.grammar
def ceiling_lamp_grid_rand(
    rng: pf.RNG, extent: tuple[float, float], energy: float
) -> CeilingGridResult:
    spacing_x = pf.random.uniform(rng, 1.5, 2.5)
    spacing_y = pf.random.uniform(rng, 1.5, 2.5)
    margin_x = min(pf.random.uniform(rng, 0.4, 1.5), 0.3 * extent[0])
    margin_y = min(pf.random.uniform(rng, 0.4, 1.5), 0.3 * extent[1])

    template_fn = pf.control.choice(
        rng,
        [
            (light_ceiling_rand, 2.5),
            (lamp.lamp_ceiling_shade_rand, 1.0),
        ],
    )
    lamp_template = template_fn(rng, energy=energy)
    lamp_template.mesh.item().name = template_fn.__name__
    mesh_template = lamp_template.mesh

    light_offset = (0.0, 0.0, 0.0)
    if lamp_template.light is not None:
        light = lamp_template.light.item()
        light_offset = tuple(light.location - mesh_template.item().location)
        pf.ops.object.set_transform(lamp_template.light, location=light_offset)

    # bake the template's authored orientation, then centre it in the ceiling plane
    pf.ops.mesh.transform_apply(mesh_template)
    bmin, bmax = pf.ops.attr.bbox_min_max(mesh_template)
    center = (np.array(bmin) + np.array(bmax)) / 2
    pf.ops.object.set_transform(mesh_template, location=(-center[0], -center[1], 0.0))
    pf.ops.mesh.transform_apply(mesh_template)

    lamp_w = np.array(bmax) - np.array(bmin)
    gap_x = max(0.1, spacing_x - lamp_w[0])
    gap_y = max(0.1, spacing_y - lamp_w[1])
    margin_x_low, margin_x_high, n_x = fit_grid_margins(
        extent[0], lamp_w[0], gap_x, margin_x, 0.5
    )
    margin_y_low, margin_y_high, n_y = fit_grid_margins(
        extent[1], lamp_w[1], gap_y, margin_y, 0.5
    )
    # re-hangs the Z-up lamp along the down normal; net identity, so light offsets hold
    lamp_hang = (np.pi / 2, 0.0, -np.pi / 2)
    return CeilingGridResult(
        instance=mesh_template,
        light=lamp_template.light,
        light_offset=light_offset,
        footprint=(float(lamp_w[0]), float(lamp_w[1])),
        spacing=(gap_x, gap_y),
        margin_low=(margin_x_low, margin_y_low),
        margin_high=(margin_x_high, margin_y_high),
        counts=(n_x, n_y),
        rotation_offset=lamp_hang,
        reveal_depth=0.0,
        recess_pct=0.0,
        chamfer=0.0,
    )


def ceiling_lamp_lights(
    light: pf.LightObject | None,
    light_offset: tuple[float, float, float],
    meshes: list[pf.MeshObject],
    energy: float,
) -> list[pf.LightObject]:
    lights: list[pf.LightObject] = []
    if light is not None and meshes:
        locations = np.array([m.item().location for m in meshes])
        lights = duplicates(light, locations + np.asarray(light_offset))
    # rescale to the actual placed count to hit the total energy
    for placed in lights:
        placed.item().data.energy = energy / len(lights)
    return lights


@pf.tracer.grammar
def ceiling_light_placement_rand(
    rng: pf.RNG,
    ceiling: pf.MeshObject,
    dimensions: pf.Vector,
) -> tuple[list[pf.MeshObject], list[pf.LightObject]]:
    lumens = dimensions.x * dimensions.y * pf.random.uniform(rng, 300, 700)
    total_energy = lumens / 177
    grid = ceiling_lamp_grid_rand(rng, _uv_extent(ceiling), total_energy)

    uv_meters = pf.nodes.geo.input_named_attribute(
        name="UVMap", data_type=pf.NodeDataType.FLOAT_VECTOR
    ).attribute
    grid_res = grid_placement.grid_from_spacing(
        uv_surface=ceiling,
        target_uv=uv_meters,
        instance=grid.instance,
        spacing=pf.Vector((*grid.spacing, 0)),
        margin_low=pf.Vector((*grid.margin_low, 0)),
        margin_high=pf.Vector((*grid.margin_high, 0)),
        x_instances_max=grid.counts[0],
        y_instances_max=grid.counts[1],
        rotation_offset=grid.rotation_offset,
    )
    instances = grid_placement.place_instances_on_uv_grid(
        surface=ceiling,
        uv_field=uv_meters,
        grid_mesh=grid_res.grid_mesh,
        query_uv=grid_res.query_uv,
        instance=grid.instance,
        secondary_axis_vector=(0, 1, 0),
        rotation_offset=grid.rotation_offset,
    )
    meshes = pf.nodes.to_aliases(instances)
    propagate_modifiers_to_instances([grid.instance], meshes)
    logger.info(
        "Placed %d ceiling lamps in %.1fx%.1fm room",
        len(meshes),
        dimensions.x,
        dimensions.y,
    )
    if not meshes:
        logger.warning("Ceiling lamp grid produced no lamps")
    lights = ceiling_lamp_lights(grid.light, grid.light_offset, meshes, total_energy)
    templates = [grid.instance, grid.light]
    delete_objects([obj.item() for obj in templates if obj is not None])
    return meshes, lights


@pf.tracer.grammar
def ceiling_skylight_grid_rand(
    rng: pf.RNG, extent: tuple[float, float]
) -> CeilingGridResult:
    # each side must fit the smaller extent (either orientation), plus headroom
    floor_margin = 0.1
    max_side = max(0.45, min(extent) - 2 * floor_margin - 0.1)

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
    window_dimensions = window.window_dimensions_rand(
        rng, width=skylight_width, height=skylight_length
    )
    # skylights never get curtains
    window_result = window.window_from_profile_rand(
        rng, window.rectangular_profile(window_dimensions)
    )
    win = window_result.mesh

    # local +X follows the down-facing normal, so the ceiling footprint is Y x Z
    bmin, bmax = pf.ops.attr.bbox_min_max(win)
    center = (np.array(bmin) + np.array(bmax)) / 2
    pf.ops.object.set_transform(win, location=-center)
    pf.ops.mesh.transform_apply(win)
    if window_result.light is not None:
        pf.ops.object.set_transform(
            window_result.light,
            location=np.array(window_result.light.item().location) - center,
        )
    bmin, bmax = pf.ops.attr.bbox_min_max(win)
    win_dims = (np.array(bmax) - np.array(bmin))[[1, 2]]

    spacing = pf.random.uniform(rng, 0.4, 1.8)
    margin_frac = pf.random.uniform(rng, 0.0, 1.0)

    # fit per-axis counts; min_margin within [floor, (extent-win)/2]
    splits = pf.random.uniform(rng, (0.4,) * 2, (0.6,) * 2)
    fits = []
    for axis, n_cap, split in ((0, 4, splits[0]), (1, 3, splits[1])):
        size = win_dims[axis]
        max_margin = max(floor_margin, (extent[axis] - size) / 2)
        min_margin = floor_margin + (max_margin - floor_margin) * margin_frac
        fit = fit_grid_margins(extent[axis], size, spacing, min_margin, split, n_cap)
        fits.append(fit)

    reveal_depth = pf.random.uniform(rng, 0.1, 1.0)
    recess_pct = pf.random.uniform(rng, 0.0, 1.0)
    chamfer = pf.random.clip_gaussian(rng, 0.006, 0.002, 0.004, 0.010)
    return CeilingGridResult(
        instance=win,
        light=window_result.light,
        light_offset=(0.0, 0.0, 0.0),
        footprint=(float(win_dims[0]), float(win_dims[1])),
        spacing=(spacing, spacing),
        margin_low=(fits[0][0], fits[1][0]),
        margin_high=(fits[0][1], fits[1][1]),
        counts=(fits[0][2], fits[1][2]),
        rotation_offset=(0.0, 0.0, 0.0),
        reveal_depth=reveal_depth,
        recess_pct=recess_pct,
        chamfer=chamfer,
    )


@pf.tracer.grammar
def ceiling_skylights_rand(
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
    grid = ceiling_skylight_grid_rand(rng, _uv_extent(ceiling))
    cutout = cutout_spaced_instances(
        surface=ceiling,
        instance=grid.instance,
        surface_material=ceiling_material,
        spacing=pf.Vector((*grid.spacing, 0)),
        margin_low=pf.Vector((*grid.margin_low, 0)),
        margin_high=pf.Vector((*grid.margin_high, 0)),
        x_instances_max=4,
        y_instances_max=3,
        canonical_up_axis="Y",
        instance_secondary_axis=(0, 1, 0),
        wall_thickness=grid.reveal_depth,
        recess_pct=grid.recess_pct,
        chamfer=grid.chamfer,
    )

    portals: list[pf.LightObject] = []
    if grid.light is not None and cutout.aliases:
        portals = arrange_window_portals(cutout.aliases, grid.instance, grid.light)
    templates = [grid.instance, grid.light, cutout.trim_edges]
    delete_objects([obj.item() for obj in templates if obj is not None])

    backs = [cutout.lightblocker] if cutout.lightblocker is not None else []
    sills = [cutout.sill] if cutout.sill is not None else []
    return cutout.geom, backs, sills, cutout.aliases, portals


@pf.tracer.grammar
def ceiling_light_bar_grid_rand(
    rng: pf.RNG, extent: tuple[float, float], ceiling_material: pf.Material
) -> CeilingGridResult:
    # bars run long along x, thin along y, rows stacked across y
    bar_length = min(pf.random.uniform(rng, 1.5, 4.0), extent[0] - 0.4)
    bar_width = pf.random.uniform(rng, 0.02, 1.0)
    bar_depth = pf.random.uniform(rng, 0.03, 0.06)

    # thin box housing, x/y-centered (symmetric -> flip-invariant)
    cube = mesh_util.box(size=(bar_length, bar_width, bar_depth))
    housing = pf.nodes.to_mesh_object(cube)
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
    margin_y_low, margin_y_high, n_y = fit_grid_margins(
        extent[1], bar_width, spacing, min_margin_y, split_y, n_cap=4
    )
    split_x = pf.random.uniform(rng, 0.4, 0.6)
    margin_x_low, margin_x_high, n_x = fit_grid_margins(
        extent[0], bar_length, spacing, min_margin_x, split_x, n_cap=1
    )

    reveal_depth = pf.random.uniform(rng, 0.1, 1.0)
    recess_pct = pf.random.uniform(rng, 0.0, 1.0)
    chamfer = pf.random.clip_gaussian(rng, 0.006, 0.002, 0.004, 0.010)
    return CeilingGridResult(
        instance=housing,
        light=None,
        light_offset=(0.0, 0.0, 0.0),
        footprint=(bar_length, bar_width),
        spacing=(spacing, spacing),
        margin_low=(margin_x_low, margin_y_low),
        margin_high=(margin_x_high, margin_y_high),
        counts=(n_x, n_y),
        rotation_offset=(np.pi / 2, 0.0, np.pi / 2),
        reveal_depth=reveal_depth,
        recess_pct=recess_pct,
        chamfer=chamfer,
    )


@pf.tracer.grammar
def ceiling_light_bar_lights_rand(
    rng: pf.RNG,
    footprint: tuple[float, float],
    bars: list[pf.MeshObject],
    area: float,
) -> list[pf.LightObject]:
    lumens = area * pf.random.uniform(rng, 300, 700)
    per_energy = lumens / 177 / max(1, len(bars))

    # shared blackbody temperature, indoor range
    temperature = pf.random.clip_gaussian(rng, 4500, 1000, 2000, 8000)

    # one area lamp per bar, just below the ceiling, facing down
    lights: list[pf.LightObject] = []
    energy_mults = pf.random.uniform(rng, (0.75,) * len(bars), (1.25,) * len(bars))
    for bar, energy_mult in zip(bars, energy_mults, strict=True):
        lamp_energy = per_energy * energy_mult
        light = pf.ops.primitives.light.area_lamp(
            shape="RECTANGLE",
            size_x=footprint[0],
            size_y=footprint[1],
            energy=lamp_energy,
        )
        blackbody = pf.nodes.color.blackbody(temperature=temperature)
        emission = pf.nodes.shader.emission(color=blackbody, strength=1.0)
        pf.nodes.to_light(light, surface=emission)
        light.item().location = bar.item().location - pf.Vector((0.0, 0.0, 0.03))
        lights.append(light)
    return lights


@pf.tracer.grammar
def ceiling_light_bars_rand(
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
    grid = ceiling_light_bar_grid_rand(rng, _uv_extent(ceiling), ceiling_material)
    cutout = cutout_spaced_instances(
        surface=ceiling,
        instance=grid.instance,
        surface_material=ceiling_material,
        spacing=pf.Vector((*grid.spacing, 0)),
        margin_low=pf.Vector((*grid.margin_low, 0)),
        margin_high=pf.Vector((*grid.margin_high, 0)),
        x_instances_max=1,
        y_instances_max=4,
        canonical_up_axis="Y",
        instance_secondary_axis=(0, 1, 0),
        wall_thickness=grid.reveal_depth,
        recess_pct=grid.recess_pct,
        chamfer=grid.chamfer,
        rotation_offset=grid.rotation_offset,
    )
    area = dimensions.x * dimensions.y
    lights = ceiling_light_bar_lights_rand(rng, grid.footprint, cutout.aliases, area)
    delete_objects([grid.instance.item(), cutout.trim_edges.item()])
    backs = [cutout.lightblocker] if cutout.lightblocker is not None else []
    sills = [cutout.sill] if cutout.sill is not None else []
    return cutout.geom, backs, sills, cutout.aliases, lights


@pf.tracer.grammar
def ceiling_feature_rand(
    rng: pf.RNG,
    shape: RoomShapeResult,
    wall_thickness: float = 0.1,
) -> CeilingFeaturesResult:
    vec_pos = pf.nodes.shader.geometry().position

    rng_floor_mat, rng_ceiling_mat, rng_choice, rng_feature = rng.spawn(4)

    floor_mat = floor_material_rand(rng_floor_mat, vec_pos)
    pf.ops.object.set_material(
        shape.floor,
        surface=floor_mat.surface,
        displacement=floor_mat.displacement,
    )
    pf.ops.modifier.subdivide_surface(
        shape.floor, levels=ROOM_SUBSURF_LEVELS, _skip_apply=True
    )

    ceiling_mat = ceiling_material_rand(rng_ceiling_mat, vec_pos)
    pf.ops.object.set_material(
        shape.ceiling,
        surface=ceiling_mat.surface,
        displacement=ceiling_mat.displacement,
    )
    pf.ops.modifier.subdivide_surface(
        shape.ceiling, levels=ROOM_SUBSURF_LEVELS, _skip_apply=True
    )

    def ceiling_plain_with_lights(rng: pf.RNG):
        ceiling_back = extrude_for_thickness(shape.ceiling, wall_thickness)
        ceiling_back.item().name = "room_wall_back"
        lamp_meshes, lamp_lights = ceiling_light_placement_rand(
            rng, shape.ceiling, dimensions=shape.dimensions
        )
        return shape.ceiling, [ceiling_back], [], lamp_meshes, lamp_lights

    def ceiling_skylights(rng: pf.RNG):
        return ceiling_skylights_rand(rng, shape.ceiling, ceiling_mat)

    def ceiling_light_bars(rng: pf.RNG):
        return ceiling_light_bars_rand(
            rng,
            shape.ceiling,
            ceiling_mat,
            dimensions=shape.dimensions,
        )

    option = pf.control.choice(
        rng_choice,
        [
            (ceiling_plain_with_lights, 4.0),
            (ceiling_skylights, 1.0),
            (ceiling_light_bars, 1.0),
        ],
    )
    ceiling_geom, backs, sills, light_meshes, ceiling_lights = option(rng_feature)

    name_objects([shape.floor], "room_floor")
    name_objects([ceiling_geom], "room_ceiling")
    name_objects(backs, "room_ceiling_back")
    name_objects(sills, "room_ceiling_sill")
    name_objects(light_meshes, "ceiling_light")
    return CeilingFeaturesResult(
        floor=shape.floor,
        ceiling=ceiling_geom,
        backs=backs,
        sills=sills,
        light_meshes=light_meshes,
        lights=ceiling_lights,
    )
