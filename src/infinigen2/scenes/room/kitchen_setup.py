# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import math
from typing import Callable, NamedTuple

import numpy as np
import procfunc as pf
import shapely
from procfunc.nodes import types as t

from infinigen2.objects import bathroom_hardware, chair, handles, sink, storage, tap
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import keep_non_colliding, place_surrounding
from infinigen2.scenes.placement.distribute import propagate_modifiers_to_instances
from infinigen2.scenes.placement.retry import repeat_attempts
from infinigen2.scenes.placement.snap import snap_to_plane
from infinigen2.scenes.room.dining_table_setup import chairs_on_edge
from infinigen2.scenes.room.room_shape import room_shape_rand
from infinigen2.scenes.room.wall_base import plane_to_posed_canonical_mesh
from infinigen2.scenes.setup_utils import (
    back_face_grounded,
    center_inside,
    inset_floor_point_rand,
    retry_place,
)
from infinigen2.shaders.base_materials import metal_brushed
from infinigen2.shaders.functionality_lists import (
    cabinet_material_rand,
    kitchen_counter_rand,
)
from infinigen2.util import mesh as mesh_util
from infinigen2.util.scene_cleanup import delete_object
from infinigen2.uv_surface import cutouts, grid_placement

__all__ = [
    "KitchenModuleArrayResult",
    "KitchenSetupResult",
    "island_rand",
    "island_setup_rand",
    "kitchen_cabinets_wall_setup_rand",
    "kitchen_module_array_rand",
    "kitchen_setup_rand",
]


class KitchenSetupResult(NamedTuple):
    all_objects: list[pf.MeshObject]
    colliders: ccol.CollisionSet
    countertops: list[pf.MeshObject]
    storage_containers: list[pf.MeshObject]
    supports: list[pf.MeshObject]
    storages: list[pf.MeshObject]


class KitchenModuleArrayResult(NamedTuple):
    mesh: pf.MeshObject
    all_objects: list[pf.MeshObject]
    cabinets: list[pf.MeshObject]
    plinths: list[pf.MeshObject]
    sinks: list[pf.MeshObject]
    taps: list[pf.MeshObject]


class _SinkTop(NamedTuple):
    top: pf.ProcNode[pf.MeshObject]
    sinks: list[pf.ProcNode[t.Instances]]
    taps: list[pf.ProcNode[t.Instances]]


class _Box(NamedTuple):
    mesh: pf.MeshObject


class _SinkCut(NamedTuple):
    top: pf.ProcNode[pf.MeshObject]
    sinks: pf.ProcNode[t.Instances]
    taps: pf.ProcNode[t.Instances]


class _RowGeo(NamedTuple):
    cabinets: pf.ProcNode[t.Instances]
    plinth: pf.ProcNode[pf.MeshObject]


def _module_array_box(
    back_overhang: float, depth: float, length: float, height: float
) -> _Box:
    box = mesh_util.box(
        size=(back_overhang + depth, length, height),
        location=(-back_overhang, 0.0, 0.0),
        anchor=(0.0, 0.0, 0.0),
    )
    return _Box(pf.nodes.to_mesh_object(box))


def _setup_result(
    arrays: list[KitchenModuleArrayResult],
    uppers: list[pf.MeshObject],
    others: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> KitchenSetupResult:
    countertops, cabinets, sinks = [], [], []
    all_objects = uppers + others
    for array in arrays:
        countertops.append(array.mesh)
        cabinets += array.cabinets
        sinks += array.sinks
        all_objects += array.all_objects
    return KitchenSetupResult(
        all_objects=all_objects,
        colliders=colliders,
        countertops=countertops,
        storage_containers=sinks + uppers,
        supports=countertops + sinks,
        storages=countertops + cabinets + uppers,
    )


def _all_clear(objs: list[pf.MeshObject], colliders: ccol.CollisionSet) -> bool:
    return not any(ccol.intersection_test(colliders, obj) for obj in objs)


def _wall_length(wall: pf.MeshObject) -> float:
    bmin, bmax = pf.ops.attr.bbox_min_max(wall, global_coords=False)
    return float(bmax[1] - bmin[1])


def _module_count_rand(rng: pf.RNG, least: int, most: int) -> int:
    return max(least, round(most * (1.0 - pf.random.uniform(rng, 0.0, 1.0) ** 2)))


def _modules_rand(
    rng: pf.RNG,
    count: int,
    weighted_fns: list[tuple[Callable[..., storage.StorageResult], float]],
    dimensions: pf.Vector,
    material: pf.Material,
    handle: pf.MeshObject,
    first: list[pf.MeshObject],
    name: str,
) -> pf.Collection:
    modules = list(first)
    for r in rng.spawn(count):
        rng_choice, rng_module = r.spawn(2)
        module_fn = pf.control.choice(rng_choice, weighted_fns)
        module = module_fn(
            rng_module,
            dimensions=dimensions,
            frame_material=material,
            handle=handle.clone(),
        )
        modules.append(module.mesh)
    for index, module in enumerate(modules):
        module.item().name = f"{name}_{index:02d}"
    return pf.types.Collection(modules, name=name)


def _handle_rand(rng: pf.RNG) -> pf.MeshObject:
    rng_choice, rng_handle = rng.spawn(2)
    handle_fn = pf.control.choice(
        rng_choice,
        [(handles.handle_bar_pull_rand, 3.0), (handles.handle_knob_rand, 1.0)],
    )
    return handle_fn(rng_handle).mesh


def _base_modules_rand(
    rng: pf.RNG,
    count: int,
    dimensions: pf.Vector,
    material: pf.Material,
    handle: pf.MeshObject,
) -> pf.Collection:
    rng_sink_base, rng_modules = rng.spawn(2)
    sink_base = storage.storage_sink_base_rand(
        rng_sink_base, dimensions, material, handle.clone()
    )
    return _modules_rand(
        rng_modules,
        count,
        [
            (storage.storage_drawer_door_rand, 2.0),
            (storage.storage_composite_rand, 1.5),
        ],
        dimensions,
        material,
        handle,
        [sink_base.mesh],
        "kitchen_base",
    )


def _upper_shelf_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    frame_material: pf.Material,
    handle: pf.MeshObject,
) -> storage.StorageResult:
    rng_board, rng_shelf = rng.spawn(2)
    board = pf.random.uniform(rng_board, 0.016, 0.019)
    n_shelves = math.ceil(dimensions.z / 0.35)
    return storage.storage_cell_shelf_rand(
        rng_shelf, dimensions, 1, n_shelves, frame_material, 0.008, board, board, board
    )


@pf.nodes.node_function
def _module_row(
    modules: t.SocketOrVal[pf.Collection],
    count: t.SocketOrVal[int],
    width: t.SocketOrVal[float],
    start_y: t.SocketOrVal[float],
    z: t.SocketOrVal[float],
    index_min: t.SocketOrVal[int],
    index_max: t.SocketOrVal[int],
    seed: t.SocketOrVal[int],
    sink_start: t.SocketOrVal[int],
    sink_end: t.SocketOrVal[int],
) -> pf.ProcNode[t.Instances]:
    points = pf.nodes.geo.mesh_line(
        start_location=pf.nodes.math.combine_xyz(y=start_y, z=z),
        offset=pf.nodes.math.combine_xyz(y=width),
        count=count,
    )
    index = pf.nodes.geo.input_index()
    is_sink = pf.nodes.func.boolean_and(
        a=pf.nodes.func.greater_equal(a=index, b=sink_start),
        b=pf.nodes.func.less_than(a=index, b=sink_end),
    )
    random_index = pf.nodes.func.random_value(
        min=index_min, max=index_max, seed=seed, data_type=pf.NodeDataType.INT
    )
    return pf.nodes.geo.instance_on_points(
        points=points,
        instance=pf.nodes.geo.collection_info(modules, separate_children=True),
        pick_instance=True,
        instance_index=pf.nodes.func.switch(
            switch=is_sink, a=random_index, b=0, data_type=pf.NodeDataType.INT
        ),
    )


@pf.nodes.node_function
def _countertop_sheet(
    x_min: t.SocketOrVal[float],
    x_max: t.SocketOrVal[float],
    length: t.SocketOrVal[float],
    z: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    sheet = pf.nodes.geo.transform(
        geometry=pf.nodes.geo.mesh_grid(
            size_x=x_max - x_min, size_y=length, vertices_x=2, vertices_y=2
        ).mesh,
        translation=pf.nodes.math.combine_xyz(
            x=(x_min + x_max) * 0.5, y=length * 0.5, z=z
        ),
    )
    position = pf.nodes.geo.input_position()
    return pf.nodes.geo.store_named_attribute(
        geometry=sheet,
        name="UVMap",
        value=pf.nodes.math.combine_xyz(x=position.y, y=position.x * -1.0),
        domain="CORNER",
        data_type="FLOAT2",
    )


@pf.nodes.node_function
def _sink_cut(
    sheet: pf.ProcNode[pf.MeshObject],
    sink_template: pf.ProcNode[pf.MeshObject],
    footprint: pf.ProcNode[pf.MeshObject],
    tap_template: pf.ProcNode[pf.MeshObject],
    first_uv: t.SocketOrVal[pf.Vector],
    last_uv: t.SocketOrVal[pf.Vector],
    tap_uv: t.SocketOrVal[pf.Vector],
    count: t.SocketOrVal[int],
    seat: t.SocketOrVal[float],
    tap_seat: t.SocketOrVal[float],
) -> _SinkCut:
    uv_field = pf.nodes.geo.input_named_attribute(
        name="UVMap", data_type=pf.NodeDataType.FLOAT_VECTOR
    ).attribute
    # sinks are authored Z-up; the cutout grid frame puts the surface normal on +X
    cut = cutouts.uv_cutout_instances(
        surface=sheet,
        uv_field=uv_field,
        grid=grid_placement.grid_between(first_uv, last_uv, count),
        instance=sink_template,
        footprint=footprint,
        rotation_offset=(0.0, math.pi / 2, 0.0),
        secondary_axis_vector=(-1.0, 0.0, 0.0),
        normal_offset=seat,
    )
    tap_grid = grid_placement.grid_between(first_uv + tap_uv, last_uv + tap_uv, count)
    taps = grid_placement.place_instances_on_uv_grid(
        surface=sheet,
        uv_field=uv_field,
        grid_mesh=tap_grid.grid_mesh,
        query_uv=tap_grid.query_uv,
        instance=tap_template,
        secondary_axis_vector=(-1.0, 0.0, 0.0),
        rotation_offset=(0.0, math.pi / 2, 0.0),
        normal_offset=tap_seat,
    )
    top = pf.nodes.geo.merge_by_distance(cut.holed, distance=0.001)
    return _SinkCut(top=top, sinks=cut.instances, taps=taps)


@pf.nodes.node_function
def _countertop_slab(
    top: pf.ProcNode[pf.MeshObject],
    thickness: t.SocketOrVal[float],
    material: t.SocketOrVal[pf.Material],
) -> pf.ProcNode[pf.MeshObject]:
    # the remeshed cutout surface has no fixed winding, so face it up first
    top = pf.nodes.geo.flip_faces(top, selection=pf.nodes.geo.input_normal().z < 0.0)
    shell = pf.nodes.geo.extrude_mesh(
        mesh=top, offset_scale=thickness, individual=False
    )
    slab = pf.nodes.geo.merge_by_distance(
        pf.nodes.geo.join_geometry([shell.mesh, pf.nodes.geo.flip_faces(top)]),
        distance=0.0001,
    )
    slab = mesh_util.crease_sharp(mesh_util.metric_box_uv(slab), threshold_degrees=30.0)
    return pf.nodes.geo.set_material(geometry=slab, material=material)


def _sink_top_rand(
    rng: pf.RNG,
    sheet: pf.ProcNode[pf.MeshObject],
    start: int,
    n_sinks: int,
    module_width: float,
    counter_depth: float,
    thickness: float,
) -> _SinkTop:
    rng_params, rng_sink, rng_tap = rng.spawn(3)
    outer_y = module_width * pf.random.uniform(rng_params, 0.62, 0.82)
    outer_x = counter_depth * pf.random.uniform(rng_params, 0.72, 0.85)
    margin = outer_y * pf.random.uniform(rng_params, 0.04, 0.07)
    tap_margin = outer_x * pf.random.uniform(rng_params, 0.2, 0.25)
    bowl = pf.Vector((outer_x - margin - tap_margin, outer_y - margin, 0.0))
    upper_height = pf.random.uniform(rng_params, 0.18, 0.25)
    lower_height = pf.random.uniform(rng_params, 0.0, 0.01)
    sink_obj = sink.sink_rand(
        rng_sink,
        width=bowl.y,
        depth=bowl.x,
        upper_height=upper_height,
        lower_height=lower_height,
        margin=margin,
        water_tap_margin=tap_margin,
    ).mesh

    shift = (tap_margin + margin) / 2.56
    rim_back = shift - (outer_x + tap_margin) * 0.5
    rim_top = upper_height - lower_height + 0.01
    origin_x = (counter_depth - outer_x) * 0.5 - rim_back
    seat = thickness + 0.003 - rim_top
    cut = _sink_cut(
        sheet=sheet,
        sink_template=sink_obj,
        footprint=mesh_util.box(
            size=(bowl.x * 1.01, bowl.y * 1.01, 0.05), location=(shift, 0.0, 0.0)
        ),
        tap_template=tap.tap_rand(rng_tap).mesh,
        first_uv=pf.Vector(((start + 0.5) * module_width, -origin_x, 0.0)),
        last_uv=pf.Vector(((start + n_sinks - 0.5) * module_width, -origin_x, 0.0)),
        tap_uv=pf.Vector((0.0, -(rim_back + 0.05), 0.0)),
        count=n_sinks,
        seat=seat,
        tap_seat=seat + rim_top - 0.01,
    )
    return _SinkTop(cut.top, [cut.sinks], [cut.taps])


def _cabinet_row_geo(
    modules: pf.Collection,
    count: int,
    n_modules: int,
    module_width: float,
    plinth: pf.Vector,
    cabinet_material: pf.Material,
    seed: int,
    sink_start: int,
    sink_end: int,
) -> _RowGeo:
    cabinets = _module_row(
        modules,
        count,
        module_width,
        0.0,
        plinth.z,
        1,
        n_modules - 1,
        seed,
        sink_start,
        sink_end,
    )
    base = mesh_util.box(
        size=(plinth.x, count * module_width, plinth.z), anchor=(0.0, 0.0, 0.0)
    )
    base = pf.nodes.geo.set_material(geometry=base, material=cabinet_material)
    return _RowGeo(cabinets, base)


@pf.tracer.grammar
def kitchen_module_array_rand(
    rng: pf.RNG,
    pose: pf.Matrix | None = None,
    count: int | None = None,
    n_sinks: int | None = None,
    back_overhang: float = 0.0,
    modules: pf.Collection | None = None,
    n_modules: int | None = None,
    module_width: float | None = None,
    counter_depth: float | None = None,
    height: float | None = None,
    thickness: float | None = None,
    plinth: pf.Vector | None = None,
    cabinet_material: pf.Material | None = None,
    countertop_material: pf.Material | None = None,
) -> KitchenModuleArrayResult:
    """A straight run of base cabinets under one countertop, `n_sinks` of them
    sink bases, built along +y from `pose` with its back at x=0. `modules[0]` is
    the sink base and `modules[1:n_modules]` the other cabinet modules."""
    rng_top_choice, rng_top, rng_params, rng_defaults = rng.spawn(4)
    rng_dims, rng_cabinet_mat, rng_counter_mat, rng_handle, rng_modules = (
        rng_defaults.spawn(5)
    )
    vec = pf.nodes.shader.coord().uv
    if pose is None:
        pose = pf.Matrix.Identity(4)
    if count is None:
        count = pf.random.randint(rng_dims, 2, 8)
    if n_sinks is None:
        n_sinks = pf.random.randint(rng_dims, 0, 2)
    if module_width is None:
        module_width = pf.random.uniform(rng_dims, 0.4, 0.9)
    if counter_depth is None:
        counter_depth = pf.random.uniform(rng_dims, 0.575, 0.64)
    if height is None:
        height = pf.random.uniform(rng_dims, 0.86, 0.94)
    if thickness is None:
        thickness = pf.random.uniform(rng_dims, 0.02, 0.045)
    if plinth is None:
        plinth = pf.Vector((counter_depth - 0.1, 0.0, 0.12))
    if cabinet_material is None:
        cabinet_material = cabinet_material_rand(rng_cabinet_mat, vec)
    if countertop_material is None:
        countertop_material = kitchen_counter_rand(rng_counter_mat, vec)
    if modules is None:
        n_modules = pf.random.randint(rng_dims, 2, 5)
        modules = _base_modules_rand(
            rng_modules,
            n_modules - 1,
            pf.Vector(
                (counter_depth - 0.03, module_width, height - thickness - plinth.z)
            ),
            cabinet_material,
            _handle_rand(rng_handle),
        )
    length = count * module_width
    start = round(pf.random.uniform(rng_params, 0.2, 0.8) * (count - n_sinks))
    seed = pf.random.randint(rng_params, 0, 2**20)

    sheet = _countertop_sheet(-back_overhang, counter_depth, length, height - thickness)
    top_fn = pf.control.choice(
        rng_top_choice,
        [(lambda _r, sheet, *_: _SinkTop(sheet, [], []), 1.0), (_sink_top_rand, 1.0)],
        chosen_idx=int(n_sinks > 0),
    )
    top = top_fn(rng_top, sheet, start, n_sinks, module_width, counter_depth, thickness)
    slab = _countertop_slab(top.top, thickness, countertop_material)
    countertop = pf.nodes.to_mesh_object(slab)

    row = _cabinet_row_geo(
        modules,
        count,
        n_modules,
        module_width,
        plinth,
        cabinet_material,
        seed,
        start,
        start + n_sinks,
    )
    cabinets = pf.nodes.to_aliases(row.cabinets)
    plinth_obj = pf.nodes.to_mesh_object(row.plinth)
    sinks = [obj for node in top.sinks for obj in pf.nodes.to_aliases(node)]
    taps = [obj for node in top.taps for obj in pf.nodes.to_aliases(node)]
    all_objects = [countertop, plinth_obj] + cabinets + sinks + taps
    for obj in [countertop] + cabinets:
        pf.ops.modifier.subdivide_surface(obj, levels=2, _skip_apply=True)
    for obj in all_objects:
        obj.item().matrix_world = pose @ obj.item().matrix_world
        obj.item().name = "kitchen_module_array"
    return KitchenModuleArrayResult(
        mesh=countertop,
        all_objects=all_objects,
        cabinets=cabinets,
        plinths=[plinth_obj],
        sinks=sinks,
        taps=taps,
    )


def island_rand(
    rng: pf.RNG,
    pose: pf.Matrix,
    count: int,
    n_sinks: int,
    modules: pf.Collection,
    n_modules: int,
    module_width: float,
    counter_depth: float,
    height: float,
    thickness: float,
    plinth: pf.Vector,
    cabinet_material: pf.Material,
    countertop_material: pf.Material,
) -> KitchenModuleArrayResult:
    """A double-sided array: a front row under the countertop and a back row facing
    the other way, both inside one countertop two cabinets deep."""
    rng_front, rng_seed = rng.spawn(2)
    front = kitchen_module_array_rand(
        rng_front,
        pose,
        count,
        n_sinks,
        counter_depth,
        modules,
        n_modules,
        module_width,
        counter_depth,
        height,
        thickness,
        plinth,
        cabinet_material,
        countertop_material,
    )

    row = _cabinet_row_geo(
        modules,
        count,
        n_modules,
        module_width,
        plinth,
        cabinet_material,
        pf.random.randint(rng_seed, 0, 2**20),
        0,
        0,
    )
    cabinets = pf.nodes.to_aliases(row.cabinets)
    plinths = [pf.nodes.to_mesh_object(row.plinth)]
    shift = pf.Matrix.Translation((0.0, count * module_width, 0.0))
    back_pose = pose @ shift @ pf.Matrix.Rotation(math.pi, 4, "Z")
    for obj in cabinets:
        pf.ops.modifier.subdivide_surface(obj, levels=2, _skip_apply=True)
    for obj in cabinets + plinths:
        obj.item().matrix_world = back_pose @ obj.item().matrix_world
        obj.item().name = "kitchen_island"
    return front._replace(
        all_objects=front.all_objects + cabinets + plinths,
        cabinets=front.cabinets + cabinets,
        plinths=front.plinths + plinths,
    )


def _wall_fraction(
    wall: pf.MeshObject, length: float, frac: float, overlap: float
) -> float:
    full = _wall_length(wall)
    inside = full - 2.0 * overlap - 0.02
    return (overlap + 0.01 + frac * (inside - length)) / (full - length)


def _snap_back_to_wall(
    rng: pf.RNG,
    box: _Box,
    wall: pf.MeshObject,
    anchors: list[pf.MeshObject],
    overlap: float,
    length: float,
) -> None:
    del anchors
    snap_to_plane(
        box.mesh,
        wall,
        placement=_wall_fraction(
            wall, length, np.clip(pf.random.uniform(rng, -0.5, 1.5), 0.0, 1.0), overlap
        ),
        child_side="back",
        parent_side="front",
        margin=0.003,
    )


def _snap_end_to_array(
    rng: pf.RNG,
    box: _Box,
    wall: pf.MeshObject,
    anchors: list[pf.MeshObject],
    overlap: float,
    length: float,
) -> None:
    """Butt one end of `box` against the front of an anchor's end, turning an L."""
    del wall, overlap, length
    rng_side, rng_end, rng_anchor = rng.spawn(3)
    side = pf.control.choice(rng_side, [("left", 1.0), ("right", 1.0)])
    end = pf.control.choice(rng_end, [(0.0, 1.0), (1.0, 1.0)])
    snap_to_plane(
        box.mesh,
        rng_anchor.choice(anchors),
        placement=end,
        child_side=side,
        parent_side="front",
        margin=0.002,
    )


def _snap_module_array_rand(
    rng: pf.RNG,
    box: _Box,
    wall: pf.MeshObject,
    anchors: list[pf.MeshObject],
    overlap: float,
    length: float,
) -> None:
    rng_choice, rng_snap = rng.spawn(2)
    snap_fn = pf.control.choice(
        rng_choice, [(_snap_back_to_wall, 1.0), (_snap_end_to_array, 1.0)]
    )
    snap_fn(rng_snap, box, wall, anchors, overlap, length)


def _upper_row_attempt_rand(
    rng: pf.RNG,
    pose: pf.Matrix,
    length: float,
    count: int,
    modules: pf.Collection,
    n_modules: int,
    module_width: float,
    bottom: float,
    colliders: ccol.CollisionSet,
) -> list[pf.MeshObject] | None:
    rng_start, rng_seed = rng.spawn(2)
    frac = np.clip(pf.random.uniform(rng_start, -0.5, 1.5), 0.0, 1.0)
    start = frac * (length - count * module_width)
    seed = pf.random.randint(rng_seed, 0, 2**20)
    row = _module_row(
        modules, count, module_width, start, bottom, 0, n_modules - 1, seed, 0, 0
    )
    uppers = pf.nodes.to_aliases(row)
    for obj in uppers:
        pf.ops.modifier.subdivide_surface(obj, levels=2, _skip_apply=True)
        obj.item().matrix_world = pose @ obj.item().matrix_world
        obj.item().name = "kitchen_setup"
    if not _all_clear(uppers, colliders):
        return None
    return uppers


def _uppers_rand(
    rng: pf.RNG,
    pose: pf.Matrix,
    length: float,
    modules: pf.Collection,
    n_modules: int,
    module_width: float,
    bottom: float,
    colliders: ccol.CollisionSet,
) -> list[pf.MeshObject]:
    rng_count, rng_attempts = rng.spawn(2)
    array_count = round(length / module_width)
    least = min(math.ceil(0.8 / module_width), array_count)
    count = _module_count_rand(rng_count, least, array_count)
    uppers = repeat_attempts(
        _upper_row_attempt_rand,
        rng_attempts,
        4,
        pose,
        length,
        count,
        modules,
        n_modules,
        module_width,
        bottom,
        colliders,
    )
    if uppers is None:
        return []
    return uppers


@pf.tracer.grammar
def kitchen_cabinets_wall_setup_rand(
    rng: pf.RNG,
    walls: list[pf.MeshObject],
    sinks_per_wall: list[int],
    base_modules: pf.Collection,
    n_base: int,
    upper_modules: pf.Collection,
    n_upper: int,
    module_width: float,
    counter_depth: float,
    height: float,
    thickness: float,
    plinth: pf.Vector,
    upper_bottom: float,
    cabinet_material: pf.Material,
    countertop_material: pf.Material,
    colliders: ccol.CollisionSet,
    wall_overlap: float,
) -> KitchenSetupResult:
    """One array of base cabinets with uppers above it per wall, each snapped back to
    its wall or with one end against an earlier array's front to turn an L."""
    arrays: list[KitchenModuleArrayResult] = []
    uppers: list[pf.MeshObject] = []
    boxes: list[pf.MeshObject] = []

    for wall, n_sinks, r in zip(
        walls, sinks_per_wall, rng.spawn(len(walls)), strict=True
    ):
        rng_count, rng_array, rng_place, rng_uppers = r.spawn(4)
        corners = counter_depth * min(len(arrays), 2)
        inside = _wall_length(wall) - 2.0 * wall_overlap - 0.02 - corners
        most = math.floor(inside / module_width)
        count = _module_count_rand(rng_count, min(2, most), most)
        length = count * module_width
        box = _module_array_box(0.0, counter_depth, length, height)

        backing = ccol.collision_set([wall])
        grounded = lambda mesh: back_face_grounded(mesh, backing, margin=0.012)  # noqa: E731, B023
        placed = retry_place(
            rng_place,
            box,
            colliders,
            _snap_module_array_rand,
            attempts=12,
            accept_fn=grounded,
            wall=wall,
            anchors=boxes + [wall],
            overlap=wall_overlap,
            length=length,
        )
        if placed is None:
            continue
        boxes.append(box.mesh)
        pose = box.mesh.item().matrix_world.copy()

        array = kitchen_module_array_rand(
            rng_array,
            pose,
            count,
            n_sinks,
            0.0,
            base_modules,
            n_base,
            module_width,
            counter_depth,
            height,
            thickness,
            plinth,
            cabinet_material,
            countertop_material,
        )
        colliders = ccol.collision_set(
            colliders.objs + array.all_objects, cache=colliders
        )
        arrays.append(array)

        array_uppers = _uppers_rand(
            rng_uppers,
            pose,
            length,
            upper_modules,
            n_upper,
            module_width,
            upper_bottom,
            colliders,
        )
        colliders = ccol.collision_set(colliders.objs + array_uppers, cache=colliders)
        uppers += array_uppers

    for box in boxes:
        delete_object(box.item())
    return _setup_result(arrays, uppers, [], colliders)


def _place_island(
    rng: pf.RNG,
    box: _Box,
    bbox_min: pf.Vector,
    bbox_max: pf.Vector,
    length: float,
    depth: float,
    margin: float,
) -> None:
    """Centre the island inside the room bbox inset by `margin` and its own half
    size, its length along the long axis."""
    extent = bbox_max - bbox_min
    along_y = extent.y >= extent.x
    half = pf.Vector((depth / 2.0, length / 2.0))
    half = half if along_y else pf.Vector((half.y, half.x))
    x, y = inset_floor_point_rand(rng, extent, (half.x, half.y), margin)
    yaw = 0.0 if along_y else math.pi / 2.0
    location = (bbox_min.x + x, bbox_min.y + y, bbox_min.z)
    pose = pf.Matrix.Translation(location) @ pf.Matrix.Rotation(yaw, 4, "Z")
    centre = pf.Matrix.Translation((0.0, -length / 2.0, 0.0))
    box.mesh.item().matrix_world = pose @ centre


def _walkway_clear(
    mesh: pf.MeshObject, colliders: ccol.CollisionSet, height: float
) -> bool:
    """The footprint grown by a 36 inch walkway misses every counter and wall."""
    lo, hi = (np.array(b) for b in pf.ops.attr.bbox_min_max(mesh, global_coords=True))
    centre = (lo + hi) / 2.0
    transform = np.eye(4)
    transform[:3, 3] = (centre[0], centre[1], height / 2.0)
    walkway = 2.0 * 0.91
    size = [hi[0] - lo[0] + walkway, hi[1] - lo[1] + walkway, height - 0.1]
    return not ccol.box_intersection_test(colliders, transform, size)


def _arrange_stools(
    rng: pf.RNG,
    parent: pf.MeshObject,
    length: float,
    overhang: float,
    height: float,
) -> list[pf.MeshObject]:
    """A row of counter stools just clear of the countertop edge at local x =
    -`overhang`, along its `length`."""
    rng_dims, rng_chair, rng_params = rng.spawn(3)
    seat = height - pf.random.uniform(rng_dims, 0.25, 0.32)
    dimensions = chair.chair_dining_dimensions_rand(rng_dims, seat_elevation=seat)
    stool = chair.chair_rand(rng_chair, dimensions=dimensions).mesh
    collection = pf.types.Collection([stool], name="kitchen_stool")
    stool_geo = pf.nodes.geo.collection_info(collection, separate_children=True)

    trim = dimensions[1] / 2.0 + pf.random.uniform(rng_params, 0.05, 0.12)
    edge = pf.nodes.geo.curve_line(
        start=(-overhang, trim, 0.0), end=(-overhang, length - trim, 0.0)
    )
    row = chairs_on_edge(
        stool_geo,
        edge,
        (1.0, 0.0, 0.0),
        dimensions[1] + pf.random.uniform(rng_params, 0.12, 0.3),
        dimensions[0] / 2.0 + 0.03,
        pf.random.uniform(rng_params, 0.0, 1.0) ** 3,
        pf.random.randint(rng_params, 0, 2**20),
    )

    pose = parent.item().matrix_world
    posed = pf.nodes.geo.transform(
        row,
        translation=pose.translation + pf.Vector((0.0, 0.0, 0.001)),
        rotation=pose.to_euler(),
    )
    stools = pf.nodes.to_aliases(posed)
    propagate_modifiers_to_instances([stool], stools)
    return stools


def _stools_rand(
    rng: pf.RNG,
    parent: pf.MeshObject,
    length: float,
    overhang: float,
    height: float,
    colliders: ccol.CollisionSet,
) -> list[pf.MeshObject]:
    stools, _ = place_surrounding(
        rng,
        parent,
        lambda r, p: _arrange_stools(r, p, length, overhang, height),
        colliders,
        min_kept=2,
    )
    return stools


@pf.tracer.grammar
def island_setup_rand(
    rng: pf.RNG,
    count: int,
    n_sinks: int,
    modules: pf.Collection,
    n_modules: int,
    module_width: float,
    counter_depth: float,
    height: float,
    thickness: float,
    plinth: pf.Vector,
    cabinet_material: pf.Material,
    countertop_material: pf.Material,
    bbox_min: pf.Vector,
    bbox_max: pf.Vector,
    region: shapely.Polygon | None,
    walkway: ccol.CollisionSet,
    colliders: ccol.CollisionSet,
) -> KitchenSetupResult:
    """Place an island mid-floor a walkway clear of `walkway`, and stools along
    its overhang half the time."""
    rng_place, rng_island, rng_stool_choice, rng_stools = rng.spawn(4)
    length = count * module_width
    box = _module_array_box(counter_depth, counter_depth, length, height)

    def accepted(mesh: pf.MeshObject) -> bool:
        if region is not None and not center_inside(region, mesh):
            return False
        return _walkway_clear(mesh, walkway, height)

    placed = retry_place(
        rng_place,
        box,
        colliders,
        _place_island,
        attempts=10,
        accept_fn=accepted,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        length=length,
        depth=2.0 * counter_depth,
        margin=counter_depth + 1.07,
    )
    if placed is None:
        return _setup_result([], [], [], colliders)

    island = island_rand(
        rng_island,
        box.mesh.item().matrix_world.copy(),
        count,
        n_sinks,
        modules,
        n_modules,
        module_width,
        counter_depth,
        height,
        thickness,
        plinth,
        cabinet_material,
        countertop_material,
    )
    stools_fn = pf.control.choice(
        rng_stool_choice, [(lambda *_: [], 1.0), (_stools_rand, 1.0)]
    )
    stools = stools_fn(rng_stools, box.mesh, length, counter_depth, height, colliders)
    delete_object(box.mesh.item())
    colliders = ccol.collision_set(
        colliders.objs + island.all_objects + stools, cache=colliders
    )
    return _setup_result([island], [], stools, colliders)


def _snap_hardware(
    rng: pf.RNG,
    hardware: bathroom_hardware.BathroomHardwareResult,
    parents: list[pf.MeshObject],
    height: float,
) -> None:
    rng_side, rng_place = rng.spawn(2)
    side = pf.control.choice(rng_side, [("left", 1.0), ("right", 1.0), ("front", 1.0)])
    hardware.mesh.item().location.z = pf.random.uniform(
        rng_place, 0.4 * height, height + 0.45
    )
    snap_to_plane(
        hardware.mesh,
        rng_place.choice(parents),
        placement=pf.random.uniform(rng_place, 0.1, 0.9),
        child_side="back",
        parent_side=side,
        margin=0.003,
    )


def _hardware_rand(
    rng: pf.RNG,
    parents: list[pf.MeshObject],
    height: float,
    colliders: ccol.CollisionSet,
) -> list[pf.MeshObject]:
    rng_count, rng_material, rng_items = rng.spawn(3)
    vec = pf.nodes.shader.coord().uv
    material = metal_brushed.metal_brushed_radial_rand(rng_material, vec)
    targets = ccol.collision_set(parents)

    placed = []
    for r in rng_items.spawn(pf.random.randint(rng_count, 0, 5)):
        rng_hardware, rng_place = r.spawn(2)
        hardware = bathroom_hardware.hardware_bathroom_rand(
            rng_hardware, material=material
        )
        grounded = lambda mesh: back_face_grounded(mesh, targets, margin=0.003)  # noqa: E731
        attached = retry_place(
            rng_place,
            hardware,
            colliders,
            _snap_hardware,
            attempts=16,
            accept_fn=grounded,
            parents=parents,
            height=height,
        )
        placed.append(attached)
    kept, _ = keep_non_colliding(placed, colliders)
    return [h.mesh for h in kept]


def _kitchen_setup_demo_rand(rng: pf.RNG) -> KitchenSetupResult:
    rng_shape, rng_setup = rng.spawn(2)
    room_dimensions = pf.Vector((4.5, 5.5, 2.7))
    shape = room_shape_rand(rng_shape, dimensions=room_dimensions)
    walls = [
        plane_to_posed_canonical_mesh(
            pf.nodes.to_mesh_object(pf.nodes.geo.object_info(wall.obj).geometry)
        )
        for wall in shape.flat_walls
    ]
    colliders = ccol.collision_set(walls + [shape.walls])
    setup = kitchen_setup_rand(
        rng_setup,
        walls,
        bbox_min=pf.Vector((0.0, 0.0, 0.0)),
        bbox_max=room_dimensions,
        colliders=colliders,
    )
    flat_walls = [wall.obj for wall in shape.flat_walls]
    for obj in walls + flat_walls + [shape.walls, shape.floor, shape.ceiling]:
        delete_object(obj.item())
    ground = pf.ops.primitives.mesh_single_vertex()
    ground.item().name = "kitchen_setup"
    all_objects = setup.all_objects + [ground]
    return setup._replace(
        all_objects=all_objects,
        colliders=ccol.collision_set(all_objects),
    )


@pf.tracer.grammar
def kitchen_setup_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject] | None = None,
    bbox_min: pf.Vector | None = None,
    bbox_max: pf.Vector | None = None,
    colliders: ccol.CollisionSet | None = None,
    wall_overlap: float = 0.0,
    reserved_wall_planes: list[pf.MeshObject] | None = None,
    region: shapely.Polygon | None = None,
) -> KitchenSetupResult:
    """Cabinet arrays on every `reserved_wall_planes` wall and some `wall_planes`
    walls, then an island and wall hardware. `wall_overlap` is how far each wall
    plane extends past the room corners."""
    if reserved_wall_planes is None:
        reserved_wall_planes = []
    if wall_planes is None:
        return _kitchen_setup_demo_rand(rng)
    if bbox_min is None:
        bbox_min = pf.Vector((0.0, 0.0, 0.0))
    if bbox_max is None:
        bbox_max = pf.Vector((4.5, 5.5, 2.7))
    if colliders is None:
        colliders = ccol.collision_set(wall_planes)
    rng_params, rng_materials, rng_modules, rng_walls, rng_island, rng_hardware = (
        rng.spawn(6)
    )

    module_width = pf.random.uniform(rng_params, 0.4, 0.9)
    depth = pf.random.uniform(rng_params, 0.56, 0.6)
    height = pf.random.uniform(rng_params, 0.86, 0.94)
    thickness = pf.random.uniform(rng_params, 0.02, 0.045)
    counter_depth = depth + pf.random.uniform(rng_params, 0.015, 0.04)
    plinth_height = pf.random.uniform(rng_params, 0.09, 0.15)
    plinth_recess = pf.random.uniform(rng_params, 0.05, 0.08)
    plinth = pf.Vector((depth - plinth_recess, 0.0, plinth_height))
    ceiling = float(bbox_max.z - bbox_min.z)
    upper_bottom = min(
        height + pf.random.uniform(rng_params, 0.45, 0.55), ceiling - 0.65
    )
    upper_depth = pf.random.uniform(rng_params, 0.3, 0.37)
    upper_height = (ceiling - upper_bottom) * pf.random.uniform(rng_params, 0.5, 0.9)

    rng_cabinet_material, rng_counter_material = rng_materials.spawn(2)
    vec = pf.nodes.shader.coord().uv
    cabinet_material = cabinet_material_rand(rng_cabinet_material, vec)
    countertop_material = kitchen_counter_rand(rng_counter_material, vec)

    rng_handle, rng_base, rng_upper, rng_counts = rng_modules.spawn(4)
    handle = _handle_rand(rng_handle)
    base_dims = pf.Vector((depth, module_width, height - thickness - plinth_height))
    upper_dims = pf.Vector((upper_depth, module_width, upper_height))
    n_base = pf.random.randint(rng_counts, 1, 4)
    n_upper = pf.random.randint(rng_counts, 1, 4)
    base_modules = _base_modules_rand(
        rng_base, n_base, base_dims, cabinet_material, handle
    )
    upper_modules = _modules_rand(
        rng_upper,
        n_upper,
        [(storage.storage_cabinet_with_door_rand, 3.0), (_upper_shelf_rand, 2.0)],
        upper_dims,
        cabinet_material,
        handle,
        [],
        "kitchen_upper",
    )

    rng_featured, rng_sinks, rng_island_sinks, rng_arrays = rng_walls.spawn(4)
    long_walls = [w for w in wall_planes if _wall_length(w) - 2.0 * wall_overlap >= 1.2]
    order = rng_featured.permutation(len(long_walls))
    n_featured = pf.random.randint(rng_featured, 0, len(long_walls) + 1)
    featured = [long_walls[i] for i in order[:n_featured]]
    walls = reserved_wall_planes + featured
    n_sinks = pf.control.choice(rng_sinks, [(0, 0.12), (1, 0.76), (2, 0.12)])
    island_sinks = pf.control.choice(rng_island_sinks, [(0, 2.0), (1, 1.0)])
    sinks_per_wall = ([n_sinks] + [0] * len(walls))[: len(walls)]
    arrays = kitchen_cabinets_wall_setup_rand(
        rng_arrays,
        walls,
        sinks_per_wall,
        base_modules,
        n_base + 1,
        upper_modules,
        n_upper,
        module_width,
        counter_depth,
        height,
        thickness,
        plinth,
        upper_bottom,
        cabinet_material,
        countertop_material,
        colliders,
        wall_overlap,
    )

    rng_island_params, rng_island_setup = rng_island.spawn(2)
    island_margin = counter_depth + 1.07
    extent = bbox_max - bbox_min
    island_free = max(extent.x, extent.y) - 2.0 * island_margin
    island_length = island_free * pf.random.uniform(rng_island_params, 0.7, 1.0)
    walkway = wall_planes + reserved_wall_planes + arrays.all_objects
    islands = island_setup_rand(
        rng_island_setup,
        max(2, round(island_length / module_width)),
        island_sinks,
        base_modules,
        n_base + 1,
        module_width,
        counter_depth,
        height,
        thickness,
        plinth,
        cabinet_material,
        countertop_material,
        bbox_min,
        bbox_max,
        region,
        ccol.collision_set(walkway),
        arrays.colliders,
    )

    hardware = _hardware_rand(
        rng_hardware,
        arrays.storages + islands.storages + wall_planes + reserved_wall_planes,
        height,
        islands.colliders,
    )
    return KitchenSetupResult(
        all_objects=arrays.all_objects + islands.all_objects + hardware,
        colliders=ccol.collision_set(
            islands.colliders.objs + hardware, cache=islands.colliders
        ),
        countertops=arrays.countertops + islands.countertops,
        storage_containers=arrays.storage_containers + islands.storage_containers,
        supports=arrays.supports + islands.supports,
        storages=arrays.storages + islands.storages,
    )
