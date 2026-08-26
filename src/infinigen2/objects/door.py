# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.objects import handles
from infinigen2.shaders.functionality_lists import (
    furniture_material_rand,
    glass_material_rand,
)
from infinigen2.util.mesh import metric_box_uv

__all__ = [
    "DoorResult",
    "door_body",
    "door_body_rand",
    "door_with_handle",
    "door_with_handle_rand",
]


class DoorResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def _door_body_geometry(
    frame_material: t.SocketOrVal[pf.Material],
    panel_material: t.SocketOrVal[pf.Material],
    width: t.SocketOrVal[float] = 0.8,
    height: t.SocketOrVal[float] = 2.0,
    thickness: t.SocketOrVal[float] = 0.04,
    n_panels: t.SocketOrVal[int] = 2,
    margin_fraction: t.SocketOrVal[float] = 0.12,
    panel_thickness: t.SocketOrVal[float] = 0.016,
) -> pf.ProcNode[pf.MeshObject]:
    n_panels_f = n_panels.astype(dtype=float)
    verts_y = (n_panels_f + 1.0).astype(dtype=int)
    grid = pf.nodes.geo.mesh_grid(
        size_x=width, size_y=height, vertices_x=2, vertices_y=verts_y
    )
    grid_mesh = pf.nodes.geo.transform(
        geometry=grid.mesh,
        translation=pf.nodes.math.combine_xyz(x=width * 0.5, y=height * 0.5),
    )
    inset = pf.nodes.geo.extrude_mesh(mesh=grid_mesh, offset_scale=0.0)

    centroid = pf.nodes.geo.capture_attribute(
        geometry=inset.mesh, domain="FACE", position=pf.nodes.geo.input_position()
    )
    panel_h = height / n_panels_f
    margin = margin_fraction * pf.nodes.math.minimum(width, panel_h)
    scale = pf.nodes.math.combine_xyz(
        x=1.0 - 2.0 * margin / width,
        y=1.0 - 2.0 * margin / panel_h,
        z=1.0,
    )
    to_vertex = pf.nodes.geo.input_position() - centroid.position
    planar = pf.nodes.geo.set_position(
        geometry=centroid.geometry,
        position=centroid.position + to_vertex * scale,
        selection=inset.top,
    )
    split = pf.nodes.geo.separate_geometry(
        geometry=planar, selection=inset.top, domain="FACE"
    )

    frame_shell = pf.nodes.geo.extrude_mesh(mesh=split.inverted, offset_scale=thickness)
    frame_cap = pf.nodes.geo.flip_faces(split.inverted)
    frame = pf.nodes.geo.join_geometry([frame_shell.mesh, frame_cap])
    frame = pf.nodes.geo.merge_by_distance(frame, distance=0.0001)
    frame = pf.nodes.geo.set_material(frame, material=frame_material)

    panel_shell = pf.nodes.geo.extrude_mesh(
        mesh=split.selection, offset_scale=panel_thickness
    )
    panel_cap = pf.nodes.geo.flip_faces(split.selection)
    panel = pf.nodes.geo.join_geometry([panel_shell.mesh, panel_cap])
    panel = pf.nodes.geo.merge_by_distance(panel, distance=0.0001)
    panel = pf.nodes.geo.transform(
        geometry=panel,
        translation=pf.nodes.math.combine_xyz(z=(thickness - panel_thickness) * 0.5),
    )
    panel = pf.nodes.geo.set_material(panel, material=panel_material)

    door = pf.nodes.geo.join_geometry([frame, panel])
    return pf.nodes.geo.transform(geometry=door, rotation=(1.5708, 0.0, 1.5708))


def _same_opaque(rng: pf.RNG, vec) -> tuple[pf.Material, pf.Material]:
    mat = furniture_material_rand(rng, vec)
    return mat, mat


def _two_opaque(rng: pf.RNG, vec) -> tuple[pf.Material, pf.Material]:
    rng_frame, rng_panel = rng.spawn(2)
    return furniture_material_rand(rng_frame, vec), furniture_material_rand(
        rng_panel, vec
    )


def _opaque_and_glass(rng: pf.RNG, vec) -> tuple[pf.Material, pf.Material]:
    rng_frame, rng_panel = rng.spawn(2)
    return furniture_material_rand(rng_frame, vec), glass_material_rand(rng_panel, vec)


def _door_body_finish(geo: pf.ProcNode, bevel_width: float) -> DoorResult:
    sharp = pf.nodes.geo.input_mesh_edge_angle().unsigned_angle > 0.5
    geo = pf.nodes.geo.store_named_attribute(
        geometry=geo,
        name="crease_edge",
        domain="EDGE",
        value=sharp.astype(dtype=float),
        data_type="FLOAT",
    )
    geo = metric_box_uv(geo)
    obj = pf.nodes.to_mesh_object(geo)
    pf.ops.modifier.bevel(obj, width=bevel_width, segments=2)
    pf.ops.modifier.subdivide_surface(obj, levels=6, _skip_apply=True)
    return DoorResult(mesh=obj)


def door_body(
    dimensions: pf.Vector | None = None,
    frame_material: pf.Material | None = None,
    panel_material: pf.Material | None = None,
) -> DoorResult:
    if dimensions is None:
        dimensions = pf.Vector((0.04, 0.8, 2.0))
    if frame_material is None:
        frame_material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    if panel_material is None:
        panel_material = frame_material
    geo = _door_body_geometry(
        frame_material=frame_material,
        panel_material=panel_material,
        width=dimensions.y,
        height=dimensions.z,
        thickness=dimensions.x,
    )
    return _door_body_finish(geo, bevel_width=0.003)


def door_body_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_material: pf.Material | None = None,
    panel_material: pf.Material | None = None,
) -> DoorResult:
    rng, rng_mat_choice, rng_mat = rng.spawn(3)
    if dimensions is None:
        width = pf.random.uniform(rng, 0.7, 0.95)
        height = pf.random.uniform(rng, 1.9, 2.1)
        thickness = pf.random.uniform(rng, 0.035, 0.045)
        dimensions = pf.Vector((thickness, width, height))

    vec = pf.nodes.shader.coord().uv
    if frame_material is None and panel_material is None:
        mat_func = pf.control.choice(
            rng_mat_choice,
            [(_same_opaque, 2.0), (_two_opaque, 0.5), (_opaque_and_glass, 0.5)],
        )
        mats = mat_func(rng_mat, vec)
        frame_material = mats[0]
        panel_material = mats[1]

    thickness = dimensions.x
    n_panels = pf.random.randint(rng, 1, 5)
    margin_fraction = pf.random.uniform(rng, 0.1, 0.22)
    geo = _door_body_geometry(
        frame_material=frame_material,
        panel_material=panel_material,
        width=dimensions.y,
        height=dimensions.z,
        thickness=thickness,
        n_panels=n_panels,
        margin_fraction=margin_fraction,
        panel_thickness=thickness * pf.random.uniform(rng, 0.35, 0.5),
    )
    bevel_width = pf.random.uniform(rng, 0.001, 0.005)
    return _door_body_finish(geo, bevel_width=bevel_width)


def _place_handle(
    door: pf.MeshObject,
    handle: pf.MeshObject,
    dimensions: pf.Vector,
    edge_offset: float,
    handle_z_frac: float,
) -> None:
    handle_x = dimensions.x
    handle_y = dimensions.y - edge_offset
    handle_z = dimensions.z * handle_z_frac
    pf.ops.object.set_transform(handle, location=(handle_x, handle_y, handle_z))
    pf.ops.object.join(door, handle)


def door_with_handle(
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
) -> DoorResult:
    if dimensions is None:
        dimensions = pf.Vector((0.04, 0.8, 2.0))
    door = door_body(
        dimensions=dimensions, frame_material=material, panel_material=material
    ).mesh
    handle = handles.lever_handle().mesh
    _place_handle(door, handle, dimensions, edge_offset=0.06, handle_z_frac=0.46)
    return DoorResult(mesh=door)


def _choose_handle_rand(rng_choice: pf.RNG, rng_handle: pf.RNG) -> pf.MeshObject:
    handle_func = pf.control.choice(
        rng_choice,
        [
            (handles.lever_handle_rand, 2.0),
            (handles.bar_pull_handle_rand, 1.5),
            (handles.curved_pull_handle_rand, 1.5),
            (handles.knob_handle_rand, 1.0),
        ],
    )
    return handle_func(rng_handle).mesh


def door_with_handle_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    handle: pf.MeshObject | None = None,
) -> DoorResult:
    rng, rng_door, rng_handle_choice, rng_handle, rng_place = rng.spawn(5)
    if dimensions is None:
        width = pf.random.uniform(rng, 0.7, 0.95)
        height = pf.random.uniform(rng, 1.9, 2.1)
        thickness = pf.random.uniform(rng, 0.035, 0.045)
        dimensions = pf.Vector((thickness, width, height))

    door = door_body_rand(
        rng_door,
        dimensions=dimensions,
        frame_material=material,
        panel_material=material,
    ).mesh

    if handle is None:
        handle = _choose_handle_rand(rng_handle_choice, rng_handle)

    edge_offset = pf.random.uniform(rng_place, 0.04, 0.08)
    handle_z_frac = pf.random.uniform(rng_place, 0.42, 0.5)
    _place_handle(door, handle, dimensions, edge_offset, handle_z_frac)
    return DoorResult(mesh=door)
