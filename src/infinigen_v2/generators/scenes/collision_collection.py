from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import bpy
import fcl
import numpy as np
import procfunc as pf
import trimesh

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollisionSet:
    objs: list[pf.Object]
    collision_manager: trimesh.collision.CollisionManager
    object_names: dict[int, str]
    mesh_colliders: dict[int, trimesh.Trimesh]
    mesh_fcl_colliders: dict[int, Any]
    object_fcl_objs: dict[int, Any]  # object_id -> fcl.CollisionObject in manager
    object_last_transforms: dict[int, np.ndarray]  # object_id -> last synced transform


def _is_collidable_object(obj: pf.Object) -> bool:
    item = obj.item()
    data = getattr(item, "data", None)
    if data is None:
        return False
    if not hasattr(data, "vertices") or not hasattr(data, "polygons"):
        return False
    return True


def _mesh_name(obj: pf.Object) -> str:
    return obj.item().name


def _mesh_id(obj: pf.Object) -> int:
    return id(obj.item().data)


def _object_id(obj: pf.Object) -> int:
    return id(obj.item())


def _mesh_from_object(obj: pf.Object) -> trimesh.Trimesh:
    bpy.context.view_layer.update()
    vertices = pf.ops.attr.vertex_positions(obj, global_coords=False)
    if vertices.shape[0] == 0:
        raise ValueError(f"{obj.item().name} has no vertices")
    obj.item().data.calc_loop_triangles()
    loop_triangles = obj.item().data.loop_triangles
    if len(loop_triangles) == 0:
        raise ValueError(f"{obj.item().name} has no loop triangles")
    faces_flat = np.zeros(len(loop_triangles) * 3, dtype=np.int32)
    loop_triangles.foreach_get("vertices", faces_flat)
    faces = faces_flat.reshape(-1, 3)

    # Filter degenerate (zero-area) faces
    areas = np.linalg.norm(
        np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        ),
        axis=1,
    )
    valid = areas > 1e-10
    n_removed = len(faces) - int(valid.sum())
    if n_removed > 0:
        faces = faces[valid]

    dims = vertices.max(axis=0) - vertices.min(axis=0)
    logger.debug(
        "collider %s: %d verts, %d faces (%d degenerate removed), "
        "dims=(%.2f, %.2f, %.2f)",
        obj.item().name,
        len(vertices),
        len(faces),
        n_removed,
        *dims,
    )

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _object_transform(obj: pf.Object) -> np.ndarray:
    bpy.context.view_layer.update()
    return np.asarray(obj.item().matrix_world, dtype=np.float64)


def _assert_rigid(obj: pf.Object, tol: float = 1e-5) -> None:
    T = _object_transform(obj)
    R = T[:3, :3]
    err = np.max(np.abs(R.T @ R - np.eye(3)))
    if err > tol:
        raise ValueError(
            f"object {obj.item().name!r} has non-unit scale/shear in "
            f"matrix_world (|R^T R - I|_inf = {err:.3g}); apply scale to "
            f"mesh data (e.g. `bpy.ops.object.transform_apply`) before "
            f"adding to CollisionSet"
        )


def _fcl_transform(transform: np.ndarray) -> fcl.Transform:
    return fcl.Transform(transform[:3, :3], transform[:3, 3])


def _add_object_cached(
    manager: trimesh.collision.CollisionManager,
    name: str,
    col_obj: fcl.CollisionObject,
    fcl_geom: Any,
) -> None:
    if name in manager._objs:
        manager._manager.unregisterObject(manager._objs[name]["obj"])
    manager._objs[name] = {"obj": col_obj, "geom": fcl_geom}
    manager._names[id(fcl_geom)] = name
    manager._manager.registerObject(col_obj)
    manager._manager.update()


def _sync_transforms(col: CollisionSet) -> None:
    """Update FCL collision objects whose Blender transforms have changed since last query."""
    if not col.objs:
        return
    bpy.context.view_layer.update()
    any_changed = False
    for obj in col.objs:
        oid = _object_id(obj)
        col_obj = col.object_fcl_objs.get(oid)
        if col_obj is None:
            continue
        _assert_rigid(obj)
        T = np.asarray(obj.item().matrix_world, dtype=np.float64)
        last_T = col.object_last_transforms.get(oid)
        if last_T is None or not np.array_equal(last_T, T):
            col_obj.setTransform(_fcl_transform(T))
            col.object_last_transforms[oid] = T
            any_changed = True
    if any_changed:
        col.collision_manager._manager.update()


def n_colliders(col: CollisionSet) -> int:
    return len(col.object_names)


def collision_set(
    objs: list[pf.Object], existing: CollisionSet | None = None
) -> CollisionSet:
    mesh_colliders: dict[int, trimesh.Trimesh] = (
        {} if existing is None else dict(existing.mesh_colliders)
    )
    mesh_fcl_colliders: dict[int, Any] = (
        {} if existing is None else dict(existing.mesh_fcl_colliders)
    )
    object_names: dict[int, str] = {}
    object_fcl_objs: dict[int, Any] = {}
    object_last_transforms: dict[int, np.ndarray] = {}
    manager = trimesh.collision.CollisionManager()
    collidable_objs: list[pf.Object] = []

    for obj in objs:
        if not _is_collidable_object(obj):
            continue
        _assert_rigid(obj)
        collidable_objs.append(obj)
        mesh_id = _mesh_id(obj)
        if mesh_id not in mesh_colliders:
            try:
                mesh_colliders[mesh_id] = _mesh_from_object(obj)
            except ValueError:
                collidable_objs.pop()
                continue

        fcl_geom = mesh_fcl_colliders.get(mesh_id)
        if fcl_geom is None:
            fcl_geom = manager._get_fcl_obj(mesh_colliders[mesh_id])
            mesh_fcl_colliders[mesh_id] = fcl_geom

        name = _mesh_name(obj)
        T = _object_transform(obj)
        col_obj = fcl.CollisionObject(fcl_geom, _fcl_transform(T))
        _add_object_cached(manager, name, col_obj, fcl_geom)
        oid = _object_id(obj)
        object_names[oid] = name
        object_fcl_objs[oid] = col_obj
        object_last_transforms[oid] = T

    return CollisionSet(
        objs=collidable_objs,
        collision_manager=manager,
        object_names=object_names,
        mesh_colliders=mesh_colliders,
        mesh_fcl_colliders=mesh_fcl_colliders,
        object_fcl_objs=object_fcl_objs,
        object_last_transforms=object_last_transforms,
    )


def intersection_test(col: CollisionSet, obj: pf.Object) -> bool:
    if n_colliders(col) == 0:
        return False
    if not _is_collidable_object(obj):
        return False
    _assert_rigid(obj)
    _sync_transforms(col)

    mesh_id = _mesh_id(obj)
    geom = col.mesh_colliders.get(mesh_id)
    if geom is None:
        try:
            geom = _mesh_from_object(obj)
        except ValueError:
            return False
        col.mesh_colliders[mesh_id] = geom

    fcl_geom = col.mesh_fcl_colliders.get(mesh_id)
    if fcl_geom is None:
        fcl_geom = col.collision_manager._get_fcl_obj(geom)
        col.mesh_fcl_colliders[mesh_id] = fcl_geom

    probe = trimesh.collision.CollisionManager()
    _add_object_cached(
        probe,
        "__probe__",
        fcl.CollisionObject(fcl_geom, _fcl_transform(_object_transform(obj))),
        fcl_geom,
    )
    hit = col.collision_manager.in_collision_other(probe, return_data=False)
    return bool(hit)


def box_intersection_test(
    col: CollisionSet, transform: np.ndarray, size: float = 1.0
) -> bool:
    if n_colliders(col) == 0:
        return False
    _sync_transforms(col)
    box = trimesh.creation.box(extents=[size, size, size])
    fcl_geom = col.collision_manager._get_fcl_obj(box)
    probe = trimesh.collision.CollisionManager()
    _add_object_cached(
        probe,
        "__probe__",
        fcl.CollisionObject(fcl_geom, _fcl_transform(transform)),
        fcl_geom,
    )
    return bool(col.collision_manager.in_collision_other(probe, return_data=False))


def any_self_collision(col: CollisionSet) -> bool:
    """Return True if any two objects in the set currently intersect each other."""
    if n_colliders(col) < 2:
        return False
    _sync_transforms(col)
    return bool(col.collision_manager.in_collision_internal())


def raycast(
    col: CollisionSet, ray_origins: np.ndarray, ray_directions: np.ndarray
) -> Any:
    """Raycast over the combined mesh from tracked colliders."""
    if n_colliders(col) == 0:
        return (
            np.empty((0, 3)),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    meshes = []
    for obj in col.objs:
        mesh = col.mesh_colliders.get(_mesh_id(obj))
        if mesh is None:
            continue
        mesh = mesh.copy()
        mesh.apply_transform(_object_transform(obj))
        meshes.append(mesh)

    if not meshes:
        return (
            np.empty((0, 3)),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    combined = trimesh.util.concatenate(meshes)
    return combined.ray.intersects_location(
        ray_origins=np.asarray(ray_origins),
        ray_directions=np.asarray(ray_directions),
        multiple_hits=False,
    )
