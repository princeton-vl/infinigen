# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

from infinigen2.exporters.util.blender_render import object_index_table_names
from infinigen2.exporters.util.format import SCENE_PASS_DEFAULTS, ExportType

__all__ = [
    "collect_object_data",
    "save_object_data",
]

logger = logging.getLogger(__name__)

OBJECT_DATA_PATH = SCENE_PASS_DEFAULTS[ExportType.OBJECT_DATA].path

STR_DTYPE = "S63"

SHEAR_TOLERANCE = 1e-4

FLOAT_FIELDS = {
    "location_meters": (3,),
    "rotation_euler_rad": (3,),
    "scale": (3,),
    "local_bbox_min": (3,),
    "local_bbox_max": (3,),
}


def _unwrap(objects) -> list[bpy.types.Object]:
    return [o.item() if hasattr(o, "item") else o for o in objects]


def _allocate_buffers(n_objects: int, n_frames: int) -> dict[str, np.ndarray]:
    return {
        name: np.full((n_objects, *axes, n_frames), np.nan, dtype=np.float32)
        for name, axes in FLOAT_FIELDS.items()
    }


def _object_index(objs: list[bpy.types.Object]) -> np.ndarray:
    """Each object's already-assigned segmentation index, which maps rows of the npz
    onto the object-index table. Row order itself is arbitrary.

    Rejects indices that are unset, shared, or no longer pointing back at their own
    object, so a row can always be joined to the object-index pass."""
    index = np.array([o.pass_index for o in objs], dtype=np.int32)

    unassigned = [o.name for o, i in zip(objs, index, strict=True) if i == 0]
    if unassigned:
        raise ValueError(
            f"{len(unassigned)} objects have pass_index 0, which the object-index "
            f"table reserves for the background: {unassigned[:5]}. Run "
            f"configure_object_index_table() before exporting object data."
        )

    values, counts = np.unique(index, return_counts=True)
    clashing = values[counts > 1]
    if len(clashing):
        raise ValueError(
            f"pass_index must be unique per object, but {clashing.tolist()} are "
            f"shared; rows could not be mapped back to the object-index table."
        )

    table = object_index_table_names([None, *bpy.data.objects])
    stale = [
        f"{o.name!r} claims index {i}"
        for o, i in zip(objs, index, strict=True)
        if i >= len(table) or table[i] != o.name
    ]
    if stale:
        raise ValueError(
            f"{len(stale)} objects have a pass_index that no longer points back at "
            f"themselves in the current {len(table)}-entry object-index table, so "
            f"their rows would join to the wrong segmentation value: {stale[:5]}. "
            f"Re-run configure_object_index_table() after creating or deleting objects."
        )
    return index


def _data_ids(objs: list[bpy.types.Object]) -> np.ndarray:
    """Objects sharing one mesh datablock share a data_id, so instances stay groupable."""
    pointers = {}
    data_id = np.full(len(objs), -1, dtype=np.int32)
    for i, o in enumerate(objs):
        if o.data is None:
            continue
        data_id[i] = pointers.setdefault(o.data.as_pointer(), len(pointers))
    return data_id


def _object_metadata(objs: list[bpy.types.Object]) -> dict[str, np.ndarray]:
    return {
        "object_index": _object_index(objs),
        "object_name": np.array([o.name for o in objs], dtype=STR_DTYPE),
        "object_type": np.array([o.type for o in objs], dtype=STR_DTYPE),
        "data_name": np.array(
            [o.data.name if o.data is not None else "" for o in objs], dtype=STR_DTYPE
        ),
        "data_id": _data_ids(objs),
    }


def _warn_if_sheared(
    o: bpy.types.Object, location: Vector, rotation: Quaternion, scale: Vector
) -> None:
    rebuilt = np.array(Matrix.LocRotScale(location, rotation, scale))
    shear = np.abs(rebuilt - np.array(o.matrix_world)).max()
    if shear <= SHEAR_TOLERANCE:
        return
    logger.warning(
        f"{o.name} has a sheared transform ({shear=:.3g}), which "
        f"location/rotation/scale cannot represent; its box will be inexact"
    )


def _capture_frame(
    buffers: dict[str, np.ndarray],
    objs: list[bpy.types.Object],
    idx: int,
    eulers: list[Euler],
) -> None:
    """Write every object's current-frame world state into column idx."""
    for i, o in enumerate(objs):
        location, rotation, scale = o.matrix_world.decompose()
        eulers[i] = rotation.to_euler("XYZ", eulers[i])
        bbox = np.asarray(o.bound_box, dtype=np.float32)
        buffers["location_meters"][i, :, idx] = location
        buffers["rotation_euler_rad"][i, :, idx] = eulers[i]
        buffers["scale"][i, :, idx] = scale
        buffers["local_bbox_min"][i, :, idx] = bbox.min(axis=0)
        buffers["local_bbox_max"][i, :, idx] = bbox.max(axis=0)
        _warn_if_sheared(o, location, rotation, scale)


def collect_object_data(
    objects: list,
    frame_start: int,
    frame_end: int,
) -> dict[str, np.ndarray | np.int32]:
    """Collect per-object 3D ground truth over the frame range in memory.

    Every array's axis 0 is one row per object in the order given. object_index carries
    each object's already-assigned segmentation index, so a row maps onto the
    object-index table and the object-index pass without depending on row order.

    Pose is location_meters, rotation_euler_rad (XYZ, kept continuous across frames) and
    scale rather than a 4x4; a world-space bbox corner is
    location + R(rotation) @ (scale * local_bbox corner)."""
    objs = _unwrap(objects)
    n_frames = frame_end - frame_start + 1
    buffers = _allocate_buffers(len(objs), n_frames)
    metadata = _object_metadata(objs)

    original_frame = bpy.context.scene.frame_current
    eulers = [Euler((0.0, 0.0, 0.0)) for _ in objs]
    for idx in range(n_frames):
        bpy.context.scene.frame_set(frame_start + idx)
        _capture_frame(buffers, objs, idx, eulers)
    bpy.context.scene.frame_set(original_frame)

    return {
        **buffers,
        **metadata,
        "frame_start": np.int32(frame_start),
        "frame_end": np.int32(frame_end),
    }


def save_object_data(
    objects: list,
    output_folder: Path,
    frame_start: int,
    frame_end: int,
    path: Path = OBJECT_DATA_PATH,
) -> dict[ExportType, list[Path]]:
    """Write per-object 3D ground truth over the frame range as object-data.npz."""
    data = collect_object_data(objects, frame_start, frame_end)

    result_path = Path(output_folder) / path
    result_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(result_path, **data)
    return {ExportType.OBJECT_DATA: [result_path]}
