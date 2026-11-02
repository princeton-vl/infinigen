# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import bpy
import numpy as np

from infinigen.core import surface


@dataclass
class PendingScalarDisplacement:
    evaluator: Callable[[np.ndarray, dict[str, np.ndarray]], np.ndarray]
    material: bpy.types.Material | None
    attributes: tuple[str, ...]
    capture_position_as: str | None


_PENDING: dict[int, PendingScalarDisplacement] = {}


def defer_scalar_displacement(
    obj: bpy.types.Object,
    evaluator: Callable[[np.ndarray, dict[str, np.ndarray]], np.ndarray],
    *,
    attributes: tuple[str, ...] = (),
    material: bpy.types.Material | None = None,
    capture_position_as: str | None = None,
) -> None:
    """Register a scalar normal displacement for `obj`, applied by `apply_pending`."""
    if obj.type != "MESH":
        raise TypeError(f"Expected mesh object, got {obj.type}")
    key = obj.as_pointer()
    if key in _PENDING:
        raise ValueError(f"Object {obj.name!r} already has a deferred surface field")
    _PENDING[key] = PendingScalarDisplacement(
        evaluator, material, tuple(attributes), capture_position_as
    )


def has_pending(obj: bpy.types.Object) -> bool:
    return obj.as_pointer() in _PENDING


def transfer_pending(source: bpy.types.Object, *targets: bpy.types.Object) -> None:
    """Move `source`'s registered field to `targets`, the pieces it was split into."""
    pending = _PENDING.pop(source.as_pointer(), None)
    if pending is None:
        return
    for target in targets:
        _PENDING[target.as_pointer()] = pending


def _point_vectors(mesh: bpy.types.Mesh, name: str) -> np.ndarray:
    attribute = mesh.attributes.get(name)
    if (
        attribute is None
        or attribute.domain != "POINT"
        or attribute.data_type != "FLOAT_VECTOR"
    ):
        raise ValueError(f"Expected POINT/FLOAT_VECTOR attribute {name!r}")
    values = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    attribute.data.foreach_get("vector", values.reshape(-1))
    return values


def apply_pending(obj: bpy.types.Object) -> bool:
    """Displace the vertices along their normals by the registered field and assign
    its material; return whether one was registered."""
    pending = _PENDING.pop(obj.as_pointer(), None)
    if pending is None:
        return False
    mesh = obj.data
    positions = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    normals = np.empty_like(positions)
    mesh.vertices.foreach_get("co", positions.reshape(-1))
    mesh.vertices.foreach_get("normal", normals.reshape(-1))
    attributes = {name: _point_vectors(mesh, name) for name in pending.attributes}
    displacement = np.asarray(
        pending.evaluator(positions, attributes), dtype=np.float32
    ).reshape(-1)
    if (
        displacement.shape[0] != positions.shape[0]
        or not np.isfinite(displacement).all()
    ):
        raise ValueError("Surface field must return one finite scalar per vertex")
    if pending.capture_position_as is not None:
        captured = mesh.attributes.get(pending.capture_position_as)
        if captured is None:
            captured = mesh.attributes.new(
                pending.capture_position_as, "FLOAT_VECTOR", "POINT"
            )
        captured.data.foreach_set("vector", positions.reshape(-1))
    positions += normals * displacement[:, None]
    mesh.vertices.foreach_set("co", positions.reshape(-1))
    mesh.update()
    if pending.material is not None:
        surface.assign_material(obj, pending.material)
    return True
