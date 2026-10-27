# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Direct access to Blender 4.2 mesh buffers, checked against RNA, with RNA
fallbacks on other versions and layouts."""

import ctypes

import bpy
import numpy as np


def _int_buffer(items, width, field):
    n = len(items)
    if bpy.app.version[:2] != (4, 2) or n < 2:
        return None
    first, second, last = items[0], items[1], items[n - 1]
    address = first.as_pointer()
    stride = width * 4
    if (
        not address
        or second.as_pointer() != address + stride
        or last.as_pointer() != address + (n - 1) * stride
    ):
        return None
    view = np.ctypeslib.as_array(
        (ctypes.c_int32 * (n * width)).from_address(address)
    ).reshape(n, width)
    if not np.array_equal(
        view[0], np.asarray(getattr(first, field)).reshape(-1)
    ) or not np.array_equal(view[-1], np.asarray(getattr(last, field)).reshape(-1)):
        return None
    return view


def write_faces(mesh, counts):
    offsets = np.cumsum(counts, dtype=np.int32) - counts
    view = _int_buffer(mesh.polygons, 1, "loop_start")
    if view is None:
        mesh.polygons.foreach_set("loop_start", offsets)
        mesh.polygons.foreach_set("loop_total", counts)
    else:
        # Blender 4.2 face_offsets has n+1 entries, the last being the corner
        # count; check it against RNA before writing the buffer. Only for newly
        # allocated, unshared face storage.
        n = len(counts)
        raw = np.ctypeslib.as_array(
            (ctypes.c_int32 * (n + 1)).from_address(view.ctypes.data)
        )
        last = mesh.polygons[-1]
        if raw[-1] == last.loop_start + last.loop_total:
            raw[:-1] = offsets
            raw[-1] = len(mesh.loops)
        else:
            mesh.polygons.foreach_set("loop_start", offsets)
            mesh.polygons.foreach_set("loop_total", counts)
    # the caller calls mesh.update after all topology writes


def read_triangles(mesh):
    mesh.calc_loop_triangles()
    view = _int_buffer(mesh.loop_triangles, 3, "loops")
    if view is None:
        tri = np.empty((len(mesh.loop_triangles), 3), np.int32)
        mesh.loop_triangles.foreach_get("vertices", tri.ravel())
        return tri
    vertices = np.empty(len(mesh.loops), np.int32)
    mesh.attributes[".corner_vert"].data.foreach_get("value", vertices)
    # the indexing copies, so the view does not escape
    return vertices[view]


def read_normals(mesh):
    items = mesh.vertex_normals
    n = len(items)
    if bpy.app.version[:2] == (4, 2) and n >= 2:
        first, second, last = items[0], items[1], items[-1]
        address = first.as_pointer()
        if (
            address
            and second.as_pointer() == address + 12
            and last.as_pointer() == address + (n - 1) * 12
        ):
            view = np.ctypeslib.as_array(
                (ctypes.c_float * (n * 3)).from_address(address)
            ).reshape(n, 3)
            if np.array_equal(view[0], first.vector) and np.array_equal(
                view[-1], last.vector
            ):
                return view.copy()
    result = np.empty((n, 3), np.float32)
    items.foreach_get("vector", result.ravel())
    return result


def write_positions(mesh, values):
    if not len(values):
        return
    mesh.attributes["position"].data.foreach_set("vector", values.ravel())
    # the attribute write does not call Mesh::tag_positions_changed; one RNA co
    # assignment does, invalidating bounds, normals and triangulation
    mesh.vertices[0].co = values[0]


def writable_attribute(mesh, name, width, dtype):
    """Writable view of a new, unshared Blender 4.2 attribute, or None. Call only
    after allocating all mesh domains and attributes; the caller keeps the mesh alive,
    does not resize it, and updates it after writing."""
    if bpy.app.version[:2] != (4, 2) or mesh.users > 1:
        return None
    attr = mesh.attributes[name]
    items = attr.data
    n = len(items)
    expected = {
        ("FLOAT_VECTOR", 3): np.float32,
        ("FLOAT", 1): np.float32,
        ("INT32_2D", 2): np.int32,
        ("INT", 1): np.int32,
    }
    if n < 2 or expected.get((attr.data_type, width)) != dtype:
        return None
    first, second, last = items[0], items[1], items[-1]
    address, stride = first.as_pointer(), width * 4
    if (
        not address
        or second.as_pointer() != address + stride
        or last.as_pointer() != address + (n - 1) * stride
    ):
        return None
    scalar = ctypes.c_float if dtype == np.float32 else ctypes.c_int32
    array = np.ctypeslib.as_array((scalar * (n * width)).from_address(address)).reshape(
        n, width
    )
    field = "vector" if attr.data_type == "FLOAT_VECTOR" else "value"
    if not np.array_equal(
        array[0], np.asarray(getattr(first, field)).reshape(-1)
    ) or not np.array_equal(array[-1], np.asarray(getattr(last, field)).reshape(-1)):
        return None
    return array.reshape(n) if width == 1 else array
