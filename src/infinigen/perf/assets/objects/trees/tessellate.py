# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Retessellation of the independent tubes that Geometry Nodes builds for tree stems."""

from __future__ import annotations

import logging

import bpy
import numpy as np

from infinigen.core.util import blender as butil

try:
    # Built only with INFINIGEN_INSTALL_PERF. The optimized trees import this
    # module either way and use the kernel only when it is built.
    from infinigen.perf.assets.objects.trees._cylinder_mesher import (
        mesh_stem as _mesh_stem,
    )
    from infinigen.perf.assets.objects.trees._cylinder_mesher import (
        smooth_stem as _smooth_stem,
    )
    from infinigen.perf.assets.objects.trees._cylinder_mesher import (
        write_mesh_buffers as _write_mesh_buffers,
    )

    KERNEL_AVAILABLE = True
except ImportError:
    _mesh_stem = _smooth_stem = _write_mesh_buffers = None
    KERNEL_AVAILABLE = False


def _require_kernel():
    if not KERNEL_AVAILABLE:
        raise RuntimeError(
            "adapt_mesh_method='stem_tessellate' needs the native cylinder "
            "mesher, which is built only with INFINIGEN_INSTALL_PERF=True. "
            "Reinstall with that set, or choose another adapt_mesh_method."
        )


logger = logging.getLogger(__name__)

_POINT_LAYOUTS = {
    "BOOLEAN": ("value", 1, np.bool_),
    "FLOAT": ("value", 1, np.float32),
    "FLOAT_VECTOR": ("vector", 3, np.float32),
    "INT": ("value", 1, np.int32),
}


def _foreach(data, field: str, width: int, dtype) -> np.ndarray:
    values = np.empty(len(data) * width, dtype=dtype)
    if len(values):
        data.foreach_get(field, values)
    return values.reshape((-1, width)) if width > 1 else values


def _stem_ranges(
    mesh: bpy.types.Mesh, profile_resolution: int
) -> list[tuple[int, int]]:
    loop_starts = _foreach(mesh.polygons, "loop_start", 1, np.int32)
    loop_totals = _foreach(mesh.polygons, "loop_total", 1, np.int32)
    loop_vertices = _foreach(mesh.loops, "vertex_index", 1, np.int32)
    cap_ranges = []
    for polygon_index in np.flatnonzero(loop_totals == profile_resolution):
        start = loop_starts[polygon_index]
        cap = loop_vertices[start : start + profile_resolution]
        cap_ranges.append((int(cap.min()), int(cap.max()) + 1))
    cap_ranges.sort()
    if len(cap_ranges) % 2:
        raise ValueError(f"Expected paired tube caps, got {len(cap_ranges)}")

    stems = []
    for index in range(0, len(cap_ranges), 2):
        first, second = cap_ranges[index : index + 2]
        start = min(first[0], second[0])
        stop = max(first[1], second[1])
        if (stop - start) % profile_resolution:
            raise ValueError(
                f"Tube vertex range {(start, stop)} is not divisible by "
                f"{profile_resolution=}"
            )
        stems.append((start, stop))

    covered = sum(stop - start for start, stop in stems)
    if covered != len(mesh.vertices):
        raise ValueError(
            f"Tube ranges cover {covered} of {len(mesh.vertices)} vertices"
        )
    return stems


def _read_point_attributes(mesh: bpy.types.Mesh) -> dict[str, tuple]:
    result = {}
    for attribute in mesh.attributes:
        if (
            attribute.domain != "POINT"
            or attribute.name == "position"
            or butil.blender_internal_attr(attribute)
        ):
            continue
        layout = _POINT_LAYOUTS.get(attribute.data_type)
        if layout is None:
            raise ValueError(
                f"Unsupported tree point attribute {attribute.name}: "
                f"{attribute.data_type}"
            )
        field, width, dtype = layout
        result[attribute.name] = (
            attribute.data_type,
            field,
            width,
            dtype,
            _foreach(attribute.data, field, width, dtype),
        )
    return result


def _catmull_clark_vertex_positions(
    mesh: bpy.types.Mesh, vertices: np.ndarray
) -> np.ndarray:
    """Return one Catmull-Clark update without creating refined topology."""

    loop_vertices = _foreach(mesh.loops, "vertex_index", 1, np.int32)
    loop_totals = _foreach(mesh.polygons, "loop_total", 1, np.int32)
    polygon_indices = np.repeat(np.arange(len(mesh.polygons)), loop_totals)

    face_sums = np.zeros((len(mesh.polygons), 3), dtype=np.float64)
    np.add.at(face_sums, polygon_indices, vertices[loop_vertices])
    face_centers = face_sums / loop_totals[:, None]
    face_sum_per_vertex = np.zeros_like(vertices)
    face_count_per_vertex = np.zeros(len(vertices), dtype=np.int32)
    np.add.at(face_sum_per_vertex, loop_vertices, face_centers[polygon_indices])
    np.add.at(face_count_per_vertex, loop_vertices, 1)

    edge_vertices = _foreach(mesh.edges, "vertices", 2, np.int32)
    edge_midpoints = vertices[edge_vertices].mean(axis=1)
    edge_sum_per_vertex = np.zeros_like(vertices)
    edge_count_per_vertex = np.zeros(len(vertices), dtype=np.int32)
    for endpoint in range(2):
        indices = edge_vertices[:, endpoint]
        np.add.at(edge_sum_per_vertex, indices, edge_midpoints)
        np.add.at(edge_count_per_vertex, indices, 1)

    if not np.array_equal(face_count_per_vertex, edge_count_per_vertex):
        raise ValueError("Expected closed independent tree tubes")
    degree = edge_count_per_vertex
    if np.any(degree == 0):
        raise ValueError("Tree tube contains isolated vertices")
    face_average = face_sum_per_vertex / degree[:, None]
    edge_average = edge_sum_per_vertex / degree[:, None]
    return (
        face_average + 2 * edge_average + (degree - 3)[:, None] * vertices
    ) / degree[:, None]


def retessellate_stems(
    obj: bpy.types.Object,
    face_size: float,
    profile_resolution: int = 20,
    edge_length_ceiling_factor: float = 1.0,
) -> None:
    _require_kernel()
    """Replace tube grids with a strict pre-displacement edge ceiling."""

    if obj.type != "MESH":
        raise TypeError(f"Expected mesh tree, got {obj.type}")
    if face_size <= 0:
        raise ValueError(f"Expected positive {face_size=}")
    if edge_length_ceiling_factor <= 0:
        raise ValueError(f"Expected positive {edge_length_ceiling_factor=}")

    source_mesh = obj.data
    source_mesh_name = source_mesh.name
    source_vertices = _foreach(source_mesh.vertices, "co", 3, np.float64)
    stems = _stem_ranges(source_mesh, profile_resolution)
    source_attributes = _read_point_attributes(source_mesh)
    materials = list(source_mesh.materials)

    vertex_chunks = []
    loop_chunks = []
    loop_total_chunks = []
    source_lower_chunks = []
    source_upper_chunks = []
    source_factor_chunks = []
    vertex_offset = 0

    for start, stop in stems:
        ring_count = (stop - start) // profile_resolution
        source_rings = _smooth_stem(
            source_vertices[start:stop].reshape((-1, profile_resolution, 3))
        )
        (
            stem_vertices,
            stem_loops,
            stem_loop_totals,
            source_indices,
            source_factors,
        ) = _mesh_stem(
            source_rings,
            face_size,
            edge_length_ceiling_factor,
        )
        vertex_chunks.append(stem_vertices)
        loop_chunks.append(stem_loops + vertex_offset)
        loop_total_chunks.append(stem_loop_totals)
        source_lower_chunks.append(start + source_indices * profile_resolution)
        source_upper_chunks.append(
            start + np.minimum(source_indices + 1, ring_count - 1) * profile_resolution
        )
        source_factor_chunks.append(source_factors)
        vertex_offset += len(stem_vertices)

    vertices = np.concatenate(vertex_chunks).astype(np.float32, copy=False)
    loop_vertices = np.concatenate(loop_chunks).astype(np.int32, copy=False)
    loop_totals = np.concatenate(loop_total_chunks)
    face_offsets = np.empty(len(loop_totals) + 1, dtype=np.int32)
    face_offsets[0] = 0
    np.cumsum(loop_totals, out=face_offsets[1:])
    source_lower = np.concatenate(source_lower_chunks)
    source_upper = np.concatenate(source_upper_chunks)
    source_factors = np.concatenate(source_factor_chunks)

    mesh = bpy.data.meshes.new(source_mesh_name + ".retessellated")
    mesh.vertices.add(len(vertices))
    mesh.loops.add(len(loop_vertices))
    mesh.polygons.add(len(loop_totals))
    direct_layout = (
        bpy.app.version[:2] == (4, 2)
        and len(vertices) > 1
        and len(loop_vertices) > 1
        and len(loop_totals) > 1
        and mesh.vertices[1].as_pointer() - mesh.vertices[0].as_pointer() == 12
        and mesh.loops[1].as_pointer() - mesh.loops[0].as_pointer() == 4
        and mesh.polygons[1].as_pointer() - mesh.polygons[0].as_pointer() == 4
    )
    if direct_layout:
        _write_mesh_buffers(
            mesh.vertices[0].as_pointer(),
            mesh.loops[0].as_pointer(),
            mesh.polygons[0].as_pointer(),
            vertices,
            loop_vertices,
            face_offsets,
        )
    else:
        mesh.vertices.foreach_set("co", vertices.reshape(-1))
        mesh.loops.foreach_set("vertex_index", loop_vertices)
        mesh.polygons.foreach_set("loop_start", face_offsets[:-1])
        mesh.polygons.foreach_set("loop_total", loop_totals)
    mesh.update(calc_edges=True)

    for material in materials:
        mesh.materials.append(material)
    for name, (data_type, field, width, dtype, _) in source_attributes.items():
        attribute = mesh.attributes.new(name, data_type, "POINT")
        source_values = source_attributes[name][4]
        if data_type in {"FLOAT", "FLOAT_VECTOR"}:
            reshape = (len(source_factors),) + (1,) * (source_values.ndim - 1)
            weights = source_factors.reshape(reshape)
            values = (
                source_values[source_lower] * (1 - weights)
                + source_values[source_upper] * weights
            )
        else:
            nearest = np.where(source_factors >= 0.5, source_upper, source_lower)
            values = source_values[nearest]
        attribute.data.foreach_set(field, values.astype(dtype, copy=False).reshape(-1))
    # The new faces do not map onto the source's, so carry its flat or smooth
    # shading over as a whole. A new boolean attribute is all False (smooth), so
    # it is set all True when most of the source's values are True.
    for name, data_type, domain in (
        ("sharp_edge", "BOOLEAN", "EDGE"),
        ("sharp_face", "BOOLEAN", "FACE"),
    ):
        source = source_mesh.attributes.get(name)
        if source is None:
            continue
        values = np.empty(len(source.data), dtype=bool)
        source.data.foreach_get("value", values)
        attribute = mesh.attributes.new(name, data_type, domain)
        if len(values) and values.mean() >= 0.5:
            attribute.data.foreach_set(
                "value", np.ones(len(attribute.data), dtype=bool)
            )

    obj.data = mesh
    bpy.data.meshes.remove(source_mesh)
    mesh.name = source_mesh_name
    logger.debug(
        "Retessellated %s into %d stems, %d vertices, and %d polygons",
        obj.name,
        len(stems),
        len(mesh.vertices),
        len(mesh.polygons),
    )
