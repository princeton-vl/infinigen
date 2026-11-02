# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as cnp
from libc.math cimport ceil, sqrt
from libc.stdint cimport uint64_t, uintptr_t
from libc.string cimport memcpy
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

# Equal circumferential and longitudinal spacing minimizes triangles under the diagonal edge ceiling.
DEF RING_SPACING_FACTOR = 1.0
# Ignore numerically negligible slivers shorter than 5% of the edge ceiling.
DEF MIN_EDGE_RATIO_SQUARED = 1.0 / 400.0
DEF MAX_REFINEMENT_DEPTH = 32
DEF MIN_RING_VERTICES = 6

cnp.import_array()


def write_mesh_buffers(
    uintptr_t vertex_pointer,
    uintptr_t loop_pointer,
    uintptr_t polygon_pointer,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] vertices,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] loops,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] face_offsets,
):
    """Copy buffers into Blender 4.2 storage after Python allocates it."""

    if vertices.shape[1] != 3:
        raise ValueError("Expected three coordinates per vertex")
    if face_offsets.shape[0] < 2:
        raise ValueError("Expected at least one face")
    memcpy(
        <void*>vertex_pointer,
        vertices.data,
        vertices.shape[0] * 3 * sizeof(float),
    )
    memcpy(<void*>loop_pointer, loops.data, loops.shape[0] * sizeof(int))
    memcpy(
        <void*>polygon_pointer,
        face_offsets.data,
        face_offsets.shape[0] * sizeof(int),
    )


cdef inline void append_vertex(
    vector[double]& vertices,
    vector[int]& source_index,
    vector[double]& source_factor,
    double x,
    double y,
    double z,
    int lower,
    double factor,
):
    vertices.push_back(x)
    vertices.push_back(y)
    vertices.push_back(z)
    source_index.push_back(lower)
    source_factor.push_back(factor)


cdef inline double triangle_thinness(
    vector[double]& vertices, int a, int b, int c
):
    cdef double ux, uy, uz, vx, vy, vz, wx, wy, wz
    cdef double ab2, bc2, ca2, cross_x, cross_y, cross_z
    ux = vertices[3 * b] - vertices[3 * a]
    uy = vertices[3 * b + 1] - vertices[3 * a + 1]
    uz = vertices[3 * b + 2] - vertices[3 * a + 2]
    vx = vertices[3 * c] - vertices[3 * a]
    vy = vertices[3 * c + 1] - vertices[3 * a + 1]
    vz = vertices[3 * c + 2] - vertices[3 * a + 2]
    wx = vertices[3 * c] - vertices[3 * b]
    wy = vertices[3 * c + 1] - vertices[3 * b + 1]
    wz = vertices[3 * c + 2] - vertices[3 * b + 2]
    ab2 = ux * ux + uy * uy + uz * uz
    bc2 = wx * wx + wy * wy + wz * wz
    ca2 = vx * vx + vy * vy + vz * vz
    cross_x = uy * vz - uz * vy
    cross_y = uz * vx - ux * vz
    cross_z = ux * vy - uy * vx
    return max(ab2, bc2, ca2) / sqrt(
        cross_x * cross_x + cross_y * cross_y + cross_z * cross_z
    )


cdef inline double edge_squared(vector[double]& vertices, int a, int b):
    cdef double x = vertices[3 * b] - vertices[3 * a]
    cdef double y = vertices[3 * b + 1] - vertices[3 * a + 1]
    cdef double z = vertices[3 * b + 2] - vertices[3 * a + 2]
    return x * x + y * y + z * z


cdef int append_midpoint(
    vector[double]& vertices,
    vector[int]& source_index,
    vector[double]& source_factor,
    unordered_map[uint64_t, int]& midpoint_cache,
    int a,
    int b,
):
    cdef uint64_t low = min(a, b)
    cdef uint64_t high = max(a, b)
    cdef uint64_t key = (low << 32) | high
    cdef double source_coordinate
    cdef int lower, midpoint
    if midpoint_cache.count(key):
        return midpoint_cache[key]
    source_coordinate = 0.5 * (
        source_index[a] + source_factor[a] + source_index[b] + source_factor[b]
    )
    lower = <int>source_coordinate
    midpoint = vertices.size() // 3
    append_vertex(
        vertices,
        source_index,
        source_factor,
        0.5 * (vertices[3 * a] + vertices[3 * b]),
        0.5 * (vertices[3 * a + 1] + vertices[3 * b + 1]),
        0.5 * (vertices[3 * a + 2] + vertices[3 * b + 2]),
        lower,
        source_coordinate - lower,
    )
    midpoint_cache[key] = midpoint
    return midpoint


cdef void emit_refined_triangle(
    vector[double]& vertices,
    vector[int]& source_index,
    vector[double]& source_factor,
    unordered_map[uint64_t, int]& midpoint_cache,
    vector[int]& loops,
    vector[int]& totals,
    int a,
    int b,
    int c,
    double edge_size_squared,
    int depth=0,
):
    cdef double ab2 = edge_squared(vertices, a, b)
    cdef double bc2 = edge_squared(vertices, b, c)
    cdef double ca2 = edge_squared(vertices, c, a)
    cdef int edge_start, edge_end, opposite, midpoint
    if min(ab2, bc2, ca2) < edge_size_squared * MIN_EDGE_RATIO_SQUARED:
        return
    if max(ab2, bc2, ca2) <= edge_size_squared:
        loops.push_back(a)
        loops.push_back(b)
        loops.push_back(c)
        totals.push_back(3)
        return
    if depth >= MAX_REFINEMENT_DEPTH:
        raise ValueError("Edge refinement did not converge")

    if ab2 >= bc2 and ab2 >= ca2:
        edge_start, edge_end, opposite = a, b, c
    elif bc2 >= ca2:
        edge_start, edge_end, opposite = b, c, a
    else:
        edge_start, edge_end, opposite = c, a, b

    midpoint = append_midpoint(
        vertices, source_index, source_factor, midpoint_cache, edge_start, edge_end
    )
    emit_refined_triangle(
        vertices, source_index, source_factor, midpoint_cache, loops, totals,
        edge_start, midpoint, opposite, edge_size_squared, depth + 1,
    )
    emit_refined_triangle(
        vertices, source_index, source_factor, midpoint_cache, loops, totals,
        midpoint, edge_end, opposite, edge_size_squared, depth + 1,
    )


cdef void connect_rings(
    vector[double]& vertices,
    vector[int]& source_index,
    vector[double]& source_factor,
    unordered_map[uint64_t, int]& midpoint_cache,
    vector[int]& loops,
    vector[int]& totals,
    int first_offset,
    int first_count,
    int second_offset,
    int second_count,
    double edge_size,
):
    cdef int i = 0, j = 0, a, b, c, d
    cdef double split_ac, split_bd, edge_size_squared = edge_size * edge_size
    while i < first_count or j < second_count:
        a = first_offset + (i % first_count)
        d = second_offset + (j % second_count)
        if (i + 1) * second_count == (j + 1) * first_count:
            b = first_offset + ((i + 1) % first_count)
            c = second_offset + ((j + 1) % second_count)
            split_ac = max(
                triangle_thinness(vertices, a, b, c),
                triangle_thinness(vertices, a, c, d),
            )
            split_bd = max(
                triangle_thinness(vertices, a, b, d),
                triangle_thinness(vertices, b, c, d),
            )
            if split_ac <= split_bd:
                emit_refined_triangle(
                    vertices, source_index, source_factor, midpoint_cache,
                    loops, totals, a, b, c, edge_size_squared,
                )
                emit_refined_triangle(
                    vertices, source_index, source_factor, midpoint_cache,
                    loops, totals, a, c, d, edge_size_squared,
                )
            else:
                emit_refined_triangle(
                    vertices, source_index, source_factor, midpoint_cache,
                    loops, totals, a, b, d, edge_size_squared,
                )
                emit_refined_triangle(
                    vertices, source_index, source_factor, midpoint_cache,
                    loops, totals, b, c, d, edge_size_squared,
                )
            i += 1
            j += 1
        elif (i + 1) * second_count < (j + 1) * first_count:
            b = first_offset + ((i + 1) % first_count)
            emit_refined_triangle(
                vertices, source_index, source_factor, midpoint_cache, loops,
                totals, a, b, d, edge_size_squared,
            )
            i += 1
        else:
            c = second_offset + ((j + 1) % second_count)
            emit_refined_triangle(
                vertices, source_index, source_factor, midpoint_cache, loops,
                totals, a, c, d, edge_size_squared,
            )
            j += 1


cdef double ring_perimeter(
    double[:, :, :] source,
    int lower,
    double factor,
    int profile_resolution,
):
    cdef int p, q, axis
    cdef double x, y, delta, edge_squared, perimeter = 0.0
    for p in range(profile_resolution):
        q = (p + 1) % profile_resolution
        edge_squared = 0.0
        for axis in range(3):
            x = source[lower, p, axis] * (1.0 - factor) + source[lower + 1, p, axis] * factor
            y = source[lower, q, axis] * (1.0 - factor) + source[lower + 1, q, axis] * factor
            delta = y - x
            edge_squared += delta * delta
        perimeter += sqrt(edge_squared)
    return perimeter


cdef void append_resampled_ring(
    vector[double]& vertices,
    vector[int]& source_index,
    vector[double]& source_factor,
    double[:, :, :] source,
    int lower,
    double factor,
    int profile_resolution,
    int output_count,
):
    cdef int output, p = 0, q, axis
    cdef double perimeter = ring_perimeter(source, lower, factor, profile_resolution)
    cdef double target, accumulated = 0.0, edge_length, blend
    cdef double x, y, delta, edge_squared
    for output in range(output_count):
        target = <double>output * perimeter / output_count
        while True:
            q = (p + 1) % profile_resolution
            edge_squared = 0.0
            for axis in range(3):
                x = source[lower, p, axis] * (1.0 - factor) + source[lower + 1, p, axis] * factor
                y = source[lower, q, axis] * (1.0 - factor) + source[lower + 1, q, axis] * factor
                delta = y - x
                edge_squared += delta * delta
            edge_length = sqrt(edge_squared)
            if target <= accumulated + edge_length or p + 1 >= profile_resolution:
                break
            accumulated += edge_length
            p += 1
        blend = 0.0 if edge_length == 0.0 else (target - accumulated) / edge_length
        append_vertex(
            vertices,
            source_index,
            source_factor,
            source[lower, p, 0] * (1.0 - factor) * (1.0 - blend)
                + source[lower + 1, p, 0] * factor * (1.0 - blend)
                + source[lower, q, 0] * (1.0 - factor) * blend
                + source[lower + 1, q, 0] * factor * blend,
            source[lower, p, 1] * (1.0 - factor) * (1.0 - blend)
                + source[lower + 1, p, 1] * factor * (1.0 - blend)
                + source[lower, q, 1] * (1.0 - factor) * blend
                + source[lower + 1, q, 1] * factor * blend,
            source[lower, p, 2] * (1.0 - factor) * (1.0 - blend)
                + source[lower + 1, p, 2] * factor * (1.0 - blend)
                + source[lower, q, 2] * (1.0 - factor) * blend
                + source[lower + 1, q, 2] * factor * blend,
            lower,
            factor,
        )



def smooth_stem(cnp.ndarray[cnp.float64_t, ndim=3] source_array):
    """Apply the exact Catmull-Clark vertex rule to a closed tube grid."""

    cdef double[:, :, :] source = source_array
    cdef int ring_count = source.shape[0]
    cdef int profile_resolution = source.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=3] output_array = np.empty_like(source_array)
    cdef double[:, :, :] output = output_array
    cdef int ring, profile, axis, previous_profile, next_profile
    cdef int segment, face_profile, face_count, edge_count
    cdef double face_sum, edge_sum, face_center, edge_midpoint, cap_center

    if ring_count < 2 or profile_resolution < 3:
        raise ValueError("A stem requires at least two rings and three profile vertices")

    for ring in range(ring_count):
        for profile in range(profile_resolution):
            previous_profile = (profile + profile_resolution - 1) % profile_resolution
            next_profile = (profile + 1) % profile_resolution
            for axis in range(3):
                face_sum = 0.0
                face_count = 0
                if ring > 0:
                    segment = ring - 1
                    for face_profile in (previous_profile, profile):
                        face_center = (
                            source[segment, face_profile, axis]
                            + source[segment, (face_profile + 1) % profile_resolution, axis]
                            + source[segment + 1, (face_profile + 1) % profile_resolution, axis]
                            + source[segment + 1, face_profile, axis]
                        ) * 0.25
                        face_sum += face_center
                        face_count += 1
                if ring + 1 < ring_count:
                    segment = ring
                    for face_profile in (previous_profile, profile):
                        face_center = (
                            source[segment, face_profile, axis]
                            + source[segment, (face_profile + 1) % profile_resolution, axis]
                            + source[segment + 1, (face_profile + 1) % profile_resolution, axis]
                            + source[segment + 1, face_profile, axis]
                        ) * 0.25
                        face_sum += face_center
                        face_count += 1
                if ring == 0 or ring + 1 == ring_count:
                    cap_center = 0.0
                    for face_profile in range(profile_resolution):
                        cap_center += source[ring, face_profile, axis]
                    face_sum += cap_center / profile_resolution
                    face_count += 1

                edge_sum = (
                    (source[ring, profile, axis] + source[ring, previous_profile, axis]) * 0.5
                    + (source[ring, profile, axis] + source[ring, next_profile, axis]) * 0.5
                )
                edge_count = 2
                if ring > 0:
                    edge_midpoint = (
                        source[ring, profile, axis] + source[ring - 1, profile, axis]
                    ) * 0.5
                    edge_sum += edge_midpoint
                    edge_count += 1
                if ring + 1 < ring_count:
                    edge_midpoint = (
                        source[ring, profile, axis] + source[ring + 1, profile, axis]
                    ) * 0.5
                    edge_sum += edge_midpoint
                    edge_count += 1

                if face_count != edge_count:
                    raise ValueError("Expected equal closed-tube face and edge degree")
                output[ring, profile, axis] = (
                    face_sum / face_count
                    + 2.0 * edge_sum / edge_count
                    + (edge_count - 3.0) * source[ring, profile, axis]
                ) / edge_count
    return output_array


def mesh_stem(
    cnp.ndarray[cnp.float64_t, ndim=3] source_array,
    double face_size,
    double edge_length_ceiling_factor,
):
    """Generate a swept cylinder with a strict pre-displacement edge ceiling."""

    cdef double[:, :, :] source = source_array
    cdef int ring_count = source.shape[0]
    cdef int profile_resolution = source.shape[1]
    cdef double edge_size = face_size * edge_length_ceiling_factor
    cdef double sampling_edge = edge_size * RING_SPACING_FACTOR / sqrt(2.0)
    cdef vector[double] vertices
    cdef vector[int] loops
    cdef vector[int] totals
    cdef vector[int] source_index
    cdef vector[double] source_factor
    cdef unordered_map[uint64_t, int] midpoint_cache
    cdef vector[int] ring_offsets
    cdef vector[int] ring_vertex_counts
    cdef int segment, step, steps, p, q
    cdef int previous_offset = -1
    cdef int previous_count = 0
    cdef int current_offset, current_count
    cdef int endpoint_steps_a, endpoint_steps_b
    cdef int axis
    cdef double factor, x, y, z, maximum, length
    cdef double maximum_circumference_edge, longitudinal_limit
    cdef double perimeter, perimeter_a, perimeter_b

    if ring_count < 2 or profile_resolution < 3:
        raise ValueError("A stem requires at least two rings and three profile vertices")

    for segment in range(ring_count - 1):
        maximum = 0.0
        for p in range(profile_resolution):
            length = 0.0
            for axis in range(3):
                x = source[segment + 1, p, axis] - source[segment, p, axis]
                length += x * x
            length = sqrt(length)
            if length > maximum:
                maximum = length

        perimeter_a = ring_perimeter(source, segment, 0.0, profile_resolution)
        perimeter_b = ring_perimeter(source, segment, 1.0, profile_resolution)
        endpoint_steps_a = max(MIN_RING_VERTICES, <int>ceil(perimeter_a / sampling_edge))
        endpoint_steps_b = max(MIN_RING_VERTICES, <int>ceil(perimeter_b / sampling_edge))
        current_count = max(endpoint_steps_a, endpoint_steps_b)
        maximum_circumference_edge = max(perimeter_a, perimeter_b) / current_count
        longitudinal_limit = min(
            sampling_edge,
            sqrt(
                edge_size * edge_size
                - maximum_circumference_edge * maximum_circumference_edge
            ),
        )
        steps = max(1, <int>ceil(maximum / longitudinal_limit))

        for step in range(steps):
            factor = <double>step / steps
            current_count = max(endpoint_steps_a, endpoint_steps_b)
            current_offset = vertices.size() // 3
            ring_offsets.push_back(current_offset)
            ring_vertex_counts.push_back(current_count)
            append_resampled_ring(
                vertices,
                source_index,
                source_factor,
                source,
                segment,
                factor,
                profile_resolution,
                current_count,
            )
            if previous_offset >= 0:
                connect_rings(
                    vertices, source_index, source_factor, midpoint_cache, loops, totals,
                    previous_offset, previous_count, current_offset, current_count,
                    edge_size,
                )
            previous_offset = current_offset
            previous_count = current_count

    segment = ring_count - 2
    factor = 1.0
    perimeter = ring_perimeter(source, segment, factor, profile_resolution)
    current_count = max(MIN_RING_VERTICES, <int>ceil(perimeter / sampling_edge))
    current_offset = vertices.size() // 3
    ring_offsets.push_back(current_offset)
    ring_vertex_counts.push_back(current_count)
    append_resampled_ring(
        vertices,
        source_index,
        source_factor,
        source,
        segment,
        factor,
        profile_resolution,
        current_count,
    )
    connect_rings(
        vertices, source_index, source_factor, midpoint_cache, loops, totals,
        previous_offset, previous_count, current_offset, current_count,
        edge_size,
    )

    # Reduce ring counts toward each cap center to avoid skinny fan triangles.
    cdef int cap, boundary_offset, boundary_count, cap_steps, cap_step
    cdef int center_index, inner_offset, inner_count, outer_offset, outer_count
    cdef int k, source_k, source_next
    cdef double cx, cy, cz, scale, sample, blend
    for cap in range(2):
        if cap == 0:
            boundary_offset = ring_offsets[0]
            boundary_count = ring_vertex_counts[0]
            segment = 0
        else:
            boundary_offset = ring_offsets[ring_offsets.size() - 1]
            boundary_count = ring_vertex_counts[ring_vertex_counts.size() - 1]
            segment = ring_count - 1
        cx = cy = cz = 0.0
        for k in range(boundary_count):
            cx += vertices[3 * (boundary_offset + k)]
            cy += vertices[3 * (boundary_offset + k) + 1]
            cz += vertices[3 * (boundary_offset + k) + 2]
        cx /= boundary_count
        cy /= boundary_count
        cz /= boundary_count
        cap_steps = max(1, boundary_count / MIN_RING_VERTICES)
        center_index = vertices.size() // 3
        append_vertex(vertices, source_index, source_factor, cx, cy, cz, segment, 0.0)
        inner_offset = -1
        inner_count = 1
        for cap_step in range(1, cap_steps):
            scale = <double>cap_step / cap_steps
            outer_offset = vertices.size() // 3
            outer_count = MIN_RING_VERTICES * cap_step
            for k in range(outer_count):
                sample = <double>k * boundary_count / outer_count
                source_k = <int>sample
                blend = sample - source_k
                source_next = (source_k + 1) % boundary_count
                if cap == 0:
                    source_k = (boundary_count - source_k) % boundary_count
                    source_next = (boundary_count - source_next) % boundary_count
                x = (
                    vertices[3 * (boundary_offset + source_k)] * (1.0 - blend)
                    + vertices[3 * (boundary_offset + source_next)] * blend
                )
                y = (
                    vertices[3 * (boundary_offset + source_k) + 1] * (1.0 - blend)
                    + vertices[3 * (boundary_offset + source_next) + 1] * blend
                )
                z = (
                    vertices[3 * (boundary_offset + source_k) + 2] * (1.0 - blend)
                    + vertices[3 * (boundary_offset + source_next) + 2] * blend
                )
                append_vertex(
                    vertices, source_index, source_factor,
                    cx + (x - cx) * scale,
                    cy + (y - cy) * scale,
                    cz + (z - cz) * scale,
                    segment, 0.0,
                )
            if inner_offset < 0:
                for k in range(outer_count):
                    emit_refined_triangle(
                        vertices, source_index, source_factor, midpoint_cache, loops, totals,
                        center_index, outer_offset + k,
                        outer_offset + (k + 1) % outer_count,
                        edge_size * edge_size,
                    )
            else:
                connect_rings(
                    vertices, source_index, source_factor, midpoint_cache,
                    loops, totals, inner_offset, inner_count, outer_offset,
                    outer_count, edge_size,
                )
            inner_offset = outer_offset
            inner_count = outer_count
        if inner_offset < 0:
            for k in range(boundary_count):
                q = boundary_count - 1 - k if cap == 0 else k
                if cap == 0:
                    source_k = boundary_count - 1 - ((k + 1) % boundary_count)
                else:
                    source_k = (k + 1) % boundary_count
                emit_refined_triangle(
                    vertices, source_index, source_factor, midpoint_cache, loops, totals,
                    center_index, boundary_offset + q,
                    boundary_offset + source_k,
                    edge_size * edge_size,
                )
        elif cap == 0:
            outer_offset = vertices.size() // 3
            for k in range(boundary_count):
                q = (boundary_count - k) % boundary_count
                append_vertex(
                    vertices, source_index, source_factor,
                    vertices[3 * (boundary_offset + q)],
                    vertices[3 * (boundary_offset + q) + 1],
                    vertices[3 * (boundary_offset + q) + 2],
                    segment, 0.0,
                )
            connect_rings(
                vertices, source_index, source_factor, midpoint_cache, loops,
                totals, inner_offset, inner_count, outer_offset, boundary_count,
                edge_size,
            )
        else:
            connect_rings(
                vertices, source_index, source_factor, midpoint_cache, loops,
                totals, inner_offset, inner_count, boundary_offset, boundary_count,
                edge_size,
            )
    cdef cnp.ndarray[cnp.float64_t, ndim=2] vertex_array = np.empty(
        (vertices.size() // 3, 3), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.int32_t, ndim=1] loop_array = np.empty(
        loops.size(), dtype=np.int32
    )
    cdef cnp.ndarray[cnp.int32_t, ndim=1] total_array = np.empty(
        totals.size(), dtype=np.int32
    )
    cdef cnp.ndarray[cnp.int32_t, ndim=1] source_index_array = np.empty(
        source_index.size(), dtype=np.int32
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1] source_factor_array = np.empty(
        source_factor.size(), dtype=np.float64
    )
    if vertices.size():
        memcpy(vertex_array.data, vertices.data(), vertices.size() * sizeof(double))
    if loops.size():
        memcpy(loop_array.data, loops.data(), loops.size() * sizeof(int))
    if totals.size():
        memcpy(total_array.data, totals.data(), totals.size() * sizeof(int))
    if source_index.size():
        memcpy(
            source_index_array.data,
            source_index.data(),
            source_index.size() * sizeof(int),
        )
        memcpy(
            source_factor_array.data,
            source_factor.data(),
            source_factor.size() * sizeof(double),
        )
    return vertex_array, loop_array, total_array, source_index_array, source_factor_array
