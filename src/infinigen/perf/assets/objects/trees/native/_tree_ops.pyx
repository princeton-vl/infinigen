# Copyright (C) 2026 Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as cnp
from libc.float cimport DBL_MAX
from libc.math cimport sqrt

cnp.import_array()


def nearest_distances(
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] attractors,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] vertices,
):
    """Return each attractor's distance and index to its nearest vertex."""
    if attractors.shape[1] != 3 or vertices.shape[1] != 3:
        raise ValueError("attractors and vertices must have shape (N, 3)")
    if vertices.shape[0] == 0:
        raise ValueError("vertices must not be empty")

    cdef Py_ssize_t attractor_count = attractors.shape[0]
    cdef Py_ssize_t vertex_count = vertices.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] distances = np.empty(
        attractor_count, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] closest = np.empty(
        attractor_count, dtype=np.int64
    )
    cdef Py_ssize_t i, j, best_index
    cdef double dx, dy, dz, distance_squared, best_squared

    with nogil:
        for i in range(attractor_count):
            best_squared = DBL_MAX
            best_index = 0
            for j in range(vertex_count):
                dx = attractors[i, 0] - vertices[j, 0]
                dy = attractors[i, 1] - vertices[j, 1]
                dz = attractors[i, 2] - vertices[j, 2]
                distance_squared = dx * dx + dy * dy + dz * dz
                if distance_squared < best_squared:
                    best_squared = distance_squared
                    best_index = j
            distances[i] = sqrt(best_squared)
            closest[i] = best_index

    return distances, closest


def filter_update_attractors(
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] attractors,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] vertices,
    double distance_threshold,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] current_minimum,
    cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] current_match,
    Py_ssize_t index_offset=0,
):
    """Filter reached attractors and update exact nearest-match state."""
    if attractors.shape[1] != 3 or vertices.shape[1] != 3:
        raise ValueError("attractors and vertices must have shape (N, 3)")
    if vertices.shape[0] == 0:
        raise ValueError("vertices must not be empty")
    if (
        current_minimum.shape[0] != attractors.shape[0]
        or current_match.shape[0] != attractors.shape[0]
    ):
        raise ValueError("attractor state must have one row per attractor")

    cdef Py_ssize_t attractor_count = attractors.shape[0]
    cdef Py_ssize_t vertex_count = vertices.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] distances = np.empty(
        attractor_count, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] closest = np.empty(
        attractor_count, dtype=np.int64
    )
    cdef Py_ssize_t i, j, best_index, kept_count = 0, output_index = 0
    cdef double dx, dy, dz, distance_squared, best_squared

    with nogil:
        for i in range(attractor_count):
            best_squared = DBL_MAX
            best_index = 0
            for j in range(vertex_count):
                dx = attractors[i, 0] - vertices[j, 0]
                dy = attractors[i, 1] - vertices[j, 1]
                dz = attractors[i, 2] - vertices[j, 2]
                distance_squared = dx * dx + dy * dy + dz * dz
                if distance_squared < best_squared:
                    best_squared = distance_squared
                    best_index = j
            distances[i] = sqrt(best_squared)
            closest[i] = best_index
            if distances[i] > distance_threshold:
                kept_count += 1

    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] kept_attractors = np.empty(
        (kept_count, 3), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] kept_minimum = np.empty(
        kept_count, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] kept_match = np.empty(
        kept_count, dtype=np.int64
    )

    with nogil:
        for i in range(attractor_count):
            if distances[i] <= distance_threshold:
                continue
            kept_attractors[output_index, 0] = attractors[i, 0]
            kept_attractors[output_index, 1] = attractors[i, 1]
            kept_attractors[output_index, 2] = attractors[i, 2]
            if distances[i] < current_minimum[i]:
                kept_minimum[output_index] = distances[i]
                kept_match[output_index] = closest[i] + index_offset
            else:
                kept_minimum[output_index] = current_minimum[i]
                kept_match[output_index] = current_match[i]
            output_index += 1

    return kept_attractors, kept_minimum, kept_match


def grouped_means(
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] attractors,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] vertices,
    cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] matches,
):
    """Mean (attractor - matched vertex) vectors in stable match order."""
    if attractors.shape[1] != 3 or vertices.shape[1] != 3:
        raise ValueError("attractors and vertices must have shape (N, 3)")
    if attractors.shape[0] != matches.shape[0]:
        raise ValueError("attractors and matches must have equal lengths")

    cdef Py_ssize_t count = matches.shape[0]
    cdef Py_ssize_t i, group_count = 0
    cdef cnp.int64_t match, max_match = -1
    for i in range(count):
        match = matches[i]
        if match >= vertices.shape[0]:
            raise ValueError("match index is outside vertices")
        if match > max_match:
            max_match = match

    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] counts = np.zeros(
        max_match + 1, dtype=np.int64
    )
    for i in range(count):
        match = matches[i]
        if match >= 0:
            if counts[match] == 0:
                group_count += 1
            counts[match] += 1

    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] matched = np.empty(
        group_count, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] means = np.zeros(
        (group_count, 3), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] output_index = np.empty(
        max_match + 1, dtype=np.int64
    )
    cdef Py_ssize_t out = 0
    for i in range(max_match + 1):
        if counts[i] != 0:
            matched[out] = i
            output_index[i] = out
            out += 1

    with nogil:
        for i in range(count):
            match = matches[i]
            if match >= 0:
                out = output_index[match]
                means[out, 0] += attractors[i, 0] - vertices[match, 0]
                means[out, 1] += attractors[i, 1] - vertices[match, 1]
                means[out, 2] += attractors[i, 2] - vertices[match, 2]
        for out in range(group_count):
            match = matched[out]
            means[out, 0] /= counts[match]
            means[out, 1] /= counts[match]
            means[out, 2] /= counts[match]

    return matched, means
