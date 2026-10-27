# SPDX-License-Identifier: BSD-3-Clause
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
from libc.stdint cimport int32_t
from libc.stddef cimport size_t
from libc.math cimport ceil, floor, fmin, fmax, fabs

cnp.import_array()

def field_residual_moments(
    cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] raster,
    cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] mask,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] points,
    cnp.ndarray[cnp.int32_t, ndim=2, mode="c"] faces,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] base,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] residual,
):
    """Integrate reconstruction residuals over covered raster samples."""
    cdef int h=raster.shape[0], w=raster.shape[1], channels=raster.shape[2]
    if mask.shape[0]!=h or mask.shape[1]!=w or points.shape[1]!=2 or faces.shape[1]!=3:
        raise ValueError("incompatible raster, mask, chart, or faces")
    if base.shape[0]!=points.shape[0] or residual.shape[0]!=points.shape[0] or base.shape[1]!=channels or residual.shape[1]!=channels:
        raise ValueError("incompatible field dimensions")
    if faces.size and (np.min(faces)<0 or np.max(faces)>=len(points)):
        raise ValueError("face index out of range")
    if not np.all(np.isfinite(points)):
        raise ValueError("chart coordinates must be finite")
    cdef cnp.ndarray[cnp.float64_t, ndim=1] num=np.zeros(channels), den=np.zeros(channels)
    cdef Py_ssize_t i
    cdef int a,b,c,x,y,k,x0,x1,y0,y1
    cdef double ax,ay,ux,uy,vx,vy,det,dx,dy,u,v,t,change,error
    with nogil:
        for i in range(faces.shape[0]):
            a=faces[i,0]; b=faces[i,1]; c=faces[i,2]
            ax=points[a,0]; ay=points[a,1]
            ux=points[b,0]-ax; uy=points[b,1]-ay
            vx=points[c,0]-ax; vy=points[c,1]-ay
            det=ux*vy-uy*vx
            if fabs(det)<1e-15: continue
            x0=<int>fmin(w,fmax(0,ceil(fmin(ax,fmin(points[b,0],points[c,0])))))
            x1=<int>fmax(-1,fmin(w-1,floor(fmax(ax,fmax(points[b,0],points[c,0])))))
            y0=<int>fmin(h,fmax(0,ceil(fmin(ay,fmin(points[b,1],points[c,1])))))
            y1=<int>fmax(-1,fmin(h-1,floor(fmax(ay,fmax(points[b,1],points[c,1])))))
            for y in range(y0,y1+1):
                for x in range(x0,x1+1):
                    if not mask[y,x]: continue
                    dx=x-ax; dy=y-ay
                    u=(dx*vy-dy*vx)/det; v=(ux*dy-uy*dx)/det; t=1-u-v
                    if u<0 or v<0 or t<0: continue
                    for k in range(channels):
                        change=t*residual[a,k]+u*residual[b,k]+v*residual[c,k]
                        error=raster[y,x,k]-(t*base[a,k]+u*base[b,k]+v*base[c,k])
                        num[k]+=change*error; den[k]+=change*change
    return num,den

cdef extern from "_flat_grid_preconditioner.hpp" namespace "infinigen::heightfield":
    cdef cppclass FlatGridResult:
        vector[unsigned int] faces
        int flat_cells
        int flat_blocks

    FlatGridResult precondition_flat_grid(
        const float *positions,
        int vertex_count,
        const int32_t *vertex_ids,
        int grid_height,
        int grid_width,
        const unsigned char *complete,
        float tolerance,
        int minimum_block,
        const float *metric_scale,
    ) except + nogil

cdef extern from "meshoptimizer.h":
    size_t meshopt_simplify(
        unsigned int *destination,
        const unsigned int *indices,
        size_t index_count,
        const float *vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride,
        size_t target_index_count,
        float target_error,
        unsigned int options,
        float *result_error,
    ) noexcept nogil

    size_t meshopt_simplifyWithAttributes(
        unsigned int *destination,
        const unsigned int *indices,
        size_t index_count,
        const float *vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride,
        const float *vertex_attributes,
        size_t vertex_attributes_stride,
        const float *attribute_weights,
        size_t attribute_count,
        const unsigned char *vertex_lock,
        size_t target_index_count,
        float target_error,
        unsigned int options,
        float *result_error,
    ) noexcept nogil

    size_t meshopt_simplifyWithUpdate(
        unsigned int *indices,
        size_t index_count,
        float *vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride,
        float *vertex_attributes,
        size_t vertex_attributes_stride,
        const float *attribute_weights,
        size_t attribute_count,
        const unsigned char *vertex_lock,
        size_t target_index_count,
        float target_error,
        unsigned int options,
        float *result_error,
    ) noexcept nogil

    size_t meshopt_simplifyWithUpdatePassive(
        unsigned int *indices,
        size_t index_count,
        float *vertex_positions,
        size_t vertex_count,
        size_t vertex_positions_stride,
        float *passive_attributes,
        size_t passive_attributes_stride,
        size_t passive_attribute_count,
        const unsigned char *vertex_lock,
        size_t target_index_count,
        float target_error,
        unsigned int options,
        float *result_error,
    ) noexcept nogil



def precondition_flat_regions(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] positions,
    cnp.ndarray[cnp.int32_t, ndim=2, mode="c"] vertex_ids,
    cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] complete,
    float tolerance,
    int minimum_block,
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] metric_scale,
):
    """Replace planar grid blocks with fans that keep their boundary vertices."""
    if positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if complete.shape[0] != vertex_ids.shape[0] - 1 or complete.shape[1] != vertex_ids.shape[1] - 1:
        raise ValueError("complete must cover the cells of vertex_ids")
    if tolerance < 0 or minimum_block < 2 or metric_scale.shape[0] != 3:
        raise ValueError("invalid flat-region options")
    cdef FlatGridResult result
    with nogil:
        result = precondition_flat_grid(
            &positions[0, 0], positions.shape[0], &vertex_ids[0, 0],
            vertex_ids.shape[0], vertex_ids.shape[1], &complete[0, 0],
            tolerance, minimum_block, &metric_scale[0]
        )
    cdef Py_ssize_t count = result.faces.size()
    cdef cnp.ndarray[cnp.uint32_t, ndim=2] output = np.empty((count // 3, 3), dtype=np.uint32)
    cdef Py_ssize_t i
    for i in range(count):
        output[i // 3, i % 3] = result.faces[i]
    return output, result.flat_cells, result.flat_blocks


def simplify_geometry(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] positions,
    cnp.ndarray[cnp.uint32_t, ndim=2, mode="c"] faces,
    int target_faces,
    bint lock_border=False,
    float target_error=1e30,
    int regularize=0,
    bint update_positions=False,
    features=None,
    feature_weights=None,
):
    """Simplify triangles using XYZ and optional per-vertex `features`; the output
    indices reference the input vertices."""
    if positions.shape[1] != 3 or faces.shape[1] != 3:
        raise ValueError("positions and faces must have shapes (N, 3) and (F, 3)")
    if target_faces <= 0:
        raise ValueError("target_faces must be positive")
    if positions.shape[0] == 0 or faces.shape[0] == 0:
        return np.empty((0, 3), dtype=np.uint32), 0.0

    cdef cnp.ndarray[cnp.uint32_t, ndim=1] output = np.empty(
        faces.size, dtype=np.uint32
    )
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] feature_array
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] weight_array
    if features is None:
        feature_array = np.empty((positions.shape[0], 0), dtype=np.float32)
        weight_array = np.empty(0, dtype=np.float32)
    else:
        feature_array = np.ascontiguousarray(features, dtype=np.float32)
        weight_array = np.ascontiguousarray(feature_weights, dtype=np.float32)
    if feature_array.shape[0] != positions.shape[0]:
        raise ValueError("features must have one row per vertex")
    if weight_array.shape[0] != feature_array.shape[1]:
        raise ValueError("feature_weights must have one value per feature channel")
    if feature_array.shape[1] > 32:
        raise ValueError("meshoptimizer supports at most 32 feature channels")

    cdef float result_error = 0.0
    cdef unsigned int options = 1 if lock_border else 0
    if regularize == 1:
        options |= 64  # meshopt_SimplifyRegularizeLight
    elif regularize == 2:
        options |= 16  # meshopt_SimplifyRegularize
    elif regularize != 0:
        raise ValueError("regularize must be 0 (none), 1 (light), or 2 (full)")
    if target_error < 0:
        raise ValueError("target_error must be non-negative")
    cdef size_t output_count
    cdef size_t target_index_count = min(<size_t>faces.size, <size_t>target_faces * 3)
    cdef size_t feature_count = feature_array.shape[1]
    cdef float *feature_ptr = NULL
    cdef float *weight_ptr = NULL
    if feature_count:
        feature_ptr = &feature_array[0, 0]
        weight_ptr = &weight_array[0]
    if update_positions:
        with nogil:
            output_count = meshopt_simplifyWithUpdate(
                &faces[0, 0],
                faces.size,
                &positions[0, 0],
                positions.shape[0],
                3 * sizeof(float),
                feature_ptr,
                feature_count * sizeof(float),
                weight_ptr,
                feature_count,
                NULL,
                target_index_count,
                target_error,
                options,
                &result_error,
            )
        return faces.reshape(-1)[:output_count].reshape((-1, 3)).copy(), result_error
    elif feature_count:
        with nogil:
            output_count = meshopt_simplifyWithAttributes(
                &output[0],
                &faces[0, 0],
                faces.size,
                &positions[0, 0],
                positions.shape[0],
                3 * sizeof(float),
                feature_ptr,
                feature_count * sizeof(float),
                weight_ptr,
                feature_count,
                NULL,
                target_index_count,
                target_error,
                options,
                &result_error,
            )
    else:
        with nogil:
            output_count = meshopt_simplify(
                &output[0],
                &faces[0, 0],
                faces.size,
                &positions[0, 0],
                positions.shape[0],
                3 * sizeof(float),
                target_index_count,
                target_error,
                options,
                &result_error,
            )
    return output[:output_count].reshape((-1, 3)), result_error


def simplify_geometry_with_passive_fields(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] positions,
    cnp.ndarray[cnp.uint32_t, ndim=2, mode="c"] faces,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] passive_fields,
    int target_faces,
    bint lock_border=False,
    float target_error=1e30,
    int regularize=0,
):
    """Geometry-only simplify that interpolates passive fields at each collapse."""
    if positions.shape[1] != 3 or faces.shape[1] != 3:
        raise ValueError("positions and faces must have shapes (N, 3) and (F, 3)")
    if passive_fields.shape[0] != positions.shape[0]:
        raise ValueError("passive_fields must have one row per vertex")
    if passive_fields.shape[1] == 0 or passive_fields.shape[1] > 32:
        raise ValueError("passive_fields must contain between 1 and 32 channels")
    if target_faces <= 0:
        raise ValueError("target_faces must be positive")
    if positions.shape[0] == 0 or faces.shape[0] == 0:
        return np.empty((0, 3), dtype=np.uint32), 0.0, passive_fields.copy()

    cdef float result_error = 0.0
    cdef unsigned int options = 1 if lock_border else 0
    if regularize == 1:
        options |= 64
    elif regularize == 2:
        options |= 16
    elif regularize != 0:
        raise ValueError("regularize must be 0 (none), 1 (light), or 2 (full)")
    if target_error < 0:
        raise ValueError("target_error must be non-negative")

    cdef size_t output_count
    cdef size_t target_index_count = min(<size_t>faces.size, <size_t>target_faces * 3)
    with nogil:
        output_count = meshopt_simplifyWithUpdatePassive(
            &faces[0, 0], faces.size, &positions[0, 0], positions.shape[0],
            3 * sizeof(float), &passive_fields[0, 0],
            passive_fields.shape[1] * sizeof(float), passive_fields.shape[1],
            NULL, target_index_count, target_error, options, &result_error,
        )
    return (
        faces.reshape(-1)[:output_count].reshape((-1, 3)).copy(),
        result_error,
        passive_fields,
    )

