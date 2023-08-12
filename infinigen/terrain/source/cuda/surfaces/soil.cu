// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


#include "header.h"
#include "../../common/surfaces/soil.h"


__global__ void soil_kernel(
    size_t size,
    float3_nonbuiltin *positions,
    float3_nonbuiltin *normals,
    float *f_params,
    float4_nonbuiltin *f4_params,
    float3_nonbuiltin *offsets
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        geometry_soil(
            positions[idx], normals[idx], f_params, f4_params, 
            offsets + idx
        );
    }
}


extern "C" {

    void call(
        size_t size,
        float3_nonbuiltin *positions,
        float3_nonbuiltin *normals,
        size_t n_f_params, float *f_params,
        size_t n_f4_params, float4_nonbuiltin *f4_params,
        float3_nonbuiltin *offsets
    ) {
        FLOAT3_VAR(d_positions, size);
        FLOAT3_VAR(d_normals, size);
        FLOAT3_VAR(d_offsets, size);
        FLOAT3_DfH(d_positions, positions, size);
        FLOAT3_DfH(d_normals, normals, size);
        FLOAT_VAR(d_f_params, n_f_params);
        FLOAT_DfH(d_f_params, f_params, n_f_params);
        FLOAT4_VAR(d_f4_params, n_f4_params);
        FLOAT4_DfH(d_f4_params, f4_params, n_f4_params);
        soil_kernel<<<ceil(size / 256.0), 256>>>(
            size, d_positions, d_normals, d_f_params, d_f4_params, d_offsets
        );
        FLOAT3_HfD(offsets, d_offsets, size);
        cudaFree(d_positions);
        cudaFree(d_normals);
        cudaFree(d_offsets);
        cudaFree(d_f_params);
        cudaFree(d_f4_params);
    }

}