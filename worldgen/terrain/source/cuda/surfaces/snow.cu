// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


#include "header.h"
#include "../../common/surfaces/snow.h"



__global__ void snow_kernel(
    size_t size,
    float3_nonbuiltin *positions,
    float3_nonbuiltin *normals,
    float3_nonbuiltin *offsets
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        geo_snowtexture(
            positions[idx], normals[idx],
            offsets + idx
        );
    }
}


extern "C" {

    void call(
        size_t size,
        float3_nonbuiltin *positions,
        float3_nonbuiltin *normals,
        float3_nonbuiltin *offsets
    ) {
        FLOAT3_VAR(d_positions, size);
        FLOAT3_VAR(d_normals, size);
        FLOAT3_VAR(d_offsets, size);
        FLOAT3_DfH(d_positions, positions, size);
        FLOAT3_DfH(d_normals, normals, size);
        snow_kernel<<<ceil(size / 256.0), 256>>>(
            size, d_positions, d_normals, d_offsets
        );
        FLOAT3_HfD(offsets, d_offsets, size);
        cudaFree(d_positions);
        cudaFree(d_normals);
        cudaFree(d_offsets);
    }

}