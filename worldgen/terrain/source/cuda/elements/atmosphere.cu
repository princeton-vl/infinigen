// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


#include "header.h"


__global__ void atmosphere_kernel(
    size_t size,
    float3_nonbuiltin *position,
    float *sdfs,
    int has_water,
    int *i_params, float *f_params,
    int *waterbody_i_params, float *waterbody_f_params
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atmosphere(position[idx], sdfs + idx, has_water, i_params, f_params, waterbody_i_params, waterbody_f_params);
    }
}



extern "C" {
    void call(
        size_t size,
        float3_nonbuiltin *positions,
        float *sdfs
    ) {
        using namespace data;
        float3_nonbuiltin *d_positions;
        cudaMalloc((void **)&d_positions, size * sizeof(float3_nonbuiltin));
        cudaMemcpy(d_positions, positions, size * sizeof(float3_nonbuiltin), cudaMemcpyHostToDevice);
        float *d_sdfs;
        cudaMalloc((void **)&d_sdfs, size * sizeof(float));

        atmosphere_kernel<<<ceil(size / 256.0), 256>>>(
            size, d_positions, d_sdfs, meta_param,
            d_i_params, d_f_params, second_d_i_params, second_d_f_params
        );

        cudaMemcpy(sdfs, d_sdfs, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_positions);
        cudaFree(d_sdfs);
    }
}