// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


#include "header.h"

__global__ void upsidedown_mountains_kernel(
    size_t size,
    float3_nonbuiltin *position,
    float *sdfs,
    float *auxs,
    int *i_params,
    float *f_params
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int n_auxiliaries = 1;
        if (auxs == NULL) n_auxiliaries = 0;
        upsidedown_mountains(position[idx], sdfs + idx, auxs + idx * n_auxiliaries, i_params, f_params);
    }
}

extern "C" {


    void call(
        size_t size,
        float3_nonbuiltin *positions,
        float *sdfs,
        float *auxs
    ) {
        using namespace data;
        const int n_auxiliaries = 1;
        float3_nonbuiltin *d_positions;
        cudaMalloc((void **)&d_positions, size * sizeof(float3_nonbuiltin));
        cudaMemcpy(d_positions, positions, size * sizeof(float3_nonbuiltin), cudaMemcpyHostToDevice);
        float *d_sdfs;
        cudaMalloc((void **)&d_sdfs, size * sizeof(float));
        float *d_auxs=NULL;
        if (auxs != NULL) cudaMalloc((void **)&d_auxs, size * n_auxiliaries * sizeof(float));
        upsidedown_mountains_kernel<<<ceil(size / 256.0), 256>>>(size, d_positions, d_sdfs, d_auxs, d_i_params, d_f_params);
        cudaMemcpy(sdfs, d_sdfs, size * sizeof(float), cudaMemcpyDeviceToHost);
        if (auxs != NULL) cudaMemcpy(auxs, d_auxs, size * n_auxiliaries * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_positions);
        if (auxs != NULL) cudaFree(d_auxs);
        cudaFree(d_sdfs);
    }

}