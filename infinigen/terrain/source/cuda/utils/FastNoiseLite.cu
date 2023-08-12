// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
using namespace std;
#define DEVICE_FUNC __device__
#define CONSTANT_ARRAY __device__ __constant__
#include "../../common/utils/vectors.h"
#include "../../common/utils/FastNoiseLite.h"

__global__ void perlin_kernel(
    size_t size,
    float3_nonbuiltin *position,
    float *values,
    int seed, int octaves, float freq
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        values[idx] = Perlin(position[idx].x, position[idx].y, position[idx].z, seed, octaves, freq);
    }
}



extern "C" {

    void perlin_call(
        size_t size,
        float3_nonbuiltin *positions,
        float *values,
        int seed, int octaves, float freq
    ) {
        float3_nonbuiltin *d_positions;
        cudaMalloc((void **)&d_positions, size * sizeof(float3_nonbuiltin));
        cudaMemcpy(d_positions, positions, size * sizeof(float3_nonbuiltin), cudaMemcpyHostToDevice);
        float *d_values;
        cudaMalloc((void **)&d_values, size * sizeof(float));

        perlin_kernel<<<ceil(size / 256.0), 256>>>(size, d_positions, d_values, seed, octaves, freq);

        cudaMemcpy(values, d_values, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_positions);
        cudaFree(d_values);
    }

}