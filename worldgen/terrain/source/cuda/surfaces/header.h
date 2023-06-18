// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma
// Date Signed: June 5 2023

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <assert.h>
using namespace std;
#define DEVICE_FUNC __device__
#define CONSTANT_ARRAY __device__ __constant__
#define POINTER_OR_REFERENCE_ARG
#include "../../common/utils/vectors.h"
#include "../../common/utils/nodes_util.h"
#include "../../common/utils/blender_noise.h"
#include "../../common/nodes/node_shader_tex_noise.h"
#include "../../common/nodes/node_shader_tex_voronoi.h"
#include "../../common/nodes/node_shader_mix_rgb.h"
#include "../../common/nodes/node_float_curve.h"
#include "../../common/nodes/node_shader_map_range.h"
#include "../../common/nodes/node_shader_sepcomb_xyz.h"
#include "../../common/nodes/node_shader_tex_wave.h"
#include "../../common/nodes/node_shader_vector_math.h"
#include "../../common/nodes/node_texture_math.h"
#include "../../common/nodes/node_texture_valToRgb.h"
#include "../../common/nodes/node_shader_tex_musgrave.h"


#define FLOAT3_VAR(X, size) float3_nonbuiltin *X;  cudaMalloc((void **)&X, size * sizeof(float3_nonbuiltin));
#define FLOAT3_DfH(D, H, size) cudaMemcpy(D, H, size * sizeof(float3_nonbuiltin), cudaMemcpyHostToDevice);
#define FLOAT3_HfD(H, D, size) cudaMemcpy(H, D, size * sizeof(float3_nonbuiltin), cudaMemcpyDeviceToHost);
#define FLOAT_VAR(X, size) float *X;  cudaMalloc((void **)&X, size * sizeof(float));
#define FLOAT_DfH(D, H, size) cudaMemcpy(D, H, size * sizeof(float), cudaMemcpyHostToDevice);
#define FLOAT_HfD(H, D, size) cudaMemcpy(H, D, size * sizeof(float), cudaMemcpyDeviceToHost);
#define FLOAT4_VAR(X, size) float4_nonbuiltin *X;  cudaMalloc((void **)&X, size * sizeof(float4_nonbuiltin));
#define FLOAT4_DfH(D, H, size) cudaMemcpy(D, H, size * sizeof(float4_nonbuiltin), cudaMemcpyHostToDevice);