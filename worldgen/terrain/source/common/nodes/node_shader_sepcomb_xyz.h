// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma
// Date Signed: June 5 2023

#ifndef __SEPCOMB__
#define __SEPCOMB__

DEVICE_FUNC void node_shader_sep_xyz(
    // input
    float3_nonbuiltin vector,
    // output
    float *x,
    float *y,
    float *z) {
    if (x != NULL) *x = vector.x;
    if (y != NULL) *y = vector.y;
    if (z != NULL) *z = vector.z;
}

DEVICE_FUNC void node_shader_comb_xyz(
    // input
    float x,
    float y,
    float z,
    // output
    float3_nonbuiltin *vector) {
    if (vector != NULL) *vector = float3_nonbuiltin(x, y, z);
}

#endif