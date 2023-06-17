/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2005 Blender Foundation. All rights reserved.
 * adapted by Zeyu Ma on date June 4, 2023 to compile geometry nodes */

#ifndef __VECTORMATH__
#define __VECTORMATH__

DEVICE_FUNC void node_shader_vector_math(
    // parameters
    int operation,
    // input
    float3_nonbuiltin in0,
    float3_nonbuiltin in1,
    float3_nonbuiltin in2,
    float in3,
    // output
    float3_nonbuiltin *out,
    float *out_scalar) {
    switch (operation) {
        case NODE_VECTOR_MATH_MULTIPLY_ADD:
            if (out != 0) *out = in0 * in1 + in2;
            break;
        case NODE_VECTOR_MATH_ADD:
            if (out != 0) *out = in0 + in1;
            break;
        case NODE_VECTOR_MATH_SUBTRACT:
            if (out != 0) *out = float3_nonbuiltin(in0.x - in1.x, in0.y - in1.y, in0.z - in1.z);
            break;
        case NODE_VECTOR_MATH_MULTIPLY:
            if (out != 0) *out = float3_nonbuiltin(in0.x * in1.x, in0.y * in1.y, in0.z * in1.z);
            break;
        case NODE_VECTOR_MATH_DIVIDE:
            if (out != 0) *out = float3_nonbuiltin(
                              in1.x != 0 ? in0.x / in1.x : 0,
                              in1.y != 0 ? in0.y / in1.y : 0,
                              in1.z != 0 ? in0.z / in1.z : 0);
            break;
        case NODE_VECTOR_MATH_CROSS_PRODUCT:
            if (out != 0) *out = float3_nonbuiltin(
                              float(double(in0.y) * double(in1.z) - double(in0.z) * double(in1.y)),
                              float(double(in0.z) * double(in1.x) - double(in0.x) * double(in1.z)),
                              float(double(in0.x) * double(in1.y) - double(in0.y) * double(in1.x)));
            break;
        case NODE_VECTOR_MATH_DOT_PRODUCT:
            if (out_scalar != 0) *out_scalar = in0.x * in1.x + in0.y * in1.y + in0.z * in1.z;
            break;
        case NODE_VECTOR_MATH_SCALE:
            if (out != 0) *out = in0 * in3;
            break;
        // later
        default: {
            assert(0);
            break;
        }
    }
}

#endif