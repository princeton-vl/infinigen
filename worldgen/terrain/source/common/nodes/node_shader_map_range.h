/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2005 Blender Foundation. All rights reserved.
 * adapted by Zeyu Ma on date June 4, 2023 to compile geometry nodes */

#ifndef __MAPRANGE__
#define __MAPRANGE__

DEVICE_FUNC void build_float_linear(
    bool Clamp,
    float value, float from_min, float from_max, float to_min, float to_max,
    float *r_value) {
    const float factor = safe_divide(value - from_min, from_max - from_min);
    float result = to_min + factor * (to_max - to_min);
    if (Clamp) {
        result = clamp_range(result, to_min, to_max);
    }
    *r_value = result;
}

DEVICE_FUNC void build_float_stepped(
    bool Clamp,
    float value, float from_min, float from_max, float to_min, float to_max, float steps,
    float *r_value) {
    float factor = safe_divide(value - from_min, from_max - from_min);
    factor = safe_divide(floorf(factor * (steps + 1.0f)), steps);
    float result = to_min + factor * (to_max - to_min);
    if (Clamp) {
        result = clamp_range(result, to_min, to_max);
    }
    *r_value = result;
}

DEVICE_FUNC void build_vector_linear(
    int Clamp,
    const float3_nonbuiltin &value,
    const float3_nonbuiltin &from_min,
    const float3_nonbuiltin &from_max,
    const float3_nonbuiltin &to_min,
    const float3_nonbuiltin &to_max,
    float3_nonbuiltin *r_value) {
    float3_nonbuiltin factor = safe_divide(value - from_min, from_max - from_min);
    float3_nonbuiltin result = factor * (to_max - to_min) + to_min;
    if (Clamp) {
        result = clamp_range(result, to_min, to_max);
    }
    *r_value = result;
}

DEVICE_FUNC void build_vector_stepped(
    int Clamp,
    const float3_nonbuiltin &value,
    const float3_nonbuiltin &from_min,
    const float3_nonbuiltin &from_max,
    const float3_nonbuiltin &to_min,
    const float3_nonbuiltin &to_max,
    const float3_nonbuiltin &steps,
    float3_nonbuiltin *r_value) {
    float3_nonbuiltin factor = safe_divide(value - from_min, from_max - from_min);
    factor = safe_divide(floor(factor * (steps + 1.0f)), steps);
    float3_nonbuiltin result = factor * (to_max - to_min) + to_min;
    if (Clamp) {
        result = clamp_range(result, to_min, to_max);
    }
    *r_value = result;
}

DEVICE_FUNC void node_shader_map_range(
    // parameters
    int data_type,
    int interpolation_type,
    int Clamp,
    // input
    float value,
    float from_min,
    float from_max,
    float to_min,
    float to_max,
    float step,
    float3_nonbuiltin value_float3,
    float3_nonbuiltin from_min_float3,
    float3_nonbuiltin from_max_float3,
    float3_nonbuiltin to_min_float3,
    float3_nonbuiltin to_max_float3,
    float3_nonbuiltin step_float3,
    // output
    float *r_value,
    float3_nonbuiltin *r_value_float3) {
    switch (data_type) {
        case FLOAT_VECTOR:
            switch (interpolation_type) {
                case NODE_MAP_RANGE_LINEAR: {
                    build_vector_linear(
                        Clamp,
                        value_float3, from_min_float3, from_max_float3, to_min_float3, to_max_float3,
                        r_value_float3);
                    break;
                }
                case NODE_MAP_RANGE_STEPPED: {
                    build_vector_stepped(
                        Clamp,
                        value_float3, from_min_float3, from_max_float3, to_min_float3, to_max_float3, step_float3,
                        r_value_float3);
                    break;
                }
                case NODE_MAP_RANGE_SMOOTHSTEP: {
                    float3_nonbuiltin factor = safe_divide(value_float3 - from_min_float3, from_max_float3 - from_min_float3);
                    factor.x = clamp(factor.x, 0.0f, 1.0f);
                    factor.y = clamp(factor.y, 0.0f, 1.0f);
                    factor.z = clamp(factor.z, 0.0f, 1.0f);
                    factor = (float3_nonbuiltin(3.0f, 3.0f, 3.0f) - 2.0f * factor) * (factor * factor);
                    *r_value_float3 = factor * (to_max_float3 - to_min_float3) + to_min_float3;
                    break;
                }
                case NODE_MAP_RANGE_SMOOTHERSTEP: {
                    float3_nonbuiltin factor = safe_divide(value_float3 - from_min_float3, from_max_float3 - from_min_float3);
                    factor.x = clamp(factor.x, 0.0f, 1.0f);
                    factor.y = clamp(factor.y, 0.0f, 1.0f);
                    factor.z = clamp(factor.z, 0.0f, 1.0f);
                    factor = factor * factor * factor * (factor * (factor * 6.0f - 15.0f) + 10.0f);
                    *r_value_float3 = factor * (to_max_float3 - to_min_float3) + to_min_float3;

                    break;
                }
                default:
                    break;
            }
            break;
        case FLOAT:
            switch (interpolation_type) {
                case NODE_MAP_RANGE_LINEAR: {
                    build_float_linear(
                        Clamp,
                        value, from_min, from_max, to_min, to_max,
                        r_value);
                    break;
                }
                case NODE_MAP_RANGE_STEPPED: {
                    build_float_stepped(
                        Clamp,
                        value, from_min, from_max, to_min, to_max, step,
                        r_value);
                    break;
                }
                case NODE_MAP_RANGE_SMOOTHSTEP: {
                    float factor = safe_divide(value - from_min, from_max - from_min);
                    factor = clamp(factor, 0.0f, 1.0f);
                    factor = (3.0f - 2.0f * factor) * (factor * factor);
                    *r_value = to_min + factor * (to_max - to_min);
                    break;
                }
                case NODE_MAP_RANGE_SMOOTHERSTEP: {
                    float factor = safe_divide(value - from_min, from_max - from_min);
                    factor = clamp(factor, 0.0f, 1.0f);
                    factor = factor * factor * factor * (factor * (factor * 6.0f - 15.0f) + 10.0f);
                    *r_value = to_min + factor * (to_max - to_min);
                    break;
                }
                default:
                    break;
            }
            break;
    }
}

#endif