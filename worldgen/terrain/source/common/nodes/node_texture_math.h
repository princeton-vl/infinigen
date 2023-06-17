/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2005 Blender Foundation. All rights reserved.
 * adapted by Zeyu Ma on date June 4, 2023 to compile geometry nodes */

#ifndef __TEXMATH__
#define __TEXMATH__

/* **************** SCALAR MATH ******************** */
DEVICE_FUNC void node_texture_math(
    // parameters
    int operation,
    int use_clamp,
    // input
    float in0,
    float in1,
    float in2,
    // output
    float *out) {
    if (out == 0) return;
    switch (operation) {
        case NODE_MATH_ADD:
            *out = in0 + in1;
            break;
        case NODE_MATH_SUBTRACT:
            *out = in0 - in1;
            break;
        case NODE_MATH_MULTIPLY:
            *out = in0 * in1;
            break;
        case NODE_MATH_DIVIDE: {
            if (in1 == 0) {
                /* We don't want to divide by zero. */
                *out = 0.0;
            } else {
                *out = in0 / in1;
            }
            break;
        }
        case NODE_MATH_SINE: {
            *out = sinf(in0);
            break;
        }
        case NODE_MATH_COSINE: {
            *out = cosf(in0);
            break;
        }
        case NODE_MATH_TANGENT: {
            *out = tanf(in0);
            break;
        }
        case NODE_MATH_SINH: {
            *out = sinhf(in0);
            break;
        }
        case NODE_MATH_COSH: {
            *out = coshf(in0);
            break;
        }
        case NODE_MATH_TANH: {
            *out = tanhf(in0);
            break;
        }
        case NODE_MATH_ARCSINE: {
            /* Can't do the impossible... */
            if (in0 <= 1 && in0 >= -1) {
                *out = asinf(in0);
            } else {
                *out = 0.0;
            }
            break;
        }
        case NODE_MATH_ARCCOSINE: {
            /* Can't do the impossible... */
            if (in0 <= 1 && in0 >= -1) {
                *out = acosf(in0);
            } else {
                *out = 0.0;
            }
            break;
        }
        case NODE_MATH_ARCTANGENT: {
            *out = atan(in0);
            break;
        }
        case NODE_MATH_POWER: {
            /* Only raise negative numbers by full integers */
            if (in0 >= 0) {
                out[0] = pow(in0, in1);
            } else {
                float y_mod_1 = fmodf(in1, 1);
                if (y_mod_1 > 0.999f || y_mod_1 < 0.001f) {
                    *out = pow(in0, floor(in1 + 0.5f));
                } else {
                    *out = 0.0;
                }
            }
            break;
        }
        case NODE_MATH_LOGARITHM: {
            /* Don't want any imaginary numbers... */
            if (in0 > 0 && in1 > 0) {
                *out = log(in0) / log(in1);
            } else {
                *out = 0.0;
            }
            break;
        }
        case NODE_MATH_MINIMUM: {
            if (in0 < in1) {
                *out = in0;
            } else {
                *out = in1;
            }
            break;
        }
        case NODE_MATH_MAXIMUM: {
            if (in0 > in1) {
                *out = in0;
            } else {
                *out = in1;
            }
            break;
        }
        case NODE_MATH_ROUND: {
            *out = (in0 < 0) ? (int)(in0 - 0.5f) : (int)(in0 + 0.5f);
            break;
        }

        case NODE_MATH_LESS_THAN: {
            if (in0 < in1) {
                *out = 1.0f;
            } else {
                *out = 0.0f;
            }
            break;
        }

        case NODE_MATH_GREATER_THAN: {
            if (in0 > in1) {
                *out = 1.0f;
            } else {
                *out = 0.0f;
            }
            break;
        }

        case NODE_MATH_MODULO: {
            if (in1 == 0.0f) {
                *out = 0.0f;
            } else {
                *out = fmodf(in0, in1);
            }
            break;
        }

        case NODE_MATH_ABSOLUTE: {
            *out = fabsf(in0);
            break;
        }

        case NODE_MATH_RADIANS: {
            *out = DEG2RADF(in0);
            break;
        }

        case NODE_MATH_DEGREES: {
            *out = RAD2DEGF(in0);
            break;
        }

        case NODE_MATH_ARCTAN2: {
            *out = atan2(in0, in1);
            break;
        }

        case NODE_MATH_SIGN: {
            *out = compatible_signf(in0);
            break;
        }

        case NODE_MATH_EXPONENT: {
            *out = expf(in0);
            break;
        }

        case NODE_MATH_FLOOR: {
            *out = floorf(in0);
            break;
        }

        case NODE_MATH_CEIL: {
            *out = ceilf(in0);
            break;
        }

        case NODE_MATH_FRACTION: {
            *out = in0 - floorf(in0);
            break;
        }

        case NODE_MATH_SQRT: {
            if (in0 > 0.0f) {
                *out = sqrtf(in0);
            } else {
                *out = 0.0f;
            }
            break;
        }

        case NODE_MATH_INV_SQRT: {
            if (in0 > 0.0f) {
                *out = 1.0f / sqrtf(in0);
            } else {
                *out = 0.0f;
            }
            break;
        }

        case NODE_MATH_TRUNC: {
            if (in0 > 0.0f) {
                *out = floorf(in0);
            } else {
                *out = ceilf(in0);
            }
            break;
        }

        case NODE_MATH_SNAP: {
            if (in1 == 0) {
                *out = 0.0;
            } else {
                *out = floorf(in0 / in1) * in1;
            }
            break;
        }

        case NODE_MATH_WRAP: {
            *out = wrapf(in0, in1, in2);
            break;
        }

        case NODE_MATH_PINGPONG: {
            *out = pingpongf(in0, in1);
            break;
        }

        case NODE_MATH_COMPARE: {
            *out = (fabsf(in0 - in1) <= MAX2(in2, 1e-5f)) ? 1.0f : 0.0f;
            break;
        }

        case NODE_MATH_MULTIPLY_ADD: {
            *out = in0 * in1 + in2;
            break;
        }

        case NODE_MATH_SMOOTH_MIN: {
            *out = smoothminf(in0, in1, in2);
            break;
        }

        case NODE_MATH_SMOOTH_MAX: {
            *out = -smoothminf(-in0, -in1, in2);
            break;
        }

        default: {
            assert(0);
            break;
        }
    }

    if (use_clamp) {
        CLAMP(*out, 0.0f, 1.0f);
    }
}

#endif