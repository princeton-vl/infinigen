/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2005 Blender Foundation
 * adapted by Zeyu Ma on date June 4, 2023 to compile terrain elements */

#ifndef _UINT32_T
#define _UINT32_T
typedef unsigned int uint32_t;
#endif /* _UINT32_T */

DEVICE_FUNC uint32_t float_as_uint_nonbuiltin(float f) {
    union {
        uint32_t i;
        float f;
    } u;
    u.f = f;
    return u.i;
}

typedef enum NodeMathOperation {
    NODE_MATH_ADD = 0,
    NODE_MATH_SUBTRACT = 1,
    NODE_MATH_MULTIPLY = 2,
    NODE_MATH_DIVIDE = 3,
    NODE_MATH_SINE = 4,
    NODE_MATH_COSINE = 5,
    NODE_MATH_TANGENT = 6,
    NODE_MATH_ARCSINE = 7,
    NODE_MATH_ARCCOSINE = 8,
    NODE_MATH_ARCTANGENT = 9,
    NODE_MATH_POWER = 10,
    NODE_MATH_LOGARITHM = 11,
    NODE_MATH_MINIMUM = 12,
    NODE_MATH_MAXIMUM = 13,
    NODE_MATH_ROUND = 14,
    NODE_MATH_LESS_THAN = 15,
    NODE_MATH_GREATER_THAN = 16,
    NODE_MATH_MODULO = 17,
    NODE_MATH_ABSOLUTE = 18,
    NODE_MATH_ARCTAN2 = 19,
    NODE_MATH_FLOOR = 20,
    NODE_MATH_CEIL = 21,
    NODE_MATH_FRACTION = 22,
    NODE_MATH_SQRT = 23,
    NODE_MATH_INV_SQRT = 24,
    NODE_MATH_SIGN = 25,
    NODE_MATH_EXPONENT = 26,
    NODE_MATH_RADIANS = 27,
    NODE_MATH_DEGREES = 28,
    NODE_MATH_SINH = 29,
    NODE_MATH_COSH = 30,
    NODE_MATH_TANH = 31,
    NODE_MATH_TRUNC = 32,
    NODE_MATH_SNAP = 33,
    NODE_MATH_WRAP = 34,
    NODE_MATH_COMPARE = 35,
    NODE_MATH_MULTIPLY_ADD = 36,
    NODE_MATH_PINGPONG = 37,
    NODE_MATH_SMOOTH_MIN = 38,
    NODE_MATH_SMOOTH_MAX = 39,
} NodeMathOperation;

/* -------------------------------------------------------------------- */
/** \name Conversion Defines
 * \{ */

#define RAD2DEG(_rad) ((_rad) * (180.0 / M_PI))
#define DEG2RAD(_deg) ((_deg) * (M_PI / 180.0))

#define RAD2DEGF(_rad) ((_rad) * (float)(180.0 / M_PI))
#define DEG2RADF(_deg) ((_deg) * (float)(M_PI / 180.0))

/** \} */

DEVICE_FUNC float signf(float f) {
    return (f < 0.0f) ? -1.0f : 1.0f;
}

/* `signum` function testing for zero. Matches GLSL and OSL functions. */
DEVICE_FUNC float compatible_signf(float f) {
    if (f == 0.0f) {
        return 0.0f;
    } else {
        return signf(f);
    }
}

/* -------------------------------------------------------------------- */
/** \name Clamp Macros
 * \{ */

#define CLAMPIS(a, b, c) ((a) < (b) ? (b) : (a) > (c) ? (c) \
                                                      : (a))

#define CLAMP(a, b, c)          \
    {                           \
        if ((a) < (b)) {        \
            (a) = (b);          \
        } else if ((a) > (c)) { \
            (a) = (c);          \
        }                       \
    }                           \
    (void)0

#define CLAMP_MAX(a, c)  \
    {                    \
        if ((a) > (c)) { \
            (a) = (c);   \
        }                \
    }                    \
    (void)0

#define CLAMP_MIN(a, b)  \
    {                    \
        if ((a) < (b)) { \
            (a) = (b);   \
        }                \
    }                    \
    (void)0

#define CLAMP2(vec, b, c)      \
    {                          \
        CLAMP((vec)[0], b, c); \
        CLAMP((vec)[1], b, c); \
    }                          \
    (void)0

#define CLAMP2_MIN(vec, b)      \
    {                           \
        CLAMP_MIN((vec)[0], b); \
        CLAMP_MIN((vec)[1], b); \
    }                           \
    (void)0

#define CLAMP2_MAX(vec, b)      \
    {                           \
        CLAMP_MAX((vec)[0], b); \
        CLAMP_MAX((vec)[1], b); \
    }                           \
    (void)0

#define CLAMP3(vec, b, c)      \
    {                          \
        CLAMP((vec)[0], b, c); \
        CLAMP((vec)[1], b, c); \
        CLAMP((vec)[2], b, c); \
    }                          \
    (void)0

#define CLAMP3_MIN(vec, b)      \
    {                           \
        CLAMP_MIN((vec)[0], b); \
        CLAMP_MIN((vec)[1], b); \
        CLAMP_MIN((vec)[2], b); \
    }                           \
    (void)0

#define CLAMP3_MAX(vec, b)      \
    {                           \
        CLAMP_MAX((vec)[0], b); \
        CLAMP_MAX((vec)[1], b); \
        CLAMP_MAX((vec)[2], b); \
    }                           \
    (void)0

#define CLAMP4(vec, b, c)      \
    {                          \
        CLAMP((vec)[0], b, c); \
        CLAMP((vec)[1], b, c); \
        CLAMP((vec)[2], b, c); \
        CLAMP((vec)[3], b, c); \
    }                          \
    (void)0

#define CLAMP4_MIN(vec, b)      \
    {                           \
        CLAMP_MIN((vec)[0], b); \
        CLAMP_MIN((vec)[1], b); \
        CLAMP_MIN((vec)[2], b); \
        CLAMP_MIN((vec)[3], b); \
    }                           \
    (void)0

#define CLAMP4_MAX(vec, b)      \
    {                           \
        CLAMP_MAX((vec)[0], b); \
        CLAMP_MAX((vec)[1], b); \
        CLAMP_MAX((vec)[2], b); \
        CLAMP_MAX((vec)[3], b); \
    }                           \
    (void)0

/** \} */

DEVICE_FUNC float fractf(float a) {
    return a - floorf(a);
}

/* Adapted from `godot-engine` math_funcs.h. */
DEVICE_FUNC float wrapf(float value, float max, float min) {
    float range = max - min;
    return (range != 0.0f) ? value - (range * floorf((value - min) / range)) : min;
}

DEVICE_FUNC float pingpongf(float value, float scale) {
    if (scale == 0.0f) {
        return 0.0f;
    }
    return fabsf(fractf((value - scale) / (scale * 2.0f)) * scale * 2.0f - scale);
}

#define MAX2(x, y) ((x) > (y) ? (x) : (y))

DEVICE_FUNC float smoothminf(float a, float b, float k) {
    if (k != 0.0f) {
        float h = fmaxf(k - fabsf(a - b), 0.0f) / k;
        return fminf(a, b) - h * h * h * k * (1.0f / 6.0f);
    } else {
        return fminf(a, b);
    }
}

typedef enum NodeVectorMathOperation {
    NODE_VECTOR_MATH_ADD = 0,
    NODE_VECTOR_MATH_SUBTRACT = 1,
    NODE_VECTOR_MATH_MULTIPLY = 2,
    NODE_VECTOR_MATH_DIVIDE = 3,

    NODE_VECTOR_MATH_CROSS_PRODUCT = 4,
    NODE_VECTOR_MATH_PROJECT = 5,
    NODE_VECTOR_MATH_REFLECT = 6,
    NODE_VECTOR_MATH_DOT_PRODUCT = 7,

    NODE_VECTOR_MATH_DISTANCE = 8,
    NODE_VECTOR_MATH_LENGTH = 9,
    NODE_VECTOR_MATH_SCALE = 10,
    NODE_VECTOR_MATH_NORMALIZE = 11,

    NODE_VECTOR_MATH_SNAP = 12,
    NODE_VECTOR_MATH_FLOOR = 13,
    NODE_VECTOR_MATH_CEIL = 14,
    NODE_VECTOR_MATH_MODULO = 15,
    NODE_VECTOR_MATH_FRACTION = 16,
    NODE_VECTOR_MATH_ABSOLUTE = 17,
    NODE_VECTOR_MATH_MINIMUM = 18,
    NODE_VECTOR_MATH_MAXIMUM = 19,
    NODE_VECTOR_MATH_WRAP = 20,
    NODE_VECTOR_MATH_SINE = 21,
    NODE_VECTOR_MATH_COSINE = 22,
    NODE_VECTOR_MATH_TANGENT = 23,
    NODE_VECTOR_MATH_REFRACT = 24,
    NODE_VECTOR_MATH_FACEFORWARD = 25,
    NODE_VECTOR_MATH_MULTIPLY_ADD = 26,
} NodeVectorMathOperation;

/* Voronoi Texture */

enum {
    SHD_VORONOI_EUCLIDEAN = 0,
    SHD_VORONOI_MANHATTAN = 1,
    SHD_VORONOI_CHEBYCHEV = 2,
    SHD_VORONOI_MINKOWSKI = 3,
};

enum {
    NOISE_SHD_VORONOI_EUCLIDEAN = 0,
    NOISE_SHD_VORONOI_MANHATTAN = 1,
    NOISE_SHD_VORONOI_CHEBYCHEV = 2,
    NOISE_SHD_VORONOI_MINKOWSKI = 3,
};

enum {
    SHD_VORONOI_F1 = 0,
    SHD_VORONOI_F2 = 1,
    SHD_VORONOI_SMOOTH_F1 = 2,
    SHD_VORONOI_DISTANCE_TO_EDGE = 3,
    SHD_VORONOI_N_SPHERE_RADIUS = 4,
};

enum {
    FLOAT = 0,
    FLOAT_VECTOR = 1,
};

template <typename T>
DEVICE_FUNC T safe_divide(const T &a, float b) {
    return (b != 0) ? a / b : T();
}

DEVICE_FUNC float3_nonbuiltin safe_divide(const float3_nonbuiltin &a, const float3_nonbuiltin &b) {
    return float3_nonbuiltin(
        (b.x != 0) ? a.x / b.x : 0,
        (b.y != 0) ? a.y / b.y : 0,
        (b.z != 0) ? a.z / b.z : 0);
}

template <typename T>
DEVICE_FUNC T interpolate(const T &a,
                          const T &b,
                          float t) {
    return a * (1 - t) + b * t;
}

DEVICE_FUNC float min_ff(float a, float b) {
    return (a < b) ? a : b;
}
DEVICE_FUNC float max_ff(float a, float b) {
    return (a > b) ? a : b;
}

DEVICE_FUNC float smoothstep(float edge0, float edge1, float x) {
    float result;
    if (x < edge0) {
        result = 0.0f;
    } else if (x >= edge1) {
        result = 1.0f;
    } else {
        float t = (x - edge0) / (edge1 - edge0);
        result = (3.0f - 2.0f * t) * (t * t);
    }
    return result;
}

DEVICE_FUNC float length_squared(const float2_nonbuiltin &a) {
    return a.x * a.x + a.y * a.y;
}

DEVICE_FUNC float length_squared(const float3_nonbuiltin &a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

DEVICE_FUNC float length_squared(const float4_nonbuiltin &a) {
    return a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
}

DEVICE_FUNC float dot(const float2_nonbuiltin &a, const float2_nonbuiltin &b) {
    return a.x * b.x + a.y * b.y;
}
DEVICE_FUNC float dot(const float3_nonbuiltin &a, const float3_nonbuiltin &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
DEVICE_FUNC float dot(const float4_nonbuiltin &a, const float4_nonbuiltin &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

template <typename T>
DEVICE_FUNC float length(const T &a) {
    return sqrt(length_squared(a));
}

template <typename T>
DEVICE_FUNC float distance(const T &a, const T &b) {
    return length(a - b);
}

DEVICE_FUNC float2_nonbuiltin floor(const float2_nonbuiltin &a) {
    return float2_nonbuiltin(floor(a.x), floor(a.y));
}
DEVICE_FUNC float3_nonbuiltin floor(const float3_nonbuiltin &a) {
    return float3_nonbuiltin(floor(a.x), floor(a.y), floor(a.z));
}
DEVICE_FUNC float4_nonbuiltin floor(const float4_nonbuiltin &a) {
    return float4_nonbuiltin(floor(a.x), floor(a.y), floor(a.z), floor(a.w));
}

DEVICE_FUNC float clamp(float x, float a, float b) {
    return min(max(x, a), b);
}

template <typename T>
DEVICE_FUNC T normalize_and_get_length(const T &v, float &out_length) {
    out_length = length_squared(v);
    /* A larger value causes normalize errors in a scaled down models with camera extreme close. */
    const float threshold = 1.0e-35f;
    if (out_length > threshold) {
        out_length = sqrt(out_length);
        return v / out_length;
    }
    /* Either the vector is small or one of it's values contained `nan`. */
    out_length = 0.0;
    return T();
}

template <typename T>
DEVICE_FUNC T normalize(const T &v) {
    float len;
    return normalize_and_get_length(v, len);
}

typedef enum NodeMapRangeType {
    NODE_MAP_RANGE_LINEAR = 0,
    NODE_MAP_RANGE_STEPPED = 1,
    NODE_MAP_RANGE_SMOOTHSTEP = 2,
    NODE_MAP_RANGE_SMOOTHERSTEP = 3,
} NodeMapRangeType;

/* musgrave texture */
#define SHD_MUSGRAVE_MULTIFRACTAL 0
#define SHD_MUSGRAVE_FBM 1
#define SHD_MUSGRAVE_HYBRID_MULTIFRACTAL 2
#define SHD_MUSGRAVE_RIDGED_MULTIFRACTAL 3
#define SHD_MUSGRAVE_HETERO_TERRAIN 4

/* ramps */
#define MA_RAMP_MIX 0
#define MA_RAMP_ADD 1
#define MA_RAMP_MULTIPLY 2
#define MA_RAMP_SUBTRACT 3
#define MA_RAMP_SCREEN 4
#define MA_RAMP_DIVIDE 5
#define MA_RAMP_DIFFERENCE 6
#define MA_RAMP_DARKEN 7
#define MA_RAMP_LIGHTEN 8
#define MA_RAMP_OVERLAY 9
#define MA_RAMP_DODGE 10
#define MA_RAMP_BURN 11
#define MA_RAMP_HUE 12
#define MA_RAMP_SATURATION 13
#define MA_RAMP_VALUE 14
#define MA_RAMP_COLOR 15
#define MA_RAMP_SOFT_LIGHT 16
#define MA_RAMP_LINEAR_LIGHT 17

#define FLT_MIN 1.175494350822287507969e-38f
#define FLT_MAX 340282346638528859811704183484516925440.0f
#define FLT_EPSILON 1.192092896e-07F

DEVICE_FUNC float clamp_range(const float value, const float min, const float max) {
    return (min > max) ? clamp(value, max, min) : clamp(value, min, max);
}

DEVICE_FUNC float3_nonbuiltin clamp_range(const float3_nonbuiltin value, const float3_nonbuiltin min, const float3_nonbuiltin max) {
    return float3_nonbuiltin(clamp_range(value.x, min.x, max.x),
                             clamp_range(value.y, min.y, max.y),
                             clamp_range(value.z, min.z, max.z));
}

#define SWAP(type, a, b) \
    {                    \
        type sw_ap;      \
        sw_ap = (a);     \
        (a) = (b);       \
        (b) = sw_ap;     \
    }                    \
    (void)0

DEVICE_FUNC void rgb_to_hsv(float r, float g, float b, float *r_h, float *r_s, float *r_v) {
    float k = 0.0f;
    float chroma;
    float min_gb;

    if (g < b) {
        SWAP(float, g, b);
        k = -1.0f;
    }
    min_gb = b;
    if (r < g) {
        SWAP(float, r, g);
        k = -2.0f / 6.0f - k;
        min_gb = min_ff(g, b);
    }

    chroma = r - min_gb;

    *r_h = fabsf(k + (g - b) / (6.0f * chroma + 1e-20f));
    *r_s = chroma / (r + 1e-20f);
    *r_v = r;
}

DEVICE_FUNC void hsv_to_rgb(float h, float s, float v, float *r_r, float *r_g, float *r_b) {
    float nr, ng, nb;

    nr = fabsf(h * 6.0f - 3.0f) - 1.0f;
    ng = 2.0f - fabsf(h * 6.0f - 2.0f);
    nb = 2.0f - fabsf(h * 6.0f - 4.0f);

    CLAMP(nr, 0.0f, 1.0f);
    CLAMP(nb, 0.0f, 1.0f);
    CLAMP(ng, 0.0f, 1.0f);

    *r_r = ((nr - 1.0f) * s + 1.0f) * v;
    *r_g = ((ng - 1.0f) * s + 1.0f) * v;
    *r_b = ((nb - 1.0f) * s + 1.0f) * v;
}

DEVICE_FUNC void ramp_blend(int type, float r_col[3], float fac, float col[3]) {
    float tmp, facm = 1.0f - fac;

    switch (type) {
        case MA_RAMP_MIX:
            r_col[0] = facm * (r_col[0]) + fac * col[0];
            r_col[1] = facm * (r_col[1]) + fac * col[1];
            r_col[2] = facm * (r_col[2]) + fac * col[2];
            break;
        case MA_RAMP_ADD:
            r_col[0] += fac * col[0];
            r_col[1] += fac * col[1];
            r_col[2] += fac * col[2];
            break;
        case MA_RAMP_MULTIPLY:
            r_col[0] *= (facm + fac * col[0]);
            r_col[1] *= (facm + fac * col[1]);
            r_col[2] *= (facm + fac * col[2]);
            break;
        case MA_RAMP_SCREEN:
            r_col[0] = 1.0f - (facm + fac * (1.0f - col[0])) * (1.0f - r_col[0]);
            r_col[1] = 1.0f - (facm + fac * (1.0f - col[1])) * (1.0f - r_col[1]);
            r_col[2] = 1.0f - (facm + fac * (1.0f - col[2])) * (1.0f - r_col[2]);
            break;
        case MA_RAMP_OVERLAY:
            if (r_col[0] < 0.5f) {
                r_col[0] *= (facm + 2.0f * fac * col[0]);
            } else {
                r_col[0] = 1.0f - (facm + 2.0f * fac * (1.0f - col[0])) * (1.0f - r_col[0]);
            }
            if (r_col[1] < 0.5f) {
                r_col[1] *= (facm + 2.0f * fac * col[1]);
            } else {
                r_col[1] = 1.0f - (facm + 2.0f * fac * (1.0f - col[1])) * (1.0f - r_col[1]);
            }
            if (r_col[2] < 0.5f) {
                r_col[2] *= (facm + 2.0f * fac * col[2]);
            } else {
                r_col[2] = 1.0f - (facm + 2.0f * fac * (1.0f - col[2])) * (1.0f - r_col[2]);
            }
            break;
        case MA_RAMP_SUBTRACT:
            r_col[0] -= fac * col[0];
            r_col[1] -= fac * col[1];
            r_col[2] -= fac * col[2];
            break;
        case MA_RAMP_DIVIDE:
            if (col[0] != 0.0f) {
                r_col[0] = facm * (r_col[0]) + fac * (r_col[0]) / col[0];
            }
            if (col[1] != 0.0f) {
                r_col[1] = facm * (r_col[1]) + fac * (r_col[1]) / col[1];
            }
            if (col[2] != 0.0f) {
                r_col[2] = facm * (r_col[2]) + fac * (r_col[2]) / col[2];
            }
            break;
        case MA_RAMP_DIFFERENCE:
            r_col[0] = facm * (r_col[0]) + fac * fabsf(r_col[0] - col[0]);
            r_col[1] = facm * (r_col[1]) + fac * fabsf(r_col[1] - col[1]);
            r_col[2] = facm * (r_col[2]) + fac * fabsf(r_col[2] - col[2]);
            break;
        case MA_RAMP_DARKEN:
            r_col[0] = min_ff(r_col[0], col[0]) * fac + r_col[0] * facm;
            r_col[1] = min_ff(r_col[1], col[1]) * fac + r_col[1] * facm;
            r_col[2] = min_ff(r_col[2], col[2]) * fac + r_col[2] * facm;
            break;
        case MA_RAMP_LIGHTEN:
            tmp = fac * col[0];
            if (tmp > r_col[0]) {
                r_col[0] = tmp;
            }
            tmp = fac * col[1];
            if (tmp > r_col[1]) {
                r_col[1] = tmp;
            }
            tmp = fac * col[2];
            if (tmp > r_col[2]) {
                r_col[2] = tmp;
            }
            break;
        case MA_RAMP_DODGE:
            if (r_col[0] != 0.0f) {
                tmp = 1.0f - fac * col[0];
                if (tmp <= 0.0f) {
                    r_col[0] = 1.0f;
                } else if ((tmp = (r_col[0]) / tmp) > 1.0f) {
                    r_col[0] = 1.0f;
                } else {
                    r_col[0] = tmp;
                }
            }
            if (r_col[1] != 0.0f) {
                tmp = 1.0f - fac * col[1];
                if (tmp <= 0.0f) {
                    r_col[1] = 1.0f;
                } else if ((tmp = (r_col[1]) / tmp) > 1.0f) {
                    r_col[1] = 1.0f;
                } else {
                    r_col[1] = tmp;
                }
            }
            if (r_col[2] != 0.0f) {
                tmp = 1.0f - fac * col[2];
                if (tmp <= 0.0f) {
                    r_col[2] = 1.0f;
                } else if ((tmp = (r_col[2]) / tmp) > 1.0f) {
                    r_col[2] = 1.0f;
                } else {
                    r_col[2] = tmp;
                }
            }
            break;
        case MA_RAMP_BURN:
            tmp = facm + fac * col[0];

            if (tmp <= 0.0f) {
                r_col[0] = 0.0f;
            } else if ((tmp = (1.0f - (1.0f - (r_col[0])) / tmp)) < 0.0f) {
                r_col[0] = 0.0f;
            } else if (tmp > 1.0f) {
                r_col[0] = 1.0f;
            } else {
                r_col[0] = tmp;
            }

            tmp = facm + fac * col[1];
            if (tmp <= 0.0f) {
                r_col[1] = 0.0f;
            } else if ((tmp = (1.0f - (1.0f - (r_col[1])) / tmp)) < 0.0f) {
                r_col[1] = 0.0f;
            } else if (tmp > 1.0f) {
                r_col[1] = 1.0f;
            } else {
                r_col[1] = tmp;
            }

            tmp = facm + fac * col[2];
            if (tmp <= 0.0f) {
                r_col[2] = 0.0f;
            } else if ((tmp = (1.0f - (1.0f - (r_col[2])) / tmp)) < 0.0f) {
                r_col[2] = 0.0f;
            } else if (tmp > 1.0f) {
                r_col[2] = 1.0f;
            } else {
                r_col[2] = tmp;
            }
            break;
        case MA_RAMP_HUE: {
            float rH, rS, rV;
            float colH, colS, colV;
            float tmpr, tmpg, tmpb;
            rgb_to_hsv(col[0], col[1], col[2], &colH, &colS, &colV);
            if (colS != 0) {
                rgb_to_hsv(r_col[0], r_col[1], r_col[2], &rH, &rS, &rV);
                hsv_to_rgb(colH, rS, rV, &tmpr, &tmpg, &tmpb);
                r_col[0] = facm * (r_col[0]) + fac * tmpr;
                r_col[1] = facm * (r_col[1]) + fac * tmpg;
                r_col[2] = facm * (r_col[2]) + fac * tmpb;
            }
            break;
        }
        case MA_RAMP_SATURATION: {
            float rH, rS, rV;
            float colH, colS, colV;
            rgb_to_hsv(r_col[0], r_col[1], r_col[2], &rH, &rS, &rV);
            if (rS != 0) {
                rgb_to_hsv(col[0], col[1], col[2], &colH, &colS, &colV);
                hsv_to_rgb(rH, (facm * rS + fac * colS), rV, r_col + 0, r_col + 1, r_col + 2);
            }
            break;
        }
        case MA_RAMP_VALUE: {
            float rH, rS, rV;
            float colH, colS, colV;
            rgb_to_hsv(r_col[0], r_col[1], r_col[2], &rH, &rS, &rV);
            rgb_to_hsv(col[0], col[1], col[2], &colH, &colS, &colV);
            hsv_to_rgb(rH, rS, (facm * rV + fac * colV), r_col + 0, r_col + 1, r_col + 2);
            break;
        }
        case MA_RAMP_COLOR: {
            float rH, rS, rV;
            float colH, colS, colV;
            float tmpr, tmpg, tmpb;
            rgb_to_hsv(col[0], col[1], col[2], &colH, &colS, &colV);
            if (colS != 0) {
                rgb_to_hsv(r_col[0], r_col[1], r_col[2], &rH, &rS, &rV);
                hsv_to_rgb(colH, colS, rV, &tmpr, &tmpg, &tmpb);
                r_col[0] = facm * (r_col[0]) + fac * tmpr;
                r_col[1] = facm * (r_col[1]) + fac * tmpg;
                r_col[2] = facm * (r_col[2]) + fac * tmpb;
            }
            break;
        }
        case MA_RAMP_SOFT_LIGHT: {
            float scr, scg, scb;

            /* first calculate non-fac based Screen mix */
            scr = 1.0f - (1.0f - col[0]) * (1.0f - r_col[0]);
            scg = 1.0f - (1.0f - col[1]) * (1.0f - r_col[1]);
            scb = 1.0f - (1.0f - col[2]) * (1.0f - r_col[2]);

            r_col[0] = facm * (r_col[0]) +
                       fac * (((1.0f - r_col[0]) * col[0] * (r_col[0])) + (r_col[0] * scr));
            r_col[1] = facm * (r_col[1]) +
                       fac * (((1.0f - r_col[1]) * col[1] * (r_col[1])) + (r_col[1] * scg));
            r_col[2] = facm * (r_col[2]) +
                       fac * (((1.0f - r_col[2]) * col[2] * (r_col[2])) + (r_col[2] * scb));
            break;
        }
        case MA_RAMP_LINEAR_LIGHT:
            if (col[0] > 0.5f) {
                r_col[0] = r_col[0] + fac * (2.0f * (col[0] - 0.5f));
            } else {
                r_col[0] = r_col[0] + fac * (2.0f * (col[0]) - 1.0f);
            }
            if (col[1] > 0.5f) {
                r_col[1] = r_col[1] + fac * (2.0f * (col[1] - 0.5f));
            } else {
                r_col[1] = r_col[1] + fac * (2.0f * (col[1]) - 1.0f);
            }
            if (col[2] > 0.5f) {
                r_col[2] = r_col[2] + fac * (2.0f * (col[2] - 0.5f));
            } else {
                r_col[2] = r_col[2] + fac * (2.0f * (col[2]) - 1.0f);
            }
            break;
    }
}

/* **************** ColorBand ********************* */

/** color-mode. */
enum {
    COLBAND_BLEND_RGB = 0,
    COLBAND_BLEND_HSV = 1,
    COLBAND_BLEND_HSL = 2,
};

/** Interpolation. */
enum {
    COLBAND_INTERP_LINEAR = 0,
    COLBAND_INTERP_EASE = 1,
    COLBAND_INTERP_B_SPLINE = 2,
    COLBAND_INTERP_CARDINAL = 3,
    COLBAND_INTERP_CONSTANT = 4,
};

/** Color interpolation. */
enum {
    COLBAND_HUE_NEAR = 0,
    COLBAND_HUE_FAR = 1,
    COLBAND_HUE_CW = 2,
    COLBAND_HUE_CCW = 3,
};

DEVICE_FUNC void rgb_to_hsv_v(float rgb[3], float r_hsv[3]) {
    rgb_to_hsv(rgb[0], rgb[1], rgb[2], &r_hsv[0], &r_hsv[1], &r_hsv[2]);
}

DEVICE_FUNC void hsv_to_rgb_v(float hsv[3], float r_rgb[3]) {
    hsv_to_rgb(hsv[0], hsv[1], hsv[2], &r_rgb[0], &r_rgb[1], &r_rgb[2]);
}

DEVICE_FUNC float max_fff(float a, float b, float c) {
    return max_ff(max_ff(a, b), c);
}
DEVICE_FUNC float min_fff(float a, float b, float c) {
    return min_ff(min_ff(a, b), c);
}

DEVICE_FUNC void rgb_to_hsl(float r, float g, float b, float *r_h, float *r_s, float *r_l) {
    const float cmax = max_fff(r, g, b);
    const float cmin = min_fff(r, g, b);
    float h, s, l = min_ff(1.0f, (cmax + cmin) / 2.0f);

    if (cmax == cmin) {
        h = s = 0.0f; /* achromatic */
    } else {
        float d = cmax - cmin;
        s = l > 0.5f ? d / (2.0f - cmax - cmin) : d / (cmax + cmin);
        if (cmax == r) {
            h = (g - b) / d + (g < b ? 6.0f : 0.0f);
        } else if (cmax == g) {
            h = (b - r) / d + 2.0f;
        } else {
            h = (r - g) / d + 4.0f;
        }
    }
    h /= 6.0f;

    *r_h = h;
    *r_s = s;
    *r_l = l;
}

DEVICE_FUNC void rgb_to_hsl_v(const float rgb[3], float r_hsl[3]) {
    rgb_to_hsl(rgb[0], rgb[1], rgb[2], &r_hsl[0], &r_hsl[1], &r_hsl[2]);
}

DEVICE_FUNC void hsl_to_rgb(float h, float s, float l, float *r_r, float *r_g, float *r_b) {
    float nr, ng, nb, chroma;

    nr = fabsf(h * 6.0f - 3.0f) - 1.0f;
    ng = 2.0f - fabsf(h * 6.0f - 2.0f);
    nb = 2.0f - fabsf(h * 6.0f - 4.0f);

    CLAMP(nr, 0.0f, 1.0f);
    CLAMP(nb, 0.0f, 1.0f);
    CLAMP(ng, 0.0f, 1.0f);

    chroma = (1.0f - fabsf(2.0f * l - 1.0f)) * s;

    *r_r = (nr - 0.5f) * chroma + l;
    *r_g = (ng - 0.5f) * chroma + l;
    *r_b = (nb - 0.5f) * chroma + l;
}

DEVICE_FUNC void hsl_to_rgb_v(const float hsl[3], float r_rgb[3]) {
    hsl_to_rgb(hsl[0], hsl[1], hsl[2], &r_rgb[0], &r_rgb[1], &r_rgb[2]);
}

/* wave texture */
#define SHD_WAVE_BANDS 0
#define SHD_WAVE_RINGS 1

enum {
    SHD_WAVE_BANDS_DIRECTION_X = 0,
    SHD_WAVE_BANDS_DIRECTION_Y = 1,
    SHD_WAVE_BANDS_DIRECTION_Z = 2,
    SHD_WAVE_BANDS_DIRECTION_DIAGONAL = 3,
};

enum {
    SHD_WAVE_RINGS_DIRECTION_X = 0,
    SHD_WAVE_RINGS_DIRECTION_Y = 1,
    SHD_WAVE_RINGS_DIRECTION_Z = 2,
    SHD_WAVE_RINGS_DIRECTION_SPHERICAL = 3,
};

enum {
    SHD_WAVE_PROFILE_SIN = 0,
    SHD_WAVE_PROFILE_SAW = 1,
    SHD_WAVE_PROFILE_TRI = 2,
};