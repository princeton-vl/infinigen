/* SPDX-License-Identifier: GPL-2.0-or-later AND BSD-3-Clause
 * Copyright 2009-2010 Sony Pictures Imageworks Inc., et al. All Rights Reserved (BSD-3-Clause).
 *           2011 Blender Foundation (GPL-2.0-or-later).
 * adapted by Zeyu Ma on date June 4, 2023 to compile terrain elements */

/* -------------------------------------------------------------------- */
/** \name Perlin Noise
 *
 * Perlin, Ken. "Improving noise." Proceedings of the 29th annual conference on Computer graphics
 * and interactive techniques. 2002.
 *
 * This implementation is functionally identical to the implementations in EEVEE, OSL, and SVM. So
 * any changes should be applied in all relevant implementations.
 * \{ */

DEVICE_FUNC float floor_fraction(float x, int &i)
{
    i = int(x) - ((x < 0) ? 1 : 0);
    return x - i;
}

/* Linear Interpolation. */
DEVICE_FUNC float mix(float v0, float v1, float x)
{
    return (1 - x) * v0 + x * v1;
}

/* Bilinear Interpolation:
 *
 * v2          v3
 *  @ + + + + @       y
 *  +         +       ^
 *  +         +       |
 *  +         +       |
 *  @ + + + + @       @------> x
 * v0          v1
 */
DEVICE_FUNC float mix(float v0, float v1, float v2, float v3, float x, float y)
{
    float x1 = 1.0 - x;
    return (1.0 - y) * (v0 * x1 + v1 * x) + y * (v2 * x1 + v3 * x);
}

/* Trilinear Interpolation:
 *
 *   v6               v7
 *     @ + + + + + + @
 *     +\            +\
 *     + \           + \
 *     +  \          +  \
 *     +   \ v4      +   \ v5
 *     +    @ + + + +++ + @          z
 *     +    +        +    +      y   ^
 *  v2 @ + +++ + + + @ v3 +       \  |
 *      \   +         \   +        \ |
 *       \  +          \  +         \|
 *        \ +           \ +          +---------> x
 *         \+            \+
 *          @ + + + + + + @
 *        v0               v1
 */
DEVICE_FUNC float mix(float v0,
          float v1,
          float v2,
          float v3,
          float v4,
          float v5,
          float v6,
          float v7,
          float x,
          float y,
          float z)
{
    float x1 = 1.0 - x;
    float y1 = 1.0 - y;
    float z1 = 1.0 - z;
    return z1 * (y1 * (v0 * x1 + v1 * x) + y * (v2 * x1 + v3 * x)) +
           z * (y1 * (v4 * x1 + v5 * x) + y * (v6 * x1 + v7 * x));
}

/* Quadrilinear Interpolation. */
DEVICE_FUNC float mix(float v0,
          float v1,
          float v2,
          float v3,
          float v4,
          float v5,
          float v6,
          float v7,
          float v8,
          float v9,
          float v10,
          float v11,
          float v12,
          float v13,
          float v14,
          float v15,
          float x,
          float y,
          float z,
          float w)
{
    return mix(mix(v0, v1, v2, v3, v4, v5, v6, v7, x, y, z),
               mix(v8, v9, v10, v11, v12, v13, v14, v15, x, y, z),
               w);
}

DEVICE_FUNC float fade(float t)
{
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

DEVICE_FUNC uint32_t hash_bit_rotate(uint32_t x, uint32_t k)
{
    return (x << k) | (x >> (32 - k));
}

DEVICE_FUNC void hash_bit_mix(uint32_t &a, uint32_t &b, uint32_t &c)
{
    a -= c;
    a ^= hash_bit_rotate(c, 4);
    c += b;
    b -= a;
    b ^= hash_bit_rotate(a, 6);
    a += c;
    c -= b;
    c ^= hash_bit_rotate(b, 8);
    b += a;
    a -= c;
    a ^= hash_bit_rotate(c, 16);
    c += b;
    b -= a;
    b ^= hash_bit_rotate(a, 19);
    a += c;
    c -= b;
    c ^= hash_bit_rotate(b, 4);
    b += a;
}

DEVICE_FUNC void hash_bit_final(uint32_t &a, uint32_t &b, uint32_t &c)
{
    c ^= b;
    c -= hash_bit_rotate(b, 14);
    a ^= c;
    a -= hash_bit_rotate(c, 11);
    b ^= a;
    b -= hash_bit_rotate(a, 25);
    c ^= b;
    c -= hash_bit_rotate(b, 16);
    a ^= c;
    a -= hash_bit_rotate(c, 4);
    b ^= a;
    b -= hash_bit_rotate(a, 14);
    c ^= b;
    c -= hash_bit_rotate(b, 24);
}

DEVICE_FUNC uint32_t myhash(uint32_t kx)
{
    uint32_t a, b, c;
    a = b = c = 0xdeadbeef + (1 << 2) + 13;

    a += kx;
    hash_bit_final(a, b, c);

    return c;
}

DEVICE_FUNC uint32_t myhash(uint32_t kx, uint32_t ky)
{
    uint32_t a, b, c;
    a = b = c = 0xdeadbeef + (2 << 2) + 13;

    b += ky;
    a += kx;
    hash_bit_final(a, b, c);

    return c;
}

DEVICE_FUNC uint32_t myhash(uint32_t kx, uint32_t ky, uint32_t kz)
{
    uint32_t a, b, c;
    a = b = c = 0xdeadbeef + (3 << 2) + 13;

    c += kz;
    b += ky;
    a += kx;
    hash_bit_final(a, b, c);

    return c;
}

DEVICE_FUNC uint32_t myhash(uint32_t kx, uint32_t ky, uint32_t kz, uint32_t kw)
{
    uint32_t a, b, c;
    a = b = c = 0xdeadbeef + (4 << 2) + 13;

    a += kx;
    b += ky;
    c += kz;
    hash_bit_mix(a, b, c);

    a += kw;
    hash_bit_final(a, b, c);

    return c;
}

DEVICE_FUNC float negate_if(float value, uint32_t condition)
{
    return (condition != 0u) ? -value : value;
}

DEVICE_FUNC float noise_grad(uint32_t myhash, float x)
{
    uint32_t h = myhash & 15u;
    float g = 1u + (h & 7u);
    return negate_if(g, h & 8u) * x;
}

DEVICE_FUNC float noise_grad(uint32_t myhash, float x, float y)
{
    uint32_t h = myhash & 7u;
    float u = h < 4u ? x : y;
    float v = 2.0 * (h < 4u ? y : x);
    return negate_if(u, h & 1u) + negate_if(v, h & 2u);
}

DEVICE_FUNC float noise_grad(uint32_t myhash, float x, float y, float z)
{
    uint32_t h = myhash & 15u;
    float u = h < 8u ? x : y;
    float vt = (h == 12u || h == 14u) ? x : z;
    float v = h < 4u ? y : vt;
    return negate_if(u, h & 1u) + negate_if(v, h & 2u);
}

DEVICE_FUNC float noise_grad(uint32_t myhash, float x, float y, float z, float w)
{
    uint32_t h = myhash & 31u;
    float u = h < 24u ? x : y;
    float v = h < 16u ? y : z;
    float s = h < 8u ? z : w;
    return negate_if(u, h & 1u) + negate_if(v, h & 2u) + negate_if(s, h & 4u);
}

DEVICE_FUNC float perlin_noise(float position)
{
    int X;

    float fx = floor_fraction(position, X);

    float u = fade(fx);

    float r = mix(noise_grad(myhash(X), fx), noise_grad(myhash(X + 1), fx - 1.0), u);

    return r;
}

DEVICE_FUNC float perlin_noise(float2_nonbuiltin position)
{
    int X, Y;

    float fx = floor_fraction(position.x, X);
    float fy = floor_fraction(position.y, Y);

    float u = fade(fx);
    float v = fade(fy);

    float r = mix(noise_grad(myhash(X, Y), fx, fy),
                  noise_grad(myhash(X + 1, Y), fx - 1.0, fy),
                  noise_grad(myhash(X, Y + 1), fx, fy - 1.0),
                  noise_grad(myhash(X + 1, Y + 1), fx - 1.0, fy - 1.0),
                  u,
                  v);

    return r;
}

DEVICE_FUNC float perlin_noise(float3_nonbuiltin position)
{
    int X, Y, Z;

    float fx = floor_fraction(position.x, X);
    float fy = floor_fraction(position.y, Y);
    float fz = floor_fraction(position.z, Z);

    float u = fade(fx);
    float v = fade(fy);
    float w = fade(fz);

    float r = mix(noise_grad(myhash(X, Y, Z), fx, fy, fz),
                  noise_grad(myhash(X + 1, Y, Z), fx - 1, fy, fz),
                  noise_grad(myhash(X, Y + 1, Z), fx, fy - 1, fz),
                  noise_grad(myhash(X + 1, Y + 1, Z), fx - 1, fy - 1, fz),
                  noise_grad(myhash(X, Y, Z + 1), fx, fy, fz - 1),
                  noise_grad(myhash(X + 1, Y, Z + 1), fx - 1, fy, fz - 1),
                  noise_grad(myhash(X, Y + 1, Z + 1), fx, fy - 1, fz - 1),
                  noise_grad(myhash(X + 1, Y + 1, Z + 1), fx - 1, fy - 1, fz - 1),
                  u,
                  v,
                  w);

    return r;
}

DEVICE_FUNC float perlin_noise(float4_nonbuiltin position)
{
    int X, Y, Z, W;

    float fx = floor_fraction(position.x, X);
    float fy = floor_fraction(position.y, Y);
    float fz = floor_fraction(position.z, Z);
    float fw = floor_fraction(position.w, W);

    float u = fade(fx);
    float v = fade(fy);
    float t = fade(fz);
    float s = fade(fw);

    float r = mix(
        noise_grad(myhash(X, Y, Z, W), fx, fy, fz, fw),
        noise_grad(myhash(X + 1, Y, Z, W), fx - 1.0, fy, fz, fw),
        noise_grad(myhash(X, Y + 1, Z, W), fx, fy - 1.0, fz, fw),
        noise_grad(myhash(X + 1, Y + 1, Z, W), fx - 1.0, fy - 1.0, fz, fw),
        noise_grad(myhash(X, Y, Z + 1, W), fx, fy, fz - 1.0, fw),
        noise_grad(myhash(X + 1, Y, Z + 1, W), fx - 1.0, fy, fz - 1.0, fw),
        noise_grad(myhash(X, Y + 1, Z + 1, W), fx, fy - 1.0, fz - 1.0, fw),
        noise_grad(myhash(X + 1, Y + 1, Z + 1, W), fx - 1.0, fy - 1.0, fz - 1.0, fw),
        noise_grad(myhash(X, Y, Z, W + 1), fx, fy, fz, fw - 1.0),
        noise_grad(myhash(X + 1, Y, Z, W + 1), fx - 1.0, fy, fz, fw - 1.0),
        noise_grad(myhash(X, Y + 1, Z, W + 1), fx, fy - 1.0, fz, fw - 1.0),
        noise_grad(myhash(X + 1, Y + 1, Z, W + 1), fx - 1.0, fy - 1.0, fz, fw - 1.0),
        noise_grad(myhash(X, Y, Z + 1, W + 1), fx, fy, fz - 1.0, fw - 1.0),
        noise_grad(myhash(X + 1, Y, Z + 1, W + 1), fx - 1.0, fy, fz - 1.0, fw - 1.0),
        noise_grad(myhash(X, Y + 1, Z + 1, W + 1), fx, fy - 1.0, fz - 1.0, fw - 1.0),
        noise_grad(myhash(X + 1, Y + 1, Z + 1, W + 1), fx - 1.0, fy - 1.0, fz - 1.0, fw - 1.0),
        u,
        v,
        t,
        s);

    return r;
}

/* Signed versions of perlin noise in the range [-1, 1]. The scale values were computed
 * experimentally by the OSL developers to remap the noise output to the correct range. */

DEVICE_FUNC float perlin_signed(float position)
{
    return perlin_noise(position) * 0.2500f;
}

DEVICE_FUNC float perlin_signed(float2_nonbuiltin position)
{
    return perlin_noise(position) * 0.6616f;
}

DEVICE_FUNC float perlin_signed(float3_nonbuiltin position)
{
    return perlin_noise(position) * 0.9820f;
}

DEVICE_FUNC float perlin_signed(float4_nonbuiltin position)
{
    return perlin_noise(position) * 0.8344f;
}

/* Positive versions of perlin noise in the range [0, 1]. */

DEVICE_FUNC float perlin(float position)
{
    return perlin_signed(position) / 2.0f + 0.5f;
}

DEVICE_FUNC float perlin(float2_nonbuiltin position)
{
    return perlin_signed(position) / 2.0f + 0.5f;
}

DEVICE_FUNC float perlin(float3_nonbuiltin position)
{
    return perlin_signed(position) / 2.0f + 0.5f;
}

DEVICE_FUNC float perlin(float4_nonbuiltin position)
{
    return perlin_signed(position) / 2.0f + 0.5f;
}

DEVICE_FUNC uint32_t hash_float(float kx)
{
    return myhash(float_as_uint_nonbuiltin(kx));
}

DEVICE_FUNC uint32_t hash_float(float2_nonbuiltin k)
{
    return myhash(float_as_uint_nonbuiltin(k.x), float_as_uint_nonbuiltin(k.y));
}

DEVICE_FUNC uint32_t hash_float(float3_nonbuiltin k)
{
    return myhash(float_as_uint_nonbuiltin(k.x), float_as_uint_nonbuiltin(k.y), float_as_uint_nonbuiltin(k.z));
}

DEVICE_FUNC uint32_t hash_float(float4_nonbuiltin k)
{
    return myhash(float_as_uint_nonbuiltin(k.x), float_as_uint_nonbuiltin(k.y), float_as_uint_nonbuiltin(k.z), float_as_uint_nonbuiltin(k.w));
}

/* Hashing a number of uint32_t into a float in the range [0, 1]. */

DEVICE_FUNC float uint_to_float_01(uint32_t k)
{
    return float(k) / float(0xFFFFFFFFu);
}

DEVICE_FUNC float hash_to_float(uint32_t kx)
{
    return uint_to_float_01(myhash(kx));
}

DEVICE_FUNC float hash_to_float(uint32_t kx, uint32_t ky)
{
    return uint_to_float_01(myhash(kx, ky));
}

DEVICE_FUNC float hash_to_float(uint32_t kx, uint32_t ky, uint32_t kz)
{
    return uint_to_float_01(myhash(kx, ky, kz));
}

DEVICE_FUNC float hash_to_float(uint32_t kx, uint32_t ky, uint32_t kz, uint32_t kw)
{
    return uint_to_float_01(myhash(kx, ky, kz, kw));
}

/* Hashing a number of floats into a float in the range [0, 1]. */

DEVICE_FUNC float hash_float_to_float(float k)
{
    return uint_to_float_01(hash_float(k));
}

DEVICE_FUNC float hash_float_to_float(float2_nonbuiltin k)
{
    return uint_to_float_01(hash_float(k));
}

DEVICE_FUNC float hash_float_to_float(float3_nonbuiltin k)
{
    return uint_to_float_01(hash_float(k));
}

DEVICE_FUNC float hash_float_to_float(float4_nonbuiltin k)
{
    return uint_to_float_01(hash_float(k));
}

DEVICE_FUNC float2_nonbuiltin hash_float_to_float2(float2_nonbuiltin k)
{
    return float2_nonbuiltin(hash_float_to_float(k), hash_float_to_float(float3_nonbuiltin(k.x, k.y, 1.0)));
}

DEVICE_FUNC float3_nonbuiltin hash_float_to_float3(float k)
{
    return float3_nonbuiltin(hash_float_to_float(k),
                  hash_float_to_float(float2_nonbuiltin(k, 1.0)),
                  hash_float_to_float(float2_nonbuiltin(k, 2.0)));
}

DEVICE_FUNC float3_nonbuiltin hash_float_to_float3(float2_nonbuiltin k)
{
    return float3_nonbuiltin(hash_float_to_float(k),
                  hash_float_to_float(float3_nonbuiltin(k.x, k.y, 1.0)),
                  hash_float_to_float(float3_nonbuiltin(k.x, k.y, 2.0)));
}

DEVICE_FUNC float3_nonbuiltin hash_float_to_float3(float3_nonbuiltin k)
{
    return float3_nonbuiltin(hash_float_to_float(k),
                  hash_float_to_float(float4_nonbuiltin(k.x, k.y, k.z, 1.0)),
                  hash_float_to_float(float4_nonbuiltin(k.x, k.y, k.z, 2.0)));
}

DEVICE_FUNC float3_nonbuiltin hash_float_to_float3(float4_nonbuiltin k)
{
    return float3_nonbuiltin(hash_float_to_float(k),
                  hash_float_to_float(float4_nonbuiltin(k.z, k.x, k.w, k.y)),
                  hash_float_to_float(float4_nonbuiltin(k.w, k.z, k.y, k.x)));
}

DEVICE_FUNC float4_nonbuiltin hash_float_to_float4(float4_nonbuiltin k)
{
    return float4_nonbuiltin(hash_float_to_float(k),
                  hash_float_to_float(float4_nonbuiltin(k.w, k.x, k.y, k.z)),
                  hash_float_to_float(float4_nonbuiltin(k.z, k.w, k.x, k.y)),
                  hash_float_to_float(float4_nonbuiltin(k.y, k.z, k.w, k.x)));
}

/** \} */

/* The following offset functions generate random offsets to be added to
 * positions to act as a seed since the noise functions don't have seed values.
 * The offset's components are in the range [100, 200], not too high to cause
 * bad precision and not too small to be noticeable. We use float seed because
 * OSL only supports float hashes and we need to maintain compatibility with it.
 */

DEVICE_FUNC float random_float_offset(float seed)
{
    return 100.0f + hash_float_to_float(seed) * 100.0f;
}

DEVICE_FUNC float2_nonbuiltin random_float2_offset(float seed)
{
    return float2_nonbuiltin(100.0f + hash_float_to_float(float2_nonbuiltin(seed, 0.0f)) * 100.0f,
                  100.0f + hash_float_to_float(float2_nonbuiltin(seed, 1.0f)) * 100.0f);
}

DEVICE_FUNC float3_nonbuiltin random_float3_offset(float seed)
{
    return float3_nonbuiltin(100.0f + hash_float_to_float(float2_nonbuiltin(seed, 0.0f)) * 100.0f,
                  100.0f + hash_float_to_float(float2_nonbuiltin(seed, 1.0f)) * 100.0f,
                  100.0f + hash_float_to_float(float2_nonbuiltin(seed, 2.0f)) * 100.0f);
}

DEVICE_FUNC float4_nonbuiltin random_float4_offset(float seed)
{
    return float4_nonbuiltin(100.0f + hash_float_to_float(float2_nonbuiltin(seed, 0.0f)) * 100.0f,
                  100.0f + hash_float_to_float(float2_nonbuiltin(seed, 1.0f)) * 100.0f,
                  100.0f + hash_float_to_float(float2_nonbuiltin(seed, 2.0f)) * 100.0f,
                  100.0f + hash_float_to_float(float2_nonbuiltin(seed, 3.0f)) * 100.0f);
}

/* Perlin noises to be added to the position to distort other noises. */

DEVICE_FUNC float perlin_distortion(float position, float strength)
{
    return perlin_signed(position + random_float_offset(0.0)) * strength;
}

DEVICE_FUNC float2_nonbuiltin perlin_distortion(float2_nonbuiltin position, float strength)
{
    return float2_nonbuiltin(perlin_signed(position + random_float2_offset(0.0f)) * strength,
                  perlin_signed(position + random_float2_offset(1.0f)) * strength);
}

DEVICE_FUNC float3_nonbuiltin perlin_distortion(float3_nonbuiltin position, float strength)
{
    return float3_nonbuiltin(perlin_signed(position + random_float3_offset(0.0f)) * strength,
                  perlin_signed(position + random_float3_offset(1.0f)) * strength,
                  perlin_signed(position + random_float3_offset(2.0f)) * strength);
}

DEVICE_FUNC float4_nonbuiltin perlin_distortion(float4_nonbuiltin position, float strength)
{
    return float4_nonbuiltin(perlin_signed(position + random_float4_offset(0.0f)) * strength,
                  perlin_signed(position + random_float4_offset(1.0f)) * strength,
                  perlin_signed(position + random_float4_offset(2.0f)) * strength,
                  perlin_signed(position + random_float4_offset(3.0f)) * strength);
}

/* Positive fractal perlin noise. */

template <typename T>
DEVICE_FUNC float perlin_fractal_template(T position, float octaves, float roughness)
{
    float fscale = 1.0f;
    float amp = 1.0f;
    float maxamp = 0.0f;
    float sum = 0.0f;
    octaves = CLAMPIS(octaves, 0.0f, 15.0f);
    int n = int(octaves);
    for (int i = 0; i <= n; i++)
    {
        float t = perlin(fscale * position);
        sum += t * amp;
        maxamp += amp;
        amp *= CLAMPIS(roughness, 0.0f, 1.0f);
        fscale *= 2.0f;
    }
    float rmd = octaves - floor(octaves);
    if (rmd == 0.0f)
    {
        return sum / maxamp;
    }

    float t = perlin(fscale * position);
    float sum2 = sum + t * amp;
    sum /= maxamp;
    sum2 /= maxamp + amp;
    return (1.0f - rmd) * sum + rmd * sum2;
}

DEVICE_FUNC float perlin_fractal(float position, float octaves, float roughness)
{
    return perlin_fractal_template(position, octaves, roughness);
}

DEVICE_FUNC float perlin_fractal(float2_nonbuiltin position, float octaves, float roughness)
{
    return perlin_fractal_template(position, octaves, roughness);
}

DEVICE_FUNC float perlin_fractal(float3_nonbuiltin position, float octaves, float roughness)
{
    return perlin_fractal_template(position, octaves, roughness);
}

DEVICE_FUNC float perlin_fractal(float4_nonbuiltin position, float octaves, float roughness)
{
    return perlin_fractal_template(position, octaves, roughness);
}

/* Positive distorted fractal perlin noise. */

DEVICE_FUNC float perlin_fractal_distorted(float position, float octaves, float roughness, float distortion)
{
    position += perlin_distortion(position, distortion);
    return perlin_fractal(position, octaves, roughness);
}

DEVICE_FUNC float perlin_fractal_distorted(float2_nonbuiltin position, float octaves, float roughness, float distortion)
{
    position += perlin_distortion(position, distortion);
    return perlin_fractal(position, octaves, roughness);
}

DEVICE_FUNC float perlin_fractal_distorted(float3_nonbuiltin position, float octaves, float roughness, float distortion)
{
    position += perlin_distortion(position, distortion);
    return perlin_fractal(position, octaves, roughness);
}

DEVICE_FUNC float perlin_fractal_distorted(float4_nonbuiltin position, float octaves, float roughness, float distortion)
{
    position += perlin_distortion(position, distortion);
    return perlin_fractal(position, octaves, roughness);
}

/* Positive distorted fractal perlin noise that outputs a float3_nonbuiltin. The arbitrary seeds are for
 * compatibility with shading functions. */

DEVICE_FUNC float3_nonbuiltin perlin_float3_fractal_distorted(float position,
                                       float octaves,
                                       float roughness,
                                       float distortion)
{
    position += perlin_distortion(position, distortion);
    return float3_nonbuiltin(perlin_fractal(position, octaves, roughness),
                  perlin_fractal(position + random_float_offset(1.0f), octaves, roughness),
                  perlin_fractal(position + random_float_offset(2.0f), octaves, roughness));
}

DEVICE_FUNC float3_nonbuiltin perlin_float3_fractal_distorted(float2_nonbuiltin position,
                                       float octaves,
                                       float roughness,
                                       float distortion)
{
    position += perlin_distortion(position, distortion);
    return float3_nonbuiltin(perlin_fractal(position, octaves, roughness),
                  perlin_fractal(position + random_float2_offset(2.0f), octaves, roughness),
                  perlin_fractal(position + random_float2_offset(3.0f), octaves, roughness));
}

DEVICE_FUNC float3_nonbuiltin perlin_float3_fractal_distorted(float3_nonbuiltin position,
                                       float octaves,
                                       float roughness,
                                       float distortion)
{
    position += perlin_distortion(position, distortion);
    return float3_nonbuiltin(perlin_fractal(position, octaves, roughness),
                  perlin_fractal(position + random_float3_offset(3.0f), octaves, roughness),
                  perlin_fractal(position + random_float3_offset(4.0f), octaves, roughness));
}

DEVICE_FUNC float3_nonbuiltin perlin_float3_fractal_distorted(float4_nonbuiltin position,
                                       float octaves,
                                       float roughness,
                                       float distortion)
{
    position += perlin_distortion(position, distortion);
    return float3_nonbuiltin(perlin_fractal(position, octaves, roughness),
                  perlin_fractal(position + random_float4_offset(4.0f), octaves, roughness),
                  perlin_fractal(position + random_float4_offset(5.0f), octaves, roughness));
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Voronoi Noise
 *
 * \note Ported from Cycles code.
 *
 * Original code is under the MIT License, Copyright (c) 2013 Inigo Quilez.
 *
 * Smooth Voronoi:
 *
 * - https://wiki.blender.org/wiki/User:OmarSquircleArt/GSoC2019/Documentation/Smooth_Voronoi
 *
 * Distance To Edge based on:
 *
 * - https://www.iquilezles.org/www/articles/voronoilines/voronoilines.htm
 * - https://www.shadertoy.com/view/ldl3W8
 *
 * With optimization to change -2..2 scan window to -1..1 for better performance,
 * as explained in https://www.shadertoy.com/view/llG3zy.
 * \{ */

/* **** 1D Voronoi **** */

DEVICE_FUNC float voronoi_distance(const float a, const float b)
{
    return abs(b - a);
}

DEVICE_FUNC void voronoi_f1(
    const float w, const float randomness, float *r_distance, float3_nonbuiltin *r_color, float *r_w)
{
    const float cellPosition = floorf(w);
    const float localPosition = w - cellPosition;

    float minDistance = 8.0f;
    float targetOffset = 0.0f;
    float targetPosition = 0.0f;
    for (int i = -1; i <= 1; i++)
    {
        const float cellOffset = i;
        const float pointPosition = cellOffset +
                                    hash_float_to_float(cellPosition + cellOffset) * randomness;
        const float distanceToPoint = voronoi_distance(pointPosition, localPosition);
        if (distanceToPoint < minDistance)
        {
            targetOffset = cellOffset;
            minDistance = distanceToPoint;
            targetPosition = pointPosition;
        }
    }
    if (r_distance != nullptr)
    {
        *r_distance = minDistance;
    }
    if (r_color != nullptr)
    {
        *r_color = hash_float_to_float3(cellPosition + targetOffset);
    }
    if (r_w != nullptr)
    {
        *r_w = targetPosition + cellPosition;
    }
}

DEVICE_FUNC void voronoi_smooth_f1(const float w,
                       const float smoothness,
                       const float randomness,
                       float *r_distance,
                       float3_nonbuiltin *r_color,
                       float *r_w)
{
    const float cellPosition = floorf(w);
    const float localPosition = w - cellPosition;
    const float smoothness_clamped = max_ff(smoothness, FLT_MIN);

    float smoothDistance = 8.0f;
    float smoothPosition = 0.0f;
    float3_nonbuiltin smoothColor = float3_nonbuiltin(0.0f, 0.0f, 0.0f);
    for (int i = -2; i <= 2; i++)
    {
        const float cellOffset = i;
        const float pointPosition = cellOffset +
                                    hash_float_to_float(cellPosition + cellOffset) * randomness;
        const float distanceToPoint = voronoi_distance(pointPosition, localPosition);
        const float h = smoothstep(
            0.0f, 1.0f, 0.5f + 0.5f * (smoothDistance - distanceToPoint) / smoothness_clamped);
        float correctionFactor = smoothness * h * (1.0f - h);
        smoothDistance = mix(smoothDistance, distanceToPoint, h) - correctionFactor;
        if (r_color != nullptr || r_w != nullptr)
        {
            correctionFactor /= 1.0f + 3.0f * smoothness;
            if (r_color != nullptr)
            {
                const float3_nonbuiltin cellColor = hash_float_to_float3(cellPosition + cellOffset);
                smoothColor = interpolate(smoothColor, cellColor, h) - correctionFactor;
            }
            if (r_w != nullptr)
            {
                smoothPosition = mix(smoothPosition, pointPosition, h) - correctionFactor;
            }
        }
    }
    if (r_distance != nullptr)
    {
        *r_distance = smoothDistance;
    }
    if (r_color != nullptr)
    {
        *r_color = smoothColor;
    }
    if (r_w != nullptr)
    {
        *r_w = cellPosition + smoothPosition;
    }
}

DEVICE_FUNC void voronoi_f2(
    const float w, const float randomness, float *r_distance, float3_nonbuiltin *r_color, float *r_w)
{
    const float cellPosition = floorf(w);
    const float localPosition = w - cellPosition;

    float distanceF1 = 8.0f;
    float distanceF2 = 8.0f;
    float offsetF1 = 0.0f;
    float positionF1 = 0.0f;
    float offsetF2 = 0.0f;
    float positionF2 = 0.0f;
    for (int i = -1; i <= 1; i++)
    {
        const float cellOffset = i;
        const float pointPosition = cellOffset +
                                    hash_float_to_float(cellPosition + cellOffset) * randomness;
        const float distanceToPoint = voronoi_distance(pointPosition, localPosition);
        if (distanceToPoint < distanceF1)
        {
            distanceF2 = distanceF1;
            distanceF1 = distanceToPoint;
            offsetF2 = offsetF1;
            offsetF1 = cellOffset;
            positionF2 = positionF1;
            positionF1 = pointPosition;
        }
        else if (distanceToPoint < distanceF2)
        {
            distanceF2 = distanceToPoint;
            offsetF2 = cellOffset;
            positionF2 = pointPosition;
        }
    }
    if (r_distance != nullptr)
    {
        *r_distance = distanceF2;
    }
    if (r_color != nullptr)
    {
        *r_color = hash_float_to_float3(cellPosition + offsetF2);
    }
    if (r_w != nullptr)
    {
        *r_w = positionF2 + cellPosition;
    }
}

DEVICE_FUNC void voronoi_distance_to_edge(const float w, const float randomness, float *r_distance)
{
    const float cellPosition = floorf(w);
    const float localPosition = w - cellPosition;

    const float midPointPosition = hash_float_to_float(cellPosition) * randomness;
    const float leftPointPosition = -1.0f + hash_float_to_float(cellPosition - 1.0f) * randomness;
    const float rightPointPosition = 1.0f + hash_float_to_float(cellPosition + 1.0f) * randomness;
    const float distanceToMidLeft = abs((midPointPosition + leftPointPosition) / 2.0f -
                                             localPosition);
    const float distanceToMidRight = abs((midPointPosition + rightPointPosition) / 2.0f -
                                              localPosition);

    *r_distance = min(distanceToMidLeft, distanceToMidRight);
}

DEVICE_FUNC void voronoi_n_sphere_radius(const float w, const float randomness, float *r_radius)
{
    const float cellPosition = floorf(w);
    const float localPosition = w - cellPosition;

    float closestPoint = 0.0f;
    float closestPointOffset = 0.0f;
    float minDistance = 8.0f;
    for (int i = -1; i <= 1; i++)
    {
        const float cellOffset = i;
        const float pointPosition = cellOffset +
                                    hash_float_to_float(cellPosition + cellOffset) * randomness;
        const float distanceToPoint = abs(pointPosition - localPosition);
        if (distanceToPoint < minDistance)
        {
            minDistance = distanceToPoint;
            closestPoint = pointPosition;
            closestPointOffset = cellOffset;
        }
    }

    minDistance = 8.0f;
    float closestPointToClosestPoint = 0.0f;
    for (int i = -1; i <= 1; i++)
    {
        if (i == 0)
        {
            continue;
        }
        const float cellOffset = i + closestPointOffset;
        const float pointPosition = cellOffset +
                                    hash_float_to_float(cellPosition + cellOffset) * randomness;
        const float distanceToPoint = abs(closestPoint - pointPosition);
        if (distanceToPoint < minDistance)
        {
            minDistance = distanceToPoint;
            closestPointToClosestPoint = pointPosition;
        }
    }
    *r_radius = abs(closestPointToClosestPoint - closestPoint) / 2.0f;
}

/* **** 2D Voronoi **** */

DEVICE_FUNC float voronoi_distance(const float2_nonbuiltin a,
                              const float2_nonbuiltin b,
                              const int metric,
                              const float exponent)
{
    switch (metric)
    {
    case NOISE_SHD_VORONOI_EUCLIDEAN:
        return distance(a, b);
    case NOISE_SHD_VORONOI_MANHATTAN:
        return abs(a.x - b.x) + abs(a.y - b.y);
    case NOISE_SHD_VORONOI_CHEBYCHEV:
        return max(abs(a.x - b.x), abs(a.y - b.y));
    case NOISE_SHD_VORONOI_MINKOWSKI:
        return pow(pow(abs(a.x - b.x), exponent) +
                            pow(abs(a.y - b.y), exponent),
                        1.0f / exponent);
    default:
        assert(0);
        break;
    }
    return 0.0f;
}

DEVICE_FUNC void voronoi_f1(const float2_nonbuiltin coord,
                const float exponent,
                const float randomness,
                const int metric,
                float *r_distance,
                float3_nonbuiltin *r_color,
                float2_nonbuiltin *r_position)
{
    const float2_nonbuiltin cellPosition = floor(coord);
    const float2_nonbuiltin localPosition = coord - cellPosition;

    float minDistance = 8.0f;
    float2_nonbuiltin targetOffset = float2_nonbuiltin(0.0f, 0.0f);
    float2_nonbuiltin targetPosition = float2_nonbuiltin(0.0f, 0.0f);
    for (int j = -1; j <= 1; j++)
    {
        for (int i = -1; i <= 1; i++)
        {
            const float2_nonbuiltin cellOffset = float2_nonbuiltin(i, j);
            const float2_nonbuiltin pointPosition = cellOffset +
                                         hash_float_to_float2(cellPosition + cellOffset) * randomness;
            float distanceToPoint = voronoi_distance(pointPosition, localPosition, metric, exponent);
            if (distanceToPoint < minDistance)
            {
                targetOffset = cellOffset;
                minDistance = distanceToPoint;
                targetPosition = pointPosition;
            }
        }
    }
    if (r_distance != nullptr)
    {
        *r_distance = minDistance;
    }
    if (r_color != nullptr)
    {
        *r_color = hash_float_to_float3(cellPosition + targetOffset);
    }
    if (r_position != nullptr)
    {
        *r_position = targetPosition + cellPosition;
    }
}

DEVICE_FUNC void voronoi_smooth_f1(const float2_nonbuiltin coord,
                       const float smoothness,
                       const float exponent,
                       const float randomness,
                       const int metric,
                       float *r_distance,
                       float3_nonbuiltin *r_color,
                       float2_nonbuiltin *r_position)
{
    const float2_nonbuiltin cellPosition = floor(coord);
    const float2_nonbuiltin localPosition = coord - cellPosition;
    const float smoothness_clamped = max_ff(smoothness, FLT_MIN);

    float smoothDistance = 8.0f;
    float3_nonbuiltin smoothColor = float3_nonbuiltin(0.0f, 0.0f, 0.0f);
    float2_nonbuiltin smoothPosition = float2_nonbuiltin(0.0f, 0.0f);
    for (int j = -2; j <= 2; j++)
    {
        for (int i = -2; i <= 2; i++)
        {
            const float2_nonbuiltin cellOffset = float2_nonbuiltin(i, j);
            const float2_nonbuiltin pointPosition = cellOffset +
                                         hash_float_to_float2(cellPosition + cellOffset) * randomness;
            const float distanceToPoint = voronoi_distance(
                pointPosition, localPosition, metric, exponent);
            const float h = smoothstep(
                0.0f, 1.0f, 0.5f + 0.5f * (smoothDistance - distanceToPoint) / smoothness_clamped);
            float correctionFactor = smoothness * h * (1.0f - h);
            smoothDistance = mix(smoothDistance, distanceToPoint, h) - correctionFactor;
            if (r_color != nullptr || r_position != nullptr)
            {
                correctionFactor /= 1.0f + 3.0f * smoothness;
                if (r_color != nullptr)
                {
                    const float3_nonbuiltin cellColor = hash_float_to_float3(cellPosition + cellOffset);
                    smoothColor = interpolate(smoothColor, cellColor, h) - correctionFactor;
                }
                if (r_position != nullptr)
                {
                    smoothPosition = interpolate(smoothPosition, pointPosition, h) - correctionFactor;
                }
            }
        }
    }
    if (r_distance != nullptr)
    {
        *r_distance = smoothDistance;
    }
    if (r_color != nullptr)
    {
        *r_color = smoothColor;
    }
    if (r_position != nullptr)
    {
        *r_position = cellPosition + smoothPosition;
    }
}

DEVICE_FUNC void voronoi_f2(const float2_nonbuiltin coord,
                const float exponent,
                const float randomness,
                const int metric,
                float *r_distance,
                float3_nonbuiltin *r_color,
                float2_nonbuiltin *r_position)
{
    const float2_nonbuiltin cellPosition = floor(coord);
    const float2_nonbuiltin localPosition = coord - cellPosition;

    float distanceF1 = 8.0f;
    float distanceF2 = 8.0f;
    float2_nonbuiltin offsetF1 = float2_nonbuiltin(0.0f, 0.0f);
    float2_nonbuiltin positionF1 = float2_nonbuiltin(0.0f, 0.0f);
    float2_nonbuiltin offsetF2 = float2_nonbuiltin(0.0f, 0.0f);
    float2_nonbuiltin positionF2 = float2_nonbuiltin(0.0f, 0.0f);
    for (int j = -1; j <= 1; j++)
    {
        for (int i = -1; i <= 1; i++)
        {
            const float2_nonbuiltin cellOffset = float2_nonbuiltin(i, j);
            const float2_nonbuiltin pointPosition = cellOffset +
                                         hash_float_to_float2(cellPosition + cellOffset) * randomness;
            const float distanceToPoint = voronoi_distance(
                pointPosition, localPosition, metric, exponent);
            if (distanceToPoint < distanceF1)
            {
                distanceF2 = distanceF1;
                distanceF1 = distanceToPoint;
                offsetF2 = offsetF1;
                offsetF1 = cellOffset;
                positionF2 = positionF1;
                positionF1 = pointPosition;
            }
            else if (distanceToPoint < distanceF2)
            {
                distanceF2 = distanceToPoint;
                offsetF2 = cellOffset;
                positionF2 = pointPosition;
            }
        }
    }
    if (r_distance != nullptr)
    {
        *r_distance = distanceF2;
    }
    if (r_color != nullptr)
    {
        *r_color = hash_float_to_float3(cellPosition + offsetF2);
    }
    if (r_position != nullptr)
    {
        *r_position = positionF2 + cellPosition;
    }
}

DEVICE_FUNC void voronoi_distance_to_edge(const float2_nonbuiltin coord, const float randomness, float *r_distance)
{
    const float2_nonbuiltin cellPosition = floor(coord);
    const float2_nonbuiltin localPosition = coord - cellPosition;

    float2_nonbuiltin vectorToClosest = float2_nonbuiltin(0.0f, 0.0f);
    float minDistance = 8.0f;
    for (int j = -1; j <= 1; j++)
    {
        for (int i = -1; i <= 1; i++)
        {
            const float2_nonbuiltin cellOffset = float2_nonbuiltin(i, j);
            const float2_nonbuiltin vectorToPoint = cellOffset +
                                         hash_float_to_float2(cellPosition + cellOffset) * randomness -
                                         localPosition;
            const float distanceToPoint = length_squared(vectorToPoint);
            if (distanceToPoint < minDistance)
            {
                minDistance = distanceToPoint;
                vectorToClosest = vectorToPoint;
            }
        }
    }

    minDistance = 8.0f;
    for (int j = -1; j <= 1; j++)
    {
        for (int i = -1; i <= 1; i++)
        {
            const float2_nonbuiltin cellOffset = float2_nonbuiltin(i, j);
            const float2_nonbuiltin vectorToPoint = cellOffset +
                                         hash_float_to_float2(cellPosition + cellOffset) * randomness -
                                         localPosition;
            const float2_nonbuiltin perpendicularToEdge = vectorToPoint - vectorToClosest;
            if (dot(perpendicularToEdge, perpendicularToEdge) > 0.0001f)
            {
                const float distanceToEdge = dot((vectorToClosest + vectorToPoint) / 2.0f,
                                                 normalize(perpendicularToEdge));
                minDistance = min(minDistance, distanceToEdge);
            }
        }
    }
    *r_distance = minDistance;
}

DEVICE_FUNC void voronoi_n_sphere_radius(const float2_nonbuiltin coord, const float randomness, float *r_radius)
{
    const float2_nonbuiltin cellPosition = floor(coord);
    const float2_nonbuiltin localPosition = coord - cellPosition;

    float2_nonbuiltin closestPoint = float2_nonbuiltin(0.0f, 0.0f);
    float2_nonbuiltin closestPointOffset = float2_nonbuiltin(0.0f, 0.0f);
    float minDistance = 8.0f;
    for (int j = -1; j <= 1; j++)
    {
        for (int i = -1; i <= 1; i++)
        {
            const float2_nonbuiltin cellOffset = float2_nonbuiltin(i, j);
            const float2_nonbuiltin pointPosition = cellOffset +
                                         hash_float_to_float2(cellPosition + cellOffset) * randomness;
            const float distanceToPoint = distance(pointPosition, localPosition);
            if (distanceToPoint < minDistance)
            {
                minDistance = distanceToPoint;
                closestPoint = pointPosition;
                closestPointOffset = cellOffset;
            }
        }
    }

    minDistance = 8.0f;
    float2_nonbuiltin closestPointToClosestPoint = float2_nonbuiltin(0.0f, 0.0f);
    for (int j = -1; j <= 1; j++)
    {
        for (int i = -1; i <= 1; i++)
        {
            if (i == 0 && j == 0)
            {
                continue;
            }
            const float2_nonbuiltin cellOffset = float2_nonbuiltin(i, j) + closestPointOffset;
            const float2_nonbuiltin pointPosition = cellOffset +
                                         hash_float_to_float2(cellPosition + cellOffset) * randomness;
            const float distanceToPoint = distance(closestPoint, pointPosition);
            if (distanceToPoint < minDistance)
            {
                minDistance = distanceToPoint;
                closestPointToClosestPoint = pointPosition;
            }
        }
    }
    *r_radius = distance(closestPointToClosestPoint, closestPoint) / 2.0f;
}

/* **** 3D Voronoi **** */

DEVICE_FUNC float voronoi_distance(const float3_nonbuiltin a,
                              const float3_nonbuiltin b,
                              const int metric,
                              const float exponent)
{
    switch (metric)
    {
    case NOISE_SHD_VORONOI_EUCLIDEAN:
        return distance(a, b);
    case NOISE_SHD_VORONOI_MANHATTAN:
        return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z);
    case NOISE_SHD_VORONOI_CHEBYCHEV:
        return max(abs(a.x - b.x), max(abs(a.y - b.y), abs(a.z - b.z)));
    case NOISE_SHD_VORONOI_MINKOWSKI:
        return pow(pow(abs(a.x - b.x), exponent) +
                            pow(abs(a.y - b.y), exponent) +
                            pow(abs(a.z - b.z), exponent),
                        1.0f / exponent);
    default:
        assert(0);
        break;
    }
    return 0.0f;
}

DEVICE_FUNC void voronoi_f1(const float3_nonbuiltin coord,
                const float exponent,
                const float randomness,
                const int metric,
                float *r_distance,
                float3_nonbuiltin *r_color,
                float3_nonbuiltin *r_position)
{
    const float3_nonbuiltin cellPosition = floor(coord);
    const float3_nonbuiltin localPosition = coord - cellPosition;

    float minDistance = 8.0f;
    float3_nonbuiltin targetOffset = float3_nonbuiltin(0.0f, 0.0f, 0.0f);
    float3_nonbuiltin targetPosition = float3_nonbuiltin(0.0f, 0.0f, 0.0f);
    for (int k = -1; k <= 1; k++)
    {
        for (int j = -1; j <= 1; j++)
        {
            for (int i = -1; i <= 1; i++)
            {
                const float3_nonbuiltin cellOffset = float3_nonbuiltin(i, j, k);
                const float3_nonbuiltin pointPosition = cellOffset +
                                             hash_float_to_float3(cellPosition + cellOffset) * randomness;
                const float distanceToPoint = voronoi_distance(
                    pointPosition, localPosition, metric, exponent);
                if (distanceToPoint < minDistance)
                {
                    targetOffset = cellOffset;
                    minDistance = distanceToPoint;
                    targetPosition = pointPosition;
                }
            }
        }
    }
    if (r_distance != nullptr)
    {
        *r_distance = minDistance;
    }
    if (r_color != nullptr)
    {
        *r_color = hash_float_to_float3(cellPosition + targetOffset);
    }
    if (r_position != nullptr)
    {
        *r_position = targetPosition + cellPosition;
    }
}

DEVICE_FUNC void voronoi_smooth_f1(const float3_nonbuiltin coord,
                       const float smoothness,
                       const float exponent,
                       const float randomness,
                       const int metric,
                       float *r_distance,
                       float3_nonbuiltin *r_color,
                       float3_nonbuiltin *r_position)
{
    const float3_nonbuiltin cellPosition = floor(coord);
    const float3_nonbuiltin localPosition = coord - cellPosition;
    const float smoothness_clamped = max_ff(smoothness, FLT_MIN);

    float smoothDistance = 8.0f;
    float3_nonbuiltin smoothColor = float3_nonbuiltin(0.0f, 0.0f, 0.0f);
    float3_nonbuiltin smoothPosition = float3_nonbuiltin(0.0f, 0.0f, 0.0f);
    for (int k = -2; k <= 2; k++)
    {
        for (int j = -2; j <= 2; j++)
        {
            for (int i = -2; i <= 2; i++)
            {
                const float3_nonbuiltin cellOffset = float3_nonbuiltin(i, j, k);
                const float3_nonbuiltin pointPosition = cellOffset +
                                             hash_float_to_float3(cellPosition + cellOffset) * randomness;
                const float distanceToPoint = voronoi_distance(
                    pointPosition, localPosition, metric, exponent);
                const float h = smoothstep(
                    0.0f, 1.0f, 0.5f + 0.5f * (smoothDistance - distanceToPoint) / smoothness_clamped);
                float correctionFactor = smoothness * h * (1.0f - h);
                smoothDistance = mix(smoothDistance, distanceToPoint, h) - correctionFactor;
                if (r_color != nullptr || r_position != nullptr)
                {
                    correctionFactor /= 1.0f + 3.0f * smoothness;
                    if (r_color != nullptr)
                    {
                        const float3_nonbuiltin cellColor = hash_float_to_float3(cellPosition + cellOffset);
                        smoothColor = interpolate(smoothColor, cellColor, h) - correctionFactor;
                    }
                    if (r_position != nullptr)
                    {
                        smoothPosition = interpolate(smoothPosition, pointPosition, h) -
                                         correctionFactor;
                    }
                }
            }
        }
    }
    if (r_distance != nullptr)
    {
        *r_distance = smoothDistance;
    }
    if (r_color != nullptr)
    {
        *r_color = smoothColor;
    }
    if (r_position != nullptr)
    {
        *r_position = cellPosition + smoothPosition;
    }
}

DEVICE_FUNC void voronoi_f2(const float3_nonbuiltin coord,
                const float exponent,
                const float randomness,
                const int metric,
                float *r_distance,
                float3_nonbuiltin *r_color,
                float3_nonbuiltin *r_position)
{
    const float3_nonbuiltin cellPosition = floor(coord);
    const float3_nonbuiltin localPosition = coord - cellPosition;

    float distanceF1 = 8.0f;
    float distanceF2 = 8.0f;
    float3_nonbuiltin offsetF1 = float3_nonbuiltin(0.0f, 0.0f, 0.0f);
    float3_nonbuiltin positionF1 = float3_nonbuiltin(0.0f, 0.0f, 0.0f);
    float3_nonbuiltin offsetF2 = float3_nonbuiltin(0.0f, 0.0f, 0.0f);
    float3_nonbuiltin positionF2 = float3_nonbuiltin(0.0f, 0.0f, 0.0f);
    for (int k = -1; k <= 1; k++)
    {
        for (int j = -1; j <= 1; j++)
        {
            for (int i = -1; i <= 1; i++)
            {
                const float3_nonbuiltin cellOffset = float3_nonbuiltin(i, j, k);
                const float3_nonbuiltin pointPosition = cellOffset +
                                             hash_float_to_float3(cellPosition + cellOffset) * randomness;
                const float distanceToPoint = voronoi_distance(
                    pointPosition, localPosition, metric, exponent);
                if (distanceToPoint < distanceF1)
                {
                    distanceF2 = distanceF1;
                    distanceF1 = distanceToPoint;
                    offsetF2 = offsetF1;
                    offsetF1 = cellOffset;
                    positionF2 = positionF1;
                    positionF1 = pointPosition;
                }
                else if (distanceToPoint < distanceF2)
                {
                    distanceF2 = distanceToPoint;
                    offsetF2 = cellOffset;
                    positionF2 = pointPosition;
                }
            }
        }
    }
    if (r_distance != nullptr)
    {
        *r_distance = distanceF2;
    }
    if (r_color != nullptr)
    {
        *r_color = hash_float_to_float3(cellPosition + offsetF2);
    }
    if (r_position != nullptr)
    {
        *r_position = positionF2 + cellPosition;
    }
}

DEVICE_FUNC void voronoi_distance_to_edge(const float3_nonbuiltin coord, const float randomness, float *r_distance)
{
    const float3_nonbuiltin cellPosition = floor(coord);
    const float3_nonbuiltin localPosition = coord - cellPosition;

    float3_nonbuiltin vectorToClosest = float3_nonbuiltin(0.0f, 0.0f, 0.0f);
    float minDistance = 8.0f;
    for (int k = -1; k <= 1; k++)
    {
        for (int j = -1; j <= 1; j++)
        {
            for (int i = -1; i <= 1; i++)
            {
                const float3_nonbuiltin cellOffset = float3_nonbuiltin(i, j, k);
                const float3_nonbuiltin vectorToPoint = cellOffset +
                                             hash_float_to_float3(cellPosition + cellOffset) * randomness -
                                             localPosition;
                const float distanceToPoint = dot(vectorToPoint, vectorToPoint);
                if (distanceToPoint < minDistance)
                {
                    minDistance = distanceToPoint;
                    vectorToClosest = vectorToPoint;
                }
            }
        }
    }

    minDistance = 8.0f;
    for (int k = -1; k <= 1; k++)
    {
        for (int j = -1; j <= 1; j++)
        {
            for (int i = -1; i <= 1; i++)
            {
                const float3_nonbuiltin cellOffset = float3_nonbuiltin(i, j, k);
                const float3_nonbuiltin vectorToPoint = cellOffset +
                                             hash_float_to_float3(cellPosition + cellOffset) * randomness -
                                             localPosition;
                const float3_nonbuiltin perpendicularToEdge = vectorToPoint - vectorToClosest;
                if (dot(perpendicularToEdge, perpendicularToEdge) > 0.0001f)
                {
                    const float distanceToEdge = dot((vectorToClosest + vectorToPoint) / 2.0f,
                                                     normalize(perpendicularToEdge));
                    minDistance = min(minDistance, distanceToEdge);
                }
            }
        }
    }
    *r_distance = minDistance;
}

DEVICE_FUNC void voronoi_n_sphere_radius(const float3_nonbuiltin coord, const float randomness, float *r_radius)
{
    const float3_nonbuiltin cellPosition = floor(coord);
    const float3_nonbuiltin localPosition = coord - cellPosition;

    float3_nonbuiltin closestPoint = float3_nonbuiltin(0.0f, 0.0f, 0.0f);
    float3_nonbuiltin closestPointOffset = float3_nonbuiltin(0.0f, 0.0f, 0.0f);
    float minDistance = 8.0f;
    for (int k = -1; k <= 1; k++)
    {
        for (int j = -1; j <= 1; j++)
        {
            for (int i = -1; i <= 1; i++)
            {
                const float3_nonbuiltin cellOffset = float3_nonbuiltin(i, j, k);
                const float3_nonbuiltin pointPosition = cellOffset +
                                             hash_float_to_float3(cellPosition + cellOffset) * randomness;
                const float distanceToPoint = distance(pointPosition, localPosition);
                if (distanceToPoint < minDistance)
                {
                    minDistance = distanceToPoint;
                    closestPoint = pointPosition;
                    closestPointOffset = cellOffset;
                }
            }
        }
    }

    minDistance = 8.0f;
    float3_nonbuiltin closestPointToClosestPoint = float3_nonbuiltin(0.0f, 0.0f, 0.0f);
    for (int k = -1; k <= 1; k++)
    {
        for (int j = -1; j <= 1; j++)
        {
            for (int i = -1; i <= 1; i++)
            {
                if (i == 0 && j == 0 && k == 0)
                {
                    continue;
                }
                const float3_nonbuiltin cellOffset = float3_nonbuiltin(i, j, k) + closestPointOffset;
                const float3_nonbuiltin pointPosition = cellOffset +
                                             hash_float_to_float3(cellPosition + cellOffset) * randomness;
                const float distanceToPoint = distance(closestPoint, pointPosition);
                if (distanceToPoint < minDistance)
                {
                    minDistance = distanceToPoint;
                    closestPointToClosestPoint = pointPosition;
                }
            }
        }
    }
    *r_radius = distance(closestPointToClosestPoint, closestPoint) / 2.0f;
}

/* **** 4D Voronoi **** */

DEVICE_FUNC float voronoi_distance(const float4_nonbuiltin a,
                              const float4_nonbuiltin b,
                              const int metric,
                              const float exponent)
{
    switch (metric)
    {
    case NOISE_SHD_VORONOI_EUCLIDEAN:
        return distance(a, b);
    case NOISE_SHD_VORONOI_MANHATTAN:
        return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z) + abs(a.w - b.w);
    case NOISE_SHD_VORONOI_CHEBYCHEV:
        return max(
            abs(a.x - b.x),
            max(abs(a.y - b.y), max(abs(a.z - b.z), abs(a.w - b.w))));
    case NOISE_SHD_VORONOI_MINKOWSKI:
        return pow(
            pow(abs(a.x - b.x), exponent) + pow(abs(a.y - b.y), exponent) +
                pow(abs(a.z - b.z), exponent) + pow(abs(a.w - b.w), exponent),
            1.0f / exponent);
    default:
        assert(0);
        break;
    }
    return 0.0f;
}

DEVICE_FUNC void voronoi_f1(const float4_nonbuiltin coord,
                const float exponent,
                const float randomness,
                const int metric,
                float *r_distance,
                float3_nonbuiltin *r_color,
                float4_nonbuiltin *r_position)
{
    const float4_nonbuiltin cellPosition = floor(coord);
    const float4_nonbuiltin localPosition = coord - cellPosition;

    float minDistance = 8.0f;
    float4_nonbuiltin targetOffset = float4_nonbuiltin(0.0f, 0.0f, 0.0f, 0.0f);
    float4_nonbuiltin targetPosition = float4_nonbuiltin(0.0f, 0.0f, 0.0f, 0.0f);
    for (int u = -1; u <= 1; u++)
    {
        for (int k = -1; k <= 1; k++)
        {
            for (int j = -1; j <= 1; j++)
            {
                for (int i = -1; i <= 1; i++)
                {
                    const float4_nonbuiltin cellOffset = float4_nonbuiltin(i, j, k, u);
                    const float4_nonbuiltin pointPosition = cellOffset +
                                                 hash_float_to_float4(cellPosition + cellOffset) *
                                                     randomness;
                    const float distanceToPoint = voronoi_distance(
                        pointPosition, localPosition, metric, exponent);
                    if (distanceToPoint < minDistance)
                    {
                        targetOffset = cellOffset;
                        minDistance = distanceToPoint;
                        targetPosition = pointPosition;
                    }
                }
            }
        }
    }
    if (r_distance != nullptr)
    {
        *r_distance = minDistance;
    }
    if (r_color != nullptr)
    {
        *r_color = hash_float_to_float3(cellPosition + targetOffset);
    }
    if (r_position != nullptr)
    {
        *r_position = targetPosition + cellPosition;
    }
}

DEVICE_FUNC void voronoi_smooth_f1(const float4_nonbuiltin coord,
                       const float smoothness,
                       const float exponent,
                       const float randomness,
                       const int metric,
                       float *r_distance,
                       float3_nonbuiltin *r_color,
                       float4_nonbuiltin *r_position)
{
    const float4_nonbuiltin cellPosition = floor(coord);
    const float4_nonbuiltin localPosition = coord - cellPosition;
    const float smoothness_clamped = max_ff(smoothness, FLT_MIN);

    float smoothDistance = 8.0f;
    float3_nonbuiltin smoothColor = float3_nonbuiltin(0.0f, 0.0f, 0.0f);
    float4_nonbuiltin smoothPosition = float4_nonbuiltin(0.0f, 0.0f, 0.0f, 0.0f);
    for (int u = -2; u <= 2; u++)
    {
        for (int k = -2; k <= 2; k++)
        {
            for (int j = -2; j <= 2; j++)
            {
                for (int i = -2; i <= 2; i++)
                {
                    const float4_nonbuiltin cellOffset = float4_nonbuiltin(i, j, k, u);
                    const float4_nonbuiltin pointPosition = cellOffset +
                                                 hash_float_to_float4(cellPosition + cellOffset) *
                                                     randomness;
                    const float distanceToPoint = voronoi_distance(
                        pointPosition, localPosition, metric, exponent);
                    const float h = smoothstep(
                        0.0f, 1.0f, 0.5f + 0.5f * (smoothDistance - distanceToPoint) / smoothness_clamped);
                    float correctionFactor = smoothness * h * (1.0f - h);
                    smoothDistance = mix(smoothDistance, distanceToPoint, h) - correctionFactor;
                    if (r_color != nullptr || r_position != nullptr)
                    {
                        correctionFactor /= 1.0f + 3.0f * smoothness;
                        if (r_color != nullptr)
                        {
                            const float3_nonbuiltin cellColor = hash_float_to_float3(cellPosition + cellOffset);
                            smoothColor = interpolate(smoothColor, cellColor, h) - correctionFactor;
                        }
                        if (r_position != nullptr)
                        {
                            smoothPosition = interpolate(smoothPosition, pointPosition, h) -
                                             correctionFactor;
                        }
                    }
                }
            }
        }
    }
    if (r_distance != nullptr)
    {
        *r_distance = smoothDistance;
    }
    if (r_color != nullptr)
    {
        *r_color = smoothColor;
    }
    if (r_position != nullptr)
    {
        *r_position = cellPosition + smoothPosition;
    }
}

DEVICE_FUNC void voronoi_f2(const float4_nonbuiltin coord,
                const float exponent,
                const float randomness,
                const int metric,
                float *r_distance,
                float3_nonbuiltin *r_color,
                float4_nonbuiltin *r_position)
{
    const float4_nonbuiltin cellPosition = floor(coord);
    const float4_nonbuiltin localPosition = coord - cellPosition;

    float distanceF1 = 8.0f;
    float distanceF2 = 8.0f;
    float4_nonbuiltin offsetF1 = float4_nonbuiltin(0.0f, 0.0f, 0.0f, 0.0f);
    float4_nonbuiltin positionF1 = float4_nonbuiltin(0.0f, 0.0f, 0.0f, 0.0f);
    float4_nonbuiltin offsetF2 = float4_nonbuiltin(0.0f, 0.0f, 0.0f, 0.0f);
    float4_nonbuiltin positionF2 = float4_nonbuiltin(0.0f, 0.0f, 0.0f, 0.0f);
    for (int u = -1; u <= 1; u++)
    {
        for (int k = -1; k <= 1; k++)
        {
            for (int j = -1; j <= 1; j++)
            {
                for (int i = -1; i <= 1; i++)
                {
                    const float4_nonbuiltin cellOffset = float4_nonbuiltin(i, j, k, u);
                    const float4_nonbuiltin pointPosition = cellOffset +
                                                 hash_float_to_float4(cellPosition + cellOffset) *
                                                     randomness;
                    const float distanceToPoint = voronoi_distance(
                        pointPosition, localPosition, metric, exponent);
                    if (distanceToPoint < distanceF1)
                    {
                        distanceF2 = distanceF1;
                        distanceF1 = distanceToPoint;
                        offsetF2 = offsetF1;
                        offsetF1 = cellOffset;
                        positionF2 = positionF1;
                        positionF1 = pointPosition;
                    }
                    else if (distanceToPoint < distanceF2)
                    {
                        distanceF2 = distanceToPoint;
                        offsetF2 = cellOffset;
                        positionF2 = pointPosition;
                    }
                }
            }
        }
    }
    if (r_distance != nullptr)
    {
        *r_distance = distanceF2;
    }
    if (r_color != nullptr)
    {
        *r_color = hash_float_to_float3(cellPosition + offsetF2);
    }
    if (r_position != nullptr)
    {
        *r_position = positionF2 + cellPosition;
    }
}

DEVICE_FUNC void voronoi_distance_to_edge(const float4_nonbuiltin coord, const float randomness, float *r_distance)
{
    const float4_nonbuiltin cellPosition = floor(coord);
    const float4_nonbuiltin localPosition = coord - cellPosition;

    float4_nonbuiltin vectorToClosest = float4_nonbuiltin(0.0f, 0.0f, 0.0f, 0.0f);
    float minDistance = 8.0f;
    for (int u = -1; u <= 1; u++)
    {
        for (int k = -1; k <= 1; k++)
        {
            for (int j = -1; j <= 1; j++)
            {
                for (int i = -1; i <= 1; i++)
                {
                    const float4_nonbuiltin cellOffset = float4_nonbuiltin(i, j, k, u);
                    const float4_nonbuiltin vectorToPoint = cellOffset +
                                                 hash_float_to_float4(cellPosition + cellOffset) *
                                                     randomness -
                                                 localPosition;
                    const float distanceToPoint = dot(vectorToPoint, vectorToPoint);
                    if (distanceToPoint < minDistance)
                    {
                        minDistance = distanceToPoint;
                        vectorToClosest = vectorToPoint;
                    }
                }
            }
        }
    }

    minDistance = 8.0f;
    for (int u = -1; u <= 1; u++)
    {
        for (int k = -1; k <= 1; k++)
        {
            for (int j = -1; j <= 1; j++)
            {
                for (int i = -1; i <= 1; i++)
                {
                    const float4_nonbuiltin cellOffset = float4_nonbuiltin(i, j, k, u);
                    const float4_nonbuiltin vectorToPoint = cellOffset +
                                                 hash_float_to_float4(cellPosition + cellOffset) *
                                                     randomness -
                                                 localPosition;
                    const float4_nonbuiltin perpendicularToEdge = vectorToPoint - vectorToClosest;
                    if (dot(perpendicularToEdge, perpendicularToEdge) > 0.0001f)
                    {
                        const float distanceToEdge = dot((vectorToClosest + vectorToPoint) / 2.0f,
                                                         normalize(perpendicularToEdge));
                        minDistance = min(minDistance, distanceToEdge);
                    }
                }
            }
        }
    }
    *r_distance = minDistance;
}

DEVICE_FUNC void voronoi_n_sphere_radius(const float4_nonbuiltin coord, const float randomness, float *r_radius)
{
    const float4_nonbuiltin cellPosition = floor(coord);
    const float4_nonbuiltin localPosition = coord - cellPosition;

    float4_nonbuiltin closestPoint = float4_nonbuiltin(0.0f, 0.0f, 0.0f, 0.0f);
    float4_nonbuiltin closestPointOffset = float4_nonbuiltin(0.0f, 0.0f, 0.0f, 0.0f);
    float minDistance = 8.0f;
    for (int u = -1; u <= 1; u++)
    {
        for (int k = -1; k <= 1; k++)
        {
            for (int j = -1; j <= 1; j++)
            {
                for (int i = -1; i <= 1; i++)
                {
                    const float4_nonbuiltin cellOffset = float4_nonbuiltin(i, j, k, u);
                    const float4_nonbuiltin pointPosition = cellOffset +
                                                 hash_float_to_float4(cellPosition + cellOffset) *
                                                     randomness;
                    const float distanceToPoint = distance(pointPosition, localPosition);
                    if (distanceToPoint < minDistance)
                    {
                        minDistance = distanceToPoint;
                        closestPoint = pointPosition;
                        closestPointOffset = cellOffset;
                    }
                }
            }
        }
    }

    minDistance = 8.0f;
    float4_nonbuiltin closestPointToClosestPoint = float4_nonbuiltin(0.0f, 0.0f, 0.0f, 0.0f);
    for (int u = -1; u <= 1; u++)
    {
        for (int k = -1; k <= 1; k++)
        {
            for (int j = -1; j <= 1; j++)
            {
                for (int i = -1; i <= 1; i++)
                {
                    if (i == 0 && j == 0 && k == 0 && u == 0)
                    {
                        continue;
                    }
                    const float4_nonbuiltin cellOffset = float4_nonbuiltin(i, j, k, u) + closestPointOffset;
                    const float4_nonbuiltin pointPosition = cellOffset +
                                                 hash_float_to_float4(cellPosition + cellOffset) *
                                                     randomness;
                    const float distanceToPoint = distance(closestPoint, pointPosition);
                    if (distanceToPoint < minDistance)
                    {
                        minDistance = distanceToPoint;
                        closestPointToClosestPoint = pointPosition;
                    }
                }
            }
        }
    }
    *r_radius = distance(closestPointToClosestPoint, closestPoint) / 2.0f;
}

/* -------------------------------------------------------------------- */
/** \name Musgrave Noise
 * \{ */

DEVICE_FUNC float musgrave_fBm(const float co,
                   const float H,
                   const float lacunarity,
                   const float octaves_unclamped)
{
    /* From "Texturing and Modelling: A procedural approach". */

    float p = co;
    float value = 0.0f;
    float pwr = 1.0f;
    const float pwHL = pow(lacunarity, -H);
    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 0; i < int(octaves); i++)
    {
        value += perlin_signed(p) * pwr;
        pwr *= pwHL;
        p *= lacunarity;
    }

    float rmd = octaves - floorf(octaves);
    if (rmd != 0.0f)
    {
        value += rmd * perlin_signed(p) * pwr;
    }

    return value;
}

DEVICE_FUNC float musgrave_multi_fractal(const float co,
                             const float H,
                             const float lacunarity,
                             const float octaves_unclamped)
{
    float p = co;
    float value = 1.0f;
    float pwr = 1.0f;
    const float pwHL = pow(lacunarity, -H);
    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 0; i < int(octaves); i++)
    {
        value *= (pwr * perlin_signed(p) + 1.0f);
        pwr *= pwHL;
        p *= lacunarity;
    }

    const float rmd = octaves - floorf(octaves);
    if (rmd != 0.0f)
    {
        value *= (rmd * pwr * perlin_signed(p) + 1.0f); /* correct? */
    }

    return value;
}

DEVICE_FUNC float musgrave_hetero_terrain(const float co,
                              const float H,
                              const float lacunarity,
                              const float octaves_unclamped,
                              const float offset)
{
    float p = co;
    const float pwHL = pow(lacunarity, -H);
    float pwr = pwHL;
    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    /* First unscaled octave of function; later octaves are scaled. */
    float value = offset + perlin_signed(p);
    p *= lacunarity;

    for (int i = 1; i < int(octaves); i++)
    {
        float increment = (perlin_signed(p) + offset) * pwr * value;
        value += increment;
        pwr *= pwHL;
        p *= lacunarity;
    }

    const float rmd = octaves - floorf(octaves);
    if (rmd != 0.0f)
    {
        float increment = (perlin_signed(p) + offset) * pwr * value;
        value += rmd * increment;
    }

    return value;
}

DEVICE_FUNC float musgrave_hybrid_multi_fractal(const float co,
                                    const float H,
                                    const float lacunarity,
                                    const float octaves_unclamped,
                                    const float offset,
                                    const float gain)
{
    float p = co;
    const float pwHL = pow(lacunarity, -H);

    float pwr = 1.0f;
    float value = 0.0f;
    float weight = 1.0f;

    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 0; (weight > 0.001f) && (i < int(octaves)); i++)
    {
        if (weight > 1.0f)
        {
            weight = 1.0f;
        }

        float signal = (perlin_signed(p) + offset) * pwr;
        pwr *= pwHL;
        value += weight * signal;
        weight *= gain * signal;
        p *= lacunarity;
    }

    const float rmd = octaves - floorf(octaves);
    if ((rmd != 0.0f) && (weight > 0.001f))
    {
        if (weight > 1.0f)
        {
            weight = 1.0f;
        }
        float signal = (perlin_signed(p) + offset) * pwr;
        value += rmd * weight * signal;
    }

    return value;
}

DEVICE_FUNC float musgrave_ridged_multi_fractal(const float co,
                                    const float H,
                                    const float lacunarity,
                                    const float octaves_unclamped,
                                    const float offset,
                                    const float gain)
{
    float p = co;
    const float pwHL = pow(lacunarity, -H);
    float pwr = pwHL;

    float signal = offset - abs(perlin_signed(p));
    signal *= signal;
    float value = signal;
    float weight = 1.0f;

    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 1; i < int(octaves); i++)
    {
        p *= lacunarity;
        weight = CLAMPIS(signal * gain, 0.0f, 1.0f);
        signal = offset - abs(perlin_signed(p));
        signal *= signal;
        signal *= weight;
        value += signal * pwr;
        pwr *= pwHL;
    }

    return value;
}

DEVICE_FUNC float musgrave_fBm(const float2_nonbuiltin co,
                   const float H,
                   const float lacunarity,
                   const float octaves_unclamped)
{
    /* From "Texturing and Modelling: A procedural approach". */

    float2_nonbuiltin p = co;
    float value = 0.0f;
    float pwr = 1.0f;
    const float pwHL = pow(lacunarity, -H);
    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 0; i < int(octaves); i++)
    {
        value += perlin_signed(p) * pwr;
        pwr *= pwHL;
        p *= lacunarity;
    }

    const float rmd = octaves - floorf(octaves);
    if (rmd != 0.0f)
    {
        value += rmd * perlin_signed(p) * pwr;
    }

    return value;
}

DEVICE_FUNC float musgrave_multi_fractal(const float2_nonbuiltin co,
                             const float H,
                             const float lacunarity,
                             const float octaves_unclamped)
{
    float2_nonbuiltin p = co;
    float value = 1.0f;
    float pwr = 1.0f;
    const float pwHL = pow(lacunarity, -H);
    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 0; i < int(octaves); i++)
    {
        value *= (pwr * perlin_signed(p) + 1.0f);
        pwr *= pwHL;
        p *= lacunarity;
    }

    const float rmd = octaves - floorf(octaves);
    if (rmd != 0.0f)
    {
        value *= (rmd * pwr * perlin_signed(p) + 1.0f); /* correct? */
    }

    return value;
}

DEVICE_FUNC float musgrave_hetero_terrain(const float2_nonbuiltin co,
                              const float H,
                              const float lacunarity,
                              const float octaves_unclamped,
                              const float offset)
{
    float2_nonbuiltin p = co;
    const float pwHL = pow(lacunarity, -H);
    float pwr = pwHL;

    /* First unscaled octave of function; later octaves are scaled. */
    float value = offset + perlin_signed(p);
    p *= lacunarity;

    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 1; i < int(octaves); i++)
    {
        float increment = (perlin_signed(p) + offset) * pwr * value;
        value += increment;
        pwr *= pwHL;
        p *= lacunarity;
    }

    const float rmd = octaves - floorf(octaves);
    if (rmd != 0.0f)
    {
        float increment = (perlin_signed(p) + offset) * pwr * value;
        value += rmd * increment;
    }

    return value;
}

DEVICE_FUNC float musgrave_hybrid_multi_fractal(const float2_nonbuiltin co,
                                    const float H,
                                    const float lacunarity,
                                    const float octaves_unclamped,
                                    const float offset,
                                    const float gain)
{
    float2_nonbuiltin p = co;
    const float pwHL = pow(lacunarity, -H);

    float pwr = 1.0f;
    float value = 0.0f;
    float weight = 1.0f;

    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 0; (weight > 0.001f) && (i < int(octaves)); i++)
    {
        if (weight > 1.0f)
        {
            weight = 1.0f;
        }

        float signal = (perlin_signed(p) + offset) * pwr;
        pwr *= pwHL;
        value += weight * signal;
        weight *= gain * signal;
        p *= lacunarity;
    }

    const float rmd = octaves - floorf(octaves);
    if ((rmd != 0.0f) && (weight > 0.001f))
    {
        if (weight > 1.0f)
        {
            weight = 1.0f;
        }
        float signal = (perlin_signed(p) + offset) * pwr;
        value += rmd * weight * signal;
    }

    return value;
}

DEVICE_FUNC float musgrave_ridged_multi_fractal(const float2_nonbuiltin co,
                                    const float H,
                                    const float lacunarity,
                                    const float octaves_unclamped,
                                    const float offset,
                                    const float gain)
{
    float2_nonbuiltin p = co;
    const float pwHL = pow(lacunarity, -H);
    float pwr = pwHL;

    float signal = offset - abs(perlin_signed(p));
    signal *= signal;
    float value = signal;
    float weight = 1.0f;

    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 1; i < int(octaves); i++)
    {
        p *= lacunarity;
        weight = CLAMPIS(signal * gain, 0.0f, 1.0f);
        signal = offset - abs(perlin_signed(p));
        signal *= signal;
        signal *= weight;
        value += signal * pwr;
        pwr *= pwHL;
    }

    return value;
}

DEVICE_FUNC float musgrave_fBm(const float3_nonbuiltin co,
                   const float H,
                   const float lacunarity,
                   const float octaves_unclamped)
{
    /* From "Texturing and Modelling: A procedural approach". */

    float3_nonbuiltin p = co;
    float value = 0.0f;
    float pwr = 1.0f;
    const float pwHL = pow(lacunarity, -H);

    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 0; i < int(octaves); i++)
    {
        value += perlin_signed(p) * pwr;
        pwr *= pwHL;
        p *= lacunarity;
    }

    const float rmd = octaves - floorf(octaves);
    if (rmd != 0.0f)
    {
        value += rmd * perlin_signed(p) * pwr;
    }

    return value;
}

DEVICE_FUNC float musgrave_multi_fractal(const float3_nonbuiltin co,
                             const float H,
                             const float lacunarity,
                             const float octaves_unclamped)
{
    float3_nonbuiltin p = co;
    float value = 1.0f;
    float pwr = 1.0f;
    const float pwHL = pow(lacunarity, -H);

    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 0; i < int(octaves); i++)
    {
        value *= (pwr * perlin_signed(p) + 1.0f);
        pwr *= pwHL;
        p *= lacunarity;
    }

    const float rmd = octaves - floorf(octaves);
    if (rmd != 0.0f)
    {
        value *= (rmd * pwr * perlin_signed(p) + 1.0f); /* correct? */
    }

    return value;
}

DEVICE_FUNC float musgrave_hetero_terrain(const float3_nonbuiltin co,
                              const float H,
                              const float lacunarity,
                              const float octaves_unclamped,
                              const float offset)
{
    float3_nonbuiltin p = co;
    const float pwHL = pow(lacunarity, -H);
    float pwr = pwHL;

    /* first unscaled octave of function; later octaves are scaled */
    float value = offset + perlin_signed(p);
    p *= lacunarity;

    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 1; i < int(octaves); i++)
    {
        float increment = (perlin_signed(p) + offset) * pwr * value;
        value += increment;
        pwr *= pwHL;
        p *= lacunarity;
    }

    const float rmd = octaves - floorf(octaves);
    if (rmd != 0.0f)
    {
        float increment = (perlin_signed(p) + offset) * pwr * value;
        value += rmd * increment;
    }

    return value;
}

DEVICE_FUNC float musgrave_hybrid_multi_fractal(const float3_nonbuiltin co,
                                    const float H,
                                    const float lacunarity,
                                    const float octaves_unclamped,
                                    const float offset,
                                    const float gain)
{
    float3_nonbuiltin p = co;
    const float pwHL = pow(lacunarity, -H);

    float pwr = 1.0f;
    float value = 0.0f;
    float weight = 1.0f;

    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 0; (weight > 0.001f) && (i < int(octaves)); i++)
    {
        if (weight > 1.0f)
        {
            weight = 1.0f;
        }

        float signal = (perlin_signed(p) + offset) * pwr;
        pwr *= pwHL;
        value += weight * signal;
        weight *= gain * signal;
        p *= lacunarity;
    }

    const float rmd = octaves - floorf(octaves);
    if ((rmd != 0.0f) && (weight > 0.001f))
    {
        if (weight > 1.0f)
        {
            weight = 1.0f;
        }
        float signal = (perlin_signed(p) + offset) * pwr;
        value += rmd * weight * signal;
    }

    return value;
}

DEVICE_FUNC float musgrave_ridged_multi_fractal(const float3_nonbuiltin co,
                                    const float H,
                                    const float lacunarity,
                                    const float octaves_unclamped,
                                    const float offset,
                                    const float gain)
{
    float3_nonbuiltin p = co;
    const float pwHL = pow(lacunarity, -H);
    float pwr = pwHL;

    float signal = offset - abs(perlin_signed(p));
    signal *= signal;
    float value = signal;
    float weight = 1.0f;

    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 1; i < int(octaves); i++)
    {
        p *= lacunarity;
        weight = CLAMPIS(signal * gain, 0.0f, 1.0f);
        signal = offset - abs(perlin_signed(p));
        signal *= signal;
        signal *= weight;
        value += signal * pwr;
        pwr *= pwHL;
    }

    return value;
}

DEVICE_FUNC float musgrave_fBm(const float4_nonbuiltin co,
                   const float H,
                   const float lacunarity,
                   const float octaves_unclamped)
{
    /* From "Texturing and Modelling: A procedural approach". */

    float4_nonbuiltin p = co;
    float value = 0.0f;
    float pwr = 1.0f;
    const float pwHL = pow(lacunarity, -H);

    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 0; i < int(octaves); i++)
    {
        value += perlin_signed(p) * pwr;
        pwr *= pwHL;
        p *= lacunarity;
    }

    const float rmd = octaves - floorf(octaves);
    if (rmd != 0.0f)
    {
        value += rmd * perlin_signed(p) * pwr;
    }

    return value;
}

DEVICE_FUNC float musgrave_multi_fractal(const float4_nonbuiltin co,
                             const float H,
                             const float lacunarity,
                             const float octaves_unclamped)
{
    float4_nonbuiltin p = co;
    float value = 1.0f;
    float pwr = 1.0f;
    const float pwHL = pow(lacunarity, -H);

    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 0; i < int(octaves); i++)
    {
        value *= (pwr * perlin_signed(p) + 1.0f);
        pwr *= pwHL;
        p *= lacunarity;
    }

    const float rmd = octaves - floorf(octaves);
    if (rmd != 0.0f)
    {
        value *= (rmd * pwr * perlin_signed(p) + 1.0f); /* correct? */
    }

    return value;
}

DEVICE_FUNC float musgrave_hetero_terrain(const float4_nonbuiltin co,
                              const float H,
                              const float lacunarity,
                              const float octaves_unclamped,
                              const float offset)
{
    float4_nonbuiltin p = co;
    const float pwHL = pow(lacunarity, -H);
    float pwr = pwHL;

    /* first unscaled octave of function; later octaves are scaled */
    float value = offset + perlin_signed(p);
    p *= lacunarity;

    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 1; i < int(octaves); i++)
    {
        float increment = (perlin_signed(p) + offset) * pwr * value;
        value += increment;
        pwr *= pwHL;
        p *= lacunarity;
    }

    const float rmd = octaves - floorf(octaves);
    if (rmd != 0.0f)
    {
        float increment = (perlin_signed(p) + offset) * pwr * value;
        value += rmd * increment;
    }

    return value;
}

DEVICE_FUNC float musgrave_hybrid_multi_fractal(const float4_nonbuiltin co,
                                    const float H,
                                    const float lacunarity,
                                    const float octaves_unclamped,
                                    const float offset,
                                    const float gain)
{
    float4_nonbuiltin p = co;
    const float pwHL = pow(lacunarity, -H);

    float pwr = 1.0f;
    float value = 0.0f;
    float weight = 1.0f;

    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 0; (weight > 0.001f) && (i < int(octaves)); i++)
    {
        if (weight > 1.0f)
        {
            weight = 1.0f;
        }

        float signal = (perlin_signed(p) + offset) * pwr;
        pwr *= pwHL;
        value += weight * signal;
        weight *= gain * signal;
        p *= lacunarity;
    }

    const float rmd = octaves - floorf(octaves);
    if ((rmd != 0.0f) && (weight > 0.001f))
    {
        if (weight > 1.0f)
        {
            weight = 1.0f;
        }
        float signal = (perlin_signed(p) + offset) * pwr;
        value += rmd * weight * signal;
    }

    return value;
}

DEVICE_FUNC float musgrave_ridged_multi_fractal(const float4_nonbuiltin co,
                                    const float H,
                                    const float lacunarity,
                                    const float octaves_unclamped,
                                    const float offset,
                                    const float gain)
{
    float4_nonbuiltin p = co;
    const float pwHL = pow(lacunarity, -H);
    float pwr = pwHL;

    float signal = offset - abs(perlin_signed(p));
    signal *= signal;
    float value = signal;
    float weight = 1.0f;

    const float octaves = CLAMPIS(octaves_unclamped, 0.0f, 15.0f);

    for (int i = 1; i < int(octaves); i++)
    {
        p *= lacunarity;
        weight = CLAMPIS(signal * gain, 0.0f, 1.0f);
        signal = offset - abs(perlin_signed(p));
        signal *= signal;
        signal *= weight;
        value += signal * pwr;
        pwr *= pwHL;
    }

    return value;
}