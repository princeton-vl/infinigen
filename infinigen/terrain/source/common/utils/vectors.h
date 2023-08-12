// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


struct float2_nonbuiltin;
struct float3_nonbuiltin;
struct float4_nonbuiltin;

struct float2_nonbuiltin {
    float x, y;
    DEVICE_FUNC float2_nonbuiltin()
        : x(0), y(0) {
    }
    DEVICE_FUNC float2_nonbuiltin(float x, float y)
        : x(x), y(y) {
    }
    DEVICE_FUNC float2_nonbuiltin(float3_nonbuiltin a);
    DEVICE_FUNC float2_nonbuiltin operator*(float s) const {
        float2_nonbuiltin res{x * s, y * s};
        return res;
    }
    DEVICE_FUNC float2_nonbuiltin operator/(float s) const {
        float2_nonbuiltin res{x / s, y / s};
        return res;
    }
    DEVICE_FUNC float2_nonbuiltin operator+(const float2_nonbuiltin &a) const {
        float2_nonbuiltin res{x + a.x, y + a.y};
        return res;
    }
    DEVICE_FUNC float2_nonbuiltin operator-(const float2_nonbuiltin &a) const {
        float2_nonbuiltin res{x - a.x, y - a.y};
        return res;
    }
    DEVICE_FUNC float2_nonbuiltin operator-(float s) const {
        float2_nonbuiltin res{x - s, y - s};
        return res;
    }
    DEVICE_FUNC float2_nonbuiltin &operator+=(const float2_nonbuiltin &a) {
        x += a.x;
        y += a.y;
        return *this;
    }
    DEVICE_FUNC float2_nonbuiltin &operator*=(float s) {
        x *= s;
        y *= s;
        return *this;
    }
};

DEVICE_FUNC float2_nonbuiltin operator*(float s, const float2_nonbuiltin &b) {
    float2_nonbuiltin res{b.x * s, b.y * s};
    return res;
}

struct float3_nonbuiltin {
    float x, y, z;
    DEVICE_FUNC float3_nonbuiltin()
        : x(0), y(0), z(0) {
    }
    DEVICE_FUNC float3_nonbuiltin(float s)
        : x(s), y(s), z(s) {
    }
    DEVICE_FUNC float3_nonbuiltin(float x, float y, float z)
        : x(x), y(y), z(z) {
    }
    DEVICE_FUNC float3_nonbuiltin(float4_nonbuiltin a);
    DEVICE_FUNC operator float() {
        return (x + y + z) / 3;
    }
    DEVICE_FUNC float3_nonbuiltin operator*(float s) const {
        float3_nonbuiltin res{x * s, y * s, z * s};
        return res;
    }
    DEVICE_FUNC float3_nonbuiltin operator/(float s) const {
        float3_nonbuiltin res{x / s, y / s, z / s};
        return res;
    }
    DEVICE_FUNC float3_nonbuiltin operator+(const float3_nonbuiltin &a) const {
        float3_nonbuiltin res{x + a.x, y + a.y, z + a.z};
        return res;
    }
    DEVICE_FUNC float3_nonbuiltin operator*(const float3_nonbuiltin &a) const {
        float3_nonbuiltin res{x * a.x, y * a.y, z * a.z};
        return res;
    }
    DEVICE_FUNC float3_nonbuiltin cross(const float3_nonbuiltin &a) const {
        float3_nonbuiltin res{
            y * a.z - z * a.y,
            z * a.x - x * a.z,
            x * a.y - y * a.x};
        return res;
    }
    DEVICE_FUNC float3_nonbuiltin operator-(const float3_nonbuiltin &a) const {
        float3_nonbuiltin res{x - a.x, y - a.y, z - a.z};
        return res;
    }
    DEVICE_FUNC float3_nonbuiltin operator+(float s) const {
        float3_nonbuiltin res{x + s, y + s, z + s};
        return res;
    }
    DEVICE_FUNC float3_nonbuiltin operator-(float s) const {
        float3_nonbuiltin res{x - s, y - s, z - s};
        return res;
    }
    DEVICE_FUNC float3_nonbuiltin &operator+=(const float3_nonbuiltin &a) {
        x += a.x;
        y += a.y;
        z += a.z;
        return *this;
    }
    DEVICE_FUNC float3_nonbuiltin &operator*=(float s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }
};

DEVICE_FUNC float3_nonbuiltin operator*(float s, const float3_nonbuiltin &b) {
    float3_nonbuiltin res{b.x * s, b.y * s, b.z * s};
    return res;
}

struct float4_nonbuiltin {
    float x, y, z, w;
    DEVICE_FUNC operator float() {
        return 0.2126f * x + 0.7152f * y + 0.0722f * z;
    }
    DEVICE_FUNC float4_nonbuiltin()
        : x(0), y(0), z(0), w(0) {
    }
    DEVICE_FUNC float4_nonbuiltin(float s)
        : x(s), y(s), z(s), w(0) {
    }
    DEVICE_FUNC float4_nonbuiltin(float3_nonbuiltin a)
        : x(a.x), y(a.y), z(a.z), w(0) {
    }
    DEVICE_FUNC float4_nonbuiltin(float x, float y, float z, float w)
        : x(x), y(y), z(z), w(w) {
    }
    DEVICE_FUNC float4_nonbuiltin operator*(float s) const {
        float4_nonbuiltin res{x * s, y * s, z * s, w * s};
        return res;
    }
    DEVICE_FUNC float4_nonbuiltin operator/(float s) const {
        float4_nonbuiltin res{x / s, y / s, z / s, w / s};
        return res;
    }
    DEVICE_FUNC float4_nonbuiltin operator+(const float4_nonbuiltin &a) const {
        float4_nonbuiltin res{x + a.x, y + a.y, z + a.z, w + a.w};
        return res;
    }
    DEVICE_FUNC float4_nonbuiltin operator-(const float4_nonbuiltin &a) const {
        float4_nonbuiltin res{x - a.x, y - a.y, z - a.z, w - a.w};
        return res;
    }
    DEVICE_FUNC float4_nonbuiltin operator-(float s) const {
        float4_nonbuiltin res{x - s, y - s, z - s, w - s};
        return res;
    }
    DEVICE_FUNC float4_nonbuiltin &operator=(const float3_nonbuiltin &a) {
        x = a.x;
        y = a.y;
        z = a.z;
        return *this;
    }
    DEVICE_FUNC float4_nonbuiltin &operator+=(const float4_nonbuiltin &a) {
        x += a.x;
        y += a.y;
        z += a.z;
        w += a.w;
        return *this;
    }
    DEVICE_FUNC float4_nonbuiltin &operator*=(float s) {
        x *= s;
        y *= s;
        z *= s;
        w *= s;
        return *this;
    }
};

DEVICE_FUNC float3_nonbuiltin::float3_nonbuiltin(float4_nonbuiltin a)
    : x(a.x), y(a.y), z(a.z) {
}
DEVICE_FUNC float2_nonbuiltin::float2_nonbuiltin(float3_nonbuiltin a)
    : x(a.x), y(a.y) {
}

DEVICE_FUNC float4_nonbuiltin operator*(float s, const float4_nonbuiltin &b) {
    float4_nonbuiltin res{b.x * s, b.y * s, b.z * s, b.w * s};
    return res;
}