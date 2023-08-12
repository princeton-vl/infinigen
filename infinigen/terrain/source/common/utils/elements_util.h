// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


DEVICE_FUNC float lerp(float s, float e, float t) { return s + (e - s) * t; }

DEVICE_FUNC float lerp_ed(float s, float e, float sd, float ed, float t) {
    float a = ed + sd - 2 * e + 2 * s, b = 3 * e - 3 * s - 2 * sd - ed;
    return s + t * (sd + t * (b + a * t));
}

DEVICE_FUNC float blerp(float c00, float c10, float c01, float c11, float tx, float ty) {
    return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}

DEVICE_FUNC float tlerp(float c000, float c100, float c010, float c110, float c001, float c101, float c011, float c111, float tx, float ty, float tz) {
    return lerp(lerp(lerp(c000, c100, tx), lerp(c010, c110, tx), ty), lerp(lerp(c001, c101, tx), lerp(c011, c111, tx), ty), tz);
}

DEVICE_FUNC float blerp(
    /*DEVICE*/ float *c,
    float x,
    float y,
    int N,
    float defa = 0) {
    if ((x < 0) | (x >= N - 1) | (y < 0) | (y >= N - 1))
        return defa;
    int x0 = int(x), y0 = int(y), x1 = x0 + 1, y1 = y0 + 1;
    float c00 = c[x0 * N + y0], c10 = c[x1 * N + y0], c01 = c[x0 * N + y1], c11 = c[x1 * N + y1];
    return blerp(c00, c10, c01, c11, x - x0, y - y0);
}

DEVICE_FUNC float cblerp(
    /*DEVICE*/ float *c,
    float x,
    float y,
    int N,
    float defa = 0) {
    if ((x < 0) | (x >= N - 1) | (y < 0) | (y >= N - 1))
        return defa;
    int x0 = int(x), y0 = int(y), x1 = x0 + 1, y1 = y0 + 1;
    float c00 = c[x0 * N + y0], c10 = c[x1 * N + y0], c01 = c[x0 * N + y1], c11 = c[x1 * N + y1];
    float dx00, dx01, dx10, dx11, dy00, dy01, dy10, dy11;
    if (x0 == 0) {
        dx00 = c10 - c00;
        dx01 = c11 - c01;
    } else {
        dx00 = (c10 - c[(x0 - 1) * N + y0]) / 2;
        dx01 = (c11 - c[(x0 - 1) * N + y1]) / 2;
    }
    if (x1 == N - 1) {
        dx10 = c10 - c00;
        dx11 = c11 - c01;
    } else {
        dx10 = (c[(x1 + 1) * N + y0] - c00) / 2;
        dx11 = (c[(x1 + 1) * N + y1] - c01) / 2;
    }
    if (y0 == 0) {
        dy00 = c01 - c00;
        dy10 = c11 - c10;
    } else {
        dy00 = (c01 - c[x0 * N + (y0 - 1)]) / 2;
        dy10 = (c11 - c[x1 * N + (y0 - 1)]) / 2;
    }
    if (y1 == N - 1) {
        dy01 = c01 - c00;
        dy11 = c11 - c10;
    } else {
        dy01 = (c[x0 * N + (y1 + 1)] - c00) / 2;
        dy11 = (c[x1 * N + (y1 + 1)] - c10) / 2;
    }
    float c_0 = lerp_ed(c00, c10, dx00, dx10, x - x0), c_1 = lerp_ed(c01, c11, dx01, dx11, x - x0);
    float dy_0 = lerp(dy00, dy10, x - x0), dy_1 = lerp(dy01, dy11, x - x0);
    return lerp_ed(c_0, c_1, dy_0, dy_1, y - y0);
}

DEVICE_FUNC float tlerp(
    /*DEVICE*/ float *c,
    float x,
    float y,
    float z,
    int N) {
    if ((x < 0) | (x >= N - 1) | (y < 0) | (y >= N - 1) | (z < 0) | (z >= N - 1))
        return 1e5;
    int x0 = int(x), y0 = int(y), z0 = int(z), x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
    float c000 = c[x0 * N * N + y0 * N + z0], c100 = c[x1 * N * N + y0 * N + z0], c010 = c[x0 * N * N + y1 * N + z0], c110 = c[x1 * N * N + y1 * N + z0];
    float c001 = c[x0 * N * N + y0 * N + z1], c101 = c[x1 * N * N + y0 * N + z1], c011 = c[x0 * N * N + y1 * N + z1], c111 = c[x1 * N * N + y1 * N + z1];
    return tlerp(c000, c100, c010, c110, c001, c101, c011, c111, x - x0, y - y0, z - z0);
}

DEVICE_FUNC float ctlerp(
    /*DEVICE*/ float *c,
    float x,
    float y,
    float z,
    int N) {
    if ((x < 0) | (x >= N - 1) | (y < 0) | (y >= N - 1) | (z < 0) | (z >= N - 1))
        return 1e5;
    int x0 = int(x), y0 = int(y), z0 = int(z), x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
    float c000 = c[x0 * N * N + y0 * N + z0], c100 = c[x1 * N * N + y0 * N + z0], c010 = c[x0 * N * N + y1 * N + z0], c110 = c[x1 * N * N + y1 * N + z0];
    float c001 = c[x0 * N * N + y0 * N + z1], c101 = c[x1 * N * N + y0 * N + z1], c011 = c[x0 * N * N + y1 * N + z1], c111 = c[x1 * N * N + y1 * N + z1];
    float c0__ = cblerp(&c[x0 * N * N], y, z, N), c1__ = cblerp(&c[x1 * N * N], y, z, N), dx000, dx001, dx010, dx011, dx100, dx101, dx110, dx111;
    if (x0 == 0) {
        dx000 = c100 - c000;
        dx001 = c101 - c001;
        dx010 = c110 - c010;
        dx011 = c111 - c011;
    } else {
        dx000 = (c100 - c[(x0 - 1) * N * N + y0 * N + z0]) / 2;
        dx001 = (c101 - c[(x0 - 1) * N * N + y0 * N + z1]) / 2;
        dx010 = (c110 - c[(x0 - 1) * N * N + y1 * N + z0]) / 2;
        dx011 = (c111 - c[(x0 - 1) * N * N + y1 * N + z1]) / 2;
    }
    if (x1 == N - 1) {
        dx100 = c100 - c000;
        dx101 = c101 - c001;
        dx110 = c110 - c010;
        dx111 = c111 - c011;
    } else {
        dx100 = (c[(x1 + 1) * N * N + y0 * N + z0] - c000) / 2;
        dx101 = (c[(x1 + 1) * N * N + y0 * N + z1] - c001) / 2;
        dx110 = (c[(x1 + 1) * N * N + y1 * N + z0] - c010) / 2;
        dx111 = (c[(x1 + 1) * N * N + y1 * N + z1] - c011) / 2;
    }
    float dx0__ = blerp(dx000, dx010, dx001, dx011, y - y0, z - z0), dx1__ = blerp(dx100, dx110, dx101, dx111, y - y0, z - z0);

    return lerp_ed(c0__, c1__, dx0__, dx1__, x - x0);
}

DEVICE_FUNC float ramp(float x, float minin, float maxin) {
    return fminf(1, fmaxf(0.0f, (x - minin) / (maxin - minin)));
}

DEVICE_FUNC float vertical_ramp(float x, float L) {
    return fminf(1e5, tan(fminf(.99, x / L) * acos(-1.) / 2) * 2 * L / acos(-1.));
}

DEVICE_FUNC int mod(int x, int y) {
    return ((x % y) + y) % y;
}

DEVICE_FUNC float sqr(float x) {
    return x * x;
}

DEVICE_FUNC float sigmoid(float x) {
    return 1 / (1 + exp(-x)) - 0.5;
}

DEVICE_FUNC float decaying_distance_weight(float x, float d1, float d2, float alpha) {
    if (x > d2)
        return 1e-5;
    else if (x > d1)
        return (x - d2) * (x - d2) + 1e-5;
    float k = 2 * (d2 - d1) / alpha / pow(d1, -alpha - 1);
    float C = (d1 - d2) * (d1 - d2) - k * pow(d1, -alpha);
    return k * pow(x, -alpha) + C + 1e-5;
}

DEVICE_FUNC float limit_and_float(int x) {
    return x % 10001 * 1.0;
}

DEVICE_FUNC float multiple(float x, float m) {
    return floor(x / m) * m;
}

DEVICE_FUNC float log_uniform(float low, float high, int seed) {
    float res = exp(log(low) + (log(high) - log(low)) * hash_to_float(seed));
    return res;
}