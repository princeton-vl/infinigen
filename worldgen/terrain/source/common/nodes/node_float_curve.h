/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation
 * adapted by Zeyu Ma on date June 4, 2023 to compile geometry nodes */

#ifndef __FLOATCURVE__
#define __FLOATCURVE__

DEVICE_FUNC void node_float_curve(
    float *ramp,
    int table_size,
    float Factor,
    float at,
    float *ValueOut
) {

    // TODO assume Factor = 1 now

    if (ValueOut == 0) return;
    const int interpolate = 1;
    const int extrapolate = 1;
    float f = at;

    if ((f < 0.0 || f > 1.0) && extrapolate)
    {
        float t0, dy;
        if (f < 0.0)
        {
            t0 = ramp[0];
            dy = t0 - ramp[1];
            f = -f;
        }
        else
        {
            t0 = ramp[table_size - 1];
            dy = t0 - ramp[table_size - 2];
            f = f - 1.0;
        }
        *ValueOut = t0 + dy * f * (table_size - 1);
    }

    f = clamp(at, 0.0, 1.0) * (table_size - 1);

    /* clamp int as well in case of NaN */
    int i = (int)f;
    if (i < 0)
        i = 0;
    if (i >= table_size)
        i = table_size - 1;
    float t = f - (float)i;

    float result = ramp[i];

    if (interpolate && t > 0.0)
        result = (1.0 - t) * result + t * ramp[i + 1];

    *ValueOut = result;
}

#endif