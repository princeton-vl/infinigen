/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2001-2002 NaN Holding BV. All rights reserved.
 * adapted by Zeyu Ma on date June 4, 2023 to compile geometry nodes */

#ifndef __VALTORGB__
#define __VALTORGB__

/**************** do the key ****************/

/* KeyBlock->type */
enum {
    KEY_LINEAR = 0,
    KEY_CARDINAL = 1,
    KEY_BSPLINE = 2,
    KEY_CATMULL_ROM = 3,
};

DEVICE_FUNC void key_curve_position_weights(float t, float data[4], int type) {
    float t2, t3, fc;

    if (type == KEY_LINEAR) {
        data[0] = 0.0f;
        data[1] = -t + 1.0f;
        data[2] = t;
        data[3] = 0.0f;
    } else if (type == KEY_CARDINAL) {
        t2 = t * t;
        t3 = t2 * t;
        fc = 0.71f;

        data[0] = -fc * t3 + 2.0f * fc * t2 - fc * t;
        data[1] = (2.0f - fc) * t3 + (fc - 3.0f) * t2 + 1.0f;
        data[2] = (fc - 2.0f) * t3 + (3.0f - 2.0f * fc) * t2 + fc * t;
        data[3] = fc * t3 - fc * t2;
    } else if (type == KEY_BSPLINE) {
        t2 = t * t;
        t3 = t2 * t;

        data[0] = -0.16666666f * t3 + 0.5f * t2 - 0.5f * t + 0.16666666f;
        data[1] = 0.5f * t3 - t2 + 0.66666666f;
        data[2] = -0.5f * t3 + 0.5f * t2 + 0.5f * t + 0.16666666f;
        data[3] = 0.16666666f * t3;
    } else if (type == KEY_CATMULL_ROM) {
        t2 = t * t;
        t3 = t2 * t;
        fc = 0.5f;

        data[0] = -fc * t3 + 2.0f * fc * t2 - fc * t;
        data[1] = (2.0f - fc) * t3 + (fc - 3.0f) * t2 + 1.0f;
        data[2] = (fc - 2.0f) * t3 + (3.0f - 2.0f * fc) * t2 + fc * t;
        data[3] = fc * t3 - fc * t2;
    }
}

/* ------------------------------------------------------------------------- */

DEVICE_FUNC float colorband_hue_interp(
    const int ipotype_hue, const float mfac, const float fac, float h1, float h2) {
    float h_interp;
    int mode = 0;

#define HUE_INTERP(h_a, h_b) ((mfac * (h_a)) + (fac * (h_b)))
#define HUE_MOD(h) (((h) < 1.0f) ? (h) : (h)-1.0f)

    h1 = HUE_MOD(h1);
    h2 = HUE_MOD(h2);

    assert(h1 >= 0.0f && h1 < 1.0f);
    assert(h2 >= 0.0f && h2 < 1.0f);

    switch (ipotype_hue) {
        case COLBAND_HUE_NEAR: {
            if ((h1 < h2) && (h2 - h1) > +0.5f) {
                mode = 1;
            } else if ((h1 > h2) && (h2 - h1) < -0.5f) {
                mode = 2;
            } else {
                mode = 0;
            }
            break;
        }
        case COLBAND_HUE_FAR: {
            /* Do full loop in Hue space in case both stops are the same... */
            if (h1 == h2) {
                mode = 1;
            } else if ((h1 < h2) && (h2 - h1) < +0.5f) {
                mode = 1;
            } else if ((h1 > h2) && (h2 - h1) > -0.5f) {
                mode = 2;
            } else {
                mode = 0;
            }
            break;
        }
        case COLBAND_HUE_CCW: {
            if (h1 > h2) {
                mode = 2;
            } else {
                mode = 0;
            }
            break;
        }
        case COLBAND_HUE_CW: {
            if (h1 < h2) {
                mode = 1;
            } else {
                mode = 0;
            }
            break;
        }
    }

    switch (mode) {
        case 0:
            h_interp = HUE_INTERP(h1, h2);
            break;
        case 1:
            h_interp = HUE_INTERP(h1 + 1.0f, h2);
            h_interp = HUE_MOD(h_interp);
            break;
        case 2:
            h_interp = HUE_INTERP(h1, h2 + 1.0f);
            h_interp = HUE_MOD(h_interp);
            break;
    }

    assert(h_interp >= 0.0f && h_interp < 1.0f);

#undef HUE_INTERP
#undef HUE_MOD

    return h_interp;
}

DEVICE_FUNC void node_texture_valToRgb(
    // params
    int tot,
    float *positions,
    float4_nonbuiltin *colors,
    int color_mode,
    int ipotype,
    int ipotype_hue,
    // input
    float in,
    // output
    float4_nonbuiltin *r_out,
    float *alpha) {
    if (r_out == NULL && alpha == NULL)
        return;
    float4_nonbuiltin out_content;
    float4_nonbuiltin *out = &out_content;
    float fac;
    int a;

    float4_nonbuiltin *cbd1 = colors, *cbd2, *cbd0, *cbd3;
    float *pbd1 = positions, *pbd2;

    /* NOTE: when ipotype >= COLBAND_INTERP_B_SPLINE,
     * we cannot do early-out with a constant color before first color stop and after last one,
     * because interpolation starts before and ends after those... */

    ipotype = (color_mode == COLBAND_BLEND_RGB) ? ipotype : COLBAND_INTERP_LINEAR;

    if (tot == 1) {
        *out = *cbd1;
    } else if ((in <= *pbd1) && (ipotype == COLBAND_INTERP_LINEAR || ipotype == COLBAND_INTERP_EASE || ipotype == COLBAND_INTERP_CONSTANT)) {
        /* We are before first color stop. */
        *out = *cbd1;
    } else {
        float4_nonbuiltin left, right;
        float left_pos, right_pos;

        /* we're looking for first pos > in */
        for (a = 0; a < tot; a++, cbd1++, pbd1++) {
            if (*pbd1 > in) {
                break;
            }
        }

        if (a == tot) {
            cbd2 = cbd1 - 1;
            pbd2 = pbd1 - 1;
            right = *cbd2;
            right_pos = 1.0f;
            cbd1 = &right;
            pbd1 = &right_pos;
        } else if (a == 0) {
            left = *cbd1;
            left_pos = 0.0f;
            cbd2 = &left;
            pbd2 = &left_pos;
        } else {
            cbd2 = cbd1 - 1;
            pbd2 = pbd1 - 1;
        }

        if ((a == tot) && (ipotype == COLBAND_INTERP_LINEAR || ipotype == COLBAND_INTERP_EASE || ipotype == COLBAND_INTERP_CONSTANT)) {
            /* We are after last color stop. */
            *out = *cbd2;
        } else if (ipotype == COLBAND_INTERP_CONSTANT) {
            /* constant */
            *out = *cbd2;
        } else {
            if (*pbd2 != *pbd1) {
                fac = (in - *pbd1) / (*pbd2 - *pbd1);
            } else {
                /* was setting to 0.0 in 2.56 & previous, but this
                 * is incorrect for the last element, see T26732. */
                fac = (a != tot) ? 0.0f : 1.0f;
            }

            if (ipotype == COLBAND_INTERP_B_SPLINE || ipotype == COLBAND_INTERP_CARDINAL) {
                /* Interpolate from right to left: `3 2 1 0`. */
                float t[4];

                if (a >= tot - 1) {
                    cbd0 = cbd1;
                } else {
                    cbd0 = cbd1 + 1;
                }
                if (a < 2) {
                    cbd3 = cbd2;
                } else {
                    cbd3 = cbd2 - 1;
                }

                CLAMP(fac, 0.0f, 1.0f);

                if (ipotype == COLBAND_INTERP_CARDINAL) {
                    key_curve_position_weights(fac, t, KEY_CARDINAL);
                } else {
                    key_curve_position_weights(fac, t, KEY_BSPLINE);
                }

                out->x = t[3] * cbd3->x + t[2] * cbd2->x + t[1] * cbd1->x + t[0] * cbd0->x;
                out->y = t[3] * cbd3->y + t[2] * cbd2->y + t[1] * cbd1->y + t[0] * cbd0->y;
                out->z = t[3] * cbd3->z + t[2] * cbd2->z + t[1] * cbd1->z + t[0] * cbd0->z;
                out->w = t[3] * cbd3->w + t[2] * cbd2->w + t[1] * cbd1->w + t[0] * cbd0->w;
                out->x = clamp_range(out->x, 0.0f, 1.0f);
                out->y = clamp_range(out->y, 0.0f, 1.0f);
                out->z = clamp_range(out->z, 0.0f, 1.0f);
                out->w = clamp_range(out->w, 0.0f, 1.0f);
            } else {
                if (ipotype == COLBAND_INTERP_EASE) {
                    const float fac2 = fac * fac;
                    fac = 3.0f * fac2 - 2.0f * fac2 * fac;
                }
                const float mfac = 1.0f - fac;

                if (color_mode == COLBAND_BLEND_HSV) {
                    float col1[3], col2[3];

                    rgb_to_hsv_v(&cbd1->x, col1);
                    rgb_to_hsv_v(&cbd2->x, col2);

                    out->x = colorband_hue_interp(ipotype_hue, mfac, fac, col1[0], col2[0]);
                    out->y = mfac * col1[1] + fac * col2[1];
                    out->z = mfac * col1[2] + fac * col2[2];
                    out->w = mfac * cbd1->w + fac * cbd2->w;

                    hsv_to_rgb_v(&out->x, &out->x);
                } else if (color_mode == COLBAND_BLEND_HSL) {
                    float col1[3], col2[3];

                    rgb_to_hsl_v(&cbd1->x, col1);
                    rgb_to_hsl_v(&cbd2->x, col2);

                    out->x = colorband_hue_interp(ipotype_hue, mfac, fac, col1[0], col2[0]);
                    out->y = mfac * col1[1] + fac * col2[1];
                    out->z = mfac * col1[2] + fac * col2[2];
                    out->w = mfac * cbd1->w + fac * cbd2->w;

                    hsl_to_rgb_v(&out->x, &out->x);
                } else {
                    /* COLBAND_BLEND_RGB */
                    out->x = mfac * cbd1->x + fac * cbd2->x;
                    out->y = mfac * cbd1->y + fac * cbd2->y;
                    out->z = mfac * cbd1->z + fac * cbd2->z;
                    out->w = mfac * cbd1->w + fac * cbd2->w;
                }
            }
        }
    }
    if (r_out != NULL)
        *r_out = *out;
    if (alpha != NULL)
        *alpha = out->w;
}

#endif