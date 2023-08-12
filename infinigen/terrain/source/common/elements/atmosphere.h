// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


DEVICE_FUNC void atmosphere(
    float3_nonbuiltin position,
    float *sdf,
    int has_water,
    int *i_params, float *f_params,
    int *waterbody_i_params, float *waterbody_f_params
) {
    /* params
    int:

    float:
    height, spherical_radius, hacky_offset
    */

    float height = f_params[0];
    float spherical_radius = f_params[1];
    float hacky_offset = f_params[2];

    float altitude;
    if (spherical_radius > 0) {
        altitude = sqrt(position.x * position.x + position.y * position.y + position.z * position.z) - spherical_radius;
    }
    else {
        altitude = position.z;
    }
    if (has_water == 0) {
        *sdf = altitude - height;
    }
    else {
        waterbody(position, sdf, NULL, 0, 0, waterbody_i_params, waterbody_f_params, NULL, NULL, NULL, NULL);
        *sdf = fmaxf(altitude - height, -*sdf + hacky_offset);
    }
}