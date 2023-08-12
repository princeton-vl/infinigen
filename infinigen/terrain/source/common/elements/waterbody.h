// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


DEVICE_FUNC void waterbody(
    float3_nonbuiltin position,
    float *sdf,
    float *boundary_sdf,
    int mark_boundary, int is_caved,
    int *i_params, float *f_params,
    int *landtiles_i_params, float *landtiles_f_params,
    int *caves_i_params, float *caves_f_params
) {
    /* params

    float:
    height, spherical_radius
    */
    float height = f_params[0];
    float spherical_radius = f_params[1];
    float altitude;
    if (spherical_radius > 0) {
        altitude = sqrt(position.x * position.x + position.y * position.y + position.z * position.z) - spherical_radius;
    }
    else {
        altitude = position.z;
    }
    *sdf = altitude - height;
    if (boundary_sdf != NULL) {
        if (!mark_boundary) *boundary_sdf = 0;
        else {
            landtiles(
                position, boundary_sdf, 0, is_caved,
                landtiles_i_params, landtiles_f_params,
                caves_i_params, caves_f_params
            );
        }
    }
}