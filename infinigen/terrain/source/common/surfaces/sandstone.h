// Code generated using version 1.0.0 of infinigen/tools/kernelize_surfaces.py;
// refer to infinigen/assets/materials/sandstone.py which has the copyright and
// authors
DEVICE_FUNC void
nodegroup_roughness(float Input_0, float Input_1, float Input_2, float Input_3,
                    float3_nonbuiltin Input_4,
                    float Value_FROM_nodegroup_roughness,
                    float noise_texture_1_w_FROM_nodegroup_roughness,
                    float noise_texture_2_w_FROM_nodegroup_roughness,
                    float3_nonbuiltin position,
                    POINTER_OR_REFERENCE_ARG float3_nonbuiltin *Output_5) {
    float noise_texture_1_w__Value;

    noise_texture_1_w__Value = noise_texture_1_w_FROM_nodegroup_roughness;
    float4_nonbuiltin Noise_SPACE_Texture__Color;
    node_shader_tex_noise(4, position, noise_texture_1_w__Value, Input_0, 2.0,
                          0.5, 0.0, NULL, &Noise_SPACE_Texture__Color);
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Noise_SPACE_Texture__Color),
                            Input_4, float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, Vector_SPACE_Math__Vector,
        float3_nonbuiltin(Input_2), float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
        &Vector_SPACE_Math_DOT_001__Vector, NULL);
    float noise_texture_2_w__Value;

    noise_texture_2_w__Value = noise_texture_2_w_FROM_nodegroup_roughness;
    float4_nonbuiltin Noise_SPACE_Texture_DOT_001__Color;
    node_shader_tex_noise(4, position, noise_texture_2_w__Value, Input_1, 0.0,
                          0.5, 0.0, NULL, &Noise_SPACE_Texture_DOT_001__Color);
    float3_nonbuiltin Vector_SPACE_Math_DOT_002__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY,
        float3_nonbuiltin(Noise_SPACE_Texture_DOT_001__Color), Input_4,
        float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
        &Vector_SPACE_Math_DOT_002__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_003__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, Vector_SPACE_Math_DOT_002__Vector,
        float3_nonbuiltin(Input_3), float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
        &Vector_SPACE_Math_DOT_003__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_004__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_001__Vector,
        Vector_SPACE_Math_DOT_003__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_004__Vector, NULL);
    float Value__Value;

    Value__Value = Value_FROM_nodegroup_roughness;
    float3_nonbuiltin Vector_SPACE_Math_DOT_005__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, Vector_SPACE_Math_DOT_004__Vector,
        float3_nonbuiltin(Value__Value), float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
        &Vector_SPACE_Math_DOT_005__Vector, NULL);

    if (Output_5 != NULL)
        *Output_5 = Vector_SPACE_Math_DOT_005__Vector;
}
DEVICE_FUNC void nodegroup_add_noise(
    float Input_0, float Input_1, float Input_2, float Input_3, float Input_4,
    float Input_5, float noise_texture_1_w_FROM_nodegroup_add_noise,
    float3_nonbuiltin position, POINTER_OR_REFERENCE_ARG float *Output_6) {
    float noise_texture_1_w__Value;

    noise_texture_1_w__Value = noise_texture_1_w_FROM_nodegroup_add_noise;
    float Noise_SPACE_Texture__Fac;
    node_shader_tex_noise(4, position, noise_texture_1_w__Value, Input_2,
                          Input_3, Input_4, Input_5, &Noise_SPACE_Texture__Fac,
                          NULL);
    float Math__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Noise_SPACE_Texture__Fac, Input_1,
                      0.5, &Math__Value);
    float Math_DOT_001__Value;
    node_texture_math(NODE_MATH_ADD, 0, Input_0, Math__Value, 0.5,
                      &Math_DOT_001__Value);

    if (Output_6 != NULL)
        *Output_6 = Math_DOT_001__Value;
}
DEVICE_FUNC void nodegroup_add_noise_DOT_001(
    float Input_0, float Input_1, float Input_2, float Input_3, float Input_4,
    float Input_5, float noise_texture_1_w_FROM_nodegroup_add_noise_DOT_001,
    float3_nonbuiltin position, POINTER_OR_REFERENCE_ARG float *Output_6) {
    float noise_texture_1_w__Value;

    noise_texture_1_w__Value =
        noise_texture_1_w_FROM_nodegroup_add_noise_DOT_001;
    float Noise_SPACE_Texture__Fac;
    node_shader_tex_noise(4, position, noise_texture_1_w__Value, Input_2,
                          Input_3, Input_4, Input_5, &Noise_SPACE_Texture__Fac,
                          NULL);
    float Math__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Noise_SPACE_Texture__Fac, Input_1,
                      0.5, &Math__Value);
    float Math_DOT_001__Value;
    node_texture_math(NODE_MATH_ADD, 0, Input_0, Math__Value, 0.5,
                      &Math_DOT_001__Value);

    if (Output_6 != NULL)
        *Output_6 = Math_DOT_001__Value;
}
DEVICE_FUNC void
nodegroup_polynomial(float Input_0, float Input_1, float Input_2, float Input_3,
                     float Input_4, float Input_5, float Input_6, float Input_7,
                     float Input_8, POINTER_OR_REFERENCE_ARG float *Output_9) {
    float Math__Value;
    node_texture_math(NODE_MATH_POWER, 0, Input_0, Input_6, 0.5, &Math__Value);
    float Math_DOT_001__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Input_3, Math__Value, 0.5,
                      &Math_DOT_001__Value);
    float Math_DOT_002__Value;
    node_texture_math(NODE_MATH_POWER, 0, Input_1, Input_7, 0.5,
                      &Math_DOT_002__Value);
    float Math_DOT_003__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Input_4, Math_DOT_002__Value, 0.5,
                      &Math_DOT_003__Value);
    float Math_DOT_004__Value;
    node_texture_math(NODE_MATH_ADD, 0, Math_DOT_001__Value,
                      Math_DOT_003__Value, 0.5, &Math_DOT_004__Value);
    float Math_DOT_005__Value;
    node_texture_math(NODE_MATH_POWER, 0, Input_2, Input_8, 0.5,
                      &Math_DOT_005__Value);
    float Math_DOT_006__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Input_5, Math_DOT_005__Value, 0.5,
                      &Math_DOT_006__Value);
    float Math_DOT_007__Value;
    node_texture_math(NODE_MATH_ADD, 0, Math_DOT_004__Value,
                      Math_DOT_006__Value, 0.5, &Math_DOT_007__Value);

    if (Output_9 != NULL)
        *Output_9 = Math_DOT_007__Value;
}
DEVICE_FUNC void nodegroup_displacement_to_offset(
    float3_nonbuiltin Input_0, float Input_1, float3_nonbuiltin normal,
    POINTER_OR_REFERENCE_ARG float3_nonbuiltin *Output_2) {
    float Math__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, float(Input_0), Input_1, 0.5,
                      &Math__Value);
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Math__Value), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);

    if (Output_2 != NULL)
        *Output_2 = Vector_SPACE_Math__Vector;
}
DEVICE_FUNC void nodegroup_displacement_to_offset_DOT_001(
    float3_nonbuiltin Input_0, float Input_1, float3_nonbuiltin normal,
    POINTER_OR_REFERENCE_ARG float3_nonbuiltin *Output_2) {
    float Math__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, float(Input_0), Input_1, 0.5,
                      &Math__Value);
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Math__Value), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);

    if (Output_2 != NULL)
        *Output_2 = Vector_SPACE_Math__Vector;
}
DEVICE_FUNC void nodegroup_cracked_with_mask(
    float Input_0, float Input_1, float Input_2,
    float noise_texture_w_FROM_nodegroup_cracked_with_mask,
    float3_nonbuiltin normal, float3_nonbuiltin position,
    float voronoi_texture_w_FROM_nodegroup_cracked_with_mask,
    POINTER_OR_REFERENCE_ARG float3_nonbuiltin *Output_3) {
    float noise_texture_w__Value;

    noise_texture_w__Value = noise_texture_w_FROM_nodegroup_cracked_with_mask;
    float4_nonbuiltin Noise_SPACE_Texture__Color;
    node_shader_tex_noise(4, position, noise_texture_w__Value, Input_0, 2.0,
                          0.5, 1.0, NULL, &Noise_SPACE_Texture__Color);
    float voronoi_texture_w__Value;

    voronoi_texture_w__Value =
        voronoi_texture_w_FROM_nodegroup_cracked_with_mask;
    float Voronoi_SPACE_Texture__Distance;
    node_shader_tex_voronoi(
        4, SHD_VORONOI_DISTANCE_TO_EDGE, SHD_VORONOI_EUCLIDEAN,
        float3_nonbuiltin(Noise_SPACE_Texture__Color), voronoi_texture_w__Value,
        Input_2, 1.0, 0.5, 1.0, &Voronoi_SPACE_Texture__Distance, NULL, NULL,
        NULL, NULL);
    float4_nonbuiltin ColorRamp__Color;

    float ColorRamp_positions[2]{0.0, 0.05999999865889549};
    float4_nonbuiltin ColorRamp_colors[2]{
        float4_nonbuiltin(0.0, 0.0, 0.0, 1.0),
        float4_nonbuiltin(1.0, 1.0, 1.0, 1.0)};
    node_texture_valToRgb(2, ColorRamp_positions, ColorRamp_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR,
                          COLBAND_HUE_NEAR, Voronoi_SPACE_Texture__Distance,
                          &ColorRamp__Color, NULL);
    float Math__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, float(ColorRamp__Color), 1.0, 0.5,
                      &Math__Value);
    float3_nonbuiltin Group__Output_2;
    nodegroup_displacement_to_offset_DOT_001(

        float3_nonbuiltin(Math__Value), Input_1, normal, &Group__Output_2);

    if (Output_3 != NULL)
        *Output_3 = Group__Output_2;
}
DEVICE_FUNC void nodegroup_displacement_to_offset_DOT_002(
    float3_nonbuiltin Input_0, float Input_1, float3_nonbuiltin normal,
    POINTER_OR_REFERENCE_ARG float3_nonbuiltin *Output_2) {
    float Math__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, float(Input_0), Input_1, 0.5,
                      &Math__Value);
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Math__Value), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);

    if (Output_2 != NULL)
        *Output_2 = Vector_SPACE_Math__Vector;
}
DEVICE_FUNC void nodegroup_cracked_with_mask_DOT_001(
    float Input_0, float Input_1, float Input_2,
    float noise_texture_w_FROM_nodegroup_cracked_with_mask_DOT_001,
    float3_nonbuiltin normal, float3_nonbuiltin position,
    float voronoi_texture_w_FROM_nodegroup_cracked_with_mask_DOT_001,
    POINTER_OR_REFERENCE_ARG float3_nonbuiltin *Output_3) {
    float noise_texture_w__Value;

    noise_texture_w__Value =
        noise_texture_w_FROM_nodegroup_cracked_with_mask_DOT_001;
    float4_nonbuiltin Noise_SPACE_Texture__Color;
    node_shader_tex_noise(4, position, noise_texture_w__Value, Input_0, 2.0,
                          0.5, 1.0, NULL, &Noise_SPACE_Texture__Color);
    float voronoi_texture_w__Value;

    voronoi_texture_w__Value =
        voronoi_texture_w_FROM_nodegroup_cracked_with_mask_DOT_001;
    float Voronoi_SPACE_Texture__Distance;
    node_shader_tex_voronoi(
        4, SHD_VORONOI_DISTANCE_TO_EDGE, SHD_VORONOI_EUCLIDEAN,
        float3_nonbuiltin(Noise_SPACE_Texture__Color), voronoi_texture_w__Value,
        Input_2, 1.0, 0.5, 1.0, &Voronoi_SPACE_Texture__Distance, NULL, NULL,
        NULL, NULL);
    float4_nonbuiltin ColorRamp__Color;

    float ColorRamp_positions[2]{0.0, 0.05999999865889549};
    float4_nonbuiltin ColorRamp_colors[2]{
        float4_nonbuiltin(0.0, 0.0, 0.0, 1.0),
        float4_nonbuiltin(1.0, 1.0, 1.0, 1.0)};
    node_texture_valToRgb(2, ColorRamp_positions, ColorRamp_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR,
                          COLBAND_HUE_NEAR, Voronoi_SPACE_Texture__Distance,
                          &ColorRamp__Color, NULL);
    float Math__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, float(ColorRamp__Color), 1.0, 0.5,
                      &Math__Value);
    float3_nonbuiltin Group__Output_2;
    nodegroup_displacement_to_offset_DOT_002(

        float3_nonbuiltin(Math__Value), Input_1, normal, &Group__Output_2);

    if (Output_3 != NULL)
        *Output_3 = Group__Output_2;
}
DEVICE_FUNC void geometry_sandstone(
    float3_nonbuiltin position, float3_nonbuiltin normal,
    POINTER_OR_REFERENCE_ARG float *float_vars,
    POINTER_OR_REFERENCE_ARG float3_nonbuiltin *float3_nonbuiltin_vars,
    POINTER_OR_REFERENCE_ARG float3_nonbuiltin *offset) {
    float Value_FROM_nodegroup_roughness = float_vars[0];
    float crack_magnitude_1_FROM_geometry_sandstone = float_vars[1];
    float crack_magnitude_2_FROM_geometry_sandstone = float_vars[2];
    float detail0_FROM_geometry_sandstone = float_vars[3];
    float detail1_FROM_geometry_sandstone = float_vars[4];
    float detail2_FROM_geometry_sandstone = float_vars[5];
    float noise_texture_1_w_FROM_geometry_sandstone = float_vars[6];
    float noise_texture_1_w_FROM_nodegroup_add_noise = float_vars[7];
    float noise_texture_1_w_FROM_nodegroup_add_noise_DOT_001 = float_vars[8];
    float noise_texture_1_w_FROM_nodegroup_roughness = float_vars[9];
    float noise_texture_2_w_FROM_geometry_sandstone = float_vars[10];
    float noise_texture_2_w_FROM_nodegroup_roughness = float_vars[11];
    float noise_texture_w_FROM_geometry_sandstone = float_vars[12];
    float noise_texture_w_FROM_nodegroup_cracked_with_mask = float_vars[13];
    float noise_texture_w_FROM_nodegroup_cracked_with_mask_DOT_001 =
        float_vars[14];
    float3_nonbuiltin position_shift0_FROM_geometry_sandstone =
        float3_nonbuiltin_vars[0];
    float3_nonbuiltin position_shift1_FROM_geometry_sandstone =
        float3_nonbuiltin_vars[1];
    float3_nonbuiltin position_shift2_FROM_geometry_sandstone =
        float3_nonbuiltin_vars[2];
    float roug_mag_FROM_geometry_sandstone = float_vars[15];
    float roughness0_FROM_geometry_sandstone = float_vars[16];
    float roughness1_FROM_geometry_sandstone = float_vars[17];
    float roughness2_FROM_geometry_sandstone = float_vars[18];
    float scale0_FROM_geometry_sandstone = float_vars[19];
    float scale1_FROM_geometry_sandstone = float_vars[20];
    float scale2_FROM_geometry_sandstone = float_vars[21];
    float side_step_displacement_to_offset_magnitude_FROM_geometry_sandstone =
        float_vars[22];
    float side_step_poly_aplha_x_FROM_geometry_sandstone = float_vars[23];
    float side_step_poly_aplha_y_FROM_geometry_sandstone = float_vars[24];
    float stripe_mag_FROM_geometry_sandstone = float_vars[25];
    float stripe_scale_FROM_geometry_sandstone = float_vars[26];
    float stripe_warp_mag_FROM_geometry_sandstone = float_vars[27];
    float stripe_warp_scale_FROM_geometry_sandstone = float_vars[28];
    float voronoi_texture_w_FROM_nodegroup_cracked_with_mask = float_vars[29];
    float voronoi_texture_w_FROM_nodegroup_cracked_with_mask_DOT_001 =
        float_vars[30];
    float zscale0_FROM_geometry_sandstone = float_vars[31];
    float zscale1_FROM_geometry_sandstone = float_vars[32];
    float zscale2_FROM_geometry_sandstone = float_vars[33];
    float roug_mag__Value;

    roug_mag__Value = roug_mag_FROM_geometry_sandstone;
    float side_step_displacement_to_offset_magnitude__Value;

    side_step_displacement_to_offset_magnitude__Value =
        side_step_displacement_to_offset_magnitude_FROM_geometry_sandstone;
    float side_step_poly_aplha_x__Value;

    side_step_poly_aplha_x__Value =
        side_step_poly_aplha_x_FROM_geometry_sandstone;
    float side_step_poly_aplha_y__Value;

    side_step_poly_aplha_y__Value =
        side_step_poly_aplha_y_FROM_geometry_sandstone;
    float crack_magnitude_1__Value;

    crack_magnitude_1__Value = crack_magnitude_1_FROM_geometry_sandstone;
    float crack_magnitude_2__Value;

    crack_magnitude_2__Value = crack_magnitude_2_FROM_geometry_sandstone;
    float3_nonbuiltin Group__Output_5;
    nodegroup_roughness(

        200.0, 8.0, 0.5, 0.15000000596046448, normal,
        Value_FROM_nodegroup_roughness,
        noise_texture_1_w_FROM_nodegroup_roughness,
        noise_texture_2_w_FROM_nodegroup_roughness, position, &Group__Output_5);
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY, Group__Output_5,
                            float3_nonbuiltin(roug_mag__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);
    float Separate_SPACE_XYZ__X;
    float Separate_SPACE_XYZ__Y;
    node_shader_sep_xyz(

        position, &Separate_SPACE_XYZ__X, &Separate_SPACE_XYZ__Y, NULL);
    float Group_DOT_001__Output_6;
    nodegroup_add_noise(

        Separate_SPACE_XYZ__X, 0.5, 2.0, 2.0, 0.5, 0.0,
        noise_texture_1_w_FROM_nodegroup_add_noise, position,
        &Group_DOT_001__Output_6);
    float Group_DOT_002__Output_6;
    nodegroup_add_noise_DOT_001(

        Separate_SPACE_XYZ__Y, 0.5, 2.0, 2.0, 0.5, 0.0,
        noise_texture_1_w_FROM_nodegroup_add_noise_DOT_001, position,
        &Group_DOT_002__Output_6);
    float Group_DOT_003__Output_9;
    nodegroup_polynomial(

        Group_DOT_001__Output_6, Group_DOT_002__Output_6, 0.5,
        side_step_poly_aplha_x__Value, side_step_poly_aplha_y__Value, 0.0, 1.0,
        1.0, 1.0, &Group_DOT_003__Output_9);
    float noise_texture_w__Value;

    noise_texture_w__Value = noise_texture_w_FROM_geometry_sandstone;
    float Noise_SPACE_Texture__Fac;
    node_shader_tex_noise(4, float3_nonbuiltin(Group_DOT_003__Output_9),
                          noise_texture_w__Value, 10.0, 2.0, 0.5, 0.0,
                          &Noise_SPACE_Texture__Fac, NULL);
    float noise_texture_2_w__Value;

    noise_texture_2_w__Value = noise_texture_2_w_FROM_geometry_sandstone;
    float Noise_SPACE_Texture_DOT_001__Fac;
    node_shader_tex_noise(4, position, noise_texture_2_w__Value, 2.0, 2.0, 0.5,
                          0.0, &Noise_SPACE_Texture_DOT_001__Fac, NULL);
    float4_nonbuiltin ColorRamp__Color;

    float ColorRamp_positions[2]{0.4000000059604645, 0.6000000238418579};
    float4_nonbuiltin ColorRamp_colors[2]{
        float4_nonbuiltin(0.0, 0.0, 0.0, 1.0),
        float4_nonbuiltin(1.0, 1.0, 1.0, 1.0)};
    node_texture_valToRgb(2, ColorRamp_positions, ColorRamp_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR,
                          COLBAND_HUE_NEAR, Noise_SPACE_Texture_DOT_001__Fac,
                          &ColorRamp__Color, NULL);
    float Math__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Noise_SPACE_Texture__Fac,
                      float(ColorRamp__Color), 0.5, &Math__Value);
    float Math_DOT_001__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math__Value, 0.019999999552965164,
                      0.5, &Math_DOT_001__Value);
    float3_nonbuiltin Group_DOT_004__Output_2;
    nodegroup_displacement_to_offset(

        float3_nonbuiltin(Math_DOT_001__Value),
        side_step_displacement_to_offset_magnitude__Value, normal,
        &Group_DOT_004__Output_2);
    float noise_texture_1_w__Value;

    noise_texture_1_w__Value = noise_texture_1_w_FROM_geometry_sandstone;
    float Noise_SPACE_Texture_DOT_002__Fac;
    node_shader_tex_noise(4, position, noise_texture_1_w__Value, 5.0, 2.0, 0.5,
                          0.0, &Noise_SPACE_Texture_DOT_002__Fac, NULL);
    float4_nonbuiltin ColorRamp_DOT_001__Color;

    float ColorRamp_DOT_001_positions[2]{0.4000000059604645,
                                         0.6000000238418579};
    float4_nonbuiltin ColorRamp_DOT_001_colors[2]{
        float4_nonbuiltin(0.0, 0.0, 0.0, 1.0),
        float4_nonbuiltin(1.0, 1.0, 1.0, 1.0)};
    node_texture_valToRgb(
        2, ColorRamp_DOT_001_positions, ColorRamp_DOT_001_colors,
        COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR, COLBAND_HUE_NEAR,
        Noise_SPACE_Texture_DOT_002__Fac, &ColorRamp_DOT_001__Color, NULL);
    float3_nonbuiltin Group_DOT_005__Output_3;
    nodegroup_cracked_with_mask(

        2.0, crack_magnitude_1__Value, 2.0,
        noise_texture_w_FROM_nodegroup_cracked_with_mask, normal, position,
        voronoi_texture_w_FROM_nodegroup_cracked_with_mask,
        &Group_DOT_005__Output_3);
    float3_nonbuiltin Group_DOT_006__Output_3;
    nodegroup_cracked_with_mask_DOT_001(

        3.0, crack_magnitude_2__Value, 3.0,
        noise_texture_w_FROM_nodegroup_cracked_with_mask_DOT_001, normal,
        position, voronoi_texture_w_FROM_nodegroup_cracked_with_mask_DOT_001,
        &Group_DOT_006__Output_3);
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, Group_DOT_005__Output_3,
                            Group_DOT_006__Output_3,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_001__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_002__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, float3_nonbuiltin(ColorRamp_DOT_001__Color),
        Vector_SPACE_Math_DOT_001__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_002__Vector, NULL);
    float stripe_warp_scale__Value;

    stripe_warp_scale__Value = stripe_warp_scale_FROM_geometry_sandstone;
    float Noise_SPACE_Texture_DOT_003__Fac;
    node_shader_tex_noise(3, position, 0.0, stripe_warp_scale__Value, 2.0, 0.5,
                          0.0, &Noise_SPACE_Texture_DOT_003__Fac, NULL);
    float stripe_warp_mag__Value;

    stripe_warp_mag__Value = stripe_warp_mag_FROM_geometry_sandstone;
    float3_nonbuiltin Vector_SPACE_Math_DOT_003__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Noise_SPACE_Texture_DOT_003__Fac),
                            float3_nonbuiltin(stripe_warp_mag__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_003__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_004__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            Vector_SPACE_Math_DOT_003__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_004__Vector, NULL);
    float stripe_scale__Value;

    stripe_scale__Value = stripe_scale_FROM_geometry_sandstone;
    float4_nonbuiltin Wave_SPACE_Texture__Color;
    node_shader_tex_wave(SHD_WAVE_BANDS, SHD_WAVE_BANDS_DIRECTION_Z,
                         SHD_WAVE_RINGS_DIRECTION_X, SHD_WAVE_PROFILE_SAW,
                         Vector_SPACE_Math_DOT_004__Vector, stripe_scale__Value,
                         0.0, 2.0, 1.0, 0.5, 0.0, &Wave_SPACE_Texture__Color,
                         NULL);
    float stripe_mag__Value;

    stripe_mag__Value = stripe_mag_FROM_geometry_sandstone;
    float3_nonbuiltin Vector_SPACE_Math_DOT_005__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(stripe_mag__Value), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_005__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_006__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, float3_nonbuiltin(Wave_SPACE_Texture__Color),
        Vector_SPACE_Math_DOT_005__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_006__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_007__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_002__Vector,
        Vector_SPACE_Math_DOT_006__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_007__Vector, NULL);
    float scale0__Value;

    scale0__Value = scale0_FROM_geometry_sandstone;
    float detail0__Value;

    detail0__Value = detail0_FROM_geometry_sandstone;
    float roughness0__Value;

    roughness0__Value = roughness0_FROM_geometry_sandstone;
    float zscale0__Value;

    zscale0__Value = zscale0_FROM_geometry_sandstone;
    float3_nonbuiltin position_shift0__Vector;

    position_shift0__Vector = position_shift0_FROM_geometry_sandstone;
    float3_nonbuiltin Vector_SPACE_Math_DOT_008__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift0__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_008__Vector, NULL);
    float Noise_SPACE_Texture_DOT_004__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_008__Vector, 0.0,
                          scale0__Value, detail0__Value, roughness0__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_004__Fac, NULL);
    float Math_DOT_002__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture_DOT_004__Fac,
                      0.5, 0.5, &Math_DOT_002__Value);
    float Math_DOT_003__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math_DOT_002__Value,
                      zscale0__Value, 0.5, &Math_DOT_003__Value);
    float scale1__Value;

    scale1__Value = scale1_FROM_geometry_sandstone;
    float detail1__Value;

    detail1__Value = detail1_FROM_geometry_sandstone;
    float roughness1__Value;

    roughness1__Value = roughness1_FROM_geometry_sandstone;
    float zscale1__Value;

    zscale1__Value = zscale1_FROM_geometry_sandstone;
    float3_nonbuiltin position_shift1__Vector;

    position_shift1__Vector = position_shift1_FROM_geometry_sandstone;
    float3_nonbuiltin Vector_SPACE_Math_DOT_009__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift1__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_009__Vector, NULL);
    float Noise_SPACE_Texture_DOT_005__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_009__Vector, 0.0,
                          scale1__Value, detail1__Value, roughness1__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_005__Fac, NULL);
    float Math_DOT_004__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture_DOT_005__Fac,
                      0.5, 0.5, &Math_DOT_004__Value);
    float Math_DOT_005__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math_DOT_004__Value,
                      zscale1__Value, 0.5, &Math_DOT_005__Value);
    float scale2__Value;

    scale2__Value = scale2_FROM_geometry_sandstone;
    float detail2__Value;

    detail2__Value = detail2_FROM_geometry_sandstone;
    float roughness2__Value;

    roughness2__Value = roughness2_FROM_geometry_sandstone;
    float zscale2__Value;

    zscale2__Value = zscale2_FROM_geometry_sandstone;
    float3_nonbuiltin position_shift2__Vector;

    position_shift2__Vector = position_shift2_FROM_geometry_sandstone;
    float3_nonbuiltin Vector_SPACE_Math_DOT_010__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift2__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_010__Vector, NULL);
    float Noise_SPACE_Texture_DOT_006__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_010__Vector, 0.0,
                          scale2__Value, detail2__Value, roughness2__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_006__Fac, NULL);
    float Math_DOT_006__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture_DOT_006__Fac,
                      0.5, 0.5, &Math_DOT_006__Value);
    float Math_DOT_007__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math_DOT_006__Value,
                      zscale2__Value, 0.5, &Math_DOT_007__Value);
    float Math_DOT_008__Value;
    node_texture_math(NODE_MATH_MAXIMUM, 0, Math_DOT_005__Value,
                      Math_DOT_007__Value, 0.5, &Math_DOT_008__Value);
    float Math_DOT_009__Value;
    node_texture_math(NODE_MATH_MAXIMUM, 0, Math_DOT_003__Value,
                      Math_DOT_008__Value, 0.5, &Math_DOT_009__Value);
    float3_nonbuiltin Vector_SPACE_Math_DOT_011__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Math_DOT_009__Value), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_011__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_012__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, Group_DOT_004__Output_2,
                            Vector_SPACE_Math_DOT_007__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_012__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_013__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, Vector_SPACE_Math__Vector,
                            Vector_SPACE_Math_DOT_012__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_013__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_014__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_011__Vector,
        Vector_SPACE_Math_DOT_013__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_014__Vector, NULL);

    if (offset != NULL)
        *offset = Vector_SPACE_Math_DOT_014__Vector;
}
