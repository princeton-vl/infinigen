// Code generated using version 0.31.1 of worldgen/tools/kernelize_surfaces.py;
// refer to worldgen/surfaces/templates/ice.py which has the copyright and
// authors
DEVICE_FUNC void
geo_ice(float3_nonbuiltin position, float3_nonbuiltin normal,
        POINTER_OR_REFERENCE_ARG float *float_vars,
        POINTER_OR_REFERENCE_ARG float4_nonbuiltin *float4_nonbuiltin_vars,
        POINTER_OR_REFERENCE_ARG float3_nonbuiltin *offset) {
    float Value_DOT_001_FROM_geo_ice = float_vars[0];
    float Value_FROM_geo_ice = float_vars[1];
    float4_nonbuiltin color_ramp_1_VAR_FROM_geo_ice_color0 =
        float4_nonbuiltin_vars[0];
    float4_nonbuiltin color_ramp_1_VAR_FROM_geo_ice_color1 =
        float4_nonbuiltin_vars[1];
    float color_ramp_1_VAR_FROM_geo_ice_pos0 = float_vars[2];
    float color_ramp_1_VAR_FROM_geo_ice_pos1 = float_vars[3];
    float noise_texture_1_w_FROM_geo_ice = float_vars[4];
    float noise_texture_w_FROM_geo_ice = float_vars[5];
    float sur_dis_FROM_geo_ice = float_vars[6];
    float sur_rou_FROM_geo_ice = float_vars[7];
    float sur_rou__Value;

    sur_rou__Value = sur_rou_FROM_geo_ice;
    float sur_dis__Value;

    sur_dis__Value = sur_dis_FROM_geo_ice;
    float noise_texture_w__Value;

    noise_texture_w__Value = noise_texture_w_FROM_geo_ice;
    float Noise_SPACE_Texture__Fac;
    node_shader_tex_noise(4, position, noise_texture_w__Value, 4.0, 15.0, 0.5,
                          0.0, &Noise_SPACE_Texture__Fac, NULL);
    float4_nonbuiltin color_ramp_1_VAR__Color;

    float color_ramp_1_VAR_positions[2]{color_ramp_1_VAR_FROM_geo_ice_pos0,
                                        color_ramp_1_VAR_FROM_geo_ice_pos1};
    float4_nonbuiltin color_ramp_1_VAR_colors[2]{
        color_ramp_1_VAR_FROM_geo_ice_color0,
        color_ramp_1_VAR_FROM_geo_ice_color1};
    node_texture_valToRgb(
        2, color_ramp_1_VAR_positions, color_ramp_1_VAR_colors,
        COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR, COLBAND_HUE_NEAR,
        Noise_SPACE_Texture__Fac, &color_ramp_1_VAR__Color, NULL);
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(color_ramp_1_VAR__Color), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);
    float Value__Value;

    Value__Value = Value_FROM_geo_ice;
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, Vector_SPACE_Math__Vector,
                            float3_nonbuiltin(Value__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_001__Vector, NULL);
    float noise_texture_1_w__Value;

    noise_texture_1_w__Value = noise_texture_1_w_FROM_geo_ice;
    float Noise_SPACE_Texture_DOT_001__Fac;
    node_shader_tex_noise(4, position, noise_texture_1_w__Value, 1.0, 15.0,
                          0.800000011920929, sur_dis__Value,
                          &Noise_SPACE_Texture_DOT_001__Fac, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_002__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Noise_SPACE_Texture_DOT_001__Fac),
                            normal, float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_002__Vector, NULL);
    float Value_DOT_001__Value;

    Value_DOT_001__Value = Value_DOT_001_FROM_geo_ice;
    float3_nonbuiltin Vector_SPACE_Math_DOT_003__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD,
                            Vector_SPACE_Math_DOT_002__Vector,
                            float3_nonbuiltin(Value_DOT_001__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_003__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_004__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_001__Vector,
        Vector_SPACE_Math_DOT_003__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_004__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_005__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, Vector_SPACE_Math_DOT_004__Vector,
        float3_nonbuiltin(sur_rou__Value), float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_005__Vector, NULL);

    if (offset != NULL)
        *offset = Vector_SPACE_Math_DOT_005__Vector;
}
