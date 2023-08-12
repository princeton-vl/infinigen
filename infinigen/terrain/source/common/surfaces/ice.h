// Code generated using version 1.0.0 of infinigen/tools/kernelize_surfaces.py;
// refer to infinigen/assets/materials/ice.py which has the copyright and
// authors
DEVICE_FUNC void geo_ice(float3_nonbuiltin position, float3_nonbuiltin normal,
                         POINTER_OR_REFERENCE_ARG float *float_vars,
                         POINTER_OR_REFERENCE_ARG float3_nonbuiltin *offset) {
    float Scale1_FROM_geo_ice = float_vars[0];
    float Scale2_FROM_geo_ice = float_vars[1];
    float W1_FROM_geo_ice = float_vars[2];
    float W2_FROM_geo_ice = float_vars[3];
    float W1__Value;

    W1__Value = W1_FROM_geo_ice;
    float Scale1__Value;

    Scale1__Value = Scale1_FROM_geo_ice;
    float Noise_SPACE_Texture__Fac;
    node_shader_tex_noise(4, position, W1__Value, Scale1__Value, 20.0, 1.0, 0.0,
                          &Noise_SPACE_Texture__Fac, NULL);
    float4_nonbuiltin ColorRamp__Color;

    float ColorRamp_positions[2]{0.5, 1.0};
    float4_nonbuiltin ColorRamp_colors[2]{
        float4_nonbuiltin(0.0, 0.0, 0.0, 1.0),
        float4_nonbuiltin(1.0, 1.0, 1.0, 1.0)};
    node_texture_valToRgb(2, ColorRamp_positions, ColorRamp_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR,
                          COLBAND_HUE_NEAR, Noise_SPACE_Texture__Fac,
                          &ColorRamp__Color, NULL);
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_SCALE, float3_nonbuiltin(ColorRamp__Color),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        0.029999999329447746, &Vector_SPACE_Math__Vector, NULL);
    float W2__Value;

    W2__Value = W2_FROM_geo_ice;
    float Scale2__Value;

    Scale2__Value = Scale2_FROM_geo_ice;
    float Noise_SPACE_Texture_DOT_001__Fac;
    node_shader_tex_noise(4, position, W2__Value, Scale2__Value, 15.0,
                          0.699999988079071, 1.5,
                          &Noise_SPACE_Texture_DOT_001__Fac, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY, normal,
                            float3_nonbuiltin(Noise_SPACE_Texture_DOT_001__Fac),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_001__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_002__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_SCALE, Vector_SPACE_Math_DOT_001__Vector,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        0.07999999821186066, &Vector_SPACE_Math_DOT_002__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_003__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY_ADD, normal,
                            Vector_SPACE_Math__Vector,
                            Vector_SPACE_Math_DOT_002__Vector, 1.0,
                            &Vector_SPACE_Math_DOT_003__Vector, NULL);

    if (offset != NULL)
        *offset = Vector_SPACE_Math_DOT_003__Vector;
}
