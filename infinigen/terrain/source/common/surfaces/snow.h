// Code generated using version 1.0.0 of infinigen/tools/kernelize_surfaces.py;
// refer to infinigen/assets/materials/snow.py which has the copyright and
// authors
DEVICE_FUNC void
geo_snowtexture(float3_nonbuiltin position, float3_nonbuiltin normal,
                POINTER_OR_REFERENCE_ARG float3_nonbuiltin *offset) {
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY, position,
                            float3_nonbuiltin(12.0, 12.0, 12.0),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);
    float Noise_SPACE_Texture__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math__Vector, 0.0, 12.0, 2.0, 0.5,
                          0.0, &Noise_SPACE_Texture__Fac, NULL);
    float Noise_SPACE_Texture_DOT_001__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math__Vector, 0.0, 2.0, 4.0, 0.5, 0.0,
                          &Noise_SPACE_Texture_DOT_001__Fac, NULL);
    float4_nonbuiltin ColorRamp__Color;

    float ColorRamp_positions[2]{0.0689999982714653, 0.7570000290870667};
    float4_nonbuiltin ColorRamp_colors[2]{
        float4_nonbuiltin(0.0, 0.0, 0.0, 1.0),
        float4_nonbuiltin(1.0, 1.0, 1.0, 1.0)};
    node_texture_valToRgb(2, ColorRamp_positions, ColorRamp_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR,
                          COLBAND_HUE_NEAR, Noise_SPACE_Texture_DOT_001__Fac,
                          &ColorRamp__Color, NULL);
    float Noise_SPACE_Texture_DOT_002__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math__Vector, 0.0, 1.0, 4.0, 0.5, 0.0,
                          &Noise_SPACE_Texture_DOT_002__Fac, NULL);
    float4_nonbuiltin ColorRamp_DOT_001__Color;

    float ColorRamp_DOT_001_positions[2]{0.0689999982714653,
                                         0.7570000290870667};
    float4_nonbuiltin ColorRamp_DOT_001_colors[2]{
        float4_nonbuiltin(0.0, 0.0, 0.0, 1.0),
        float4_nonbuiltin(1.0, 1.0, 1.0, 1.0)};
    node_texture_valToRgb(
        2, ColorRamp_DOT_001_positions, ColorRamp_DOT_001_colors,
        COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR, COLBAND_HUE_NEAR,
        Noise_SPACE_Texture_DOT_002__Fac, &ColorRamp_DOT_001__Color, NULL);
    float Math__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, 0.6000000238418579,
                      Noise_SPACE_Texture__Fac, 0.5, &Math__Value);
    float Math_DOT_001__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, 0.4000000059604645,
                      float(ColorRamp__Color), 0.5, &Math_DOT_001__Value);
    float Math_DOT_002__Value;
    node_texture_math(NODE_MATH_ADD, 0, Math_DOT_001__Value,
                      float(ColorRamp_DOT_001__Color), 0.5,
                      &Math_DOT_002__Value);
    float Math_DOT_003__Value;
    node_texture_math(NODE_MATH_ADD, 0, Math__Value, Math_DOT_002__Value, 0.5,
                      &Math_DOT_003__Value);
    float Map_SPACE_Range__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Math_DOT_003__Value, 0.0, 2.0,
        -0.029999999329447746, 0.029999999329447746, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range__Result, NULL);
    float Noise_SPACE_Texture_DOT_003__Fac;
    node_shader_tex_noise(3, position, 0.0, 0.5, 2.0, 0.5, 0.0,
                          &Noise_SPACE_Texture_DOT_003__Fac, NULL);
    float4_nonbuiltin ColorRamp_DOT_002__Color;

    float ColorRamp_DOT_002_positions[2]{0.25, 0.75};
    float4_nonbuiltin ColorRamp_DOT_002_colors[2]{
        float4_nonbuiltin(0.0, 0.0, 0.0, 1.0),
        float4_nonbuiltin(1.0, 1.0, 1.0, 1.0)};
    node_texture_valToRgb(
        2, ColorRamp_DOT_002_positions, ColorRamp_DOT_002_colors,
        COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR, COLBAND_HUE_NEAR,
        Noise_SPACE_Texture_DOT_003__Fac, &ColorRamp_DOT_002__Color, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Map_SPACE_Range__Result),
                            float3_nonbuiltin(ColorRamp_DOT_002__Color),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_001__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_002__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY, normal,
                            Vector_SPACE_Math_DOT_001__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_002__Vector, NULL);

    if (offset != NULL)
        *offset = Vector_SPACE_Math_DOT_002__Vector;
}
