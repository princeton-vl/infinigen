// Code generated using version 1.0.0 of infinigen/tools/kernelize_surfaces.py;
// refer to infinigen/assets/materials/cobble_stone.py which has the copyright
// and authors
DEVICE_FUNC void geo_cobblestone(
    float3_nonbuiltin position, float3_nonbuiltin normal,
    POINTER_OR_REFERENCE_ARG float *float_vars,
    POINTER_OR_REFERENCE_ARG float4_nonbuiltin *float4_nonbuiltin_vars,
    POINTER_OR_REFERENCE_ARG float3_nonbuiltin *offset) {
    float Scale1_FROM_geo_cobblestone = float_vars[0];
    float Scale2_FROM_geo_cobblestone = float_vars[1];
    float Scale3_FROM_geo_cobblestone = float_vars[2];
    float W1_FROM_geo_cobblestone = float_vars[3];
    float W2_FROM_geo_cobblestone = float_vars[4];
    float W3_FROM_geo_cobblestone = float_vars[5];
    float W4_FROM_geo_cobblestone = float_vars[6];
    float4_nonbuiltin colorramp_VAR_FROM_geo_cobblestone_color0 =
        float4_nonbuiltin_vars[0];
    float4_nonbuiltin colorramp_VAR_FROM_geo_cobblestone_color1 =
        float4_nonbuiltin_vars[1];
    float colorramp_VAR_FROM_geo_cobblestone_pos0 = float_vars[7];
    float colorramp_VAR_FROM_geo_cobblestone_pos1 = float_vars[8];
    float dep_sto_FROM_geo_cobblestone = float_vars[9];
    float sca_sto_FROM_geo_cobblestone = float_vars[10];
    float uni_sto_FROM_geo_cobblestone = float_vars[11];
    float value_8_FROM_geo_cobblestone = float_vars[12];
    float sca_sto__Value;

    sca_sto__Value = sca_sto_FROM_geo_cobblestone;
    float uni_sto__Value;

    uni_sto__Value = uni_sto_FROM_geo_cobblestone;
    float dep_sto__Value;

    dep_sto__Value = dep_sto_FROM_geo_cobblestone;
    float W1__Value;

    W1__Value = W1_FROM_geo_cobblestone;
    float Scale1__Value;

    Scale1__Value = Scale1_FROM_geo_cobblestone;
    float Noise_SPACE_Texture__Fac;
    node_shader_tex_noise(4, position, W1__Value, Scale1__Value, 2.0, 0.5, 0.0,
                          &Noise_SPACE_Texture__Fac, NULL);
    float W2__Value;

    W2__Value = W2_FROM_geo_cobblestone;
    float3_nonbuiltin Voronoi_SPACE_Texture__Position;
    node_shader_tex_voronoi(4, SHD_VORONOI_F1, SHD_VORONOI_EUCLIDEAN, position,
                            W2__Value, sca_sto__Value, 1.0, 0.5, uni_sto__Value,
                            NULL, NULL, &Voronoi_SPACE_Texture__Position, NULL,
                            NULL);
    float Scale2__Value;

    Scale2__Value = Scale2_FROM_geo_cobblestone;
    float Noise_SPACE_Texture_DOT_001__Fac;
    node_shader_tex_noise(3, Voronoi_SPACE_Texture__Position, 0.0,
                          Scale2__Value, 2.0, 0.5, 0.0,
                          &Noise_SPACE_Texture_DOT_001__Fac, NULL);
    float4_nonbuiltin ColorRamp__Color;

    float ColorRamp_positions[2]{0.11590000241994858, 0.4749999940395355};
    float4_nonbuiltin ColorRamp_colors[2]{
        float4_nonbuiltin(0.0, 0.0, 0.0, 1.0),
        float4_nonbuiltin(1.0, 1.0, 1.0, 1.0)};
    node_texture_valToRgb(2, ColorRamp_positions, ColorRamp_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_CONSTANT,
                          COLBAND_HUE_NEAR, Noise_SPACE_Texture_DOT_001__Fac,
                          &ColorRamp__Color, NULL);
    float W3__Value;

    W3__Value = W3_FROM_geo_cobblestone;
    float Voronoi_SPACE_Texture_DOT_001__Distance;
    node_shader_tex_voronoi(
        4, SHD_VORONOI_DISTANCE_TO_EDGE, SHD_VORONOI_EUCLIDEAN, position,
        W3__Value, sca_sto__Value, 1.0, 0.5, uni_sto__Value,
        &Voronoi_SPACE_Texture_DOT_001__Distance, NULL, NULL, NULL, NULL);
    float Math__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, 1.5, sca_sto__Value, 0.5,
                      &Math__Value);
    float W4__Value;

    W4__Value = W4_FROM_geo_cobblestone;
    float Voronoi_SPACE_Texture_DOT_002__Distance;
    node_shader_tex_voronoi(
        4, SHD_VORONOI_DISTANCE_TO_EDGE, SHD_VORONOI_EUCLIDEAN, position,
        W4__Value, Math__Value, 1.0, 0.5, uni_sto__Value,
        &Voronoi_SPACE_Texture_DOT_002__Distance, NULL, NULL, NULL, NULL);
    float4_nonbuiltin Mix__Color;
    node_shader_mix_rgb(
        0, MA_RAMP_MIX, float(ColorRamp__Color),
        float4_nonbuiltin(Voronoi_SPACE_Texture_DOT_001__Distance),
        float4_nonbuiltin(Voronoi_SPACE_Texture_DOT_002__Distance),
        &Mix__Color);
    float4_nonbuiltin Mix_DOT_001__Color;
    node_shader_mix_rgb(0, MA_RAMP_MIX, Noise_SPACE_Texture__Fac, Mix__Color,
                        float4_nonbuiltin(0.5, 0.5, 0.5, 1.0),
                        &Mix_DOT_001__Color);
    float4_nonbuiltin colorramp_VAR__Color;

    float colorramp_VAR_positions[2]{colorramp_VAR_FROM_geo_cobblestone_pos0,
                                     colorramp_VAR_FROM_geo_cobblestone_pos1};
    float4_nonbuiltin colorramp_VAR_colors[2]{
        colorramp_VAR_FROM_geo_cobblestone_color0,
        colorramp_VAR_FROM_geo_cobblestone_color1};
    node_texture_valToRgb(2, colorramp_VAR_positions, colorramp_VAR_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR,
                          COLBAND_HUE_NEAR, float(Mix_DOT_001__Color),
                          &colorramp_VAR__Color, NULL);
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(colorramp_VAR__Color), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, Vector_SPACE_Math__Vector,
        float3_nonbuiltin(dep_sto__Value), float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_001__Vector, NULL);
    float Scale3__Value;

    Scale3__Value = Scale3_FROM_geo_cobblestone;
    float Noise_SPACE_Texture_DOT_002__Fac;
    node_shader_tex_noise(3, position, 0.0, Scale3__Value, 10.0, 0.5, 2.0,
                          &Noise_SPACE_Texture_DOT_002__Fac, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_002__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_SUBTRACT,
                            float3_nonbuiltin(Noise_SPACE_Texture_DOT_002__Fac),
                            float3_nonbuiltin(0.5, 0.5, 0.5),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_002__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_003__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            Vector_SPACE_Math_DOT_002__Vector, normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_003__Vector, NULL);
    float value_8__Value;

    value_8__Value = value_8_FROM_geo_cobblestone;
    float3_nonbuiltin Vector_SPACE_Math_DOT_004__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, Vector_SPACE_Math_DOT_003__Vector,
        float3_nonbuiltin(value_8__Value), float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_004__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_005__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_004__Vector,
        Vector_SPACE_Math_DOT_001__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_005__Vector, NULL);

    if (offset != NULL)
        *offset = Vector_SPACE_Math_DOT_005__Vector;
}
