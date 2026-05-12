// Code generated using version 1.0.0 of infinigen/tools/kernelize_surfaces.py;
// refer to infinigen/assets/materials/dirt.py which has the copyright and
// authors
DEVICE_FUNC void
geo_dirt(float3_nonbuiltin position, float3_nonbuiltin normal,
         POINTER_OR_REFERENCE_ARG float *float_vars,
         POINTER_OR_REFERENCE_ARG float3_nonbuiltin *float3_nonbuiltin_vars,
         POINTER_OR_REFERENCE_ARG float3_nonbuiltin *offset) {
    float Value_DOT_001_FROM_geo_dirt = float_vars[0];
    float Value_DOT_002_FROM_geo_dirt = float_vars[1];
    float Value_DOT_003_FROM_geo_dirt = float_vars[2];
    float Value_FROM_geo_dirt = float_vars[3];
    float colorramp_1_FROM_geo_dirt = float_vars[4];
    float colorramp_2_a_FROM_geo_dirt = float_vars[5];
    float colorramp_2_b_FROM_geo_dirt = float_vars[6];
    float detail0_FROM_geo_dirt = float_vars[7];
    float detail1_FROM_geo_dirt = float_vars[8];
    float detail2_FROM_geo_dirt = float_vars[9];
    float noise_texture_1_w_FROM_geo_dirt = float_vars[10];
    float noise_texture_2_w_FROM_geo_dirt = float_vars[11];
    float noise_texture_3_w_FROM_geo_dirt = float_vars[12];
    float noise_texture_4_w_FROM_geo_dirt = float_vars[13];
    float3_nonbuiltin position_shift0_FROM_geo_dirt = float3_nonbuiltin_vars[0];
    float3_nonbuiltin position_shift1_FROM_geo_dirt = float3_nonbuiltin_vars[1];
    float3_nonbuiltin position_shift2_FROM_geo_dirt = float3_nonbuiltin_vars[2];
    float roughness0_FROM_geo_dirt = float_vars[14];
    float roughness1_FROM_geo_dirt = float_vars[15];
    float roughness2_FROM_geo_dirt = float_vars[16];
    float scal_crack_FROM_geo_dirt = float_vars[17];
    float scale0_FROM_geo_dirt = float_vars[18];
    float scale1_FROM_geo_dirt = float_vars[19];
    float scale2_FROM_geo_dirt = float_vars[20];
    float zscale0_FROM_geo_dirt = float_vars[21];
    float zscale1_FROM_geo_dirt = float_vars[22];
    float zscale2_FROM_geo_dirt = float_vars[23];
    float noise_texture_2_w__Value;

    noise_texture_2_w__Value = noise_texture_2_w_FROM_geo_dirt;
    float Noise_SPACE_Texture_DOT_001__Fac;
    node_shader_tex_noise(4, position, noise_texture_2_w__Value, 2.5, 2.0, 0.5,
                          0.0, &Noise_SPACE_Texture_DOT_001__Fac, NULL);
    float colorramp_2_a__Value;

    colorramp_2_a__Value = colorramp_2_a_FROM_geo_dirt;
    float colorramp_2_b__Value;

    colorramp_2_b__Value = colorramp_2_b_FROM_geo_dirt;
    float Map_SPACE_Range__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Noise_SPACE_Texture_DOT_001__Fac,
        colorramp_2_a__Value, colorramp_2_b__Value, 0.0, 1.0, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range__Result, NULL);
    float noise_texture_1_w__Value;

    noise_texture_1_w__Value = noise_texture_1_w_FROM_geo_dirt;
    float4_nonbuiltin Noise_SPACE_Texture_DOT_002__Color;
    node_shader_tex_noise(4, position, noise_texture_1_w__Value, 0.5, 16.0, 0.5,
                          0.0, NULL, &Noise_SPACE_Texture_DOT_002__Color);
    float scal_crack__Value;

    scal_crack__Value = scal_crack_FROM_geo_dirt;
    float Voronoi_SPACE_Texture__Distance;
    node_shader_tex_voronoi(
        3, SHD_VORONOI_DISTANCE_TO_EDGE, SHD_VORONOI_EUCLIDEAN,
        float3_nonbuiltin(Noise_SPACE_Texture_DOT_002__Color), 0.0,
        scal_crack__Value, 1.0, 0.5, 1.0, &Voronoi_SPACE_Texture__Distance,
        NULL, NULL, NULL, NULL);
    float colorramp_1__Value;

    colorramp_1__Value = colorramp_1_FROM_geo_dirt;
    float Map_SPACE_Range_DOT_001__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Voronoi_SPACE_Texture__Distance, 0.0,
        colorramp_1__Value, 0.0, 1.0, 4.0, float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(4.0, 4.0, 4.0), &Map_SPACE_Range_DOT_001__Result,
        NULL);
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_SUBTRACT,
                            float3_nonbuiltin(1.0, 1.0, 1.0),
                            float3_nonbuiltin(Map_SPACE_Range__Result),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            Vector_SPACE_Math__Vector,
                            float3_nonbuiltin(Map_SPACE_Range_DOT_001__Result),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_001__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_002__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, float3_nonbuiltin(Map_SPACE_Range__Result),
        float3_nonbuiltin(0.5, 0.5, 0.5), float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
        &Vector_SPACE_Math_DOT_002__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_003__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_001__Vector,
        Vector_SPACE_Math_DOT_002__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_003__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_004__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            Vector_SPACE_Math_DOT_003__Vector, normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_004__Vector, NULL);
    float Value__Value;

    Value__Value = Value_FROM_geo_dirt;
    float3_nonbuiltin Vector_SPACE_Math_DOT_005__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, Vector_SPACE_Math_DOT_004__Vector,
        float3_nonbuiltin(Value__Value), float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
        &Vector_SPACE_Math_DOT_005__Vector, NULL);
    float Value_DOT_001__Value;

    Value_DOT_001__Value = Value_DOT_001_FROM_geo_dirt;
    float3_nonbuiltin Vector_SPACE_Math_DOT_006__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            Vector_SPACE_Math_DOT_005__Vector,
                            float3_nonbuiltin(Value_DOT_001__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_006__Vector, NULL);
    float noise_texture_3_w__Value;

    noise_texture_3_w__Value = noise_texture_3_w_FROM_geo_dirt;
    float Noise_SPACE_Texture_DOT_003__Fac;
    node_shader_tex_noise(4, position, noise_texture_3_w__Value,
                          6.0958356857299805, 2.0, 0.5, 0.0,
                          &Noise_SPACE_Texture_DOT_003__Fac, NULL);
    float Math__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture_DOT_003__Fac,
                      0.5, 0.5, &Math__Value);
    float3_nonbuiltin Vector_SPACE_Math_DOT_007__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Math__Value), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_007__Vector, NULL);
    float Value_DOT_002__Value;

    Value_DOT_002__Value = Value_DOT_002_FROM_geo_dirt;
    float3_nonbuiltin Vector_SPACE_Math_DOT_008__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            Vector_SPACE_Math_DOT_007__Vector,
                            float3_nonbuiltin(Value_DOT_002__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_008__Vector, NULL);
    float noise_texture_4_w__Value;

    noise_texture_4_w__Value = noise_texture_4_w_FROM_geo_dirt;
    float Noise_SPACE_Texture_DOT_004__Fac;
    node_shader_tex_noise(4, position, noise_texture_4_w__Value,
                          23.199947357177734, 2.0, 0.5, 0.0,
                          &Noise_SPACE_Texture_DOT_004__Fac, NULL);
    float4_nonbuiltin ColorRamp__Color;

    float ColorRamp_positions[4]{0.0, 0.30000001192092896, 0.699999988079071,
                                 1.0};
    float4_nonbuiltin ColorRamp_colors[4]{
        float4_nonbuiltin(0.0, 0.0, 0.0, 1.0),
        float4_nonbuiltin(0.5, 0.5, 0.5, 1.0),
        float4_nonbuiltin(0.5, 0.5, 0.5, 1.0),
        float4_nonbuiltin(1.0, 1.0, 1.0, 1.0)};
    node_texture_valToRgb(4, ColorRamp_positions, ColorRamp_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR,
                          COLBAND_HUE_NEAR, Noise_SPACE_Texture_DOT_004__Fac,
                          &ColorRamp__Color, NULL);
    float Math_DOT_001__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, float(ColorRamp__Color), 0.5, 0.5,
                      &Math_DOT_001__Value);
    float3_nonbuiltin Vector_SPACE_Math_DOT_009__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Math_DOT_001__Value), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_009__Vector, NULL);
    float Value_DOT_003__Value;

    Value_DOT_003__Value = Value_DOT_003_FROM_geo_dirt;
    float3_nonbuiltin Vector_SPACE_Math_DOT_010__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            Vector_SPACE_Math_DOT_009__Vector,
                            float3_nonbuiltin(Value_DOT_003__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_010__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_011__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_008__Vector,
        Vector_SPACE_Math_DOT_006__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_011__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_012__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_010__Vector,
        Vector_SPACE_Math_DOT_011__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_012__Vector, NULL);
    float scale0__Value;

    scale0__Value = scale0_FROM_geo_dirt;
    float detail0__Value;

    detail0__Value = detail0_FROM_geo_dirt;
    float roughness0__Value;

    roughness0__Value = roughness0_FROM_geo_dirt;
    float zscale0__Value;

    zscale0__Value = zscale0_FROM_geo_dirt;
    float3_nonbuiltin position_shift0__Vector;

    position_shift0__Vector = position_shift0_FROM_geo_dirt;
    float3_nonbuiltin Vector_SPACE_Math_DOT_013__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift0__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_013__Vector, NULL);
    float Noise_SPACE_Texture_DOT_005__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_013__Vector, 0.0,
                          scale0__Value, detail0__Value, roughness0__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_005__Fac, NULL);
    float Math_DOT_002__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture_DOT_005__Fac,
                      0.5, 0.5, &Math_DOT_002__Value);
    float Math_DOT_003__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math_DOT_002__Value,
                      zscale0__Value, 0.5, &Math_DOT_003__Value);
    float scale1__Value;

    scale1__Value = scale1_FROM_geo_dirt;
    float detail1__Value;

    detail1__Value = detail1_FROM_geo_dirt;
    float roughness1__Value;

    roughness1__Value = roughness1_FROM_geo_dirt;
    float zscale1__Value;

    zscale1__Value = zscale1_FROM_geo_dirt;
    float3_nonbuiltin position_shift1__Vector;

    position_shift1__Vector = position_shift1_FROM_geo_dirt;
    float3_nonbuiltin Vector_SPACE_Math_DOT_014__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift1__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_014__Vector, NULL);
    float Noise_SPACE_Texture_DOT_006__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_014__Vector, 0.0,
                          scale1__Value, detail1__Value, roughness1__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_006__Fac, NULL);
    float Math_DOT_004__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture_DOT_006__Fac,
                      0.5, 0.5, &Math_DOT_004__Value);
    float Math_DOT_005__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math_DOT_004__Value,
                      zscale1__Value, 0.5, &Math_DOT_005__Value);
    float scale2__Value;

    scale2__Value = scale2_FROM_geo_dirt;
    float detail2__Value;

    detail2__Value = detail2_FROM_geo_dirt;
    float roughness2__Value;

    roughness2__Value = roughness2_FROM_geo_dirt;
    float zscale2__Value;

    zscale2__Value = zscale2_FROM_geo_dirt;
    float3_nonbuiltin position_shift2__Vector;

    position_shift2__Vector = position_shift2_FROM_geo_dirt;
    float3_nonbuiltin Vector_SPACE_Math_DOT_015__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift2__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_015__Vector, NULL);
    float Noise_SPACE_Texture_DOT_007__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_015__Vector, 0.0,
                          scale2__Value, detail2__Value, roughness2__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_007__Fac, NULL);
    float Math_DOT_006__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture_DOT_007__Fac,
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
    float3_nonbuiltin Vector_SPACE_Math_DOT_016__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Math_DOT_009__Value), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_016__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_017__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_016__Vector,
        Vector_SPACE_Math_DOT_012__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_017__Vector, NULL);

    if (offset != NULL)
        *offset = Vector_SPACE_Math_DOT_017__Vector;
}
