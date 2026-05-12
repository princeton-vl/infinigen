// Code generated using version 1.0.0 of infinigen/tools/kernelize_surfaces.py;
// refer to infinigen/assets/materials/chunkyrock.py which has the copyright
// and authors
DEVICE_FUNC void
geo_rocks(float3_nonbuiltin position, float3_nonbuiltin normal,
          POINTER_OR_REFERENCE_ARG float *float_vars,
          POINTER_OR_REFERENCE_ARG float3_nonbuiltin *float3_nonbuiltin_vars,
          POINTER_OR_REFERENCE_ARG float4_nonbuiltin *float4_nonbuiltin_vars,
          POINTER_OR_REFERENCE_ARG float3_nonbuiltin *offset) {
    float Value_FROM_geo_rocks = float_vars[0];
    float4_nonbuiltin colorramp_VAR_FROM_geo_rocks_color0 =
        float4_nonbuiltin_vars[0];
    float4_nonbuiltin colorramp_VAR_FROM_geo_rocks_color1 =
        float4_nonbuiltin_vars[1];
    float colorramp_VAR_FROM_geo_rocks_pos0 = float_vars[1];
    float colorramp_VAR_FROM_geo_rocks_pos1 = float_vars[2];
    float detail0_FROM_geo_rocks = float_vars[3];
    float detail1_FROM_geo_rocks = float_vars[4];
    float detail2_FROM_geo_rocks = float_vars[5];
    float3_nonbuiltin position_shift0_FROM_geo_rocks =
        float3_nonbuiltin_vars[0];
    float3_nonbuiltin position_shift1_FROM_geo_rocks =
        float3_nonbuiltin_vars[1];
    float3_nonbuiltin position_shift2_FROM_geo_rocks =
        float3_nonbuiltin_vars[2];
    float roughness0_FROM_geo_rocks = float_vars[6];
    float roughness1_FROM_geo_rocks = float_vars[7];
    float roughness2_FROM_geo_rocks = float_vars[8];
    float scale0_FROM_geo_rocks = float_vars[9];
    float scale1_FROM_geo_rocks = float_vars[10];
    float scale2_FROM_geo_rocks = float_vars[11];
    float voronoi_texture_scale_FROM_geo_rocks = float_vars[12];
    float voronoi_texture_w_FROM_geo_rocks = float_vars[13];
    float zscale0_FROM_geo_rocks = float_vars[14];
    float zscale1_FROM_geo_rocks = float_vars[15];
    float zscale2_FROM_geo_rocks = float_vars[16];
    float4_nonbuiltin Noise_SPACE_Texture__Color;
    node_shader_tex_noise(3, position, 0.0, 5.0, 2.0, 0.5, 0.0, NULL,
                          &Noise_SPACE_Texture__Color);
    float4_nonbuiltin Mix__Color;
    node_shader_mix_rgb(0, MA_RAMP_MIX, 0.800000011920929,
                        Noise_SPACE_Texture__Color, float4_nonbuiltin(position),
                        &Mix__Color);
    float voronoi_texture_scale__Value;

    voronoi_texture_scale__Value = voronoi_texture_scale_FROM_geo_rocks;
    float voronoi_texture_w__Value;

    voronoi_texture_w__Value = voronoi_texture_w_FROM_geo_rocks;
    float Voronoi_SPACE_Texture__Distance;
    node_shader_tex_voronoi(
        4, SHD_VORONOI_DISTANCE_TO_EDGE, SHD_VORONOI_EUCLIDEAN,
        float3_nonbuiltin(Mix__Color), voronoi_texture_w__Value,
        voronoi_texture_scale__Value, 1.0, 0.5, 1.0,
        &Voronoi_SPACE_Texture__Distance, NULL, NULL, NULL, NULL);
    float4_nonbuiltin colorramp_VAR__Color;

    float colorramp_VAR_positions[2]{colorramp_VAR_FROM_geo_rocks_pos0,
                                     colorramp_VAR_FROM_geo_rocks_pos1};
    float4_nonbuiltin colorramp_VAR_colors[2]{
        colorramp_VAR_FROM_geo_rocks_color0,
        colorramp_VAR_FROM_geo_rocks_color1};
    node_texture_valToRgb(2, colorramp_VAR_positions, colorramp_VAR_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR,
                          COLBAND_HUE_NEAR, Voronoi_SPACE_Texture__Distance,
                          &colorramp_VAR__Color, NULL);
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(colorramp_VAR__Color), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);
    float Value__Value;

    Value__Value = Value_FROM_geo_rocks;
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, Vector_SPACE_Math__Vector,
        float3_nonbuiltin(Value__Value), float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
        &Vector_SPACE_Math_DOT_001__Vector, NULL);
    float scale0__Value;

    scale0__Value = scale0_FROM_geo_rocks;
    float detail0__Value;

    detail0__Value = detail0_FROM_geo_rocks;
    float roughness0__Value;

    roughness0__Value = roughness0_FROM_geo_rocks;
    float zscale0__Value;

    zscale0__Value = zscale0_FROM_geo_rocks;
    float3_nonbuiltin position_shift0__Vector;

    position_shift0__Vector = position_shift0_FROM_geo_rocks;
    float3_nonbuiltin Vector_SPACE_Math_DOT_002__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift0__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_002__Vector, NULL);
    float Noise_SPACE_Texture_DOT_001__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_002__Vector, 0.0,
                          scale0__Value, detail0__Value, roughness0__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_001__Fac, NULL);
    float Math__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture_DOT_001__Fac,
                      0.5, 0.5, &Math__Value);
    float Math_DOT_001__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math__Value, zscale0__Value, 0.5,
                      &Math_DOT_001__Value);
    float scale1__Value;

    scale1__Value = scale1_FROM_geo_rocks;
    float detail1__Value;

    detail1__Value = detail1_FROM_geo_rocks;
    float roughness1__Value;

    roughness1__Value = roughness1_FROM_geo_rocks;
    float zscale1__Value;

    zscale1__Value = zscale1_FROM_geo_rocks;
    float3_nonbuiltin position_shift1__Vector;

    position_shift1__Vector = position_shift1_FROM_geo_rocks;
    float3_nonbuiltin Vector_SPACE_Math_DOT_003__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift1__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_003__Vector, NULL);
    float Noise_SPACE_Texture_DOT_002__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_003__Vector, 0.0,
                          scale1__Value, detail1__Value, roughness1__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_002__Fac, NULL);
    float Math_DOT_002__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture_DOT_002__Fac,
                      0.5, 0.5, &Math_DOT_002__Value);
    float Math_DOT_003__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math_DOT_002__Value,
                      zscale1__Value, 0.5, &Math_DOT_003__Value);
    float scale2__Value;

    scale2__Value = scale2_FROM_geo_rocks;
    float detail2__Value;

    detail2__Value = detail2_FROM_geo_rocks;
    float roughness2__Value;

    roughness2__Value = roughness2_FROM_geo_rocks;
    float zscale2__Value;

    zscale2__Value = zscale2_FROM_geo_rocks;
    float3_nonbuiltin position_shift2__Vector;

    position_shift2__Vector = position_shift2_FROM_geo_rocks;
    float3_nonbuiltin Vector_SPACE_Math_DOT_004__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift2__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_004__Vector, NULL);
    float Noise_SPACE_Texture_DOT_003__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_004__Vector, 0.0,
                          scale2__Value, detail2__Value, roughness2__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_003__Fac, NULL);
    float Math_DOT_004__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture_DOT_003__Fac,
                      0.5, 0.5, &Math_DOT_004__Value);
    float Math_DOT_005__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math_DOT_004__Value,
                      zscale2__Value, 0.5, &Math_DOT_005__Value);
    float Math_DOT_006__Value;
    node_texture_math(NODE_MATH_MAXIMUM, 0, Math_DOT_003__Value,
                      Math_DOT_005__Value, 0.5, &Math_DOT_006__Value);
    float Math_DOT_007__Value;
    node_texture_math(NODE_MATH_MAXIMUM, 0, Math_DOT_001__Value,
                      Math_DOT_006__Value, 0.5, &Math_DOT_007__Value);
    float3_nonbuiltin Vector_SPACE_Math_DOT_005__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Math_DOT_007__Value), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_005__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_006__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_001__Vector,
        Vector_SPACE_Math_DOT_005__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_006__Vector, NULL);

    if (offset != NULL)
        *offset = Vector_SPACE_Math_DOT_006__Vector;
}
