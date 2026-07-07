// Code generated using version 1.0.0 of infinigen/tools/kernelize_surfaces.py;
// refer to infinigen/assets/materials/stone.py which has the copyright and
// authors
DEVICE_FUNC void
geo_stone(float3_nonbuiltin position, float3_nonbuiltin normal,
          POINTER_OR_REFERENCE_ARG float *float_vars,
          POINTER_OR_REFERENCE_ARG float3_nonbuiltin *float3_nonbuiltin_vars,
          POINTER_OR_REFERENCE_ARG float4_nonbuiltin *float4_nonbuiltin_vars,
          POINTER_OR_REFERENCE_ARG float3_nonbuiltin *offset) {
    float Value_DOT_001_FROM_geo_stone = float_vars[0];
    float Value_DOT_002_FROM_geo_stone = float_vars[1];
    float Value_DOT_003_FROM_geo_stone = float_vars[2];
    float Value_DOT_004_FROM_geo_stone = float_vars[3];
    float Value_FROM_geo_stone = float_vars[4];
    float4_nonbuiltin colorramp_1_VAR_FROM_geo_stone_color0 =
        float4_nonbuiltin_vars[0];
    float4_nonbuiltin colorramp_1_VAR_FROM_geo_stone_color1 =
        float4_nonbuiltin_vars[1];
    float colorramp_1_VAR_FROM_geo_stone_pos0 = float_vars[5];
    float colorramp_1_VAR_FROM_geo_stone_pos1 = float_vars[6];
    float4_nonbuiltin colorramp_2_VAR_FROM_geo_stone_color0 =
        float4_nonbuiltin_vars[2];
    float4_nonbuiltin colorramp_2_VAR_FROM_geo_stone_color1 =
        float4_nonbuiltin_vars[3];
    float colorramp_2_VAR_FROM_geo_stone_pos0 = float_vars[7];
    float colorramp_2_VAR_FROM_geo_stone_pos1 = float_vars[8];
    float detail0_FROM_geo_stone = float_vars[9];
    float detail1_FROM_geo_stone = float_vars[10];
    float detail2_FROM_geo_stone = float_vars[11];
    float heig_bumps_lf_FROM_geo_stone = float_vars[12];
    float musgrave_texture_w_FROM_geo_stone = float_vars[13];
    float noise_texture_1_w_FROM_geo_stone = float_vars[14];
    float noise_texture_2_w_FROM_geo_stone = float_vars[15];
    float noise_texture_3_scale_FROM_geo_stone = float_vars[16];
    float noise_texture_3_w_FROM_geo_stone = float_vars[17];
    float noise_texture_4_scale_FROM_geo_stone = float_vars[18];
    float noise_texture_4_w_FROM_geo_stone = float_vars[19];
    float noise_texture_w_FROM_geo_stone = float_vars[20];
    float3_nonbuiltin position_shift0_FROM_geo_stone =
        float3_nonbuiltin_vars[0];
    float3_nonbuiltin position_shift1_FROM_geo_stone =
        float3_nonbuiltin_vars[1];
    float3_nonbuiltin position_shift2_FROM_geo_stone =
        float3_nonbuiltin_vars[2];
    float roughness0_FROM_geo_stone = float_vars[21];
    float roughness1_FROM_geo_stone = float_vars[22];
    float roughness2_FROM_geo_stone = float_vars[23];
    float scale0_FROM_geo_stone = float_vars[24];
    float scale1_FROM_geo_stone = float_vars[25];
    float scale2_FROM_geo_stone = float_vars[26];
    float size_bumps_lf_FROM_geo_stone = float_vars[27];
    float wave_texture_detail_FROM_geo_stone = float_vars[28];
    float wave_texture_distortion_FROM_geo_stone = float_vars[29];
    float wave_texture_scale_FROM_geo_stone = float_vars[30];
    float zscale0_FROM_geo_stone = float_vars[31];
    float zscale1_FROM_geo_stone = float_vars[32];
    float zscale2_FROM_geo_stone = float_vars[33];
    float heig_bumps_lf__Value;

    heig_bumps_lf__Value = heig_bumps_lf_FROM_geo_stone;
    float size_bumps_lf__Value;

    size_bumps_lf__Value = size_bumps_lf_FROM_geo_stone;
    float musgrave_texture_w__Value;

    musgrave_texture_w__Value = musgrave_texture_w_FROM_geo_stone;
    float Musgrave_SPACE_Texture__Fac;
    node_shader_tex_musgrave(4, SHD_MUSGRAVE_FBM, position,
                             musgrave_texture_w__Value, size_bumps_lf__Value,
                             2.0, 2.0, 2.0, 0.0, 1.0,
                             &Musgrave_SPACE_Texture__Fac);
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Musgrave_SPACE_Texture__Fac),
                            normal, float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            Vector_SPACE_Math__Vector,
                            float3_nonbuiltin(heig_bumps_lf__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_001__Vector, NULL);
    float noise_texture_w__Value;

    noise_texture_w__Value = noise_texture_w_FROM_geo_stone;
    float Noise_SPACE_Texture__Fac;
    node_shader_tex_noise(4, position, noise_texture_w__Value, 6.0, 16.0, 0.5,
                          0.0, &Noise_SPACE_Texture__Fac, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_002__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Noise_SPACE_Texture__Fac), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_002__Vector, NULL);
    float Value__Value;

    Value__Value = Value_FROM_geo_stone;
    float3_nonbuiltin Vector_SPACE_Math_DOT_003__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, Vector_SPACE_Math_DOT_002__Vector,
        float3_nonbuiltin(Value__Value), float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
        &Vector_SPACE_Math_DOT_003__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_004__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_001__Vector,
        Vector_SPACE_Math_DOT_003__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_004__Vector, NULL);
    float noise_texture_2_w__Value;

    noise_texture_2_w__Value = noise_texture_2_w_FROM_geo_stone;
    float Noise_SPACE_Texture_DOT_001__Fac;
    node_shader_tex_noise(4, position, noise_texture_2_w__Value, 2.5, 2.0, 0.5,
                          0.0, &Noise_SPACE_Texture_DOT_001__Fac, NULL);
    float4_nonbuiltin colorramp_2_VAR__Color;

    float colorramp_2_VAR_positions[2]{colorramp_2_VAR_FROM_geo_stone_pos0,
                                       colorramp_2_VAR_FROM_geo_stone_pos1};
    float4_nonbuiltin colorramp_2_VAR_colors[2]{
        colorramp_2_VAR_FROM_geo_stone_color0,
        colorramp_2_VAR_FROM_geo_stone_color1};
    node_texture_valToRgb(2, colorramp_2_VAR_positions, colorramp_2_VAR_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR,
                          COLBAND_HUE_NEAR, Noise_SPACE_Texture_DOT_001__Fac,
                          &colorramp_2_VAR__Color, NULL);
    float noise_texture_1_w__Value;

    noise_texture_1_w__Value = noise_texture_1_w_FROM_geo_stone;
    float4_nonbuiltin Noise_SPACE_Texture_DOT_002__Color;
    node_shader_tex_noise(4, position, noise_texture_1_w__Value, 0.5, 16.0, 0.5,
                          0.0, NULL, &Noise_SPACE_Texture_DOT_002__Color);
    float wave_texture_scale__Value;

    wave_texture_scale__Value = wave_texture_scale_FROM_geo_stone;
    float wave_texture_distortion__Value;

    wave_texture_distortion__Value = wave_texture_distortion_FROM_geo_stone;
    float wave_texture_detail__Value;

    wave_texture_detail__Value = wave_texture_detail_FROM_geo_stone;
    float Wave_SPACE_Texture__Fac;
    node_shader_tex_wave(SHD_WAVE_BANDS, SHD_WAVE_BANDS_DIRECTION_X,
                         SHD_WAVE_RINGS_DIRECTION_X, SHD_WAVE_PROFILE_SIN,
                         float3_nonbuiltin(Noise_SPACE_Texture_DOT_002__Color),
                         wave_texture_scale__Value,
                         wave_texture_distortion__Value,
                         wave_texture_detail__Value, 1.0, 0.5, 0.0, NULL,
                         &Wave_SPACE_Texture__Fac);
    float4_nonbuiltin colorramp_1_VAR__Color;

    float colorramp_1_VAR_positions[2]{colorramp_1_VAR_FROM_geo_stone_pos0,
                                       colorramp_1_VAR_FROM_geo_stone_pos1};
    float4_nonbuiltin colorramp_1_VAR_colors[2]{
        colorramp_1_VAR_FROM_geo_stone_color0,
        colorramp_1_VAR_FROM_geo_stone_color1};
    node_texture_valToRgb(2, colorramp_1_VAR_positions, colorramp_1_VAR_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR,
                          COLBAND_HUE_NEAR, Wave_SPACE_Texture__Fac,
                          &colorramp_1_VAR__Color, NULL);
    float4_nonbuiltin Mix__Color;
    node_shader_mix_rgb(0, MA_RAMP_MIX, float(colorramp_2_VAR__Color),
                        colorramp_1_VAR__Color,
                        float4_nonbuiltin(0.5, 0.5, 0.5, 1.0), &Mix__Color);
    float3_nonbuiltin Vector_SPACE_Math_DOT_005__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Mix__Color), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_005__Vector, NULL);
    float Value_DOT_001__Value;

    Value_DOT_001__Value = Value_DOT_001_FROM_geo_stone;
    float3_nonbuiltin Vector_SPACE_Math_DOT_006__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            Vector_SPACE_Math_DOT_005__Vector,
                            float3_nonbuiltin(Value_DOT_001__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_006__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_007__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_004__Vector,
        Vector_SPACE_Math_DOT_006__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_007__Vector, NULL);
    float Value_DOT_002__Value;

    Value_DOT_002__Value = Value_DOT_002_FROM_geo_stone;
    float3_nonbuiltin Vector_SPACE_Math_DOT_008__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            Vector_SPACE_Math_DOT_007__Vector,
                            float3_nonbuiltin(Value_DOT_002__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_008__Vector, NULL);
    float noise_texture_3_w__Value;

    noise_texture_3_w__Value = noise_texture_3_w_FROM_geo_stone;
    float noise_texture_3_scale__Value;

    noise_texture_3_scale__Value = noise_texture_3_scale_FROM_geo_stone;
    float Noise_SPACE_Texture_DOT_003__Fac;
    node_shader_tex_noise(4, position, noise_texture_3_w__Value,
                          noise_texture_3_scale__Value, 2.0, 0.5, 0.0,
                          &Noise_SPACE_Texture_DOT_003__Fac, NULL);
    float Math__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture_DOT_003__Fac,
                      0.5, 0.5, &Math__Value);
    float3_nonbuiltin Vector_SPACE_Math_DOT_009__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Math__Value), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_009__Vector, NULL);
    float Value_DOT_003__Value;

    Value_DOT_003__Value = Value_DOT_003_FROM_geo_stone;
    float3_nonbuiltin Vector_SPACE_Math_DOT_010__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            Vector_SPACE_Math_DOT_009__Vector,
                            float3_nonbuiltin(Value_DOT_003__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_010__Vector, NULL);
    float noise_texture_4_scale__Value;

    noise_texture_4_scale__Value = noise_texture_4_scale_FROM_geo_stone;
    float noise_texture_4_w__Value;

    noise_texture_4_w__Value = noise_texture_4_w_FROM_geo_stone;
    float Noise_SPACE_Texture_DOT_004__Fac;
    node_shader_tex_noise(4, position, noise_texture_4_w__Value,
                          noise_texture_4_scale__Value, 2.0, 0.5, 0.0,
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
    float3_nonbuiltin Vector_SPACE_Math_DOT_011__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Math_DOT_001__Value), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_011__Vector, NULL);
    float Value_DOT_004__Value;

    Value_DOT_004__Value = Value_DOT_004_FROM_geo_stone;
    float3_nonbuiltin Vector_SPACE_Math_DOT_012__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            Vector_SPACE_Math_DOT_011__Vector,
                            float3_nonbuiltin(Value_DOT_004__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_012__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_013__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_008__Vector,
        Vector_SPACE_Math_DOT_012__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_013__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_014__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_010__Vector,
        Vector_SPACE_Math_DOT_013__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_014__Vector, NULL);
    float scale0__Value;

    scale0__Value = scale0_FROM_geo_stone;
    float detail0__Value;

    detail0__Value = detail0_FROM_geo_stone;
    float roughness0__Value;

    roughness0__Value = roughness0_FROM_geo_stone;
    float zscale0__Value;

    zscale0__Value = zscale0_FROM_geo_stone;
    float3_nonbuiltin position_shift0__Vector;

    position_shift0__Vector = position_shift0_FROM_geo_stone;
    float3_nonbuiltin Vector_SPACE_Math_DOT_015__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift0__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_015__Vector, NULL);
    float Noise_SPACE_Texture_DOT_005__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_015__Vector, 0.0,
                          scale0__Value, detail0__Value, roughness0__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_005__Fac, NULL);
    float Math_DOT_002__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture_DOT_005__Fac,
                      0.5, 0.5, &Math_DOT_002__Value);
    float Math_DOT_003__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math_DOT_002__Value,
                      zscale0__Value, 0.5, &Math_DOT_003__Value);
    float scale1__Value;

    scale1__Value = scale1_FROM_geo_stone;
    float detail1__Value;

    detail1__Value = detail1_FROM_geo_stone;
    float roughness1__Value;

    roughness1__Value = roughness1_FROM_geo_stone;
    float zscale1__Value;

    zscale1__Value = zscale1_FROM_geo_stone;
    float3_nonbuiltin position_shift1__Vector;

    position_shift1__Vector = position_shift1_FROM_geo_stone;
    float3_nonbuiltin Vector_SPACE_Math_DOT_016__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift1__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_016__Vector, NULL);
    float Noise_SPACE_Texture_DOT_006__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_016__Vector, 0.0,
                          scale1__Value, detail1__Value, roughness1__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_006__Fac, NULL);
    float Math_DOT_004__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture_DOT_006__Fac,
                      0.5, 0.5, &Math_DOT_004__Value);
    float Math_DOT_005__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math_DOT_004__Value,
                      zscale1__Value, 0.5, &Math_DOT_005__Value);
    float scale2__Value;

    scale2__Value = scale2_FROM_geo_stone;
    float detail2__Value;

    detail2__Value = detail2_FROM_geo_stone;
    float roughness2__Value;

    roughness2__Value = roughness2_FROM_geo_stone;
    float zscale2__Value;

    zscale2__Value = zscale2_FROM_geo_stone;
    float3_nonbuiltin position_shift2__Vector;

    position_shift2__Vector = position_shift2_FROM_geo_stone;
    float3_nonbuiltin Vector_SPACE_Math_DOT_017__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift2__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_017__Vector, NULL);
    float Noise_SPACE_Texture_DOT_007__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_017__Vector, 0.0,
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
    float3_nonbuiltin Vector_SPACE_Math_DOT_018__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Math_DOT_009__Value), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_018__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_019__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_014__Vector,
        Vector_SPACE_Math_DOT_018__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_019__Vector, NULL);

    if (offset != NULL)
        *offset = Vector_SPACE_Math_DOT_019__Vector;
}
