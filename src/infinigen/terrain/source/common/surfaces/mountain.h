// Code generated using version 1.0.0 of infinigen/tools/kernelize_surfaces.py;
// refer to infinigen/assets/materials/mountain.py which has the copyright and
// authors
DEVICE_FUNC void
geo_MOUNTAIN(float3_nonbuiltin position, float3_nonbuiltin normal,
             POINTER_OR_REFERENCE_ARG float *float_vars,
             POINTER_OR_REFERENCE_ARG float3_nonbuiltin *float3_nonbuiltin_vars,
             POINTER_OR_REFERENCE_ARG float3_nonbuiltin *offset) {
    float crack_mask_rampmax0_FROM_geo_MOUNTAIN = float_vars[0];
    float crack_mask_rampmax1_FROM_geo_MOUNTAIN = float_vars[1];
    float crack_mask_rampmax2_FROM_geo_MOUNTAIN = float_vars[2];
    float crack_mask_rampmax3_FROM_geo_MOUNTAIN = float_vars[3];
    float crack_mask_rampmax4_FROM_geo_MOUNTAIN = float_vars[4];
    float crack_mask_rampmax5_FROM_geo_MOUNTAIN = float_vars[5];
    float crack_mask_rampmax6_FROM_geo_MOUNTAIN = float_vars[6];
    float crack_mask_rampmax7_FROM_geo_MOUNTAIN = float_vars[7];
    float crack_mask_rampmin0_FROM_geo_MOUNTAIN = float_vars[8];
    float crack_mask_rampmin1_FROM_geo_MOUNTAIN = float_vars[9];
    float crack_mask_rampmin2_FROM_geo_MOUNTAIN = float_vars[10];
    float crack_mask_rampmin3_FROM_geo_MOUNTAIN = float_vars[11];
    float crack_mask_rampmin4_FROM_geo_MOUNTAIN = float_vars[12];
    float crack_mask_rampmin5_FROM_geo_MOUNTAIN = float_vars[13];
    float crack_mask_rampmin6_FROM_geo_MOUNTAIN = float_vars[14];
    float crack_mask_rampmin7_FROM_geo_MOUNTAIN = float_vars[15];
    float crack_modulation_detail0_FROM_geo_MOUNTAIN = float_vars[16];
    float crack_modulation_detail1_FROM_geo_MOUNTAIN = float_vars[17];
    float crack_modulation_detail2_FROM_geo_MOUNTAIN = float_vars[18];
    float crack_modulation_detail3_FROM_geo_MOUNTAIN = float_vars[19];
    float crack_modulation_detail4_FROM_geo_MOUNTAIN = float_vars[20];
    float crack_modulation_detail5_FROM_geo_MOUNTAIN = float_vars[21];
    float crack_modulation_detail6_FROM_geo_MOUNTAIN = float_vars[22];
    float crack_modulation_detail7_FROM_geo_MOUNTAIN = float_vars[23];
    float crack_modulation_roughness0_FROM_geo_MOUNTAIN = float_vars[24];
    float crack_modulation_roughness1_FROM_geo_MOUNTAIN = float_vars[25];
    float crack_modulation_roughness2_FROM_geo_MOUNTAIN = float_vars[26];
    float crack_modulation_roughness3_FROM_geo_MOUNTAIN = float_vars[27];
    float crack_modulation_roughness4_FROM_geo_MOUNTAIN = float_vars[28];
    float crack_modulation_roughness5_FROM_geo_MOUNTAIN = float_vars[29];
    float crack_modulation_roughness6_FROM_geo_MOUNTAIN = float_vars[30];
    float crack_modulation_roughness7_FROM_geo_MOUNTAIN = float_vars[31];
    float crack_modulation_scale0_FROM_geo_MOUNTAIN = float_vars[32];
    float crack_modulation_scale1_FROM_geo_MOUNTAIN = float_vars[33];
    float crack_modulation_scale2_FROM_geo_MOUNTAIN = float_vars[34];
    float crack_modulation_scale3_FROM_geo_MOUNTAIN = float_vars[35];
    float crack_modulation_scale4_FROM_geo_MOUNTAIN = float_vars[36];
    float crack_modulation_scale5_FROM_geo_MOUNTAIN = float_vars[37];
    float crack_modulation_scale6_FROM_geo_MOUNTAIN = float_vars[38];
    float crack_modulation_scale7_FROM_geo_MOUNTAIN = float_vars[39];
    float crack_scale0_FROM_geo_MOUNTAIN = float_vars[40];
    float crack_scale1_FROM_geo_MOUNTAIN = float_vars[41];
    float crack_scale2_FROM_geo_MOUNTAIN = float_vars[42];
    float crack_scale3_FROM_geo_MOUNTAIN = float_vars[43];
    float crack_scale4_FROM_geo_MOUNTAIN = float_vars[44];
    float crack_scale5_FROM_geo_MOUNTAIN = float_vars[45];
    float crack_scale6_FROM_geo_MOUNTAIN = float_vars[46];
    float crack_scale7_FROM_geo_MOUNTAIN = float_vars[47];
    float crack_slope_base0_FROM_geo_MOUNTAIN = float_vars[48];
    float crack_slope_base1_FROM_geo_MOUNTAIN = float_vars[49];
    float crack_slope_base2_FROM_geo_MOUNTAIN = float_vars[50];
    float crack_slope_base3_FROM_geo_MOUNTAIN = float_vars[51];
    float crack_slope_base4_FROM_geo_MOUNTAIN = float_vars[52];
    float crack_slope_base5_FROM_geo_MOUNTAIN = float_vars[53];
    float crack_slope_base6_FROM_geo_MOUNTAIN = float_vars[54];
    float crack_slope_base7_FROM_geo_MOUNTAIN = float_vars[55];
    float crack_slope_scale0_FROM_geo_MOUNTAIN = float_vars[56];
    float crack_slope_scale1_FROM_geo_MOUNTAIN = float_vars[57];
    float crack_slope_scale2_FROM_geo_MOUNTAIN = float_vars[58];
    float crack_slope_scale3_FROM_geo_MOUNTAIN = float_vars[59];
    float crack_slope_scale4_FROM_geo_MOUNTAIN = float_vars[60];
    float crack_slope_scale5_FROM_geo_MOUNTAIN = float_vars[61];
    float crack_slope_scale6_FROM_geo_MOUNTAIN = float_vars[62];
    float crack_slope_scale7_FROM_geo_MOUNTAIN = float_vars[63];
    float crack_zscale_scale0_FROM_geo_MOUNTAIN = float_vars[64];
    float crack_zscale_scale1_FROM_geo_MOUNTAIN = float_vars[65];
    float crack_zscale_scale2_FROM_geo_MOUNTAIN = float_vars[66];
    float crack_zscale_scale3_FROM_geo_MOUNTAIN = float_vars[67];
    float crack_zscale_scale4_FROM_geo_MOUNTAIN = float_vars[68];
    float crack_zscale_scale5_FROM_geo_MOUNTAIN = float_vars[69];
    float crack_zscale_scale6_FROM_geo_MOUNTAIN = float_vars[70];
    float crack_zscale_scale7_FROM_geo_MOUNTAIN = float_vars[71];
    float detail0_FROM_geo_MOUNTAIN = float_vars[72];
    float detail1_FROM_geo_MOUNTAIN = float_vars[73];
    float detail2_FROM_geo_MOUNTAIN = float_vars[74];
    float3_nonbuiltin position_shift0_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[0];
    float3_nonbuiltin position_shift1_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[1];
    float3_nonbuiltin position_shift2_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[2];
    float3_nonbuiltin position_shift_crack0_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[3];
    float3_nonbuiltin position_shift_crack1_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[4];
    float3_nonbuiltin position_shift_crack2_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[5];
    float3_nonbuiltin position_shift_crack3_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[6];
    float3_nonbuiltin position_shift_crack4_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[7];
    float3_nonbuiltin position_shift_crack5_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[8];
    float3_nonbuiltin position_shift_crack6_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[9];
    float3_nonbuiltin position_shift_crack7_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[10];
    float3_nonbuiltin position_shift_mask0_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[11];
    float3_nonbuiltin position_shift_mask1_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[12];
    float3_nonbuiltin position_shift_mask2_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[13];
    float3_nonbuiltin position_shift_mask3_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[14];
    float3_nonbuiltin position_shift_mask4_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[15];
    float3_nonbuiltin position_shift_mask5_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[16];
    float3_nonbuiltin position_shift_mask6_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[17];
    float3_nonbuiltin position_shift_mask7_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[18];
    float3_nonbuiltin position_shift_slope_modulation0_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[19];
    float3_nonbuiltin position_shift_slope_modulation1_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[20];
    float3_nonbuiltin position_shift_slope_modulation2_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[21];
    float3_nonbuiltin position_shift_slope_modulation3_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[22];
    float3_nonbuiltin position_shift_slope_modulation4_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[23];
    float3_nonbuiltin position_shift_slope_modulation5_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[24];
    float3_nonbuiltin position_shift_slope_modulation6_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[25];
    float3_nonbuiltin position_shift_slope_modulation7_FROM_geo_MOUNTAIN =
        float3_nonbuiltin_vars[26];
    float roughness0_FROM_geo_MOUNTAIN = float_vars[75];
    float roughness1_FROM_geo_MOUNTAIN = float_vars[76];
    float roughness2_FROM_geo_MOUNTAIN = float_vars[77];
    float scale0_FROM_geo_MOUNTAIN = float_vars[78];
    float scale1_FROM_geo_MOUNTAIN = float_vars[79];
    float scale2_FROM_geo_MOUNTAIN = float_vars[80];
    float zscale0_FROM_geo_MOUNTAIN = float_vars[81];
    float zscale1_FROM_geo_MOUNTAIN = float_vars[82];
    float zscale2_FROM_geo_MOUNTAIN = float_vars[83];
    float scale0__Value;

    scale0__Value = scale0_FROM_geo_MOUNTAIN;
    float detail0__Value;

    detail0__Value = detail0_FROM_geo_MOUNTAIN;
    float roughness0__Value;

    roughness0__Value = roughness0_FROM_geo_MOUNTAIN;
    float zscale0__Value;

    zscale0__Value = zscale0_FROM_geo_MOUNTAIN;
    float3_nonbuiltin position_shift0__Vector;

    position_shift0__Vector = position_shift0_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift0__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);
    float Noise_SPACE_Texture__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math__Vector, 0.0, scale0__Value,
                          detail0__Value, roughness0__Value, 0.0,
                          &Noise_SPACE_Texture__Fac, NULL);
    float Math__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture__Fac, 0.5, 0.5,
                      &Math__Value);
    float Math_DOT_001__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math__Value, zscale0__Value, 0.5,
                      &Math_DOT_001__Value);
    float scale1__Value;

    scale1__Value = scale1_FROM_geo_MOUNTAIN;
    float detail1__Value;

    detail1__Value = detail1_FROM_geo_MOUNTAIN;
    float roughness1__Value;

    roughness1__Value = roughness1_FROM_geo_MOUNTAIN;
    float zscale1__Value;

    zscale1__Value = zscale1_FROM_geo_MOUNTAIN;
    float3_nonbuiltin position_shift1__Vector;

    position_shift1__Vector = position_shift1_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift1__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_001__Vector, NULL);
    float Noise_SPACE_Texture_DOT_001__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_001__Vector, 0.0,
                          scale1__Value, detail1__Value, roughness1__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_001__Fac, NULL);
    float Math_DOT_002__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture_DOT_001__Fac,
                      0.5, 0.5, &Math_DOT_002__Value);
    float Math_DOT_003__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math_DOT_002__Value,
                      zscale1__Value, 0.5, &Math_DOT_003__Value);
    float scale2__Value;

    scale2__Value = scale2_FROM_geo_MOUNTAIN;
    float detail2__Value;

    detail2__Value = detail2_FROM_geo_MOUNTAIN;
    float roughness2__Value;

    roughness2__Value = roughness2_FROM_geo_MOUNTAIN;
    float zscale2__Value;

    zscale2__Value = zscale2_FROM_geo_MOUNTAIN;
    float3_nonbuiltin position_shift2__Vector;

    position_shift2__Vector = position_shift2_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_002__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift2__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_002__Vector, NULL);
    float Noise_SPACE_Texture_DOT_002__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_002__Vector, 0.0,
                          scale2__Value, detail2__Value, roughness2__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_002__Fac, NULL);
    float Math_DOT_004__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture_DOT_002__Fac,
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
    float crack_modulation_scale0__Value;

    crack_modulation_scale0__Value = crack_modulation_scale0_FROM_geo_MOUNTAIN;
    float crack_modulation_detail0__Value;

    crack_modulation_detail0__Value =
        crack_modulation_detail0_FROM_geo_MOUNTAIN;
    float crack_modulation_roughness0__Value;

    crack_modulation_roughness0__Value =
        crack_modulation_roughness0_FROM_geo_MOUNTAIN;
    float3_nonbuiltin position_shift_mask0__Vector;

    position_shift_mask0__Vector = position_shift_mask0_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_003__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_mask0__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_003__Vector, NULL);
    float Noise_SPACE_Texture_DOT_003__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_003__Vector, 0.0,
                          crack_modulation_scale0__Value,
                          crack_modulation_detail0__Value,
                          crack_modulation_roughness0__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_003__Fac, NULL);
    float3_nonbuiltin position_shift_slope_modulation0__Vector;

    position_shift_slope_modulation0__Vector =
        position_shift_slope_modulation0_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_004__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_slope_modulation0__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_004__Vector, NULL);
    float Noise_SPACE_Texture_DOT_004__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_004__Vector, 0.0,
                          crack_modulation_scale0__Value,
                          crack_modulation_detail0__Value,
                          crack_modulation_roughness0__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_004__Fac, NULL);
    float Map_SPACE_Range__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Noise_SPACE_Texture_DOT_004__Fac,
        0.44999998807907104, 0.550000011920929, 0.0, 1.0, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range__Result, NULL);
    float crack_scale0__Value;

    crack_scale0__Value = crack_scale0_FROM_geo_MOUNTAIN;
    float crack_zscale_scale0__Value;

    crack_zscale_scale0__Value = crack_zscale_scale0_FROM_geo_MOUNTAIN;
    float crack_slope_base0__Value;

    crack_slope_base0__Value = crack_slope_base0_FROM_geo_MOUNTAIN;
    float crack_slope_scale0__Value;

    crack_slope_scale0__Value = crack_slope_scale0_FROM_geo_MOUNTAIN;
    float crack_mask_rampmin0__Value;

    crack_mask_rampmin0__Value = crack_mask_rampmin0_FROM_geo_MOUNTAIN;
    float crack_mask_rampmax0__Value;

    crack_mask_rampmax0__Value = crack_mask_rampmax0_FROM_geo_MOUNTAIN;
    float Math_DOT_008__Value;
    node_texture_math(NODE_MATH_ADD, 0, crack_mask_rampmin0__Value, 0.5, 0.5,
                      &Math_DOT_008__Value);
    float Math_DOT_009__Value;
    node_texture_math(NODE_MATH_ADD, 0, crack_mask_rampmax0__Value, 0.5, 0.5,
                      &Math_DOT_009__Value);
    float Map_SPACE_Range_DOT_001__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Noise_SPACE_Texture_DOT_003__Fac,
        Math_DOT_008__Value, Math_DOT_009__Value, 0.0, 1.0, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range_DOT_001__Result, NULL);
    float Math_DOT_010__Value;
    node_texture_math(NODE_MATH_POWER, 0, crack_slope_base0__Value,
                      Map_SPACE_Range__Result, 0.5, &Math_DOT_010__Value);
    float Math_DOT_011__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, crack_zscale_scale0__Value,
                      Math_DOT_010__Value, 0.5, &Math_DOT_011__Value);
    float Math_DOT_012__Value;
    node_texture_math(NODE_MATH_DIVIDE, 0, 1.0, crack_slope_scale0__Value, 0.5,
                      &Math_DOT_012__Value);
    float Math_DOT_013__Value;
    node_texture_math(NODE_MATH_DIVIDE, 0, 1.0, crack_slope_base0__Value, 0.5,
                      &Math_DOT_013__Value);
    float Math_DOT_014__Value;
    node_texture_math(NODE_MATH_POWER, 0, Math_DOT_013__Value,
                      Map_SPACE_Range__Result, 0.5, &Math_DOT_014__Value);
    float Math_DOT_015__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math_DOT_012__Value,
                      Math_DOT_014__Value, 0.5, &Math_DOT_015__Value);
    float3_nonbuiltin position_shift_crack0__Vector;

    position_shift_crack0__Vector = position_shift_crack0_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_005__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_crack0__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_005__Vector, NULL);
    float Voronoi_SPACE_Texture__Distance;
    node_shader_tex_voronoi(
        3, SHD_VORONOI_DISTANCE_TO_EDGE, SHD_VORONOI_EUCLIDEAN,
        Vector_SPACE_Math_DOT_005__Vector, 0.0, crack_scale0__Value, 1.0, 0.5,
        1.0, &Voronoi_SPACE_Texture__Distance, NULL, NULL, NULL, NULL);
    float Map_SPACE_Range_DOT_002__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Voronoi_SPACE_Texture__Distance, 0.0,
        Math_DOT_015__Value, -1.0, 0.0, 4.0, float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(4.0, 4.0, 4.0), &Map_SPACE_Range_DOT_002__Result,
        NULL);
    float Math_DOT_016__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Map_SPACE_Range_DOT_001__Result,
                      Math_DOT_011__Value, 0.5, &Math_DOT_016__Value);
    float Math_DOT_017__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Map_SPACE_Range_DOT_002__Result,
                      Math_DOT_016__Value, 0.5, &Math_DOT_017__Value);
    float crack_modulation_scale1__Value;

    crack_modulation_scale1__Value = crack_modulation_scale1_FROM_geo_MOUNTAIN;
    float crack_modulation_detail1__Value;

    crack_modulation_detail1__Value =
        crack_modulation_detail1_FROM_geo_MOUNTAIN;
    float crack_modulation_roughness1__Value;

    crack_modulation_roughness1__Value =
        crack_modulation_roughness1_FROM_geo_MOUNTAIN;
    float3_nonbuiltin position_shift_mask1__Vector;

    position_shift_mask1__Vector = position_shift_mask1_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_006__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_mask1__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_006__Vector, NULL);
    float Noise_SPACE_Texture_DOT_005__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_006__Vector, 0.0,
                          crack_modulation_scale1__Value,
                          crack_modulation_detail1__Value,
                          crack_modulation_roughness1__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_005__Fac, NULL);
    float3_nonbuiltin position_shift_slope_modulation1__Vector;

    position_shift_slope_modulation1__Vector =
        position_shift_slope_modulation1_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_007__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_slope_modulation1__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_007__Vector, NULL);
    float Noise_SPACE_Texture_DOT_006__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_007__Vector, 0.0,
                          crack_modulation_scale1__Value,
                          crack_modulation_detail1__Value,
                          crack_modulation_roughness1__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_006__Fac, NULL);
    float Map_SPACE_Range_DOT_003__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Noise_SPACE_Texture_DOT_006__Fac,
        0.44999998807907104, 0.550000011920929, 0.0, 1.0, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range_DOT_003__Result, NULL);
    float crack_scale1__Value;

    crack_scale1__Value = crack_scale1_FROM_geo_MOUNTAIN;
    float crack_zscale_scale1__Value;

    crack_zscale_scale1__Value = crack_zscale_scale1_FROM_geo_MOUNTAIN;
    float crack_slope_base1__Value;

    crack_slope_base1__Value = crack_slope_base1_FROM_geo_MOUNTAIN;
    float crack_slope_scale1__Value;

    crack_slope_scale1__Value = crack_slope_scale1_FROM_geo_MOUNTAIN;
    float crack_mask_rampmin1__Value;

    crack_mask_rampmin1__Value = crack_mask_rampmin1_FROM_geo_MOUNTAIN;
    float crack_mask_rampmax1__Value;

    crack_mask_rampmax1__Value = crack_mask_rampmax1_FROM_geo_MOUNTAIN;
    float Math_DOT_018__Value;
    node_texture_math(NODE_MATH_ADD, 0, crack_mask_rampmin1__Value, 0.5, 0.5,
                      &Math_DOT_018__Value);
    float Math_DOT_019__Value;
    node_texture_math(NODE_MATH_ADD, 0, crack_mask_rampmax1__Value, 0.5, 0.5,
                      &Math_DOT_019__Value);
    float Map_SPACE_Range_DOT_004__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Noise_SPACE_Texture_DOT_005__Fac,
        Math_DOT_018__Value, Math_DOT_019__Value, 0.0, 1.0, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range_DOT_004__Result, NULL);
    float Math_DOT_020__Value;
    node_texture_math(NODE_MATH_POWER, 0, crack_slope_base1__Value,
                      Map_SPACE_Range_DOT_003__Result, 0.5,
                      &Math_DOT_020__Value);
    float Math_DOT_021__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, crack_zscale_scale1__Value,
                      Math_DOT_020__Value, 0.5, &Math_DOT_021__Value);
    float Math_DOT_022__Value;
    node_texture_math(NODE_MATH_DIVIDE, 0, 1.0, crack_slope_scale1__Value, 0.5,
                      &Math_DOT_022__Value);
    float Math_DOT_023__Value;
    node_texture_math(NODE_MATH_DIVIDE, 0, 1.0, crack_slope_base1__Value, 0.5,
                      &Math_DOT_023__Value);
    float Math_DOT_024__Value;
    node_texture_math(NODE_MATH_POWER, 0, Math_DOT_023__Value,
                      Map_SPACE_Range_DOT_003__Result, 0.5,
                      &Math_DOT_024__Value);
    float Math_DOT_025__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math_DOT_022__Value,
                      Math_DOT_024__Value, 0.5, &Math_DOT_025__Value);
    float3_nonbuiltin position_shift_crack1__Vector;

    position_shift_crack1__Vector = position_shift_crack1_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_008__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_crack1__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_008__Vector, NULL);
    float Voronoi_SPACE_Texture_DOT_001__Distance;
    node_shader_tex_voronoi(
        3, SHD_VORONOI_DISTANCE_TO_EDGE, SHD_VORONOI_EUCLIDEAN,
        Vector_SPACE_Math_DOT_008__Vector, 0.0, crack_scale1__Value, 1.0, 0.5,
        1.0, &Voronoi_SPACE_Texture_DOT_001__Distance, NULL, NULL, NULL, NULL);
    float Map_SPACE_Range_DOT_005__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1,
        Voronoi_SPACE_Texture_DOT_001__Distance, 0.0, Math_DOT_025__Value, -1.0,
        0.0, 4.0, float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(4.0, 4.0, 4.0), &Map_SPACE_Range_DOT_005__Result,
        NULL);
    float Math_DOT_026__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Map_SPACE_Range_DOT_004__Result,
                      Math_DOT_021__Value, 0.5, &Math_DOT_026__Value);
    float Math_DOT_027__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Map_SPACE_Range_DOT_005__Result,
                      Math_DOT_026__Value, 0.5, &Math_DOT_027__Value);
    float crack_modulation_scale2__Value;

    crack_modulation_scale2__Value = crack_modulation_scale2_FROM_geo_MOUNTAIN;
    float crack_modulation_detail2__Value;

    crack_modulation_detail2__Value =
        crack_modulation_detail2_FROM_geo_MOUNTAIN;
    float crack_modulation_roughness2__Value;

    crack_modulation_roughness2__Value =
        crack_modulation_roughness2_FROM_geo_MOUNTAIN;
    float3_nonbuiltin position_shift_mask2__Vector;

    position_shift_mask2__Vector = position_shift_mask2_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_009__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_mask2__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_009__Vector, NULL);
    float Noise_SPACE_Texture_DOT_007__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_009__Vector, 0.0,
                          crack_modulation_scale2__Value,
                          crack_modulation_detail2__Value,
                          crack_modulation_roughness2__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_007__Fac, NULL);
    float3_nonbuiltin position_shift_slope_modulation2__Vector;

    position_shift_slope_modulation2__Vector =
        position_shift_slope_modulation2_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_010__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_slope_modulation2__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_010__Vector, NULL);
    float Noise_SPACE_Texture_DOT_008__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_010__Vector, 0.0,
                          crack_modulation_scale2__Value,
                          crack_modulation_detail2__Value,
                          crack_modulation_roughness2__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_008__Fac, NULL);
    float Map_SPACE_Range_DOT_006__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Noise_SPACE_Texture_DOT_008__Fac,
        0.44999998807907104, 0.550000011920929, 0.0, 1.0, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range_DOT_006__Result, NULL);
    float crack_scale2__Value;

    crack_scale2__Value = crack_scale2_FROM_geo_MOUNTAIN;
    float crack_zscale_scale2__Value;

    crack_zscale_scale2__Value = crack_zscale_scale2_FROM_geo_MOUNTAIN;
    float crack_slope_base2__Value;

    crack_slope_base2__Value = crack_slope_base2_FROM_geo_MOUNTAIN;
    float crack_slope_scale2__Value;

    crack_slope_scale2__Value = crack_slope_scale2_FROM_geo_MOUNTAIN;
    float crack_mask_rampmin2__Value;

    crack_mask_rampmin2__Value = crack_mask_rampmin2_FROM_geo_MOUNTAIN;
    float crack_mask_rampmax2__Value;

    crack_mask_rampmax2__Value = crack_mask_rampmax2_FROM_geo_MOUNTAIN;
    float Math_DOT_028__Value;
    node_texture_math(NODE_MATH_ADD, 0, crack_mask_rampmin2__Value, 0.5, 0.5,
                      &Math_DOT_028__Value);
    float Math_DOT_029__Value;
    node_texture_math(NODE_MATH_ADD, 0, crack_mask_rampmax2__Value, 0.5, 0.5,
                      &Math_DOT_029__Value);
    float Map_SPACE_Range_DOT_007__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Noise_SPACE_Texture_DOT_007__Fac,
        Math_DOT_028__Value, Math_DOT_029__Value, 0.0, 1.0, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range_DOT_007__Result, NULL);
    float Math_DOT_030__Value;
    node_texture_math(NODE_MATH_POWER, 0, crack_slope_base2__Value,
                      Map_SPACE_Range_DOT_006__Result, 0.5,
                      &Math_DOT_030__Value);
    float Math_DOT_031__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, crack_zscale_scale2__Value,
                      Math_DOT_030__Value, 0.5, &Math_DOT_031__Value);
    float Math_DOT_032__Value;
    node_texture_math(NODE_MATH_DIVIDE, 0, 1.0, crack_slope_scale2__Value, 0.5,
                      &Math_DOT_032__Value);
    float Math_DOT_033__Value;
    node_texture_math(NODE_MATH_DIVIDE, 0, 1.0, crack_slope_base2__Value, 0.5,
                      &Math_DOT_033__Value);
    float Math_DOT_034__Value;
    node_texture_math(NODE_MATH_POWER, 0, Math_DOT_033__Value,
                      Map_SPACE_Range_DOT_006__Result, 0.5,
                      &Math_DOT_034__Value);
    float Math_DOT_035__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math_DOT_032__Value,
                      Math_DOT_034__Value, 0.5, &Math_DOT_035__Value);
    float3_nonbuiltin position_shift_crack2__Vector;

    position_shift_crack2__Vector = position_shift_crack2_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_011__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_crack2__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_011__Vector, NULL);
    float Voronoi_SPACE_Texture_DOT_002__Distance;
    node_shader_tex_voronoi(
        3, SHD_VORONOI_DISTANCE_TO_EDGE, SHD_VORONOI_EUCLIDEAN,
        Vector_SPACE_Math_DOT_011__Vector, 0.0, crack_scale2__Value, 1.0, 0.5,
        1.0, &Voronoi_SPACE_Texture_DOT_002__Distance, NULL, NULL, NULL, NULL);
    float Map_SPACE_Range_DOT_008__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1,
        Voronoi_SPACE_Texture_DOT_002__Distance, 0.0, Math_DOT_035__Value, -1.0,
        0.0, 4.0, float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(4.0, 4.0, 4.0), &Map_SPACE_Range_DOT_008__Result,
        NULL);
    float Math_DOT_036__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Map_SPACE_Range_DOT_007__Result,
                      Math_DOT_031__Value, 0.5, &Math_DOT_036__Value);
    float Math_DOT_037__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Map_SPACE_Range_DOT_008__Result,
                      Math_DOT_036__Value, 0.5, &Math_DOT_037__Value);
    float crack_modulation_scale3__Value;

    crack_modulation_scale3__Value = crack_modulation_scale3_FROM_geo_MOUNTAIN;
    float crack_modulation_detail3__Value;

    crack_modulation_detail3__Value =
        crack_modulation_detail3_FROM_geo_MOUNTAIN;
    float crack_modulation_roughness3__Value;

    crack_modulation_roughness3__Value =
        crack_modulation_roughness3_FROM_geo_MOUNTAIN;
    float3_nonbuiltin position_shift_mask3__Vector;

    position_shift_mask3__Vector = position_shift_mask3_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_012__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_mask3__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_012__Vector, NULL);
    float Noise_SPACE_Texture_DOT_009__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_012__Vector, 0.0,
                          crack_modulation_scale3__Value,
                          crack_modulation_detail3__Value,
                          crack_modulation_roughness3__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_009__Fac, NULL);
    float3_nonbuiltin position_shift_slope_modulation3__Vector;

    position_shift_slope_modulation3__Vector =
        position_shift_slope_modulation3_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_013__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_slope_modulation3__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_013__Vector, NULL);
    float Noise_SPACE_Texture_DOT_010__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_013__Vector, 0.0,
                          crack_modulation_scale3__Value,
                          crack_modulation_detail3__Value,
                          crack_modulation_roughness3__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_010__Fac, NULL);
    float Map_SPACE_Range_DOT_009__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Noise_SPACE_Texture_DOT_010__Fac,
        0.44999998807907104, 0.550000011920929, 0.0, 1.0, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range_DOT_009__Result, NULL);
    float crack_scale3__Value;

    crack_scale3__Value = crack_scale3_FROM_geo_MOUNTAIN;
    float crack_zscale_scale3__Value;

    crack_zscale_scale3__Value = crack_zscale_scale3_FROM_geo_MOUNTAIN;
    float crack_slope_base3__Value;

    crack_slope_base3__Value = crack_slope_base3_FROM_geo_MOUNTAIN;
    float crack_slope_scale3__Value;

    crack_slope_scale3__Value = crack_slope_scale3_FROM_geo_MOUNTAIN;
    float crack_mask_rampmin3__Value;

    crack_mask_rampmin3__Value = crack_mask_rampmin3_FROM_geo_MOUNTAIN;
    float crack_mask_rampmax3__Value;

    crack_mask_rampmax3__Value = crack_mask_rampmax3_FROM_geo_MOUNTAIN;
    float Math_DOT_038__Value;
    node_texture_math(NODE_MATH_ADD, 0, crack_mask_rampmin3__Value, 0.5, 0.5,
                      &Math_DOT_038__Value);
    float Math_DOT_039__Value;
    node_texture_math(NODE_MATH_ADD, 0, crack_mask_rampmax3__Value, 0.5, 0.5,
                      &Math_DOT_039__Value);
    float Map_SPACE_Range_DOT_010__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Noise_SPACE_Texture_DOT_009__Fac,
        Math_DOT_038__Value, Math_DOT_039__Value, 0.0, 1.0, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range_DOT_010__Result, NULL);
    float Math_DOT_040__Value;
    node_texture_math(NODE_MATH_POWER, 0, crack_slope_base3__Value,
                      Map_SPACE_Range_DOT_009__Result, 0.5,
                      &Math_DOT_040__Value);
    float Math_DOT_041__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, crack_zscale_scale3__Value,
                      Math_DOT_040__Value, 0.5, &Math_DOT_041__Value);
    float Math_DOT_042__Value;
    node_texture_math(NODE_MATH_DIVIDE, 0, 1.0, crack_slope_scale3__Value, 0.5,
                      &Math_DOT_042__Value);
    float Math_DOT_043__Value;
    node_texture_math(NODE_MATH_DIVIDE, 0, 1.0, crack_slope_base3__Value, 0.5,
                      &Math_DOT_043__Value);
    float Math_DOT_044__Value;
    node_texture_math(NODE_MATH_POWER, 0, Math_DOT_043__Value,
                      Map_SPACE_Range_DOT_009__Result, 0.5,
                      &Math_DOT_044__Value);
    float Math_DOT_045__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math_DOT_042__Value,
                      Math_DOT_044__Value, 0.5, &Math_DOT_045__Value);
    float3_nonbuiltin position_shift_crack3__Vector;

    position_shift_crack3__Vector = position_shift_crack3_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_014__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_crack3__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_014__Vector, NULL);
    float Voronoi_SPACE_Texture_DOT_003__Distance;
    node_shader_tex_voronoi(
        3, SHD_VORONOI_DISTANCE_TO_EDGE, SHD_VORONOI_EUCLIDEAN,
        Vector_SPACE_Math_DOT_014__Vector, 0.0, crack_scale3__Value, 1.0, 0.5,
        1.0, &Voronoi_SPACE_Texture_DOT_003__Distance, NULL, NULL, NULL, NULL);
    float Map_SPACE_Range_DOT_011__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1,
        Voronoi_SPACE_Texture_DOT_003__Distance, 0.0, Math_DOT_045__Value, -1.0,
        0.0, 4.0, float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(4.0, 4.0, 4.0), &Map_SPACE_Range_DOT_011__Result,
        NULL);
    float Math_DOT_046__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Map_SPACE_Range_DOT_010__Result,
                      Math_DOT_041__Value, 0.5, &Math_DOT_046__Value);
    float Math_DOT_047__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Map_SPACE_Range_DOT_011__Result,
                      Math_DOT_046__Value, 0.5, &Math_DOT_047__Value);
    float crack_modulation_scale4__Value;

    crack_modulation_scale4__Value = crack_modulation_scale4_FROM_geo_MOUNTAIN;
    float crack_modulation_detail4__Value;

    crack_modulation_detail4__Value =
        crack_modulation_detail4_FROM_geo_MOUNTAIN;
    float crack_modulation_roughness4__Value;

    crack_modulation_roughness4__Value =
        crack_modulation_roughness4_FROM_geo_MOUNTAIN;
    float3_nonbuiltin position_shift_mask4__Vector;

    position_shift_mask4__Vector = position_shift_mask4_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_015__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_mask4__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_015__Vector, NULL);
    float Noise_SPACE_Texture_DOT_011__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_015__Vector, 0.0,
                          crack_modulation_scale4__Value,
                          crack_modulation_detail4__Value,
                          crack_modulation_roughness4__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_011__Fac, NULL);
    float3_nonbuiltin position_shift_slope_modulation4__Vector;

    position_shift_slope_modulation4__Vector =
        position_shift_slope_modulation4_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_016__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_slope_modulation4__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_016__Vector, NULL);
    float Noise_SPACE_Texture_DOT_012__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_016__Vector, 0.0,
                          crack_modulation_scale4__Value,
                          crack_modulation_detail4__Value,
                          crack_modulation_roughness4__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_012__Fac, NULL);
    float Map_SPACE_Range_DOT_012__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Noise_SPACE_Texture_DOT_012__Fac,
        0.44999998807907104, 0.550000011920929, 0.0, 1.0, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range_DOT_012__Result, NULL);
    float crack_scale4__Value;

    crack_scale4__Value = crack_scale4_FROM_geo_MOUNTAIN;
    float crack_zscale_scale4__Value;

    crack_zscale_scale4__Value = crack_zscale_scale4_FROM_geo_MOUNTAIN;
    float crack_slope_base4__Value;

    crack_slope_base4__Value = crack_slope_base4_FROM_geo_MOUNTAIN;
    float crack_slope_scale4__Value;

    crack_slope_scale4__Value = crack_slope_scale4_FROM_geo_MOUNTAIN;
    float crack_mask_rampmin4__Value;

    crack_mask_rampmin4__Value = crack_mask_rampmin4_FROM_geo_MOUNTAIN;
    float crack_mask_rampmax4__Value;

    crack_mask_rampmax4__Value = crack_mask_rampmax4_FROM_geo_MOUNTAIN;
    float Math_DOT_048__Value;
    node_texture_math(NODE_MATH_ADD, 0, crack_mask_rampmin4__Value, 0.5, 0.5,
                      &Math_DOT_048__Value);
    float Math_DOT_049__Value;
    node_texture_math(NODE_MATH_ADD, 0, crack_mask_rampmax4__Value, 0.5, 0.5,
                      &Math_DOT_049__Value);
    float Map_SPACE_Range_DOT_013__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Noise_SPACE_Texture_DOT_011__Fac,
        Math_DOT_048__Value, Math_DOT_049__Value, 0.0, 1.0, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range_DOT_013__Result, NULL);
    float Math_DOT_050__Value;
    node_texture_math(NODE_MATH_POWER, 0, crack_slope_base4__Value,
                      Map_SPACE_Range_DOT_012__Result, 0.5,
                      &Math_DOT_050__Value);
    float Math_DOT_051__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, crack_zscale_scale4__Value,
                      Math_DOT_050__Value, 0.5, &Math_DOT_051__Value);
    float Math_DOT_052__Value;
    node_texture_math(NODE_MATH_DIVIDE, 0, 1.0, crack_slope_scale4__Value, 0.5,
                      &Math_DOT_052__Value);
    float Math_DOT_053__Value;
    node_texture_math(NODE_MATH_DIVIDE, 0, 1.0, crack_slope_base4__Value, 0.5,
                      &Math_DOT_053__Value);
    float Math_DOT_054__Value;
    node_texture_math(NODE_MATH_POWER, 0, Math_DOT_053__Value,
                      Map_SPACE_Range_DOT_012__Result, 0.5,
                      &Math_DOT_054__Value);
    float Math_DOT_055__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math_DOT_052__Value,
                      Math_DOT_054__Value, 0.5, &Math_DOT_055__Value);
    float3_nonbuiltin position_shift_crack4__Vector;

    position_shift_crack4__Vector = position_shift_crack4_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_017__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_crack4__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_017__Vector, NULL);
    float Voronoi_SPACE_Texture_DOT_004__Distance;
    node_shader_tex_voronoi(
        3, SHD_VORONOI_DISTANCE_TO_EDGE, SHD_VORONOI_EUCLIDEAN,
        Vector_SPACE_Math_DOT_017__Vector, 0.0, crack_scale4__Value, 1.0, 0.5,
        1.0, &Voronoi_SPACE_Texture_DOT_004__Distance, NULL, NULL, NULL, NULL);
    float Map_SPACE_Range_DOT_014__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1,
        Voronoi_SPACE_Texture_DOT_004__Distance, 0.0, Math_DOT_055__Value, -1.0,
        0.0, 4.0, float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(4.0, 4.0, 4.0), &Map_SPACE_Range_DOT_014__Result,
        NULL);
    float Math_DOT_056__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Map_SPACE_Range_DOT_013__Result,
                      Math_DOT_051__Value, 0.5, &Math_DOT_056__Value);
    float Math_DOT_057__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Map_SPACE_Range_DOT_014__Result,
                      Math_DOT_056__Value, 0.5, &Math_DOT_057__Value);
    float crack_modulation_scale5__Value;

    crack_modulation_scale5__Value = crack_modulation_scale5_FROM_geo_MOUNTAIN;
    float crack_modulation_detail5__Value;

    crack_modulation_detail5__Value =
        crack_modulation_detail5_FROM_geo_MOUNTAIN;
    float crack_modulation_roughness5__Value;

    crack_modulation_roughness5__Value =
        crack_modulation_roughness5_FROM_geo_MOUNTAIN;
    float3_nonbuiltin position_shift_mask5__Vector;

    position_shift_mask5__Vector = position_shift_mask5_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_018__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_mask5__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_018__Vector, NULL);
    float Noise_SPACE_Texture_DOT_013__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_018__Vector, 0.0,
                          crack_modulation_scale5__Value,
                          crack_modulation_detail5__Value,
                          crack_modulation_roughness5__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_013__Fac, NULL);
    float3_nonbuiltin position_shift_slope_modulation5__Vector;

    position_shift_slope_modulation5__Vector =
        position_shift_slope_modulation5_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_019__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_slope_modulation5__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_019__Vector, NULL);
    float Noise_SPACE_Texture_DOT_014__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_019__Vector, 0.0,
                          crack_modulation_scale5__Value,
                          crack_modulation_detail5__Value,
                          crack_modulation_roughness5__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_014__Fac, NULL);
    float Map_SPACE_Range_DOT_015__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Noise_SPACE_Texture_DOT_014__Fac,
        0.44999998807907104, 0.550000011920929, 0.0, 1.0, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range_DOT_015__Result, NULL);
    float crack_scale5__Value;

    crack_scale5__Value = crack_scale5_FROM_geo_MOUNTAIN;
    float crack_zscale_scale5__Value;

    crack_zscale_scale5__Value = crack_zscale_scale5_FROM_geo_MOUNTAIN;
    float crack_slope_base5__Value;

    crack_slope_base5__Value = crack_slope_base5_FROM_geo_MOUNTAIN;
    float crack_slope_scale5__Value;

    crack_slope_scale5__Value = crack_slope_scale5_FROM_geo_MOUNTAIN;
    float crack_mask_rampmin5__Value;

    crack_mask_rampmin5__Value = crack_mask_rampmin5_FROM_geo_MOUNTAIN;
    float crack_mask_rampmax5__Value;

    crack_mask_rampmax5__Value = crack_mask_rampmax5_FROM_geo_MOUNTAIN;
    float Math_DOT_058__Value;
    node_texture_math(NODE_MATH_ADD, 0, crack_mask_rampmin5__Value, 0.5, 0.5,
                      &Math_DOT_058__Value);
    float Math_DOT_059__Value;
    node_texture_math(NODE_MATH_ADD, 0, crack_mask_rampmax5__Value, 0.5, 0.5,
                      &Math_DOT_059__Value);
    float Map_SPACE_Range_DOT_016__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Noise_SPACE_Texture_DOT_013__Fac,
        Math_DOT_058__Value, Math_DOT_059__Value, 0.0, 1.0, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range_DOT_016__Result, NULL);
    float Math_DOT_060__Value;
    node_texture_math(NODE_MATH_POWER, 0, crack_slope_base5__Value,
                      Map_SPACE_Range_DOT_015__Result, 0.5,
                      &Math_DOT_060__Value);
    float Math_DOT_061__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, crack_zscale_scale5__Value,
                      Math_DOT_060__Value, 0.5, &Math_DOT_061__Value);
    float Math_DOT_062__Value;
    node_texture_math(NODE_MATH_DIVIDE, 0, 1.0, crack_slope_scale5__Value, 0.5,
                      &Math_DOT_062__Value);
    float Math_DOT_063__Value;
    node_texture_math(NODE_MATH_DIVIDE, 0, 1.0, crack_slope_base5__Value, 0.5,
                      &Math_DOT_063__Value);
    float Math_DOT_064__Value;
    node_texture_math(NODE_MATH_POWER, 0, Math_DOT_063__Value,
                      Map_SPACE_Range_DOT_015__Result, 0.5,
                      &Math_DOT_064__Value);
    float Math_DOT_065__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math_DOT_062__Value,
                      Math_DOT_064__Value, 0.5, &Math_DOT_065__Value);
    float3_nonbuiltin position_shift_crack5__Vector;

    position_shift_crack5__Vector = position_shift_crack5_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_020__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_crack5__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_020__Vector, NULL);
    float Voronoi_SPACE_Texture_DOT_005__Distance;
    node_shader_tex_voronoi(
        3, SHD_VORONOI_DISTANCE_TO_EDGE, SHD_VORONOI_EUCLIDEAN,
        Vector_SPACE_Math_DOT_020__Vector, 0.0, crack_scale5__Value, 1.0, 0.5,
        1.0, &Voronoi_SPACE_Texture_DOT_005__Distance, NULL, NULL, NULL, NULL);
    float Map_SPACE_Range_DOT_017__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1,
        Voronoi_SPACE_Texture_DOT_005__Distance, 0.0, Math_DOT_065__Value, -1.0,
        0.0, 4.0, float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(4.0, 4.0, 4.0), &Map_SPACE_Range_DOT_017__Result,
        NULL);
    float Math_DOT_066__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Map_SPACE_Range_DOT_016__Result,
                      Math_DOT_061__Value, 0.5, &Math_DOT_066__Value);
    float Math_DOT_067__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Map_SPACE_Range_DOT_017__Result,
                      Math_DOT_066__Value, 0.5, &Math_DOT_067__Value);
    float crack_modulation_scale6__Value;

    crack_modulation_scale6__Value = crack_modulation_scale6_FROM_geo_MOUNTAIN;
    float crack_modulation_detail6__Value;

    crack_modulation_detail6__Value =
        crack_modulation_detail6_FROM_geo_MOUNTAIN;
    float crack_modulation_roughness6__Value;

    crack_modulation_roughness6__Value =
        crack_modulation_roughness6_FROM_geo_MOUNTAIN;
    float3_nonbuiltin position_shift_mask6__Vector;

    position_shift_mask6__Vector = position_shift_mask6_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_021__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_mask6__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_021__Vector, NULL);
    float Noise_SPACE_Texture_DOT_015__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_021__Vector, 0.0,
                          crack_modulation_scale6__Value,
                          crack_modulation_detail6__Value,
                          crack_modulation_roughness6__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_015__Fac, NULL);
    float3_nonbuiltin position_shift_slope_modulation6__Vector;

    position_shift_slope_modulation6__Vector =
        position_shift_slope_modulation6_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_022__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_slope_modulation6__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_022__Vector, NULL);
    float Noise_SPACE_Texture_DOT_016__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_022__Vector, 0.0,
                          crack_modulation_scale6__Value,
                          crack_modulation_detail6__Value,
                          crack_modulation_roughness6__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_016__Fac, NULL);
    float Map_SPACE_Range_DOT_018__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Noise_SPACE_Texture_DOT_016__Fac,
        0.44999998807907104, 0.550000011920929, 0.0, 1.0, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range_DOT_018__Result, NULL);
    float crack_scale6__Value;

    crack_scale6__Value = crack_scale6_FROM_geo_MOUNTAIN;
    float crack_zscale_scale6__Value;

    crack_zscale_scale6__Value = crack_zscale_scale6_FROM_geo_MOUNTAIN;
    float crack_slope_base6__Value;

    crack_slope_base6__Value = crack_slope_base6_FROM_geo_MOUNTAIN;
    float crack_slope_scale6__Value;

    crack_slope_scale6__Value = crack_slope_scale6_FROM_geo_MOUNTAIN;
    float crack_mask_rampmin6__Value;

    crack_mask_rampmin6__Value = crack_mask_rampmin6_FROM_geo_MOUNTAIN;
    float crack_mask_rampmax6__Value;

    crack_mask_rampmax6__Value = crack_mask_rampmax6_FROM_geo_MOUNTAIN;
    float Math_DOT_068__Value;
    node_texture_math(NODE_MATH_ADD, 0, crack_mask_rampmin6__Value, 0.5, 0.5,
                      &Math_DOT_068__Value);
    float Math_DOT_069__Value;
    node_texture_math(NODE_MATH_ADD, 0, crack_mask_rampmax6__Value, 0.5, 0.5,
                      &Math_DOT_069__Value);
    float Map_SPACE_Range_DOT_019__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Noise_SPACE_Texture_DOT_015__Fac,
        Math_DOT_068__Value, Math_DOT_069__Value, 0.0, 1.0, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range_DOT_019__Result, NULL);
    float Math_DOT_070__Value;
    node_texture_math(NODE_MATH_POWER, 0, crack_slope_base6__Value,
                      Map_SPACE_Range_DOT_018__Result, 0.5,
                      &Math_DOT_070__Value);
    float Math_DOT_071__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, crack_zscale_scale6__Value,
                      Math_DOT_070__Value, 0.5, &Math_DOT_071__Value);
    float Math_DOT_072__Value;
    node_texture_math(NODE_MATH_DIVIDE, 0, 1.0, crack_slope_scale6__Value, 0.5,
                      &Math_DOT_072__Value);
    float Math_DOT_073__Value;
    node_texture_math(NODE_MATH_DIVIDE, 0, 1.0, crack_slope_base6__Value, 0.5,
                      &Math_DOT_073__Value);
    float Math_DOT_074__Value;
    node_texture_math(NODE_MATH_POWER, 0, Math_DOT_073__Value,
                      Map_SPACE_Range_DOT_018__Result, 0.5,
                      &Math_DOT_074__Value);
    float Math_DOT_075__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math_DOT_072__Value,
                      Math_DOT_074__Value, 0.5, &Math_DOT_075__Value);
    float3_nonbuiltin position_shift_crack6__Vector;

    position_shift_crack6__Vector = position_shift_crack6_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_023__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_crack6__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_023__Vector, NULL);
    float Voronoi_SPACE_Texture_DOT_006__Distance;
    node_shader_tex_voronoi(
        3, SHD_VORONOI_DISTANCE_TO_EDGE, SHD_VORONOI_EUCLIDEAN,
        Vector_SPACE_Math_DOT_023__Vector, 0.0, crack_scale6__Value, 1.0, 0.5,
        1.0, &Voronoi_SPACE_Texture_DOT_006__Distance, NULL, NULL, NULL, NULL);
    float Map_SPACE_Range_DOT_020__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1,
        Voronoi_SPACE_Texture_DOT_006__Distance, 0.0, Math_DOT_075__Value, -1.0,
        0.0, 4.0, float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(4.0, 4.0, 4.0), &Map_SPACE_Range_DOT_020__Result,
        NULL);
    float Math_DOT_076__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Map_SPACE_Range_DOT_019__Result,
                      Math_DOT_071__Value, 0.5, &Math_DOT_076__Value);
    float Math_DOT_077__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Map_SPACE_Range_DOT_020__Result,
                      Math_DOT_076__Value, 0.5, &Math_DOT_077__Value);
    float crack_modulation_scale7__Value;

    crack_modulation_scale7__Value = crack_modulation_scale7_FROM_geo_MOUNTAIN;
    float crack_modulation_detail7__Value;

    crack_modulation_detail7__Value =
        crack_modulation_detail7_FROM_geo_MOUNTAIN;
    float crack_modulation_roughness7__Value;

    crack_modulation_roughness7__Value =
        crack_modulation_roughness7_FROM_geo_MOUNTAIN;
    float3_nonbuiltin position_shift_mask7__Vector;

    position_shift_mask7__Vector = position_shift_mask7_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_024__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_mask7__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_024__Vector, NULL);
    float Noise_SPACE_Texture_DOT_017__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_024__Vector, 0.0,
                          crack_modulation_scale7__Value,
                          crack_modulation_detail7__Value,
                          crack_modulation_roughness7__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_017__Fac, NULL);
    float3_nonbuiltin position_shift_slope_modulation7__Vector;

    position_shift_slope_modulation7__Vector =
        position_shift_slope_modulation7_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_025__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_slope_modulation7__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_025__Vector, NULL);
    float Noise_SPACE_Texture_DOT_018__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_025__Vector, 0.0,
                          crack_modulation_scale7__Value,
                          crack_modulation_detail7__Value,
                          crack_modulation_roughness7__Value, 0.0,
                          &Noise_SPACE_Texture_DOT_018__Fac, NULL);
    float Map_SPACE_Range_DOT_021__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Noise_SPACE_Texture_DOT_018__Fac,
        0.44999998807907104, 0.550000011920929, 0.0, 1.0, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range_DOT_021__Result, NULL);
    float crack_scale7__Value;

    crack_scale7__Value = crack_scale7_FROM_geo_MOUNTAIN;
    float crack_zscale_scale7__Value;

    crack_zscale_scale7__Value = crack_zscale_scale7_FROM_geo_MOUNTAIN;
    float crack_slope_base7__Value;

    crack_slope_base7__Value = crack_slope_base7_FROM_geo_MOUNTAIN;
    float crack_slope_scale7__Value;

    crack_slope_scale7__Value = crack_slope_scale7_FROM_geo_MOUNTAIN;
    float crack_mask_rampmin7__Value;

    crack_mask_rampmin7__Value = crack_mask_rampmin7_FROM_geo_MOUNTAIN;
    float crack_mask_rampmax7__Value;

    crack_mask_rampmax7__Value = crack_mask_rampmax7_FROM_geo_MOUNTAIN;
    float Math_DOT_078__Value;
    node_texture_math(NODE_MATH_ADD, 0, crack_mask_rampmin7__Value, 0.5, 0.5,
                      &Math_DOT_078__Value);
    float Math_DOT_079__Value;
    node_texture_math(NODE_MATH_ADD, 0, crack_mask_rampmax7__Value, 0.5, 0.5,
                      &Math_DOT_079__Value);
    float Map_SPACE_Range_DOT_022__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Noise_SPACE_Texture_DOT_017__Fac,
        Math_DOT_078__Value, Math_DOT_079__Value, 0.0, 1.0, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range_DOT_022__Result, NULL);
    float Math_DOT_080__Value;
    node_texture_math(NODE_MATH_POWER, 0, crack_slope_base7__Value,
                      Map_SPACE_Range_DOT_021__Result, 0.5,
                      &Math_DOT_080__Value);
    float Math_DOT_081__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, crack_zscale_scale7__Value,
                      Math_DOT_080__Value, 0.5, &Math_DOT_081__Value);
    float Math_DOT_082__Value;
    node_texture_math(NODE_MATH_DIVIDE, 0, 1.0, crack_slope_scale7__Value, 0.5,
                      &Math_DOT_082__Value);
    float Math_DOT_083__Value;
    node_texture_math(NODE_MATH_DIVIDE, 0, 1.0, crack_slope_base7__Value, 0.5,
                      &Math_DOT_083__Value);
    float Math_DOT_084__Value;
    node_texture_math(NODE_MATH_POWER, 0, Math_DOT_083__Value,
                      Map_SPACE_Range_DOT_021__Result, 0.5,
                      &Math_DOT_084__Value);
    float Math_DOT_085__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Math_DOT_082__Value,
                      Math_DOT_084__Value, 0.5, &Math_DOT_085__Value);
    float3_nonbuiltin position_shift_crack7__Vector;

    position_shift_crack7__Vector = position_shift_crack7_FROM_geo_MOUNTAIN;
    float3_nonbuiltin Vector_SPACE_Math_DOT_026__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_crack7__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_026__Vector, NULL);
    float Voronoi_SPACE_Texture_DOT_007__Distance;
    node_shader_tex_voronoi(
        3, SHD_VORONOI_DISTANCE_TO_EDGE, SHD_VORONOI_EUCLIDEAN,
        Vector_SPACE_Math_DOT_026__Vector, 0.0, crack_scale7__Value, 1.0, 0.5,
        1.0, &Voronoi_SPACE_Texture_DOT_007__Distance, NULL, NULL, NULL, NULL);
    float Map_SPACE_Range_DOT_023__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1,
        Voronoi_SPACE_Texture_DOT_007__Distance, 0.0, Math_DOT_085__Value, -1.0,
        0.0, 4.0, float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(4.0, 4.0, 4.0), &Map_SPACE_Range_DOT_023__Result,
        NULL);
    float Math_DOT_086__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Map_SPACE_Range_DOT_022__Result,
                      Math_DOT_081__Value, 0.5, &Math_DOT_086__Value);
    float Math_DOT_087__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Map_SPACE_Range_DOT_023__Result,
                      Math_DOT_086__Value, 0.5, &Math_DOT_087__Value);
    float Math_DOT_088__Value;
    node_texture_math(NODE_MATH_ADD, 0, Math_DOT_077__Value,
                      Math_DOT_087__Value, 0.5, &Math_DOT_088__Value);
    float Math_DOT_089__Value;
    node_texture_math(NODE_MATH_ADD, 0, Math_DOT_067__Value,
                      Math_DOT_088__Value, 0.5, &Math_DOT_089__Value);
    float Math_DOT_090__Value;
    node_texture_math(NODE_MATH_ADD, 0, Math_DOT_057__Value,
                      Math_DOT_089__Value, 0.5, &Math_DOT_090__Value);
    float Math_DOT_091__Value;
    node_texture_math(NODE_MATH_ADD, 0, Math_DOT_047__Value,
                      Math_DOT_090__Value, 0.5, &Math_DOT_091__Value);
    float Math_DOT_092__Value;
    node_texture_math(NODE_MATH_ADD, 0, Math_DOT_037__Value,
                      Math_DOT_091__Value, 0.5, &Math_DOT_092__Value);
    float Math_DOT_093__Value;
    node_texture_math(NODE_MATH_ADD, 0, Math_DOT_027__Value,
                      Math_DOT_092__Value, 0.5, &Math_DOT_093__Value);
    float Math_DOT_094__Value;
    node_texture_math(NODE_MATH_ADD, 0, Math_DOT_017__Value,
                      Math_DOT_093__Value, 0.5, &Math_DOT_094__Value);
    float Math_DOT_095__Value;
    node_texture_math(NODE_MATH_ADD, 0, Math_DOT_007__Value,
                      Math_DOT_094__Value, 0.5, &Math_DOT_095__Value);
    float3_nonbuiltin Vector_SPACE_Math_DOT_027__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Math_DOT_095__Value), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_027__Vector, NULL);

    if (offset != NULL)
        *offset = Vector_SPACE_Math_DOT_027__Vector;
}
