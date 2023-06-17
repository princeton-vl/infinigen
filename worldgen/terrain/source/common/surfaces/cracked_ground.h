// Code generated using version 0.31.1 of worldgen/tools/kernelize_surfaces.py;
// refer to worldgen/surfaces/templates/cracked_ground.py which has the
// copyright and authors
DEVICE_FUNC void geo_cracked_ground(
    float3_nonbuiltin position, float3_nonbuiltin normal,
    POINTER_OR_REFERENCE_ARG float *float_vars,
    POINTER_OR_REFERENCE_ARG float4_nonbuiltin *float4_nonbuiltin_vars,
    POINTER_OR_REFERENCE_ARG float3_nonbuiltin *offset) {
    float Value_DOT_001_FROM_geo_cracked_ground = float_vars[0];
    float Value_DOT_002_FROM_geo_cracked_ground = float_vars[1];
    float Value_DOT_003_FROM_geo_cracked_ground = float_vars[2];
    float Value_DOT_004_FROM_geo_cracked_ground = float_vars[3];
    float Value_DOT_005_FROM_geo_cracked_ground = float_vars[4];
    float Value_FROM_geo_cracked_ground = float_vars[5];
    float4_nonbuiltin colorramp_4_VAR_FROM_geo_cracked_ground_color0 =
        float4_nonbuiltin_vars[0];
    float4_nonbuiltin colorramp_4_VAR_FROM_geo_cracked_ground_color1 =
        float4_nonbuiltin_vars[1];
    float colorramp_4_VAR_FROM_geo_cracked_ground_pos0 = float_vars[6];
    float colorramp_4_VAR_FROM_geo_cracked_ground_pos1 = float_vars[7];
    float4_nonbuiltin colorramp_5_VAR_FROM_geo_cracked_ground_color0 =
        float4_nonbuiltin_vars[2];
    float4_nonbuiltin colorramp_5_VAR_FROM_geo_cracked_ground_color1 =
        float4_nonbuiltin_vars[3];
    float colorramp_5_VAR_FROM_geo_cracked_ground_pos0 = float_vars[8];
    float colorramp_5_VAR_FROM_geo_cracked_ground_pos1 = float_vars[9];
    float4_nonbuiltin colorramp_VAR_FROM_geo_cracked_ground_color0 =
        float4_nonbuiltin_vars[4];
    float4_nonbuiltin colorramp_VAR_FROM_geo_cracked_ground_color1 =
        float4_nonbuiltin_vars[5];
    float4_nonbuiltin colorramp_VAR_FROM_geo_cracked_ground_color2 =
        float4_nonbuiltin_vars[6];
    float colorramp_VAR_FROM_geo_cracked_ground_pos0 = float_vars[10];
    float colorramp_VAR_FROM_geo_cracked_ground_pos1 = float_vars[11];
    float colorramp_VAR_FROM_geo_cracked_ground_pos2 = float_vars[12];
    float noise_texture_1_w_FROM_geo_cracked_ground = float_vars[13];
    float noise_texture_2_w_FROM_geo_cracked_ground = float_vars[14];
    float noise_texture_3_w_FROM_geo_cracked_ground = float_vars[15];
    float noise_texture_w_FROM_geo_cracked_ground = float_vars[16];
    float sca_crac_FROM_geo_cracked_ground = float_vars[17];
    float sca_gra_FROM_geo_cracked_ground = float_vars[18];
    float voronoi_texture_2_w_FROM_geo_cracked_ground = float_vars[19];
    float voronoi_texture_3_w_FROM_geo_cracked_ground = float_vars[20];
    float voronoi_texture_w_FROM_geo_cracked_ground = float_vars[21];
    float sca_crac__Value;

    sca_crac__Value = sca_crac_FROM_geo_cracked_ground;
    float sca_gra__Value;

    sca_gra__Value = sca_gra_FROM_geo_cracked_ground;
    float noise_texture_w__Value;

    noise_texture_w__Value = noise_texture_w_FROM_geo_cracked_ground;
    float Noise_SPACE_Texture__Fac;
    node_shader_tex_noise(4, position, noise_texture_w__Value, 6.0, 16.0, 0.5,
                          0.0, &Noise_SPACE_Texture__Fac, NULL);
    float4_nonbuiltin colorramp_VAR__Color;

    float colorramp_VAR_positions[3]{
        colorramp_VAR_FROM_geo_cracked_ground_pos0,
        colorramp_VAR_FROM_geo_cracked_ground_pos1,
        colorramp_VAR_FROM_geo_cracked_ground_pos2};
    float4_nonbuiltin colorramp_VAR_colors[3]{
        colorramp_VAR_FROM_geo_cracked_ground_color0,
        colorramp_VAR_FROM_geo_cracked_ground_color1,
        colorramp_VAR_FROM_geo_cracked_ground_color2};
    node_texture_valToRgb(3, colorramp_VAR_positions, colorramp_VAR_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR,
                          COLBAND_HUE_NEAR, Noise_SPACE_Texture__Fac,
                          &colorramp_VAR__Color, NULL);
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(colorramp_VAR__Color), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);
    float Value__Value;

    Value__Value = Value_FROM_geo_cracked_ground;
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, Vector_SPACE_Math__Vector,
        float3_nonbuiltin(Value__Value), float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
        &Vector_SPACE_Math_DOT_001__Vector, NULL);
    float voronoi_texture_w__Value;

    voronoi_texture_w__Value = voronoi_texture_w_FROM_geo_cracked_ground;
    float Voronoi_SPACE_Texture__Distance;
    node_shader_tex_voronoi(4, SHD_VORONOI_SMOOTH_F1, SHD_VORONOI_EUCLIDEAN,
                            position, voronoi_texture_w__Value, 50.0, 1.0, 0.5,
                            1.0, &Voronoi_SPACE_Texture__Distance, NULL, NULL,
                            NULL, NULL);
    float4_nonbuiltin ColorRamp__Color;

    float ColorRamp_positions[2]{0.11400000005960464, 0.335999995470047};
    float4_nonbuiltin ColorRamp_colors[2]{
        float4_nonbuiltin(0.0, 0.0, 0.0, 1.0),
        float4_nonbuiltin(1.0, 1.0, 1.0, 1.0)};
    node_texture_valToRgb(2, ColorRamp_positions, ColorRamp_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR,
                          COLBAND_HUE_NEAR, Voronoi_SPACE_Texture__Distance,
                          &ColorRamp__Color, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_002__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(ColorRamp__Color), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_002__Vector, NULL);
    float Value_DOT_001__Value;

    Value_DOT_001__Value = Value_DOT_001_FROM_geo_cracked_ground;
    float3_nonbuiltin Vector_SPACE_Math_DOT_003__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            Vector_SPACE_Math_DOT_002__Vector,
                            float3_nonbuiltin(Value_DOT_001__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_003__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_004__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_001__Vector,
        Vector_SPACE_Math_DOT_003__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_004__Vector, NULL);
    float noise_texture_1_w__Value;

    noise_texture_1_w__Value = noise_texture_1_w_FROM_geo_cracked_ground;
    float4_nonbuiltin Noise_SPACE_Texture_DOT_001__Color;
    node_shader_tex_noise(4, position, noise_texture_1_w__Value,
                          3.5999999046325684, 16.0, 0.47999998927116394, 0.0,
                          NULL, &Noise_SPACE_Texture_DOT_001__Color);
    float voronoi_texture_2_w__Value;

    voronoi_texture_2_w__Value = voronoi_texture_2_w_FROM_geo_cracked_ground;
    float Voronoi_SPACE_Texture_DOT_001__Distance;
    node_shader_tex_voronoi(
        4, SHD_VORONOI_DISTANCE_TO_EDGE, SHD_VORONOI_EUCLIDEAN,
        float3_nonbuiltin(Noise_SPACE_Texture_DOT_001__Color),
        voronoi_texture_2_w__Value, 5.0, 1.0, 0.5, 1.0,
        &Voronoi_SPACE_Texture_DOT_001__Distance, NULL, NULL, NULL, NULL);
    float4_nonbuiltin ColorRamp_DOT_001__Color;

    float ColorRamp_DOT_001_positions[2]{0.23199999332427979, 1.0};
    float4_nonbuiltin ColorRamp_DOT_001_colors[2]{
        float4_nonbuiltin(1.0, 1.0, 1.0, 1.0),
        float4_nonbuiltin(0.0, 0.0, 0.0, 1.0)};
    node_texture_valToRgb(2, ColorRamp_DOT_001_positions,
                          ColorRamp_DOT_001_colors, COLBAND_BLEND_RGB,
                          COLBAND_INTERP_LINEAR, COLBAND_HUE_NEAR,
                          Voronoi_SPACE_Texture_DOT_001__Distance,
                          &ColorRamp_DOT_001__Color, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_005__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(ColorRamp_DOT_001__Color), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_005__Vector, NULL);
    float Value_DOT_002__Value;

    Value_DOT_002__Value = Value_DOT_002_FROM_geo_cracked_ground;
    float3_nonbuiltin Vector_SPACE_Math_DOT_006__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            Vector_SPACE_Math_DOT_005__Vector,
                            float3_nonbuiltin(Value_DOT_002__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_006__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_007__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_004__Vector,
        Vector_SPACE_Math_DOT_006__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_007__Vector, NULL);
    float4_nonbuiltin Voronoi_SPACE_Texture_DOT_002__Color;
    node_shader_tex_voronoi(3, SHD_VORONOI_SMOOTH_F1, SHD_VORONOI_EUCLIDEAN,
                            position, 0.0, sca_gra__Value, 1.0, 0.5, 1.0, NULL,
                            &Voronoi_SPACE_Texture_DOT_002__Color, NULL, NULL,
                            NULL);
    float4_nonbuiltin ColorRamp_DOT_002__Color;

    float ColorRamp_DOT_002_positions[2]{0.3230000138282776,
                                         0.4359999895095825};
    float4_nonbuiltin ColorRamp_DOT_002_colors[2]{
        float4_nonbuiltin(0.0, 0.0, 0.0, 1.0),
        float4_nonbuiltin(1.0, 1.0, 1.0, 1.0)};
    node_texture_valToRgb(2, ColorRamp_DOT_002_positions,
                          ColorRamp_DOT_002_colors, COLBAND_BLEND_RGB,
                          COLBAND_INTERP_LINEAR, COLBAND_HUE_NEAR,
                          float(Voronoi_SPACE_Texture_DOT_002__Color),
                          &ColorRamp_DOT_002__Color, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_008__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(ColorRamp_DOT_002__Color), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_008__Vector, NULL);
    float Value_DOT_003__Value;

    Value_DOT_003__Value = Value_DOT_003_FROM_geo_cracked_ground;
    float3_nonbuiltin Vector_SPACE_Math_DOT_009__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            Vector_SPACE_Math_DOT_008__Vector,
                            float3_nonbuiltin(Value_DOT_003__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_009__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_010__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_007__Vector,
        Vector_SPACE_Math_DOT_009__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_010__Vector, NULL);
    float noise_texture_3_w__Value;

    noise_texture_3_w__Value = noise_texture_3_w_FROM_geo_cracked_ground;
    float Noise_SPACE_Texture_DOT_002__Fac;
    node_shader_tex_noise(4, position, noise_texture_3_w__Value, 5.0, 2.0,
                          0.5199999809265137, 0.0,
                          &Noise_SPACE_Texture_DOT_002__Fac, NULL);
    float4_nonbuiltin colorramp_4_VAR__Color;

    float colorramp_4_VAR_positions[2]{
        colorramp_4_VAR_FROM_geo_cracked_ground_pos0,
        colorramp_4_VAR_FROM_geo_cracked_ground_pos1};
    float4_nonbuiltin colorramp_4_VAR_colors[2]{
        colorramp_4_VAR_FROM_geo_cracked_ground_color0,
        colorramp_4_VAR_FROM_geo_cracked_ground_color1};
    node_texture_valToRgb(2, colorramp_4_VAR_positions, colorramp_4_VAR_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR,
                          COLBAND_HUE_NEAR, Noise_SPACE_Texture_DOT_002__Fac,
                          &colorramp_4_VAR__Color, NULL);
    float noise_texture_2_w__Value;

    noise_texture_2_w__Value = noise_texture_2_w_FROM_geo_cracked_ground;
    float4_nonbuiltin Noise_SPACE_Texture_DOT_003__Color;
    node_shader_tex_noise(4, position, noise_texture_2_w__Value,
                          2.299999952316284, 16.0, 0.5, 0.0, NULL,
                          &Noise_SPACE_Texture_DOT_003__Color);
    float voronoi_texture_3_w__Value;

    voronoi_texture_3_w__Value = voronoi_texture_3_w_FROM_geo_cracked_ground;
    float Voronoi_SPACE_Texture_DOT_003__Distance;
    node_shader_tex_voronoi(
        4, SHD_VORONOI_DISTANCE_TO_EDGE, SHD_VORONOI_EUCLIDEAN,
        float3_nonbuiltin(Noise_SPACE_Texture_DOT_003__Color),
        voronoi_texture_3_w__Value, sca_crac__Value, 1.0, 0.5, 1.0,
        &Voronoi_SPACE_Texture_DOT_003__Distance, NULL, NULL, NULL, NULL);
    float4_nonbuiltin colorramp_5_VAR__Color;

    float colorramp_5_VAR_positions[2]{
        colorramp_5_VAR_FROM_geo_cracked_ground_pos0,
        colorramp_5_VAR_FROM_geo_cracked_ground_pos1};
    float4_nonbuiltin colorramp_5_VAR_colors[2]{
        colorramp_5_VAR_FROM_geo_cracked_ground_color0,
        colorramp_5_VAR_FROM_geo_cracked_ground_color1};
    node_texture_valToRgb(
        2, colorramp_5_VAR_positions, colorramp_5_VAR_colors, COLBAND_BLEND_RGB,
        COLBAND_INTERP_LINEAR, COLBAND_HUE_NEAR,
        Voronoi_SPACE_Texture_DOT_003__Distance, &colorramp_5_VAR__Color, NULL);
    float4_nonbuiltin Mix__Color;
    node_shader_mix_rgb(0, MA_RAMP_MIX, float(colorramp_4_VAR__Color),
                        colorramp_5_VAR__Color,
                        float4_nonbuiltin(1.0, 1.0, 1.0, 1.0), &Mix__Color);
    float3_nonbuiltin Vector_SPACE_Math_DOT_011__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Mix__Color), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_011__Vector, NULL);
    float Value_DOT_004__Value;

    Value_DOT_004__Value = Value_DOT_004_FROM_geo_cracked_ground;
    float3_nonbuiltin Vector_SPACE_Math_DOT_012__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            Vector_SPACE_Math_DOT_011__Vector,
                            float3_nonbuiltin(Value_DOT_004__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_012__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_013__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_010__Vector,
        Vector_SPACE_Math_DOT_012__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_013__Vector, NULL);
    float Value_DOT_005__Value;

    Value_DOT_005__Value = Value_DOT_005_FROM_geo_cracked_ground;
    float3_nonbuiltin Vector_SPACE_Math_DOT_014__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            Vector_SPACE_Math_DOT_013__Vector,
                            float3_nonbuiltin(Value_DOT_005__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_014__Vector, NULL);

    if (offset != NULL)
        *offset = Vector_SPACE_Math_DOT_014__Vector;
}
