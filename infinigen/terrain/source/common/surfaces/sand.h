// Code generated using version 1.0.0 of infinigen/tools/kernelize_surfaces.py;
// refer to infinigen/assets/materials/sand.py which has the copyright and
// authors
DEVICE_FUNC void
geo_SAND(float3_nonbuiltin position, float3_nonbuiltin normal,
         POINTER_OR_REFERENCE_ARG float *float_vars,
         POINTER_OR_REFERENCE_ARG float3_nonbuiltin *float3_nonbuiltin_vars,
         POINTER_OR_REFERENCE_ARG float3_nonbuiltin *offset) {
    float3_nonbuiltin position_shift_0_0_FROM_geo_SAND =
        float3_nonbuiltin_vars[0];
    float3_nonbuiltin position_shift_0_1_FROM_geo_SAND =
        float3_nonbuiltin_vars[1];
    float3_nonbuiltin position_shift_0_2_FROM_geo_SAND =
        float3_nonbuiltin_vars[2];
    float3_nonbuiltin position_shift_1_0_FROM_geo_SAND =
        float3_nonbuiltin_vars[3];
    float3_nonbuiltin position_shift_1_1_FROM_geo_SAND =
        float3_nonbuiltin_vars[4];
    float3_nonbuiltin position_shift_1_2_FROM_geo_SAND =
        float3_nonbuiltin_vars[5];
    float3_nonbuiltin position_shift_2_0_FROM_geo_SAND =
        float3_nonbuiltin_vars[6];
    float3_nonbuiltin position_shift_2_1_FROM_geo_SAND =
        float3_nonbuiltin_vars[7];
    float3_nonbuiltin position_shift_2_2_FROM_geo_SAND =
        float3_nonbuiltin_vars[8];
    float3_nonbuiltin position_shift_3_0_FROM_geo_SAND =
        float3_nonbuiltin_vars[9];
    float3_nonbuiltin position_shift_3_1_FROM_geo_SAND =
        float3_nonbuiltin_vars[10];
    float3_nonbuiltin position_shift_3_2_FROM_geo_SAND =
        float3_nonbuiltin_vars[11];
    float warp_detail_DOT_001_FROM_geo_SAND = float_vars[0];
    float warp_detail_DOT_002_FROM_geo_SAND = float_vars[1];
    float warp_detail_FROM_geo_SAND = float_vars[2];
    float warp_scale_DOT_001_FROM_geo_SAND = float_vars[3];
    float warp_scale_DOT_002_FROM_geo_SAND = float_vars[4];
    float warp_scale_FROM_geo_SAND = float_vars[5];
    float wave_scale_0_FROM_geo_SAND = float_vars[6];
    float wave_scale_1_FROM_geo_SAND = float_vars[7];
    float wave_scale_2_FROM_geo_SAND = float_vars[8];
    float wave_scale_0__Value;

    wave_scale_0__Value = wave_scale_0_FROM_geo_SAND;
    float3_nonbuiltin position_shift_0_0__Vector;

    position_shift_0_0__Vector = position_shift_0_0_FROM_geo_SAND;
    float3_nonbuiltin position_shift_1_0__Vector;

    position_shift_1_0__Vector = position_shift_1_0_FROM_geo_SAND;
    float3_nonbuiltin position_shift_2_0__Vector;

    position_shift_2_0__Vector = position_shift_2_0_FROM_geo_SAND;
    float3_nonbuiltin position_shift_3_0__Vector;

    position_shift_3_0__Vector = position_shift_3_0_FROM_geo_SAND;
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_3_0__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);
    float Noise_SPACE_Texture__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math__Vector, 0.0,
                          0.10000000149011612, 2.0, 0.5, 0.0,
                          &Noise_SPACE_Texture__Fac, NULL);
    float Math__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture__Fac,
                      0.6000000238418579, 0.5, &Math__Value);
    float Math_DOT_001__Value;
    node_texture_math(NODE_MATH_POWER, 1, 100000.0, Math__Value, 0.5,
                      &Math_DOT_001__Value);
    float warp_scale__Value;

    warp_scale__Value = warp_scale_FROM_geo_SAND;
    float warp_detail__Value;

    warp_detail__Value = warp_detail_FROM_geo_SAND;
    float4_nonbuiltin Noise_SPACE_Texture_DOT_001__Color;
    node_shader_tex_noise(3, position, 0.0, warp_scale__Value,
                          warp_detail__Value, 0.5, 0.0, NULL,
                          &Noise_SPACE_Texture_DOT_001__Color);
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, position_shift_0_0__Vector,
        float3_nonbuiltin(Noise_SPACE_Texture_DOT_001__Color),
        float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
        &Vector_SPACE_Math_DOT_001__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_002__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            Vector_SPACE_Math_DOT_001__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_002__Vector, NULL);
    float4_nonbuiltin Wave_SPACE_Texture__Color;
    node_shader_tex_wave(SHD_WAVE_BANDS, SHD_WAVE_BANDS_DIRECTION_X,
                         SHD_WAVE_RINGS_DIRECTION_X, SHD_WAVE_PROFILE_SIN,
                         Vector_SPACE_Math_DOT_002__Vector, wave_scale_0__Value,
                         4.0, 2.0, 1.0, 0.5, 0.0, &Wave_SPACE_Texture__Color,
                         NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_003__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_1_0__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_003__Vector, NULL);
    float Math_DOT_002__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, wave_scale_0__Value,
                      0.9800000190734863, 0.5, &Math_DOT_002__Value);
    float4_nonbuiltin Wave_SPACE_Texture_DOT_001__Color;
    node_shader_tex_wave(SHD_WAVE_BANDS, SHD_WAVE_BANDS_DIRECTION_X,
                         SHD_WAVE_RINGS_DIRECTION_X, SHD_WAVE_PROFILE_SIN,
                         Vector_SPACE_Math_DOT_003__Vector, Math_DOT_002__Value,
                         4.0, 2.0, 1.0, 0.5, 0.0,
                         &Wave_SPACE_Texture_DOT_001__Color, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_004__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_2_0__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_004__Vector, NULL);
    float Noise_SPACE_Texture_DOT_002__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_004__Vector, 0.0, 125.0, 9.0,
                          0.8999999761581421, 0.0,
                          &Noise_SPACE_Texture_DOT_002__Fac, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_005__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Noise_SPACE_Texture_DOT_002__Fac),
                            float3_nonbuiltin(1.0, 1.0, 1.0),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_005__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_006__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD,
        float3_nonbuiltin(Wave_SPACE_Texture_DOT_001__Color),
        Vector_SPACE_Math_DOT_005__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_006__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_007__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, float3_nonbuiltin(Wave_SPACE_Texture__Color),
        Vector_SPACE_Math_DOT_006__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_007__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_008__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, float3_nonbuiltin(Math_DOT_001__Value),
        float3_nonbuiltin(0.009999999776482582, 0.009999999776482582,
                          0.009999999776482582),
        float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
        &Vector_SPACE_Math_DOT_008__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_009__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY, normal,
                            Vector_SPACE_Math_DOT_008__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_009__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_010__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, Vector_SPACE_Math_DOT_007__Vector,
        Vector_SPACE_Math_DOT_009__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_010__Vector, NULL);
    float wave_scale_1__Value;

    wave_scale_1__Value = wave_scale_1_FROM_geo_SAND;
    float3_nonbuiltin position_shift_0_1__Vector;

    position_shift_0_1__Vector = position_shift_0_1_FROM_geo_SAND;
    float3_nonbuiltin position_shift_1_1__Vector;

    position_shift_1_1__Vector = position_shift_1_1_FROM_geo_SAND;
    float3_nonbuiltin position_shift_2_1__Vector;

    position_shift_2_1__Vector = position_shift_2_1_FROM_geo_SAND;
    float3_nonbuiltin position_shift_3_1__Vector;

    position_shift_3_1__Vector = position_shift_3_1_FROM_geo_SAND;
    float3_nonbuiltin Vector_SPACE_Math_DOT_011__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_3_1__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_011__Vector, NULL);
    float Noise_SPACE_Texture_DOT_003__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_011__Vector, 0.0,
                          0.10000000149011612, 2.0, 0.5, 0.0,
                          &Noise_SPACE_Texture_DOT_003__Fac, NULL);
    float Math_DOT_003__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture_DOT_003__Fac,
                      0.6000000238418579, 0.5, &Math_DOT_003__Value);
    float Math_DOT_004__Value;
    node_texture_math(NODE_MATH_POWER, 1, 100000.0, Math_DOT_003__Value, 0.5,
                      &Math_DOT_004__Value);
    float warp_scale_DOT_001__Value;

    warp_scale_DOT_001__Value = warp_scale_DOT_001_FROM_geo_SAND;
    float warp_detail_DOT_001__Value;

    warp_detail_DOT_001__Value = warp_detail_DOT_001_FROM_geo_SAND;
    float4_nonbuiltin Noise_SPACE_Texture_DOT_004__Color;
    node_shader_tex_noise(3, position, 0.0, warp_scale_DOT_001__Value,
                          warp_detail_DOT_001__Value, 0.5, 0.0, NULL,
                          &Noise_SPACE_Texture_DOT_004__Color);
    float3_nonbuiltin Vector_SPACE_Math_DOT_012__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, position_shift_0_1__Vector,
        float3_nonbuiltin(Noise_SPACE_Texture_DOT_004__Color),
        float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
        &Vector_SPACE_Math_DOT_012__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_013__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            Vector_SPACE_Math_DOT_012__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_013__Vector, NULL);
    float4_nonbuiltin Wave_SPACE_Texture_DOT_002__Color;
    node_shader_tex_wave(SHD_WAVE_BANDS, SHD_WAVE_BANDS_DIRECTION_X,
                         SHD_WAVE_RINGS_DIRECTION_X, SHD_WAVE_PROFILE_SIN,
                         Vector_SPACE_Math_DOT_013__Vector, wave_scale_1__Value,
                         4.0, 2.0, 1.0, 0.5, 0.0,
                         &Wave_SPACE_Texture_DOT_002__Color, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_014__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_1_1__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_014__Vector, NULL);
    float Math_DOT_005__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, wave_scale_1__Value,
                      0.9800000190734863, 0.5, &Math_DOT_005__Value);
    float4_nonbuiltin Wave_SPACE_Texture_DOT_003__Color;
    node_shader_tex_wave(SHD_WAVE_BANDS, SHD_WAVE_BANDS_DIRECTION_X,
                         SHD_WAVE_RINGS_DIRECTION_X, SHD_WAVE_PROFILE_SIN,
                         Vector_SPACE_Math_DOT_014__Vector, Math_DOT_005__Value,
                         4.0, 2.0, 1.0, 0.5, 0.0,
                         &Wave_SPACE_Texture_DOT_003__Color, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_015__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_2_1__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_015__Vector, NULL);
    float Noise_SPACE_Texture_DOT_005__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_015__Vector, 0.0, 125.0, 9.0,
                          0.8999999761581421, 0.0,
                          &Noise_SPACE_Texture_DOT_005__Fac, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_016__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Noise_SPACE_Texture_DOT_005__Fac),
                            float3_nonbuiltin(1.0, 1.0, 1.0),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_016__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_017__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD,
        float3_nonbuiltin(Wave_SPACE_Texture_DOT_003__Color),
        Vector_SPACE_Math_DOT_016__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_017__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_018__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD,
        float3_nonbuiltin(Wave_SPACE_Texture_DOT_002__Color),
        Vector_SPACE_Math_DOT_017__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_018__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_019__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, float3_nonbuiltin(Math_DOT_004__Value),
        float3_nonbuiltin(0.009999999776482582, 0.009999999776482582,
                          0.009999999776482582),
        float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
        &Vector_SPACE_Math_DOT_019__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_020__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY, normal,
                            Vector_SPACE_Math_DOT_019__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_020__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_021__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, Vector_SPACE_Math_DOT_018__Vector,
        Vector_SPACE_Math_DOT_020__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_021__Vector, NULL);
    float wave_scale_2__Value;

    wave_scale_2__Value = wave_scale_2_FROM_geo_SAND;
    float3_nonbuiltin position_shift_0_2__Vector;

    position_shift_0_2__Vector = position_shift_0_2_FROM_geo_SAND;
    float3_nonbuiltin position_shift_1_2__Vector;

    position_shift_1_2__Vector = position_shift_1_2_FROM_geo_SAND;
    float3_nonbuiltin position_shift_2_2__Vector;

    position_shift_2_2__Vector = position_shift_2_2_FROM_geo_SAND;
    float3_nonbuiltin position_shift_3_2__Vector;

    position_shift_3_2__Vector = position_shift_3_2_FROM_geo_SAND;
    float3_nonbuiltin Vector_SPACE_Math_DOT_022__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_3_2__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_022__Vector, NULL);
    float Noise_SPACE_Texture_DOT_006__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_022__Vector, 0.0,
                          0.10000000149011612, 2.0, 0.5, 0.0,
                          &Noise_SPACE_Texture_DOT_006__Fac, NULL);
    float Math_DOT_006__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Noise_SPACE_Texture_DOT_006__Fac,
                      0.6000000238418579, 0.5, &Math_DOT_006__Value);
    float Math_DOT_007__Value;
    node_texture_math(NODE_MATH_POWER, 1, 100000.0, Math_DOT_006__Value, 0.5,
                      &Math_DOT_007__Value);
    float warp_scale_DOT_002__Value;

    warp_scale_DOT_002__Value = warp_scale_DOT_002_FROM_geo_SAND;
    float warp_detail_DOT_002__Value;

    warp_detail_DOT_002__Value = warp_detail_DOT_002_FROM_geo_SAND;
    float4_nonbuiltin Noise_SPACE_Texture_DOT_007__Color;
    node_shader_tex_noise(3, position, 0.0, warp_scale_DOT_002__Value,
                          warp_detail_DOT_002__Value, 0.5, 0.0, NULL,
                          &Noise_SPACE_Texture_DOT_007__Color);
    float3_nonbuiltin Vector_SPACE_Math_DOT_023__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, position_shift_0_2__Vector,
        float3_nonbuiltin(Noise_SPACE_Texture_DOT_007__Color),
        float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
        &Vector_SPACE_Math_DOT_023__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_024__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            Vector_SPACE_Math_DOT_023__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_024__Vector, NULL);
    float4_nonbuiltin Wave_SPACE_Texture_DOT_004__Color;
    node_shader_tex_wave(SHD_WAVE_BANDS, SHD_WAVE_BANDS_DIRECTION_X,
                         SHD_WAVE_RINGS_DIRECTION_X, SHD_WAVE_PROFILE_SIN,
                         Vector_SPACE_Math_DOT_024__Vector, wave_scale_2__Value,
                         4.0, 2.0, 1.0, 0.5, 0.0,
                         &Wave_SPACE_Texture_DOT_004__Color, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_025__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_1_2__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_025__Vector, NULL);
    float Math_DOT_008__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, wave_scale_2__Value,
                      0.9800000190734863, 0.5, &Math_DOT_008__Value);
    float4_nonbuiltin Wave_SPACE_Texture_DOT_005__Color;
    node_shader_tex_wave(SHD_WAVE_BANDS, SHD_WAVE_BANDS_DIRECTION_X,
                         SHD_WAVE_RINGS_DIRECTION_X, SHD_WAVE_PROFILE_SIN,
                         Vector_SPACE_Math_DOT_025__Vector, Math_DOT_008__Value,
                         4.0, 2.0, 1.0, 0.5, 0.0,
                         &Wave_SPACE_Texture_DOT_005__Color, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_026__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, position,
                            position_shift_2_2__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_026__Vector, NULL);
    float Noise_SPACE_Texture_DOT_008__Fac;
    node_shader_tex_noise(3, Vector_SPACE_Math_DOT_026__Vector, 0.0, 125.0, 9.0,
                          0.8999999761581421, 0.0,
                          &Noise_SPACE_Texture_DOT_008__Fac, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_027__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Noise_SPACE_Texture_DOT_008__Fac),
                            float3_nonbuiltin(1.0, 1.0, 1.0),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_027__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_028__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD,
        float3_nonbuiltin(Wave_SPACE_Texture_DOT_005__Color),
        Vector_SPACE_Math_DOT_027__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_028__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_029__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD,
        float3_nonbuiltin(Wave_SPACE_Texture_DOT_004__Color),
        Vector_SPACE_Math_DOT_028__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_029__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_030__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, float3_nonbuiltin(Math_DOT_007__Value),
        float3_nonbuiltin(0.009999999776482582, 0.009999999776482582,
                          0.009999999776482582),
        float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
        &Vector_SPACE_Math_DOT_030__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_031__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY, normal,
                            Vector_SPACE_Math_DOT_030__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_031__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_032__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, Vector_SPACE_Math_DOT_029__Vector,
        Vector_SPACE_Math_DOT_031__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_032__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_033__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_021__Vector,
        Vector_SPACE_Math_DOT_032__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_033__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_034__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_010__Vector,
        Vector_SPACE_Math_DOT_033__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_034__Vector, NULL);

    if (offset != NULL)
        *offset = Vector_SPACE_Math_DOT_034__Vector;
}
