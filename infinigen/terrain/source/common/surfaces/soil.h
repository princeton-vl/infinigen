// Code generated using version 1.0.0 of infinigen/tools/kernelize_surfaces.py;
// refer to infinigen/assets/materials/soil.py which has the copyright and
// authors
DEVICE_FUNC void nodegroup_pebble(
    float Input_0, float Input_1,
    float
        noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble,
    float3_nonbuiltin position,
    float
        vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble,
    POINTER_OR_REFERENCE_ARG float *Output_2) {
    float noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR___Value;

    noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR___Value =
        noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble;
    float4_nonbuiltin Noise_SPACE_Texture__Color;
    node_shader_tex_noise(
        4, position,
        noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR___Value,
        Input_0, 2.0, 0.5, 0.0, NULL, &Noise_SPACE_Texture__Color);
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Noise_SPACE_Texture__Color),
                            float3_nonbuiltin(Input_1),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, Vector_SPACE_Math__Vector,
                            position, float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_001__Vector, NULL);
    float vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR___Value;

    vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR___Value =
        vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble;
    float Voronoi_SPACE_Texture__Distance;
    node_shader_tex_voronoi(
        4, SHD_VORONOI_F1, SHD_VORONOI_EUCLIDEAN,
        Vector_SPACE_Math_DOT_001__Vector,
        vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR___Value,
        Input_0, 1.0, 0.5, 1.0, &Voronoi_SPACE_Texture__Distance, NULL, NULL,
        NULL, NULL);

    if (Output_2 != NULL)
        *Output_2 = Voronoi_SPACE_Texture__Distance;
}
DEVICE_FUNC void nodegroup_pebble_DOT_001(
    float Input_0, float Input_1,
    float
        noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble_DOT_001,
    float3_nonbuiltin position,
    float
        vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble_DOT_001,
    POINTER_OR_REFERENCE_ARG float *Output_2) {
    float noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR___Value;

    noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR___Value =
        noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble_DOT_001;
    float4_nonbuiltin Noise_SPACE_Texture__Color;
    node_shader_tex_noise(
        4, position,
        noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR___Value,
        Input_0, 2.0, 0.5, 0.0, NULL, &Noise_SPACE_Texture__Color);
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Noise_SPACE_Texture__Color),
                            float3_nonbuiltin(Input_1),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, Vector_SPACE_Math__Vector,
                            position, float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_001__Vector, NULL);
    float vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR___Value;

    vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR___Value =
        vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble_DOT_001;
    float Voronoi_SPACE_Texture__Distance;
    node_shader_tex_voronoi(
        4, SHD_VORONOI_F1, SHD_VORONOI_EUCLIDEAN,
        Vector_SPACE_Math_DOT_001__Vector,
        vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR___Value,
        Input_0, 1.0, 0.5, 1.0, &Voronoi_SPACE_Texture__Distance, NULL, NULL,
        NULL, NULL);

    if (Output_2 != NULL)
        *Output_2 = Voronoi_SPACE_Texture__Distance;
}
DEVICE_FUNC void nodegroup_pebble_DOT_002(
    float Input_0, float Input_1,
    float
        noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble_DOT_002,
    float3_nonbuiltin position,
    float
        vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble_DOT_002,
    POINTER_OR_REFERENCE_ARG float *Output_2) {
    float noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR___Value;

    noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR___Value =
        noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble_DOT_002;
    float4_nonbuiltin Noise_SPACE_Texture__Color;
    node_shader_tex_noise(
        4, position,
        noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR___Value,
        Input_0, 2.0, 0.5, 0.0, NULL, &Noise_SPACE_Texture__Color);
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Noise_SPACE_Texture__Color),
                            float3_nonbuiltin(Input_1),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, Vector_SPACE_Math__Vector,
                            position, float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_001__Vector, NULL);
    float vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR___Value;

    vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR___Value =
        vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble_DOT_002;
    float Voronoi_SPACE_Texture__Distance;
    node_shader_tex_voronoi(
        4, SHD_VORONOI_F1, SHD_VORONOI_EUCLIDEAN,
        Vector_SPACE_Math_DOT_001__Vector,
        vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR___Value,
        Input_0, 1.0, 0.5, 1.0, &Voronoi_SPACE_Texture__Distance, NULL, NULL,
        NULL, NULL);

    if (Output_2 != NULL)
        *Output_2 = Voronoi_SPACE_Texture__Distance;
}
DEVICE_FUNC void nodegroup_displacement_to_offset(
    float3_nonbuiltin Input_0, float Input_1, float3_nonbuiltin normal,
    POINTER_OR_REFERENCE_ARG float3_nonbuiltin *Output_2) {
    float Math__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, float(Input_0), Input_1, 0.5,
                      &Math__Value);
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Math__Value), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);

    if (Output_2 != NULL)
        *Output_2 = Vector_SPACE_Math__Vector;
}
DEVICE_FUNC void geometry_soil(
    float3_nonbuiltin position, float3_nonbuiltin normal,
    POINTER_OR_REFERENCE_ARG float *float_vars,
    POINTER_OR_REFERENCE_ARG float4_nonbuiltin *float4_nonbuiltin_vars,
    POINTER_OR_REFERENCE_ARG float3_nonbuiltin *offset) {
    float4_nonbuiltin colorramp_2_VAR_FROM_geometry_soil_color0 =
        float4_nonbuiltin_vars[0];
    float4_nonbuiltin colorramp_2_VAR_FROM_geometry_soil_color1 =
        float4_nonbuiltin_vars[1];
    float4_nonbuiltin colorramp_2_VAR_FROM_geometry_soil_color2 =
        float4_nonbuiltin_vars[2];
    float colorramp_2_VAR_FROM_geometry_soil_pos0 = float_vars[0];
    float colorramp_2_VAR_FROM_geometry_soil_pos1 = float_vars[1];
    float colorramp_2_VAR_FROM_geometry_soil_pos2 = float_vars[2];
    float4_nonbuiltin colorramp_VAR_FROM_geometry_soil_color0 =
        float4_nonbuiltin_vars[3];
    float4_nonbuiltin colorramp_VAR_FROM_geometry_soil_color1 =
        float4_nonbuiltin_vars[4];
    float4_nonbuiltin colorramp_VAR_FROM_geometry_soil_color2 =
        float4_nonbuiltin_vars[5];
    float colorramp_VAR_FROM_geometry_soil_pos0 = float_vars[3];
    float colorramp_VAR_FROM_geometry_soil_pos1 = float_vars[4];
    float colorramp_VAR_FROM_geometry_soil_pos2 = float_vars[5];
    float
        noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble =
            float_vars[6];
    float
        noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble_DOT_001 =
            float_vars[7];
    float
        noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble_DOT_002 =
            float_vars[8];
    float
        peb1_noise_mag_SPACE__WAVE__SPACE_U_LBR_0_DOT_1_COMMA__SPACE_0_DOT_5_RBR__FROM_geometry_soil =
            float_vars[9];
    float
        peb1_size_SPACE__WAVE__SPACE_U_LBR_2_COMMA__SPACE_5_RBR__FROM_geometry_soil =
            float_vars[10];
    float
        peb2_noise_scale_SPACE__WAVE__SPACE_U_LBR_0_DOT_05_COMMA__SPACE_0_DOT_2_RBR__FROM_geometry_soil =
            float_vars[11];
    float
        peb2_size_SPACE__WAVE__SPACE_U_LBR_5_COMMA__SPACE_9_RBR__FROM_geometry_soil =
            float_vars[12];
    float
        peb3_noise_scale_SPACE__WAVE__SPACE_U_LBR_0_DOT_05_COMMA__SPACE_0_DOT_35_RBR__FROM_geometry_soil =
            float_vars[13];
    float
        peb3_size_SPACE__WAVE__SPACE_U_LBR_12_COMMA__SPACE_18_RBR__FROM_geometry_soil =
            float_vars[14];
    float
        vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble =
            float_vars[15];
    float
        vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble_DOT_001 =
            float_vars[16];
    float
        vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble_DOT_002 =
            float_vars[17];
    float peb1_size_SPACE__WAVE__SPACE_U_LBR_2_COMMA__SPACE_5_RBR___Value;

    peb1_size_SPACE__WAVE__SPACE_U_LBR_2_COMMA__SPACE_5_RBR___Value =
        peb1_size_SPACE__WAVE__SPACE_U_LBR_2_COMMA__SPACE_5_RBR__FROM_geometry_soil;
    float
        peb1_noise_mag_SPACE__WAVE__SPACE_U_LBR_0_DOT_1_COMMA__SPACE_0_DOT_5_RBR___Value;

    peb1_noise_mag_SPACE__WAVE__SPACE_U_LBR_0_DOT_1_COMMA__SPACE_0_DOT_5_RBR___Value =
        peb1_noise_mag_SPACE__WAVE__SPACE_U_LBR_0_DOT_1_COMMA__SPACE_0_DOT_5_RBR__FROM_geometry_soil;
    float Group__Output_2;
    nodegroup_pebble(

        peb1_size_SPACE__WAVE__SPACE_U_LBR_2_COMMA__SPACE_5_RBR___Value,
        peb1_noise_mag_SPACE__WAVE__SPACE_U_LBR_0_DOT_1_COMMA__SPACE_0_DOT_5_RBR___Value,
        noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble,
        position,
        vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble,
        &Group__Output_2);
    float4_nonbuiltin colorramp_VAR__Color;

    float colorramp_VAR_positions[3]{colorramp_VAR_FROM_geometry_soil_pos0,
                                     colorramp_VAR_FROM_geometry_soil_pos1,
                                     colorramp_VAR_FROM_geometry_soil_pos2};
    float4_nonbuiltin colorramp_VAR_colors[3]{
        colorramp_VAR_FROM_geometry_soil_color0,
        colorramp_VAR_FROM_geometry_soil_color1,
        colorramp_VAR_FROM_geometry_soil_color2};
    node_texture_valToRgb(3, colorramp_VAR_positions, colorramp_VAR_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR,
                          COLBAND_HUE_NEAR, Group__Output_2,
                          &colorramp_VAR__Color, NULL);
    float peb2_size_SPACE__WAVE__SPACE_U_LBR_5_COMMA__SPACE_9_RBR___Value;

    peb2_size_SPACE__WAVE__SPACE_U_LBR_5_COMMA__SPACE_9_RBR___Value =
        peb2_size_SPACE__WAVE__SPACE_U_LBR_5_COMMA__SPACE_9_RBR__FROM_geometry_soil;
    float
        peb2_noise_scale_SPACE__WAVE__SPACE_U_LBR_0_DOT_05_COMMA__SPACE_0_DOT_2_RBR___Value;

    peb2_noise_scale_SPACE__WAVE__SPACE_U_LBR_0_DOT_05_COMMA__SPACE_0_DOT_2_RBR___Value =
        peb2_noise_scale_SPACE__WAVE__SPACE_U_LBR_0_DOT_05_COMMA__SPACE_0_DOT_2_RBR__FROM_geometry_soil;
    float Group_DOT_001__Output_2;
    nodegroup_pebble_DOT_001(

        peb2_size_SPACE__WAVE__SPACE_U_LBR_5_COMMA__SPACE_9_RBR___Value,
        peb2_noise_scale_SPACE__WAVE__SPACE_U_LBR_0_DOT_05_COMMA__SPACE_0_DOT_2_RBR___Value,
        noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble_DOT_001,
        position,
        vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble_DOT_001,
        &Group_DOT_001__Output_2);
    float4_nonbuiltin colorramp_2_VAR__Color;

    float colorramp_2_VAR_positions[3]{colorramp_2_VAR_FROM_geometry_soil_pos0,
                                       colorramp_2_VAR_FROM_geometry_soil_pos1,
                                       colorramp_2_VAR_FROM_geometry_soil_pos2};
    float4_nonbuiltin colorramp_2_VAR_colors[3]{
        colorramp_2_VAR_FROM_geometry_soil_color0,
        colorramp_2_VAR_FROM_geometry_soil_color1,
        colorramp_2_VAR_FROM_geometry_soil_color2};
    node_texture_valToRgb(3, colorramp_2_VAR_positions, colorramp_2_VAR_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR,
                          COLBAND_HUE_NEAR, Group_DOT_001__Output_2,
                          &colorramp_2_VAR__Color, NULL);
    float Math__Value;
    node_texture_math(NODE_MATH_ADD, 0, float(colorramp_VAR__Color),
                      float(colorramp_2_VAR__Color), 0.5, &Math__Value);
    float peb3_size_SPACE__WAVE__SPACE_U_LBR_12_COMMA__SPACE_18_RBR___Value;

    peb3_size_SPACE__WAVE__SPACE_U_LBR_12_COMMA__SPACE_18_RBR___Value =
        peb3_size_SPACE__WAVE__SPACE_U_LBR_12_COMMA__SPACE_18_RBR__FROM_geometry_soil;
    float
        peb3_noise_scale_SPACE__WAVE__SPACE_U_LBR_0_DOT_05_COMMA__SPACE_0_DOT_35_RBR___Value;

    peb3_noise_scale_SPACE__WAVE__SPACE_U_LBR_0_DOT_05_COMMA__SPACE_0_DOT_35_RBR___Value =
        peb3_noise_scale_SPACE__WAVE__SPACE_U_LBR_0_DOT_05_COMMA__SPACE_0_DOT_35_RBR__FROM_geometry_soil;
    float Group_DOT_002__Output_2;
    nodegroup_pebble_DOT_002(

        peb3_size_SPACE__WAVE__SPACE_U_LBR_12_COMMA__SPACE_18_RBR___Value,
        peb3_noise_scale_SPACE__WAVE__SPACE_U_LBR_0_DOT_05_COMMA__SPACE_0_DOT_35_RBR___Value,
        noise1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble_DOT_002,
        position,
        vornoi1_w_SPACE__WAVE__SPACE_U_LBR_0_COMMA__SPACE_10_RBR__FROM_nodegroup_pebble_DOT_002,
        &Group_DOT_002__Output_2);
    float4_nonbuiltin ColorRamp__Color;

    float ColorRamp_positions[2]{0.0, 0.8999999761581421};
    float4_nonbuiltin ColorRamp_colors[2]{
        float4_nonbuiltin(0.15000000596046448, 0.15000000596046448,
                          0.15000000596046448, 1.0),
        float4_nonbuiltin(0.0, 0.0, 0.0, 1.0)};
    node_texture_valToRgb(2, ColorRamp_positions, ColorRamp_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR,
                          COLBAND_HUE_NEAR, Group_DOT_002__Output_2,
                          &ColorRamp__Color, NULL);
    float Math_DOT_001__Value;
    node_texture_math(NODE_MATH_ADD, 0, Math__Value, float(ColorRamp__Color),
                      0.5, &Math_DOT_001__Value);
    float3_nonbuiltin Group_DOT_003__Output_2;
    nodegroup_displacement_to_offset(

        float3_nonbuiltin(Math_DOT_001__Value), 0.10000000149011612, normal,
        &Group_DOT_003__Output_2);

    if (offset != NULL)
        *offset = Group_DOT_003__Output_2;
}
