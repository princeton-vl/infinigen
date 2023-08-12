// Code generated using version 1.0.0 of infinigen/tools/kernelize_surfaces.py;
// refer to infinigen/assets/materials/cracked_ground.py which has the
// copyright and authors
DEVICE_FUNC void nodegroup_apply_value_to_normal(
    float Input_0, float Input_1, float3_nonbuiltin normal,
    POINTER_OR_REFERENCE_ARG float3_nonbuiltin *Output_2) {
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_SCALE, normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0),
                            float3_nonbuiltin(0.0, 0.0, 0.0), Input_0,
                            &Vector_SPACE_Math__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_SCALE, Vector_SPACE_Math__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0),
                            float3_nonbuiltin(0.0, 0.0, 0.0), Input_1,
                            &Vector_SPACE_Math_DOT_001__Vector, NULL);

    if (Output_2 != NULL)
        *Output_2 = Vector_SPACE_Math_DOT_001__Vector;
}
DEVICE_FUNC void nodegroup_apply_value_to_normal_DOT_001(
    float Input_0, float Input_1, float3_nonbuiltin normal,
    POINTER_OR_REFERENCE_ARG float3_nonbuiltin *Output_2) {
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_SCALE, normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0),
                            float3_nonbuiltin(0.0, 0.0, 0.0), Input_0,
                            &Vector_SPACE_Math__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_SCALE, Vector_SPACE_Math__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0),
                            float3_nonbuiltin(0.0, 0.0, 0.0), Input_1,
                            &Vector_SPACE_Math_DOT_001__Vector, NULL);

    if (Output_2 != NULL)
        *Output_2 = Vector_SPACE_Math_DOT_001__Vector;
}
DEVICE_FUNC void nodegroup_apply_value_to_normal_DOT_002(
    float Input_0, float Input_1, float3_nonbuiltin normal,
    POINTER_OR_REFERENCE_ARG float3_nonbuiltin *Output_2) {
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_SCALE, normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0),
                            float3_nonbuiltin(0.0, 0.0, 0.0), Input_0,
                            &Vector_SPACE_Math__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_SCALE, Vector_SPACE_Math__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0),
                            float3_nonbuiltin(0.0, 0.0, 0.0), Input_1,
                            &Vector_SPACE_Math_DOT_001__Vector, NULL);

    if (Output_2 != NULL)
        *Output_2 = Vector_SPACE_Math_DOT_001__Vector;
}
DEVICE_FUNC void
geo_cracked_ground(float3_nonbuiltin position, float3_nonbuiltin normal,
                   POINTER_OR_REFERENCE_ARG float *float_vars,
                   POINTER_OR_REFERENCE_ARG float3_nonbuiltin *offset) {
    float crack_density_FROM_geo_cracked_ground = float_vars[0];
    float dep_crac_FROM_geo_cracked_ground = float_vars[1];
    float dep_landscape_FROM_geo_cracked_ground = float_vars[2];
    float sca_crac_FROM_geo_cracked_ground = float_vars[3];
    float sca_gra_FROM_geo_cracked_ground = float_vars[4];
    float sca_mask_DOT_001_FROM_geo_cracked_ground = float_vars[5];
    float sca_mask_FROM_geo_cracked_ground = float_vars[6];
    float seed_FROM_geo_cracked_ground = float_vars[7];
    float wid_crac_FROM_geo_cracked_ground = float_vars[8];
    float sca_crac__Value;

    sca_crac__Value = sca_crac_FROM_geo_cracked_ground;
    float sca_mask__Value;

    sca_mask__Value = sca_mask_FROM_geo_cracked_ground;
    float sca_mask_DOT_001__Value;

    sca_mask_DOT_001__Value = sca_mask_DOT_001_FROM_geo_cracked_ground;
    float crack_density__Value;

    crack_density__Value = crack_density_FROM_geo_cracked_ground;
    float wid_crac__Value;

    wid_crac__Value = wid_crac_FROM_geo_cracked_ground;
    float sca_gra__Value;

    sca_gra__Value = sca_gra_FROM_geo_cracked_ground;
    float dep_crac__Value;

    dep_crac__Value = dep_crac_FROM_geo_cracked_ground;
    float dep_landscape__Value;

    dep_landscape__Value = dep_landscape_FROM_geo_cracked_ground;
    float seed__Value;

    seed__Value = seed_FROM_geo_cracked_ground;
    float Noise_SPACE_Texture__Fac;
    node_shader_tex_noise(4, position, seed__Value, sca_mask_DOT_001__Value,
                          15.0, 0.5375000238418579, 0.0,
                          &Noise_SPACE_Texture__Fac, NULL);
    float4_nonbuiltin Noise_SPACE_Texture_DOT_001__Color;
    node_shader_tex_noise(4, position, seed__Value, sca_crac__Value, 15.0, 0.5,
                          0.0, NULL, &Noise_SPACE_Texture_DOT_001__Color);
    float Voronoi_SPACE_Texture__Distance;
    node_shader_tex_voronoi(
        4, SHD_VORONOI_DISTANCE_TO_EDGE, SHD_VORONOI_EUCLIDEAN,
        float3_nonbuiltin(Noise_SPACE_Texture_DOT_001__Color), seed__Value,
        2.299999952316284, 1.0, 0.5, 1.0, &Voronoi_SPACE_Texture__Distance,
        NULL, NULL, NULL, NULL);
    float Map_SPACE_Range__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Voronoi_SPACE_Texture__Distance, 0.0,
        wid_crac__Value, 1.0, 0.0, 4.0, float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(4.0, 4.0, 4.0), &Map_SPACE_Range__Result, NULL);
    float Noise_SPACE_Texture_DOT_002__Fac;
    node_shader_tex_noise(4, position, seed__Value, sca_mask__Value, 15.0, 0.5,
                          0.0, &Noise_SPACE_Texture_DOT_002__Fac, NULL);
    float Math__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, 1.0, crack_density__Value, 0.5,
                      &Math__Value);
    float Math_DOT_001__Value;
    node_texture_math(NODE_MATH_SUBTRACT, 0, Math__Value, 0.019999999552965164,
                      0.5, &Math_DOT_001__Value);
    float Math_DOT_002__Value;
    node_texture_math(NODE_MATH_ADD, 0, Math__Value, 0.019999999552965164, 0.5,
                      &Math_DOT_002__Value);
    float Map_SPACE_Range_DOT_001__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1, Noise_SPACE_Texture_DOT_002__Fac,
        Math_DOT_001__Value, Math_DOT_002__Value, 0.0, 1.0, 4.0,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(1.0, 1.0, 1.0), float3_nonbuiltin(4.0, 4.0, 4.0),
        &Map_SPACE_Range_DOT_001__Result, NULL);
    float Math_DOT_003__Value;
    node_texture_math(NODE_MATH_MULTIPLY, 0, Map_SPACE_Range__Result,
                      Map_SPACE_Range_DOT_001__Result, 0.5,
                      &Math_DOT_003__Value);
    float Voronoi_SPACE_Texture_DOT_001__Distance;
    node_shader_tex_voronoi(4, SHD_VORONOI_F1, SHD_VORONOI_EUCLIDEAN, position,
                            seed__Value, sca_gra__Value, 1.0, 0.5, 1.0,
                            &Voronoi_SPACE_Texture_DOT_001__Distance, NULL,
                            NULL, NULL, NULL);
    float Map_SPACE_Range_DOT_002__Result;
    node_shader_map_range(
        FLOAT, NODE_MAP_RANGE_LINEAR, 1,
        Voronoi_SPACE_Texture_DOT_001__Distance, 0.8999999761581421, 1.0, 0.0,
        1.0, 4.0, float3_nonbuiltin(0.0, 0.0, 0.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(1.0, 1.0, 1.0),
        float3_nonbuiltin(4.0, 4.0, 4.0), &Map_SPACE_Range_DOT_002__Result,
        NULL);
    float3_nonbuiltin Group__Output_2;
    nodegroup_apply_value_to_normal(

        Noise_SPACE_Texture__Fac, 0.30000001192092896, normal,
        &Group__Output_2);
    float3_nonbuiltin Group_DOT_001__Output_2;
    nodegroup_apply_value_to_normal_DOT_001(

        Math_DOT_003__Value, dep_crac__Value, normal, &Group_DOT_001__Output_2);
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, Group__Output_2,
                            Group_DOT_001__Output_2,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);
    float3_nonbuiltin Group_DOT_002__Output_2;
    nodegroup_apply_value_to_normal_DOT_002(

        Map_SPACE_Range_DOT_002__Result, 0.019999999552965164, normal,
        &Group_DOT_002__Output_2);
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD, Vector_SPACE_Math__Vector,
                            Group_DOT_002__Output_2,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_001__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_002__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_SCALE, Vector_SPACE_Math_DOT_001__Vector,
        float3_nonbuiltin(0.0, 0.0, 0.0), float3_nonbuiltin(0.0, 0.0, 0.0),
        dep_landscape__Value, &Vector_SPACE_Math_DOT_002__Vector, NULL);

    if (offset != NULL)
        *offset = Vector_SPACE_Math_DOT_002__Vector;
}
