// Code generated using version 0.31.1 of worldgen/tools/kernelize_surfaces.py;
// refer to worldgen/surfaces/templates/mud.py which has the copyright and
// authors
DEVICE_FUNC void
geo_mud(float3_nonbuiltin position, float3_nonbuiltin normal,
        POINTER_OR_REFERENCE_ARG float *float_vars,
        POINTER_OR_REFERENCE_ARG float4_nonbuiltin *float4_nonbuiltin_vars,
        POINTER_OR_REFERENCE_ARG float3_nonbuiltin *offset) {
    float Value_DOT_001_FROM_geo_mud = float_vars[0];
    float Value_DOT_002_FROM_geo_mud = float_vars[1];
    float Value_DOT_003_FROM_geo_mud = float_vars[2];
    float Value_DOT_004_FROM_geo_mud = float_vars[3];
    float Value_FROM_geo_mud = float_vars[4];
    float4_nonbuiltin color_ramp_VAR_FROM_geo_mud_color0 =
        float4_nonbuiltin_vars[0];
    float4_nonbuiltin color_ramp_VAR_FROM_geo_mud_color1 =
        float4_nonbuiltin_vars[1];
    float color_ramp_VAR_FROM_geo_mud_pos0 = float_vars[5];
    float color_ramp_VAR_FROM_geo_mud_pos1 = float_vars[6];
    float dep_pud_FROM_geo_mud = float_vars[7];
    float lar_bum_sca_FROM_geo_mud = float_vars[8];
    float noise_texture_1_w_FROM_geo_mud = float_vars[9];
    float noise_texture_3_w_FROM_geo_mud = float_vars[10];
    float noise_texture_w_FROM_geo_mud = float_vars[11];
    float sma_bum_sca_FROM_geo_mud = float_vars[12];
    float voronoi_texture_1_w_FROM_geo_mud = float_vars[13];
    float voronoi_texture_w_FROM_geo_mud = float_vars[14];
    float lar_bum_sca__Value;

    lar_bum_sca__Value = lar_bum_sca_FROM_geo_mud;
    float sma_bum_sca__Value;

    sma_bum_sca__Value = sma_bum_sca_FROM_geo_mud;
    float dep_pud__Value;

    dep_pud__Value = dep_pud_FROM_geo_mud;
    float4_nonbuiltin ColorRamp__Color;

    float ColorRamp_positions[3]{0.949999988079071, 0.9900000095367432, 1.0};
    float4_nonbuiltin ColorRamp_colors[3]{
        float4_nonbuiltin(0.0, 0.0, 0.0, 1.0),
        float4_nonbuiltin(1.0, 1.0, 1.0, 1.0),
        float4_nonbuiltin(1.0, 1.0, 1.0, 1.0)};
    node_texture_valToRgb(3, ColorRamp_positions, ColorRamp_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR,
                          COLBAND_HUE_NEAR, 1.0, &ColorRamp__Color, NULL);
    float noise_texture_w__Value;

    noise_texture_w__Value = noise_texture_w_FROM_geo_mud;
    float Noise_SPACE_Texture__Fac;
    node_shader_tex_noise(4, position, noise_texture_w__Value, 1.0, 9.0,
                          0.4000000059604645, 0.0, &Noise_SPACE_Texture__Fac,
                          NULL);
    float4_nonbuiltin color_ramp_VAR__Color;

    float color_ramp_VAR_positions[2]{color_ramp_VAR_FROM_geo_mud_pos0,
                                      color_ramp_VAR_FROM_geo_mud_pos1};
    float4_nonbuiltin color_ramp_VAR_colors[2]{
        color_ramp_VAR_FROM_geo_mud_color0, color_ramp_VAR_FROM_geo_mud_color1};
    node_texture_valToRgb(2, color_ramp_VAR_positions, color_ramp_VAR_colors,
                          COLBAND_BLEND_RGB, COLBAND_INTERP_LINEAR,
                          COLBAND_HUE_NEAR, Noise_SPACE_Texture__Fac,
                          &color_ramp_VAR__Color, NULL);
    float4_nonbuiltin Mix__Color;
    node_shader_mix_rgb(0, MA_RAMP_MIX, float(ColorRamp__Color),
                        float4_nonbuiltin(1.0, 1.0, 1.0, 1.0),
                        color_ramp_VAR__Color, &Mix__Color);
    float Value__Value;

    Value__Value = Value_FROM_geo_mud;
    float3_nonbuiltin Vector_SPACE_Math__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Value__Value), normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math__Vector, NULL);
    float voronoi_texture_w__Value;

    voronoi_texture_w__Value = voronoi_texture_w_FROM_geo_mud;
    float Voronoi_SPACE_Texture__Distance;
    node_shader_tex_voronoi(
        4, SHD_VORONOI_SMOOTH_F1, SHD_VORONOI_EUCLIDEAN, position,
        voronoi_texture_w__Value, sma_bum_sca__Value, 1.0, 0.5, 10.0,
        &Voronoi_SPACE_Texture__Distance, NULL, NULL, NULL, NULL);
    float voronoi_texture_1_w__Value;

    voronoi_texture_1_w__Value = voronoi_texture_1_w_FROM_geo_mud;
    float Voronoi_SPACE_Texture_DOT_001__Distance;
    node_shader_tex_voronoi(
        4, SHD_VORONOI_SMOOTH_F1, SHD_VORONOI_EUCLIDEAN, position,
        voronoi_texture_1_w__Value, lar_bum_sca__Value, 1.0, 0.5, 1.0,
        &Voronoi_SPACE_Texture_DOT_001__Distance, NULL, NULL, NULL, NULL);
    float Value_DOT_001__Value;

    Value_DOT_001__Value = Value_DOT_001_FROM_geo_mud;
    float3_nonbuiltin Vector_SPACE_Math_DOT_001__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY,
        float3_nonbuiltin(Voronoi_SPACE_Texture_DOT_001__Distance),
        float3_nonbuiltin(Value_DOT_001__Value),
        float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
        &Vector_SPACE_Math_DOT_001__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_002__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_ADD,
                            float3_nonbuiltin(Voronoi_SPACE_Texture__Distance),
                            Vector_SPACE_Math_DOT_001__Vector,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_002__Vector, NULL);
    float noise_texture_1_w__Value;

    noise_texture_1_w__Value = noise_texture_1_w_FROM_geo_mud;
    float Noise_SPACE_Texture_DOT_001__Fac;
    node_shader_tex_noise(4, position, noise_texture_1_w__Value, 5.0, 2.0, 0.5,
                          0.0, &Noise_SPACE_Texture_DOT_001__Fac, NULL);
    float Value_DOT_002__Value;

    Value_DOT_002__Value = Value_DOT_002_FROM_geo_mud;
    float3_nonbuiltin Vector_SPACE_Math_DOT_003__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Noise_SPACE_Texture_DOT_001__Fac),
                            float3_nonbuiltin(Value_DOT_002__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_003__Vector, NULL);
    float noise_texture_3_w__Value;

    noise_texture_3_w__Value = noise_texture_3_w_FROM_geo_mud;
    float Noise_SPACE_Texture_DOT_002__Fac;
    node_shader_tex_noise(4, position, noise_texture_3_w__Value, 10.0, 16.0,
                          0.5, 0.0, &Noise_SPACE_Texture_DOT_002__Fac, NULL);
    float Value_DOT_003__Value;

    Value_DOT_003__Value = Value_DOT_003_FROM_geo_mud;
    float3_nonbuiltin Vector_SPACE_Math_DOT_004__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            float3_nonbuiltin(Noise_SPACE_Texture_DOT_002__Fac),
                            float3_nonbuiltin(Value_DOT_003__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_004__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_005__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_003__Vector,
        Vector_SPACE_Math_DOT_004__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_005__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_006__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_ADD, Vector_SPACE_Math_DOT_002__Vector,
        Vector_SPACE_Math_DOT_005__Vector, float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_006__Vector, NULL);
    float Value_DOT_004__Value;

    Value_DOT_004__Value = Value_DOT_004_FROM_geo_mud;
    float3_nonbuiltin Vector_SPACE_Math_DOT_007__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_DIVIDE,
                            Vector_SPACE_Math_DOT_006__Vector,
                            float3_nonbuiltin(Value_DOT_004__Value),
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_007__Vector, NULL);
    float Float_SPACE_Curve__Value;

    float Float_SPACE_Curve_values[256]{0.0,
                                        0.0016043668147176504,
                                        0.003208733396604657,
                                        0.004813142586499453,
                                        0.0064210896380245686,
                                        0.008029169403016567,
                                        0.009637443348765373,
                                        0.011253203265368938,
                                        0.012869006022810936,
                                        0.014486467465758324,
                                        0.016114145517349243,
                                        0.017742028459906578,
                                        0.01937498338520527,
                                        0.021019084379076958,
                                        0.022663796320557594,
                                        0.024319667369127274,
                                        0.02598494663834572,
                                        0.02765166386961937,
                                        0.029338931664824486,
                                        0.0310308076441288,
                                        0.032730065286159515,
                                        0.0344528891146183,
                                        0.03617759793996811,
                                        0.03792363777756691,
                                        0.03968346118927002,
                                        0.04144925996661186,
                                        0.04325121268630028,
                                        0.045055996626615524,
                                        0.04688709229230881,
                                        0.04873772710561752,
                                        0.05059902369976044,
                                        0.0525028221309185,
                                        0.05441176891326904,
                                        0.05635911226272583,
                                        0.05832522362470627,
                                        0.06031513586640358,
                                        0.062343962490558624,
                                        0.06438267976045609,
                                        0.0664803683757782,
                                        0.06858783215284348,
                                        0.07074134051799774,
                                        0.07292257994413376,
                                        0.0751364603638649,
                                        0.07739631831645966,
                                        0.07967518270015717,
                                        0.08202134817838669,
                                        0.08438093215227127,
                                        0.08680500090122223,
                                        0.08925358951091766,
                                        0.09175460040569305,
                                        0.09429611265659332,
                                        0.09687777608633041,
                                        0.09951572120189667,
                                        0.10218139737844467,
                                        0.10492168366909027,
                                        0.10768212378025055,
                                        0.11051925271749496,
                                        0.11338293552398682,
                                        0.11631070822477341,
                                        0.11927946656942368,
                                        0.12229926884174347,
                                        0.12537431716918945,
                                        0.1284870207309723,
                                        0.13167127966880798,
                                        0.1348818987607956,
                                        0.138173907995224,
                                        0.14149227738380432,
                                        0.14487674832344055,
                                        0.1483020931482315,
                                        0.15177783370018005,
                                        0.155308797955513,
                                        0.15887421369552612,
                                        0.16251787543296814,
                                        0.1661847084760666,
                                        0.16991885006427765,
                                        0.17369011044502258,
                                        0.1775229275226593,
                                        0.18141934275627136,
                                        0.18535347282886505,
                                        0.18938976526260376,
                                        0.19347301125526428,
                                        0.19762849807739258,
                                        0.2018873393535614,
                                        0.20619745552539825,
                                        0.21062737703323364,
                                        0.2151203453540802,
                                        0.21971985697746277,
                                        0.2244117558002472,
                                        0.2292041778564453,
                                        0.23411160707473755,
                                        0.23912036418914795,
                                        0.24425995349884033,
                                        0.24950766563415527,
                                        0.25489526987075806,
                                        0.26040273904800415,
                                        0.2660525441169739,
                                        0.2718372344970703,
                                        0.27776068449020386,
                                        0.2838652431964874,
                                        0.29008713364601135,
                                        0.29647910594940186,
                                        0.3030191659927368,
                                        0.3096989393234253,
                                        0.3165503144264221,
                                        0.3235352039337158,
                                        0.3306483030319214,
                                        0.3379075527191162,
                                        0.3452858328819275,
                                        0.352755069732666,
                                        0.3603247404098511,
                                        0.3679826259613037,
                                        0.3756886124610901,
                                        0.3834362030029297,
                                        0.3912111520767212,
                                        0.3989912271499634,
                                        0.4067641794681549,
                                        0.414509654045105,
                                        0.422209769487381,
                                        0.42986902594566345,
                                        0.4374603033065796,
                                        0.44496944546699524,
                                        0.4524158239364624,
                                        0.45976728200912476,
                                        0.4670262634754181,
                                        0.4741993248462677,
                                        0.48126763105392456,
                                        0.48825109004974365,
                                        0.4951298236846924,
                                        0.5019197463989258,
                                        0.5087183117866516,
                                        0.5155714154243469,
                                        0.5226062536239624,
                                        0.5298738479614258,
                                        0.5373189449310303,
                                        0.5450230836868286,
                                        0.5529754161834717,
                                        0.5611304044723511,
                                        0.5695615410804749,
                                        0.5782675743103027,
                                        0.5871987342834473,
                                        0.5963503122329712,
                                        0.6057105660438538,
                                        0.6152567863464355,
                                        0.6249498128890991,
                                        0.6347288489341736,
                                        0.644528865814209,
                                        0.6542873382568359,
                                        0.6639405488967896,
                                        0.6734288334846497,
                                        0.6827014684677124,
                                        0.6917197704315186,
                                        0.7004590034484863,
                                        0.7088879346847534,
                                        0.7169663906097412,
                                        0.7247180938720703,
                                        0.7321847081184387,
                                        0.7393674850463867,
                                        0.7462114691734314,
                                        0.7527865171432495,
                                        0.7591356635093689,
                                        0.7651966214179993,
                                        0.7710633277893066,
                                        0.7766870260238647,
                                        0.7821179032325745,
                                        0.7873613238334656,
                                        0.7924315333366394,
                                        0.7973339557647705,
                                        0.8020923733711243,
                                        0.8066907525062561,
                                        0.8111834526062012,
                                        0.8155117034912109,
                                        0.819768488407135,
                                        0.8238635063171387,
                                        0.8278666734695435,
                                        0.8318034410476685,
                                        0.8356401920318604,
                                        0.8394440412521362,
                                        0.8431496620178223,
                                        0.8468108177185059,
                                        0.8504165410995483,
                                        0.8539533019065857,
                                        0.8574584126472473,
                                        0.8608693480491638,
                                        0.8642441630363464,
                                        0.8675463199615479,
                                        0.8707906603813171,
                                        0.8739917278289795,
                                        0.8771107196807861,
                                        0.8802037239074707,
                                        0.8832027912139893,
                                        0.8861608505249023,
                                        0.8890620470046997,
                                        0.8918980360031128,
                                        0.8947031497955322,
                                        0.8974191546440125,
                                        0.9001080989837646,
                                        0.9027222394943237,
                                        0.9052907228469849,
                                        0.9078165888786316,
                                        0.9102724194526672,
                                        0.9127112627029419,
                                        0.9150595664978027,
                                        0.9173811078071594,
                                        0.9196515083312988,
                                        0.9218699336051941,
                                        0.9240665435791016,
                                        0.926186203956604,
                                        0.9282881021499634,
                                        0.9303330779075623,
                                        0.9323399066925049,
                                        0.9343235492706299,
                                        0.9362432360649109,
                                        0.93815016746521,
                                        0.9400035738945007,
                                        0.9418265223503113,
                                        0.9436311721801758,
                                        0.9453791379928589,
                                        0.9471170902252197,
                                        0.9488133788108826,
                                        0.9504806399345398,
                                        0.952140212059021,
                                        0.953744649887085,
                                        0.9553403854370117,
                                        0.9569128155708313,
                                        0.9584519863128662,
                                        0.9599868059158325,
                                        0.9614845514297485,
                                        0.9629665017127991,
                                        0.9644445776939392,
                                        0.965881884098053,
                                        0.9673143029212952,
                                        0.968735933303833,
                                        0.9701295495033264,
                                        0.9715204238891602,
                                        0.9728959202766418,
                                        0.9742527604103088,
                                        0.9756080508232117,
                                        0.9769469499588013,
                                        0.9782734513282776,
                                        0.979599118232727,
                                        0.9809098243713379,
                                        0.9822119474411011,
                                        0.983513593673706,
                                        0.984803318977356,
                                        0.9860866665840149,
                                        0.9873696565628052,
                                        0.9886447191238403,
                                        0.9899144768714905,
                                        0.991183876991272,
                                        0.9924492835998535,
                                        0.9937101006507874,
                                        0.9949707388877869,
                                        0.99623042345047,
                                        0.9974868893623352,
                                        0.99874347448349,
                                        0.9999998807907104};
    int Float_SPACE_Curve_table_size = 256;
    node_float_curve(Float_SPACE_Curve_values, Float_SPACE_Curve_table_size,
                     1.0, float(Vector_SPACE_Math_DOT_007__Vector),
                     &Float_SPACE_Curve__Value);
    float3_nonbuiltin Vector_SPACE_Math_DOT_008__Vector;
    node_shader_vector_math(
        NODE_VECTOR_MATH_MULTIPLY, float3_nonbuiltin(Float_SPACE_Curve__Value),
        float3_nonbuiltin(dep_pud__Value), float3_nonbuiltin(0.0, 0.0, 0.0),
        1.0, &Vector_SPACE_Math_DOT_008__Vector, NULL);
    float3_nonbuiltin Vector_SPACE_Math_DOT_009__Vector;
    node_shader_vector_math(NODE_VECTOR_MATH_MULTIPLY,
                            Vector_SPACE_Math_DOT_008__Vector, normal,
                            float3_nonbuiltin(0.0, 0.0, 0.0), 1.0,
                            &Vector_SPACE_Math_DOT_009__Vector, NULL);
    float4_nonbuiltin Mix_DOT_001__Color;
    node_shader_mix_rgb(0, MA_RAMP_MIX, float(Mix__Color),
                        float4_nonbuiltin(Vector_SPACE_Math__Vector),
                        float4_nonbuiltin(Vector_SPACE_Math_DOT_009__Vector),
                        &Mix_DOT_001__Color);

    if (offset != NULL)
        *offset = Mix_DOT_001__Color;
}
