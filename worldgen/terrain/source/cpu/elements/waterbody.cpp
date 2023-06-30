extern "C" {
    void call(
        size_t size,
        float3_nonbuiltin *positions,
        float *sdfs,
        float *auxs
    ) {
        using namespace data;
        int n_auxiliaries = 1;
        if (auxs == NULL) n_auxiliaries = 0;
        #pragma omp parallel for
        for (size_t idx = 0; idx < size; idx++) {
            waterbody(
                positions[idx], sdfs + idx, auxs + n_auxiliaries * idx, meta_param, second_meta_param,
                d_i_params, d_f_params, second_d_i_params, second_d_f_params, third_d_i_params, third_d_f_params
            );
        }
    }
}