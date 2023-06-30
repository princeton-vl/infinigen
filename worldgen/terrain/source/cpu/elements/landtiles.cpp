extern "C" {

    void call(
        size_t size,
        float3_nonbuiltin *positions,
        float *sdfs,
        float *auxs
    ) {
        using namespace data;
        if (auxs == NULL) n_auxiliaries = 0;
        #pragma omp parallel for
        for (size_t idx = 0; idx < size; idx++) {
            landtiles(
                positions[idx], sdfs + idx, auxs + n_auxiliaries * idx, meta_param,
                d_i_params, d_f_params, second_d_i_params, second_d_f_params
            );
        }
    }

}