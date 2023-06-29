// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma


namespace data {
    int meta_param, second_meta_param;
    int *d_i_params=NULL, *second_d_i_params=NULL, *third_d_i_params=NULL;
    float *d_f_params=NULL, *second_d_f_params=NULL, *third_d_f_params=NULL;
}

extern "C" {
    void init(
        int meta_param_, int second_meta_param_,
        size_t i_size, int *i_params, size_t f_size, float *f_params,
        size_t second_i_size, int *second_i_params, size_t second_f_size, float *second_f_params,
        size_t third_i_size, int *third_i_params, size_t third_f_size, float *third_f_params
    ) {
        using namespace data;
        meta_param = meta_param_;
        second_meta_param = second_meta_param_;
        if (i_size > 0) {
            d_i_params = i_params;
        }
        if (f_size > 0) {
            d_f_params = f_params;
        }
        if (second_i_size > 0) {
            second_d_i_params = second_i_params;
        }
        if (second_f_size > 0) {
            second_d_f_params = second_f_params;
        }
        if (third_i_size > 0) {
            third_d_i_params = third_i_params;
        }
        if (third_f_size > 0) {
            third_d_f_params = third_f_params;
        }
    }

    void cleanup() {
    }


}