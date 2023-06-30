extern "C" {
    void call(
        size_t size,
        float3_nonbuiltin *positions,
    ) {
        using namespace data;
        #pragma omp parallel for
        for (size_t idx = 0; idx < size; idx++) {
        }
    }

}