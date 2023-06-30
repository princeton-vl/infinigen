__global__ void upsidedown_mountains_kernel(
    size_t size,
    float3_nonbuiltin *position,
    float *sdfs,
    int *i_params,
    float *f_params
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
    }
}

extern "C" {


    void call(
        size_t size,
        float3_nonbuiltin *positions,
    ) {
        using namespace data;
        float3_nonbuiltin *d_positions;
        cudaMalloc((void **)&d_positions, size * sizeof(float3_nonbuiltin));
        cudaMemcpy(d_positions, positions, size * sizeof(float3_nonbuiltin), cudaMemcpyHostToDevice);
        float *d_sdfs;
        cudaMalloc((void **)&d_sdfs, size * sizeof(float));
        cudaMemcpy(sdfs, d_sdfs, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_positions);
        cudaFree(d_sdfs);
    }

}