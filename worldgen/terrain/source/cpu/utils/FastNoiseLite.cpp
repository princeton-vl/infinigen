using namespace std;
#define DEVICE_FUNC
#define CONSTANT_ARRAY const
#include "../../common/utils/vectors.h"
#include "../../common/utils/FastNoiseLite.h"


extern "C" {

    void perlin_call(
        size_t size,
        float3_nonbuiltin *positions,
        float *values,
        int seed, int octaves, float freq
    ) {
        for (size_t idx = 0; idx < size; idx++) {
            values[idx] = Perlin(positions[idx].x, positions[idx].y, positions[idx].z, seed, octaves, freq);
        }
    }

}