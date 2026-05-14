#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

namespace lmrs {

struct Bf16Pair {
    __nv_bfloat16 x;
    __nv_bfloat16 y;
};

struct Int8Pair {
    int8_t x;
    int8_t y;
};

struct Float4 {
    float x;
    float y;
    float z;
    float w;
};

} // namespace lmrs
