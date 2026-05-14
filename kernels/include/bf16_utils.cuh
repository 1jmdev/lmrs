#pragma once

#include <cuda_bf16.h>

namespace lmrs {

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float value) {
    return __float2bfloat16(value);
}

} // namespace lmrs
