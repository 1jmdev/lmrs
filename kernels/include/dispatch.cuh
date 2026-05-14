#pragma once

#include <cuda_runtime.h>

namespace lmrs {

constexpr int kBlockSize = 256;

inline int div_ceil_int(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

inline dim3 grid_1d(int elements, int block_size = kBlockSize) {
    return dim3(div_ceil_int(elements, block_size), 1, 1);
}

} // namespace lmrs
