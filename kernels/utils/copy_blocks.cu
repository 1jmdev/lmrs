#include <stdint.h>

extern "C" __global__ void copy_blocks_u8(
    const uint8_t *__restrict__ src,
    uint8_t *__restrict__ dst,
    const int *__restrict__ block_map,
    const int block_bytes,
    const int blocks
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = blocks * block_bytes;
    if (idx >= total) return;
    const int block = idx / block_bytes;
    const int offset = idx % block_bytes;
    dst[block * block_bytes + offset] = src[block_map[block] * block_bytes + offset];
}
