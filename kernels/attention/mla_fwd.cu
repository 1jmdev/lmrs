#include <cuda_bf16.h>

extern "C" __global__ void mla_fwd_bf16(
    const __nv_bfloat16 *__restrict__ latent,
    const __nv_bfloat16 *__restrict__ proj,
    __nv_bfloat16 *__restrict__ out,
    const int rows,
    const int latent_dim,
    const int out_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * out_dim;
    if (idx >= total) return;
    const int row = idx / out_dim;
    const int col = idx % out_dim;
    float acc = 0.0f;
    for (int k = 0; k < latent_dim; ++k) {
        acc += __bfloat162float(latent[row * latent_dim + k]) * __bfloat162float(proj[k * out_dim + col]);
    }
    out[idx] = __float2bfloat16(acc);
}
