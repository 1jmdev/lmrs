#include <cuda_bf16.h>

extern "C" __global__ void flash_mha_fwd_bf16(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v,
    __nv_bfloat16 *__restrict__ out,
    const int rows,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    const int row = blockIdx.x;
    const int d = threadIdx.x;
    if (row >= rows || d >= head_dim) return;
    const int q_base = row * head_dim;
    float weighted = 0.0f;
    float denom = 0.0f;
    for (int s = 0; s < seq_len; ++s) {
        float score = 0.0f;
        const int kv_base = (row * seq_len + s) * head_dim;
        for (int i = 0; i < head_dim; ++i) {
            score += __bfloat162float(q[q_base + i]) * __bfloat162float(k[kv_base + i]);
        }
        const float p = expf(score * scale);
        weighted += p * __bfloat162float(v[kv_base + d]);
        denom += p;
    }
    out[q_base + d] = __float2bfloat16(weighted / fmaxf(denom, 1.0e-20f));
}
