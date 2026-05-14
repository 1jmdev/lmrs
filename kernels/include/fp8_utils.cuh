#pragma once

#include <stdint.h>

namespace lmrs {

__device__ __forceinline__ float e4m3_to_float(uint8_t value, float scale) {
    if (value == 0) return 0.0f;
    const int sign = (value & 0x80) ? -1 : 1;
    const int exponent = (value >> 3) & 0x0f;
    const int mantissa = value & 0x07;
    if (exponent == 0) {
        return sign * scale * ldexpf(static_cast<float>(mantissa), -9);
    }
    return sign * scale * ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
}

__device__ __forceinline__ uint8_t float_to_e4m3(float value, float inv_scale) {
    float scaled = value * inv_scale;
    if (scaled == 0.0f) return 0;
    const uint8_t sign = scaled < 0.0f ? 0x80 : 0;
    scaled = fabsf(scaled);
    int exponent;
    float normalized = frexpf(scaled, &exponent) * 2.0f;
    exponent -= 1;
    const int biased = max(0, min(15, exponent + 7));
    const int mantissa = max(0, min(7, static_cast<int>((normalized - 1.0f) * 8.0f + 0.5f)));
    return sign | static_cast<uint8_t>((biased << 3) | mantissa);
}

} // namespace lmrs
