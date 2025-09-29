#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace wire {

// Simplest constant-shift inpaint per mask:
// For mask!=0, copy NV12 from donor at (x+dx, y+dy)
void apply_mask_shift_nv12(
    uint8_t* dY,  int pitchY,
    uint8_t* dUV, int pitchUV,
    int W, int H,
    const uint8_t* dMask, int maskPitch,
    float dx, float dy,
    cudaStream_t stream);

} // namespace wire
