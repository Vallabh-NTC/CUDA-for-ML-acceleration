#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace wire {

// Apply mask with a *two-donor* inpaint:
// For every pixel where mask!=0, sample NV12 from (x+dx,y+dy) *and* (x-dx,y-dy)
//   - Y (luma): bilinear from both, 50/50 blend (t weakly tunable in .cu)
//   - UV (chroma): average of the two nearest-neighbour samples
void apply_mask_shift_nv12(
    uint8_t* dY,  int pitchY,
    uint8_t* dUV, int pitchUV,
    int W, int H,
    const uint8_t* dMask, int maskPitch,
    float dx, float dy,
    cudaStream_t stream);

} // namespace wire
