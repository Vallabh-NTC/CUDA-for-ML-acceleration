// color_ops.cuh
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace icp {

// Manual-only color/tone parameters. All values are clamped in the launcher.
struct ColorParams {
    bool  enable      = true;

    // Manual exposure in EV (negative = darker).
    float exposure_ev = 0.0f;   // neutral

    // Tone (neutral defaults)
    float contrast    = 1.00f;  // neutral (UI 0)
    float highlights  = 0.00f;  // neutral
    float shadows     = 0.00f;  // neutral
    float whites      = 0.00f;  // neutral
    float gamma       = 1.00f;  // neutral

    // Added controls (neutral)
    float brightness  = 0.0f;   // neutral
    float brilliance  = 0.0f;   // neutral
    float sharpness   = 0.0f;   // off

    // Color
    float saturation  = 1.00f;  // neutral (UI 0 -> 1.0)
    bool  tv_range    = true;   
};

void launch_tone_saturation_nv12(
    uint8_t* dY, int W, int H, int pitchY,
    uint8_t* dUV,            int pitchUV,
    const ColorParams& p,
    cudaStream_t stream);

} // namespace icp
