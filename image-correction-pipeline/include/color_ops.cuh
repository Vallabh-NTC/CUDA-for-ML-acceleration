#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace icp {

// Manual-only color/tone parameters. All values are clamped in the launcher.
struct ColorParams {
    bool  enable      = true;

    // Manual exposure in EV (negative = darker). Range: [-2.0 .. +2.0]
    float exposure_ev = 0.0f;

    // Tone (iPhone-like)
    float contrast    = 1.10f;  // [0.50 .. 1.80] S-curve around mid-gray
    float highlights  = -0.30f; // [-1.0 .. +1.0] <0 = earlier soft-knee
    float shadows     = 0.25f;  // [-1.0 .. +1.0] >0 = lift shadows
    float whites      = -0.20f; // [-1.0 .. +1.0] <0 = lower white ceiling
    float gamma       = 1.00f;  // [0.70 .. 1.30]

    // Added controls
    float brightness  = 0.0f;   // [-1.0 .. +1.0] mid-shift
    float brilliance  = 0.0f;   // [-1.0 .. +1.0] mid-boost with edge-protection
    float sharpness   = 0.0f;   // [0.0 .. 1.0] unsharp mask amount

    // Color
    float saturation  = 1.15f;  // [0.50 .. 1.50]
    bool  tv_range    = true;   // true: clamp Y to [16..235]
};

// Launch: in-place on NV12 planes
void launch_tone_saturation_nv12(
    uint8_t* dY, int W, int H, int pitchY,
    uint8_t* dUV,            int pitchUV,
    const ColorParams& p,
    cudaStream_t stream);

} // namespace icp
