#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace icp {

// Fixed config (no CLI). Tweak here if you ever recalibrate.
struct RectifyConfig {
    float fish_fov_deg = 195.1f; // equidistant fisheye FOV (deg)
    float out_hfov_deg = 90.0f;  // desired perspective HFOV (deg)
    float cx_f = 959.50f;        // fisheye circle center (pixels)
    float cy_f = 539.50f;
    float r_f  = 1100.77f;       // fisheye circle radius (pixels)
    int   out_width = 1920;      // keep aspect from source

     // --- Color controls (host-configurable) ---
    // Contrast & brightness are applied in linear 8-bit space.

    float brightness = 0.0f;   // additive offset per channel in [0..255] units
    float contrast   = 1.0f;   // multiplicative scale (>1 more contrast, <1 less)

    // Saturation is applied by mixing with luma (Rec.709). 1.0 = unchanged.
    // 0.0 = grayscale, >1 boosts colors.
    float saturation = 1.0f;   // recommended range [0.0, 2.0]

    // Gamma is applied at the end on normalized [0,1] channels.
    // gamma < 1 brightens mid-tones; gamma > 1 darkens them. 1.0 = unchanged.
    float gamma = 1.0f;        // recommended range [0.5, 2.2]

    // Simple white-balance multipliers per channel (applied in linear 8-bit space before gamma).
    // 1.0 = unchanged. Tweak to bias temperature/tint.
    float wb_r = 1.0f;
    float wb_g = 1.0f;
    float wb_b = 1.0f;
};

// Launch undistortion (equidistant -> perspective) on RGBA frame.
// d_src_rgba/d_dst_rgba are *device* pointers (aliases of pinned host via cudaHostGetDevicePointer).
// src_stride/dst_stride in BYTES.
void fisheye_rectify_rgba(
    const uint8_t* d_src_rgba, int src_w, int src_h, size_t src_stride,
    uint8_t* d_dst_rgba,       int dst_w, int dst_h, size_t dst_stride,
    const RectifyConfig& cfg,
    cudaStream_t stream);

} 
