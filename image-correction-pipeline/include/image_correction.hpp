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
