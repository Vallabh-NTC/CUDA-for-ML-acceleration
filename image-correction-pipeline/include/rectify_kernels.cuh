// image-correction-pipeline/include/rectify_kernels.cuh
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "rectify_config.hpp"

// ----------------------------------------------------------------------------
// launch_rectify_kernel
// ----------------------------------------------------------------------------
// Host-side launcher for the CUDA kernel performing:
//   (a) Equidistant fisheye → perspective remap
//   (b) Color pipeline: brightness, contrast, saturation, white balance, gamma
//
// IMPORTANT INPUT FORMAT:
//   - d_src / d_dst must reference PITCH-LINEAR ABGR (one 4-byte pixel per texel).
//   - ABGR here means bytes in order [A, B, G, R] per pixel.
//   - nvivafilter gives us CU_EGL_COLOR_FORMAT_ABGR for RGBA in NVMM.
//   - If your camera outputs UYVY/YUY2/NV12, convert with `nvvidconv` to RGBA
//     *before* the filter.
//
// IN-PLACE OPERATION:
//   - d_src and d_dst may alias the same base pointer to process in-place,
//     provided the kernel only reads src coordinates and writes dst coords
//     once per pixel.
//
// STRIDES:
//   - src_pitch and dst_pitch are in BYTES (not pixels).
// ----------------------------------------------------------------------------
void launch_rectify_kernel(
    const uint8_t* d_src, int src_w, int src_h, int src_pitch,
    uint8_t* d_dst,       int dst_w, int dst_h, int dst_pitch,
    const RectifyConfig& cfg,
    cudaStream_t stream);
