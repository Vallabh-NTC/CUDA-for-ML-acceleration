/**
 * @file kernel_rectify.cuh
 * @brief CUDA interface for fisheye rectification (NV12 → NV12).
 *
 * Provides a single launcher:
 *   - launch_rectify_nv12(): warps fisheye NV12 input into perspective output.
 *
 * Parameters:
 *   - dY/dUV: destination pointers (NV12 frame in-place).
 *   - sY/sUV: scratch copy of the source frame (avoids overwrite).
 *   - Calibration params: fisheye FOV, circle radius, optical center, output FOV.
 *
 * Notes:
 *   - Geometry math: equidistant fisheye projection.
 *   - Output is NV12, same size as input.
 */

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace icp {

// NV12 → NV12 (rettifica geometrica; nessun colore)
void launch_rectify_nv12(
    const uint8_t* d_src_y,  int src_w, int src_h, int src_pitch_y,
    const uint8_t* d_src_uv,                   int src_pitch_uv,
    uint8_t* d_dst_y,        int dst_w, int dst_h, int dst_pitch_y,
    uint8_t* d_dst_uv,                       int dst_pitch_uv,
    // Geometria
    float cx_f, float cy_f, float r_f,
    float f_fish, float fx, float cx_rect, float cy_rect,
    cudaStream_t stream);

} // namespace icp
