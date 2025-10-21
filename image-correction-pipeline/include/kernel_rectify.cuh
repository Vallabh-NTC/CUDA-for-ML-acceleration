/**
 * @file kernel_rectify.cuh
 * @brief CUDA interfaces for:
 *   1) fisheye rectification (NV12 → NV12, equidistant model r = f_fish * theta)
 *   2) post-rectification center-crop/zoom (NV12 → NV12, same size)
 */

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace icp {

/**
 * @brief Launch CUDA kernel for fisheye rectification (NV12 → NV12).
 * Equidistant projection: r = f_fish * theta.
 */
void launch_rectify_nv12(
    const uint8_t* d_src_y,  int src_w, int src_h, int src_pitch_y,
    const uint8_t* d_src_uv,                   int src_pitch_uv,
    uint8_t* d_dst_y,        int dst_w, int dst_h, int dst_pitch_y,
    uint8_t* d_dst_uv,                       int dst_pitch_uv,
    // Geometry parameters (fisheye image circle on source)
    float cx_f, float cy_f, float r_f,
    // Projection params
    float f_fish, float fx, float cx_rect, float cy_rect,
    cudaStream_t stream);

/**
 * @brief Center-crop/zoom an already-rectified NV12 frame to the same size.
 * @param crop_frac fraction to remove on each side in rectified domain [0..0.45].
 *                  Example: 0.20 → keep central 60% and scale it to full frame.
 *
 * Source and destination can be different buffers (recommended). If the same,
 * the read/modify overlap would cause artifacts.
 */
void launch_crop_center_nv12(
    const uint8_t* d_src_y,  int W, int H, int src_pitch_y,
    const uint8_t* d_src_uv,             int src_pitch_uv,
    uint8_t* d_dst_y,                    int dst_pitch_y,
    uint8_t* d_dst_uv,                   int dst_pitch_uv,
    float crop_frac,
    cudaStream_t stream);

} // namespace icp
