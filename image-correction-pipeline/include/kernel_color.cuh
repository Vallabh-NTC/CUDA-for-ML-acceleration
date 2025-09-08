/**
 * @file kernel_color.cuh
 * @brief CUDA kernel interfaces for color processing (NV12).
 *
 * This header declares all GPU kernels for:
 *  - Temporal denoise (reduce flicker/grain using previous frame).
 *  - Local tone mapping (CLAHE-lite style).
 *  - Color grading (contrast/brightness/saturation/gamma + WB).
 *  - Histogram + chroma stats (used for auto-exposure / auto-WB).
 *
 * All functions operate directly on NV12 planes (Y + UV).
 */

#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

namespace icp {

// temporal denoise (NV12)
void launch_temporal_denoise_nv12(
    uint8_t* dY, uint8_t* dUV,
    int pitchY, int pitchUV,
    int W, int H,
    uint8_t* prevY, uint8_t* prevUV, int ppY, int ppUV,
    float sigmaY, float sigmaUV,
    int thrY, int thrUV,
    cudaStream_t stream);

// local tonemap (NV12)
void launch_local_tonemap_nv12(
    uint8_t* dY, int pitchY, int W, int H,
    int radius,
    float amount,
    float hi_start, float hi_end,
    cudaStream_t stream);

// color grade in-place (NV12)
void launch_color_grade_nv12_inplace(
    uint8_t* dY, uint8_t* dUV, int pitchY, int pitchUV, int W, int H,
    const uint8_t lutY[256],
    float contrast, float addY,
    float sat_base, float gamma,
    float wb_r, float wb_g, float wb_b,
    float sat_hi_start, float sat_hi_end, float sat_hi_min,
    cudaStream_t stream);

// histogram + mean UV stats (NV12)
void compute_stats_nv12(const uint8_t* dY, int pitchY,
                        const uint8_t* dUV, int pitchUV,
                        int W, int H,
                        int roi_x, int roi_y, int roi_w, int roi_h,
                        int step,
                        uint32_t hist[256], float* meanU, float* meanV,
                        cudaStream_t stream);

} // namespace icp
