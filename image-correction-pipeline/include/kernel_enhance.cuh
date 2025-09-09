#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace icp {

/**
 * @brief Parameters for image enhancement.
 *
 * Control gamma correction, tone mapping, sharpening, and color saturation.
 */
struct EnhanceParams {
    float gamma;          ///< Gamma correction factor (1.0 = identity)
    float local_tm;       ///< Local tone mapping strength
    float tm_white;       ///< Tone mapping white point
    float sharpen_amount; ///< Sharpening strength
    float sharpen_clip;   ///< Sharpening soft-limit to avoid halos
    float saturation;     ///< Color saturation multiplier
};

/**
 * @brief Launch enhancement pipeline on an NV12 frame.
 *
 * Operations:
 *   - Gamma correction
 *   - Local tone mapping
 *   - Sharpening
 *   - Saturation adjustment
 */
void launch_enhance_nv12(
    uint8_t* dY, int W, int H, int pitchY,
    uint8_t* dUV, int pitchUV,
    const EnhanceParams& p,
    cudaStream_t stream);

/**
 * @brief Adaptive exposure guard to avoid clipping highlights.
 *
 * Builds a histogram (ignores top 25% rows), estimates 99th percentile,
 * and applies a multiplicative gain factor to normalize exposure.
 */
void launch_highlight_guard(
    uint8_t* dY, int W, int H, int pitchY,
    float* pGain,
    cudaStream_t stream);

/**
 * @brief Apply highlight rolloff in top-weighted regions.
 *
 * Useful for compressing overexposed skies.
 */
void launch_highlight_rolloff_top(
    uint8_t* dY, int W, int H, int pitchY,
    float startN, float ymaxN, float strength,
    cudaStream_t stream);

} // namespace icp
