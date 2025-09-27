#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace wire {

// ---------- One-time: derive per-column [top..bot] from the binary mask -----
void top_bottom_from_mask(
    const uint8_t* dMask, int maskPitch,
    int W, int H,
    int* dTop, int* dBot,
    cudaStream_t stream);

// ---------- Per-frame: make masked band disappear by copying neighbors -------
void disappear_band_nv12(
    uint8_t* dY,  int pitchY,
    uint8_t* dUV, int pitchUV,
    int W, int H,
    const int* dTop, const int* dBot,
    int offTop, int offBot,       // number of rows above/below to copy
    float sigmaY, float sigmaUV,  // ignored, ABI compatibility
    cudaStream_t stream);

// ---------- Optional debug overlays -----------------------------------------
void overlay_mask_nv12(
    uint8_t* dY,  int W, int H, int pitchY,
    uint8_t* dUV, int pitchUV,
    const uint8_t* dMask, int maskPitch,
    uint8_t y_on, uint8_t u_on, uint8_t v_on,
    cudaStream_t stream);

void overlay_polyline_nv12(
    uint8_t* dY,  int pY,
    uint8_t* dUV, int pUV,
    int W, int H,
    const int* dTop,
    uint8_t y_on, uint8_t u_on, uint8_t v_on,
    cudaStream_t stream);

} // namespace wire
