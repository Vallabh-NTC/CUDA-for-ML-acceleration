#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace wire {

// ------------------- Optional: overlay helpers -------------------

void overlay_mask_nv12(
    uint8_t* dY, int W, int H, int pitchY,
    uint8_t* dUV, int pitchUV,
    const uint8_t* dMask, int maskPitch,
    uint8_t y_on, uint8_t u_on, uint8_t v_on,
    cudaStream_t stream);

void overlay_polyline_nv12(
    uint8_t* dY, int pY,
    uint8_t* dUV, int pUV,
    int W, int H,
    const int* dTop,          // kept for ABI; unused by radial path
    uint8_t y_on, uint8_t u_on, uint8_t v_on,
    cudaStream_t stream);

// ------------------- Radial (fisheye-aligned) disappearance -----

// Per ogni pixel con mask!=0, calcola n = normalize([x-cx, y-cy]).
// Campiona i donor: p_in = p - offIn*n, p_out = p + offOut*n.
// Y = interpolazione lineare tra Y(p_in) e Y(p_out).
// UV = copia dal donor più vicino (no blending).
void disappear_mask_radial_nv12(
    uint8_t* dY, int pitchY,
    uint8_t* dUV, int pitchUV,
    int W, int H,
    const uint8_t* dMask, int maskPitch,
    float cx, float cy,          // fisheye center (pixels)
    float offIn, float offOut,   // donor offsets (pixels)
    cudaStream_t stream);

} // namespace wire
