#pragma once
// overlay.hpp — flow field + resultant vector on NV12 EGLImage

#include <cuda_runtime.h>
#include <cstdint>

// Draw flow field arrows (green) over the ROI
void overlay_draw_flow(
    uint8_t      *d_y, uint8_t *d_uv,
    int           pitchY, int pitchUV, int W, int H,
    const float  *d_flow,
    float roi_x0, float roi_x1, float roi_y0, float roi_y1,
    int step, float arrow_scale, float min_mag,
    cudaStream_t stream = 0);

// Draw resultant vector (blue, thick) at center of ROI
// mean_u, mean_v   : mean displacement in px/frame from flow_reduce
// result_scale     : visual amplification (e.g. 8.0)
void overlay_draw_resultant(
    uint8_t    *d_y, uint8_t *d_uv,
    int         pitchY, int pitchUV, int W, int H,
    float       mean_u, float mean_v,
    float       roi_x0, float roi_x1,
    float       roi_y0, float roi_y1,
    float       result_scale,
    cudaStream_t stream = 0);