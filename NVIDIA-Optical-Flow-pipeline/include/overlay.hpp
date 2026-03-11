#pragma once
// overlay.hpp — flow field + resultant vector on NV12 EGLImage

#include <cuda_runtime.h>
#include <cstdint>

// Apply FOE pitch correction to the RAFT flow field in-place.
// Flow layout: [1, 2, H, W] — ch0 = u (horizontal), ch1 = v (vertical)
// For each pixel:  v_corrected = v - (foe_a * u + foe_b)
// Call AFTER flow_reduce (numerical output unaffected) and BEFORE
// overlay_draw_flow / overlay_draw_resultant. No-op if foe_a==0 && foe_b==0.
void foe_correct_flow(float *d_flow, int H, int W,
                      float foe_a, float foe_b,
                      cudaStream_t stream = 0);

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