#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

namespace draw {

// Simple 2D point used for hand keypoints on the image.
struct Point2D {
    int   x;
    int   y;
    float conf;  // confidence in [0,1] (can be 1.0 if you don't use it)
};

/**
 * Draw hand keypoints + simple skeleton on an NV12 frame.
 *
 * - dY, dUV : NV12 planes (device pointers)
 * - W, H    : frame size
 * - pitch   : pitch of Y/UV planes
 * - hPoints : host array of keypoints (image coordinates)
 * - numPoints: number of keypoints
 * - isPeace : if true, draw in green-ish color; otherwise white
 * - stream  : CUDA stream
 */
void launch_draw_hand_points_nv12(
    uint8_t* dY,
    uint8_t* dUV,
    int W,
    int H,
    int pitch,
    const Point2D* hPoints,
    int numPoints,
    bool isPeace,
    cudaStream_t stream
);

} // namespace draw
