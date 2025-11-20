#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace draw {

// Simple 2D point in image coordinates
struct Point2D {
    int   x;     // x coordinate in image space (pixels)
    int   y;     // y coordinate in image space (pixels)
    float conf;  // confidence score (e.g. heatmap peak value)
};

/**
 * Draws a small "dot" for each hand keypoint directly on an NV12 frame.
 *
 * The function:
 *   - uploads the keypoints (host array) to a small device buffer,
 *   - launches a CUDA kernel that draws a filled circle for each point
 *     on the Y plane (luma) of the NV12 image,
 *   - optionally you can later extend it to modify UV for colored dots.
 *
 * NOTE:
 *   - This is a simple implementation meant for clarity.
 *   - For high performance, you may want to keep the device buffer
 *     persistent inside your per-instance state instead of allocating/
 *     freeing it every frame.
 *
 * @param dY       Device pointer to Y plane (NV12)
 * @param dUV      Device pointer to UV plane (NV12, interleaved)
 * @param W        Image width in pixels
 * @param H        Image height in pixels
 * @param pitch    Line pitch in bytes (for both Y and UV planes)
 * @param hPoints  Host pointer to an array of Point2D (size = numPoints)
 * @param numPoints Number of points in hPoints
 * @param stream   CUDA stream to enqueue the work on
 */
void launch_draw_hand_points_nv12(
    uint8_t* dY,
    uint8_t* dUV,
    int W,
    int H,
    int pitch,
    const Point2D* hPoints,
    int numPoints,
    cudaStream_t stream
);

} // namespace draw
