#include "kernel_draw_hand_points_nv12.cuh"

#include <cuda_runtime.h>

namespace draw {

// -----------------------------------------------------------------------------
// Device helpers
// -----------------------------------------------------------------------------

// Clamp integer x to [lo, hi]
__device__ inline int clampi(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// Write a single pixel on the Y plane (luma).
// For now we only touch the Y plane to avoid unexpected color shifts.
// You can extend this later to tint the skeleton by modifying UV as well.
__device__ inline void set_nv12_luma_pixel(
    uint8_t* dY,
    int W,
    int H,
    int pitch,
    int x,
    int y,
    uint8_t yVal
) {
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    dY[y * pitch + x] = yVal;
}

// Simple Bresenham-like line drawer on the Y plane
__device__ void draw_line_luma(
    uint8_t* dY,
    int W,
    int H,
    int pitch,
    int x0,
    int y0,
    int x1,
    int y1,
    uint8_t yVal
) {
    // Trivial reject for clearly out-of-bounds lines (optional, conservative)
    if ((x0 < 0 && x1 < 0) || (x0 >= W && x1 >= W) ||
        (y0 < 0 && y1 < 0) || (y0 >= H && y1 >= H)) {
        return;
    }

    int dx = abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;

    int x = x0;
    int y = y0;

    while (true) {
        set_nv12_luma_pixel(dY, W, H, pitch, x, y, yVal);
        if (x == x1 && y == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y += sy;
        }
    }
}

// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

// Kernel: for each point, draw a small filled circle on the Y plane.
__global__ void k_draw_points_nv12(
    uint8_t* dY,
    uint8_t* dUV,   // currently unused, kept for future color extensions
    int W,
    int H,
    int pitch,
    const Point2D* points,
    int numPoints
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    Point2D p = points[idx];

    // Skip points with non-positive confidence (can be used as a mask)
    if (p.conf <= 0.0f) return;

    const int radius = 4;
    const int r2 = radius * radius;

    int cx = p.x;
    int cy = p.y;

    // Bounding box for the circle
    int x0 = clampi(cx - radius, 0, W - 1);
    int x1 = clampi(cx + radius, 0, W - 1);
    int y0 = clampi(cy - radius, 0, H - 1);
    int y1 = clampi(cy + radius, 0, H - 1);

    // Simple disk fill
    for (int y = y0; y <= y1; ++y) {
        int dy = y - cy;
        for (int x = x0; x <= x1; ++x) {
            int dx = x - cx;
            if (dx * dx + dy * dy <= r2) {
                // 255 = bright luma (white-ish dot)
                set_nv12_luma_pixel(dY, W, H, pitch, x, y, 255);
            }
        }
    }
}

// Kernel: draw simple "chain" skeleton lines between consecutive keypoints.
// Edge list: (0-1), (1-2), (2-3), ... (numPoints-2, numPoints-1).
__global__ void k_draw_skeleton_nv12(
    uint8_t* dY,
    uint8_t* dUV,   // currently unused
    int W,
    int H,
    int pitch,
    const Point2D* points,
    int numPoints
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= numPoints - 1) return;

    int i0 = e;
    int i1 = e + 1;

    Point2D p0 = points[i0];
    Point2D p1 = points[i1];

    // Only draw if both endpoints are valid / confident
    if (p0.conf <= 0.0f || p1.conf <= 0.0f) return;

    draw_line_luma(dY, W, H, pitch, p0.x, p0.y, p1.x, p1.y, 255);
}

// -----------------------------------------------------------------------------
// Host launcher
// -----------------------------------------------------------------------------

void launch_draw_hand_points_nv12(
    uint8_t* dY,
    uint8_t* dUV,
    int W,
    int H,
    int pitch,
    const Point2D* hPoints,
    int numPoints,
    cudaStream_t stream
) {
    if (!dY || !dUV || !hPoints || numPoints <= 0 || W <= 0 || H <= 0 || pitch <= 0) {
        return;
    }

    // NOTE:
    //  This implementation allocates a temporary device buffer for the points
    //  on each call. For a high-frequency pipeline, consider storing this
    //  buffer in your per-instance state and reusing it instead.
    Point2D* dPoints = nullptr;
    size_t bytes = static_cast<size_t>(numPoints) * sizeof(Point2D);

    if (cudaMalloc(&dPoints, bytes) != cudaSuccess) {
        return;
    }

    // Async upload of keypoints to device
    if (cudaMemcpyAsync(dPoints, hPoints, bytes,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        cudaFree(dPoints);
        return;
    }

    // --- 1) Draw circular points for each keypoint ---
    {
        const int blockSizePts = 64;
        const int gridSizePts  = (numPoints + blockSizePts - 1) / blockSizePts;

        k_draw_points_nv12<<<gridSizePts, blockSizePts, 0, stream>>>(
            dY,
            dUV,
            W,
            H,
            pitch,
            dPoints,
            numPoints
        );
    }

    // --- 2) Draw "chain" skeleton lines between consecutive keypoints ---
    if (numPoints > 1) {
        const int numEdges = numPoints - 1;
        const int blockSizeEdges = 64;
        const int gridSizeEdges  = (numEdges + blockSizeEdges - 1) / blockSizeEdges;

        k_draw_skeleton_nv12<<<gridSizeEdges, blockSizeEdges, 0, stream>>>(
            dY,
            dUV,
            W,
            H,
            pitch,
            dPoints,
            numPoints
        );
    }

    // Free the temporary buffer (for higher performance, keep it persistent).
    cudaFree(dPoints);
}

} // namespace draw
