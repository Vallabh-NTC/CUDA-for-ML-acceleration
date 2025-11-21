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

// Write a single pixel on the Y plane (luma) only.
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

// Write a single pixel with luma + chroma, approximating a color.
// Here we use it to draw a "green-ish" skeleton in NV12.
__device__ inline void set_nv12_colored_pixel(
    uint8_t* dY,
    uint8_t* dUV,
    int W,
    int H,
    int pitch,
    int x,
    int y,
    uint8_t yVal,
    uint8_t uVal,
    uint8_t vVal
) {
    if (x < 0 || y < 0 || x >= W || y >= H) return;

    // Set luma
    dY[y * pitch + x] = yVal;

    // NV12: UV is 4:2:0, interleaved, one pair per 2x2 block.
    // Only update chroma on even x,y to avoid fighting with neighbors too much.
    if ((x & 1) == 0 && (y & 1) == 0) {
        int uv_y = y / 2;
        int uv_x = x;
        if (uv_y >= 0 && uv_y < H/2 && uv_x >= 0 && uv_x+1 < W) {
            uint8_t* rowUV = dUV + uv_y * pitch;
            rowUV[uv_x + 0] = uVal;
            rowUV[uv_x + 1] = vVal;
        }
    }
}

// Simple Bresenham-like line drawer.
// If useColor == true, we also write UV to get a green-ish, **thicker** line.
// Otherwise we only touch Y (white/gray, thin skeleton).
__device__ void draw_line(
    uint8_t* dY,
    uint8_t* dUV,
    int W,
    int H,
    int pitch,
    int x0,
    int y0,
    int x1,
    int y1,
    bool useColor
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

    // White-ish Y and green-ish UV (BT.601 approx for RGB(0,255,0)):
    const uint8_t yWhite = 255;
    const uint8_t yGreen = 180;  // slightly lower luma
    const uint8_t uGreen = 43;
    const uint8_t vGreen = 21;

    while (true) {
        if (useColor) {
            // Thicker green line: draw a small 3x3 block around the center.
            for (int oy = -1; oy <= 1; ++oy) {
                for (int ox = -1; ox <= 1; ++ox) {
                    set_nv12_colored_pixel(
                        dY, dUV, W, H, pitch,
                        x + ox, y + oy,
                        yGreen, uGreen, vGreen
                    );
                }
            }
        } else {
            // Thin white line: single pixel
            set_nv12_luma_pixel(dY, W, H, pitch, x, y, yWhite);
        }

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

// Kernel: for each point, draw a small filled circle.
// If useColor == true → bigger green circle; otherwise smaller white circle.
__global__ void k_draw_points_nv12(
    uint8_t* dY,
    uint8_t* dUV,
    int W,
    int H,
    int pitch,
    const Point2D* points,
    int numPoints,
    bool useColor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    Point2D p = points[idx];

    // Skip points with non-positive confidence (can be used as a mask)
    if (p.conf <= 0.0f) return;

    // Thicker/bigger circles when useColor==true (PEACE)
    const int radius = useColor ? 7 : 4;
    const int r2 = radius * radius;

    int cx = p.x;
    int cy = p.y;

    // Bounding box for the circle
    int x0 = clampi(cx - radius, 0, W - 1);
    int x1 = clampi(cx + radius, 0, W - 1);
    int y0 = clampi(cy - radius, 0, H - 1);
    int y1 = clampi(cy + radius, 0, H - 1);

    // Colors
    const uint8_t yWhite = 255;
    const uint8_t yGreen = 180;
    const uint8_t uGreen = 43;
    const uint8_t vGreen = 21;

    // Simple disk fill
    for (int y = y0; y <= y1; ++y) {
        int dy = y - cy;
        for (int x = x0; x <= x1; ++x) {
            int dx = x - cx;
            if (dx * dx + dy * dy <= r2) {
                if (useColor) {
                    set_nv12_colored_pixel(
                        dY, dUV, W, H, pitch,
                        x, y,
                        yGreen, uGreen, vGreen
                    );
                } else {
                    set_nv12_luma_pixel(dY, W, H, pitch, x, y, yWhite);
                }
            }
        }
    }
}

// Kernel: draw simple "chain" skeleton lines between consecutive keypoints.
// Edge list: (0-1), (1-2), (2-3), ... (numPoints-2, numPoints-1).
__global__ void k_draw_skeleton_nv12(
    uint8_t* dY,
    uint8_t* dUV,
    int W,
    int H,
    int pitch,
    const Point2D* points,
    int numPoints,
    bool useColor
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= numPoints - 1) return;

    int i0 = e;
    int i1 = e + 1;

    Point2D p0 = points[i0];
    Point2D p1 = points[i1];

    // Only draw if both endpoints are valid / confident
    if (p0.conf <= 0.0f || p1.conf <= 0.0f) return;

    draw_line(
        dY,
        dUV,
        W,
        H,
        pitch,
        p0.x,
        p0.y,
        p1.x,
        p1.y,
        useColor
    );
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
    bool isPeace,          // if true, draw points/skeleton green & thicker
    cudaStream_t stream
) {
    if (!dY || !dUV || !hPoints || numPoints <= 0 || W <= 0 || H <= 0 || pitch <= 0) {
        return;
    }

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
            numPoints,
            isPeace   // useColor -> green + thick if PEACE, white if not
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
            numPoints,
            isPeace   // thicker green if PEACE
        );
    }

    cudaFree(dPoints);
}

} // namespace draw
