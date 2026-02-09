// overlay.cu
// CUDA overlay on NV12 EGLImage (Jetson NVMM surface-array friendly).
// Reads MV field from a CUDA pitch-linear buffer (2S16 interleaved S10.5) and draws arrows.
//
// NEW: optional IMU resultant arrow (green) drawn alongside OF resultant (blue).
// IMU vector is in m/s^2 (linear accel) and is visual-only. imuScale controls pixels per (m/s^2).
//

#include <cstdint>
#include <cstring>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>

#include "overlay.hpp"

__device__ __forceinline__ float s10_5_to_px(int16_t v) { return (float)v * (1.0f/32.0f); }

// Safe rsqrt to avoid division by 0.
__device__ __forceinline__ float fast_rsqrtf_safe(float x)
{
    return rsqrtf(fmaxf(x, 1e-12f));
}

// Deterministic hash -> float in [0, 1).
// Used to add stable per-cell jitter (and optionally per-frame).
__device__ __forceinline__ float hash01_u32(uint32_t x)
{
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return (x & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

// Rotate a 2D vector (x,y) by angle with cos/sin.
__device__ __forceinline__ void rotate2(float x, float y, float c, float s, float &ox, float &oy)
{
    ox = c * x - s * y;
    oy = s * x + c * y;
}

__device__ inline void putY_ptr(uint8_t *Y, int pitch, int W, int H, int x, int y, uint8_t v)
{
    if ((unsigned)x < (unsigned)W && (unsigned)y < (unsigned)H) Y[y * pitch + x] = v;
}

__device__ inline void putUV_ptr(uint8_t *UV, int pitchUV, int W, int H, int x, int y, uint8_t U, uint8_t V)
{
    int uvx = x >> 1;
    int uvy = y >> 1;
    int uvW = W >> 1;
    int uvH = H >> 1;
    if ((unsigned)uvx < (unsigned)uvW && (unsigned)uvy < (unsigned)uvH) {
        uint8_t *p = &UV[uvy * pitchUV + uvx * 2];
        p[0] = U; p[1] = V;
    }
}

__device__ inline void drawLine_ptr(uint8_t *Y, uint8_t *UV,
                                    int pitchY, int pitchUV,
                                    int W, int H,
                                    int x0, int y0, int x1, int y1,
                                    uint8_t Yc, uint8_t Uc, uint8_t Vc)
{
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    while (true) {
        putY_ptr(Y, pitchY, W, H, x0, y0, Yc);
        putUV_ptr(UV, pitchUV, W, H, x0, y0, Uc, Vc);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

// Draw one arrow with a simple arrowhead (pointer path)
__device__ inline void drawArrow_ptr(uint8_t *Y, uint8_t *UV,
                                     int pitchY, int pitchUV,
                                     int W, int H,
                                     int x0, int y0, int x1, int y1,
                                     uint8_t Yc, uint8_t Uc, uint8_t Vc)
{
    drawLine_ptr(Y, UV, pitchY, pitchUV, W, H, x0, y0, x1, y1, Yc, Uc, Vc);

    int hx = x1 - x0;
    int hy = y1 - y0;
    if (hx == 0 && hy == 0) return;

    int px = -hy, py = hx;
    int len = max(1, abs(hx) + abs(hy));
    int ahx = (hx * 6) / len;
    int ahy = (hy * 6) / len;
    int apx = (px * 4) / len;
    int apy = (py * 4) / len;

    int xh1 = x1 - ahx + apx;
    int yh1 = y1 - ahy + apy;
    int xh2 = x1 - ahx - apx;
    int yh2 = y1 - ahy - apy;

    drawLine_ptr(Y, UV, pitchY, pitchUV, W, H, x1, y1, xh1, yh1, Yc, Uc, Vc);
    drawLine_ptr(Y, UV, pitchY, pitchUV, W, H, x1, y1, xh2, yh2, Yc, Uc, Vc);
}

// Kernel: draw MV field arrows from pitch-linear MV buffer.
// Optional visual "direction forcing" towards resultant direction with ±forceDeg jitter.
__global__ void drawFieldPitchKernel(uint8_t *Y, uint8_t *UV,
                                     int pitchY, int pitchUV,
                                     int W, int H,
                                     const int16_t *mvPtr, int mvPitchBytes,
                                     int mvW, int mvH, int grid,
                                     int x0, int x1, int y0, int y1,
                                     int step, float scale,
                                     float minMagDraw,
                                     // resultant direction (px/frame) for visual forcing
                                     float resDxPx, float resDyPx,
                                     // forcing parameters
                                     float forceDeg,
                                     float forceMinResMag,
                                     uint32_t frameTag,
                                     // colors
                                     uint8_t Yc, uint8_t Uc, uint8_t Vc)
{
    int sx = blockIdx.x * blockDim.x + threadIdx.x;
    int sy = blockIdx.y * blockDim.y + threadIdx.y;

    int mx = x0 + sx * step;
    int my = y0 + sy * step;
    if (mx >= x1 || my >= y1) return;
    if (mx < 0 || my < 0 || mx >= mvW || my >= mvH) return;

    const uint8_t *rowB = (const uint8_t*)mvPtr + (size_t)my * (size_t)mvPitchBytes;
    const int16_t *row  = (const int16_t*)rowB;

    int16_t fx = row[mx*2 + 0];
    int16_t fy = row[mx*2 + 1];

    float dx = s10_5_to_px(fx);
    float dy = s10_5_to_px(fy);

    float magL1 = fabsf(dx) + fabsf(dy);
    if (magL1 < minMagDraw) return;

    // ------------------------------------------------------------------
    // VISUAL ONLY: If a local MV deviates too much from the OF resultant direction,
    // draw it aligned to the resultant direction, with a small jitter ±forceDeg.
    // ------------------------------------------------------------------
    float resMagL1 = fabsf(resDxPx) + fabsf(resDyPx);
    if (forceDeg > 0.0f && resMagL1 >= forceMinResMag)
    {
        // Normalize resultant direction.
        float rx = resDxPx;
        float ry = resDyPx;
        float rinv = fast_rsqrtf_safe(rx*rx + ry*ry);
        rx *= rinv;
        ry *= rinv;

        // Normalize local vector direction.
        float v2 = dx*dx + dy*dy;
        float vinv = fast_rsqrtf_safe(v2);
        float vx = dx * vinv;
        float vy = dy * vinv;

        // dot = cos(theta). If dot < cos(forceDeg) then deviation > forceDeg.
        float maxRad = forceDeg * 0.01745329252f;
        float cosTh  = cosf(maxRad);
        float dot    = vx*rx + vy*ry;

        if (dot < cosTh)
        {
            // Deterministic jitter per (mx,my,frameTag).
            uint32_t h = (uint32_t)(mx * 73856093u) ^ (uint32_t)(my * 19349663u) ^ (uint32_t)(frameTag * 83492791u);
            float u = hash01_u32(h);                 // [0,1)
            float a = (u * 2.0f - 1.0f) * maxRad;    // [-maxRad, +maxRad]
            float c = cosf(a), s = sinf(a);

            // Rotate resultant direction by jitter.
            float fx2, fy2;
            rotate2(rx, ry, c, s, fx2, fy2);

            // Keep original magnitude (L2) so arrows still look "natural".
            float vmag = sqrtf(v2);
            dx = fx2 * vmag;
            dy = fy2 * vmag;
        }
    }

    // Arrow in pixel space
    int px0 = mx * grid;
    int py0 = my * grid;

    int px1 = px0 + (int)lrintf(dx * scale);
    int py1 = py0 + (int)lrintf(dy * scale);

    px1 = max(0, min(W - 1, px1));
    py1 = max(0, min(H - 1, py1));

    drawArrow_ptr(Y, UV, pitchY, pitchUV, W, H, px0, py0, px1, py1, Yc, Uc, Vc);
}

// Kernel: draw OF resultant arrow at center
__global__ void drawResultantPitchKernel(uint8_t *Y, uint8_t *UV,
                                         int pitchY, int pitchUV,
                                         int W, int H,
                                         float resDxPx, float resDyPx,
                                         float scale,
                                         uint8_t Yc, uint8_t Uc, uint8_t Vc)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int cx = W / 2;
    int cy = H / 2;

    int x1 = cx + (int)lrintf(resDxPx * scale);
    int y1 = cy + (int)lrintf(resDyPx * scale);

    x1 = max(0, min(W - 1, x1));
    y1 = max(0, min(H - 1, y1));

    drawArrow_ptr(Y, UV, pitchY, pitchUV, W, H, cx, cy, x1, y1, Yc, Uc, Vc);
}

// Kernel: draw IMU resultant arrow (visual-only) with slight offset
__global__ void drawImuPitchKernel(uint8_t *Y, uint8_t *UV,
                                   int pitchY, int pitchUV,
                                   int W, int H,
                                   float imuDx, float imuDy,
                                   float imuScale,
                                   uint8_t Yc, uint8_t Uc, uint8_t Vc)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    if ((fabsf(imuDx) + fabsf(imuDy)) < 1e-6f) return;

    // Offset to avoid perfect overlap with OF arrow
    int cx = W / 2 + 14;
    int cy = H / 2 + 14;

    int x1 = cx + (int)lrintf(imuDx * imuScale);
    int y1 = cy + (int)lrintf(imuDy * imuScale);

    x1 = max(0, min(W - 1, x1));
    y1 = max(0, min(H - 1, y1));

    drawArrow_ptr(Y, UV, pitchY, pitchUV, W, H, cx, cy, x1, y1, Yc, Uc, Vc);
}

extern "C" void overlay_draw_mvs_nv12(EGLImageKHR eglImage,
                                      int W, int H,
                                      const int16_t *mvPtr, int mvPitchBytes,
                                      int mvW, int mvH, int grid,
                                      int x0, int x1, int y0, int y1,
                                      int step, float scale,
                                      float minMagDraw,
                                      uint8_t fieldY, uint8_t fieldU, uint8_t fieldV,
                                      float resDxPx, float resDyPx,
                                      uint8_t resY, uint8_t resU, uint8_t resV,
                                      // IMU resultant (optional)
                                      float imuDx, float imuDy,
                                      float imuScale,
                                      uint8_t imuY, uint8_t imuU, uint8_t imuV,
                                      // visual forcing
                                      float forceDeg,
                                      float forceMinResMag,
                                      uint32_t frameTag)
{
    if (!eglImage || !mvPtr) return;

    static bool cuInitDone = false;
    if (!cuInitDone) { if (cuInit(0) != CUDA_SUCCESS) return; cuInitDone = true; }

    static cudaStream_t stream = nullptr;
    if (!stream) {
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return;
    }

    CUgraphicsResource cuRes = nullptr;
    CUeglFrame eglFrame;

    if (cuGraphicsEGLRegisterImage(&cuRes, (EGLImageKHR)eglImage,
                                   CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE) != CUDA_SUCCESS)
        return;

    if (cuGraphicsResourceGetMappedEglFrame(&eglFrame, cuRes, 0, 0) != CUDA_SUCCESS) {
        cuGraphicsUnregisterResource(cuRes);
        return;
    }

    if (eglFrame.frameType == CU_EGL_FRAME_TYPE_PITCH) {
        uint8_t *Y  = (uint8_t*)eglFrame.frame.pPitch[0];
        uint8_t *UV = (uint8_t*)eglFrame.frame.pPitch[1];
        int pitchY  = (int)eglFrame.pitch;
        int pitchUV = (int)eglFrame.pitch;

        // Grid for sampled points in MV space
        int sxN = (x1 - x0 + step - 1) / step;
        int syN = (y1 - y0 + step - 1) / step;

        dim3 block(8, 8);
        dim3 gridD((sxN + block.x - 1) / block.x,
                   (syN + block.y - 1) / block.y);

        // Field overlay with optional visual forcing towards OF resultant direction
        drawFieldPitchKernel<<<gridD, block, 0, stream>>>(
            Y, UV, pitchY, pitchUV, W, H,
            mvPtr, mvPitchBytes, mvW, mvH, grid,
            x0, x1, y0, y1,
            step, scale, minMagDraw,
            resDxPx, resDyPx,
            forceDeg, forceMinResMag, frameTag,
            fieldY, fieldU, fieldV);

        // OF resultant arrow (blue)
        drawResultantPitchKernel<<<1, 1, 0, stream>>>(
            Y, UV, pitchY, pitchUV, W, H,
            resDxPx, resDyPx, scale,
            resY, resU, resV);

        // IMU resultant arrow (green) - optional; disabled if imuDx/imuDy ~ 0
        drawImuPitchKernel<<<1, 1, 0, stream>>>(
            Y, UV, pitchY, pitchUV, W, H,
            imuDx, imuDy, imuScale,
            imuY, imuU, imuV);

        cudaGetLastError(); // swallow
        cudaStreamSynchronize(stream);
    }
    // ARRAY surface path can be added similarly if needed.

    cuGraphicsUnregisterResource(cuRes);
}
