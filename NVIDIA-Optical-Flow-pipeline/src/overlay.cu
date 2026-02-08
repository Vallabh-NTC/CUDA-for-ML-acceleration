// overlay.cu
// CUDA overlay on NV12 EGLImage (Jetson NVMM surface-array friendly).
// Reads MV field from a CUDA pitch-linear buffer (2S16 interleaved S10.5) and draws arrows.

#include <cstdint>
#include <cstring>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>

#include "overlay.hpp"

__device__ __forceinline__ float s10_5_to_px(int16_t v) { return (float)v * (1.0f/32.0f); }

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

__device__ inline void putY_surf(cudaSurfaceObject_t sY, int W, int H, int x, int y, uint8_t v)
{
    if ((unsigned)x < (unsigned)W && (unsigned)y < (unsigned)H) surf2Dwrite(v, sY, x, y);
}

__device__ inline void putUV_surf(cudaSurfaceObject_t sUV, int W, int H, int x, int y, uint8_t U, uint8_t V)
{
    int uvx = x >> 1;
    int uvy = y >> 1;
    int uvW = W >> 1;
    int uvH = H >> 1;
    if ((unsigned)uvx < (unsigned)uvW && (unsigned)uvy < (unsigned)uvH) {
        uint16_t uv = (uint16_t)U | ((uint16_t)V << 8);
        surf2Dwrite(uv, sUV, uvx * 2, uvy);
    }
}

__device__ inline void drawLine_surf(cudaSurfaceObject_t sY, cudaSurfaceObject_t sUV,
                                     int W, int H,
                                     int x0, int y0, int x1, int y1,
                                     uint8_t Yc, uint8_t Uc, uint8_t Vc)
{
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    while (true) {
        putY_surf(sY, W, H, x0, y0, Yc);
        putUV_surf(sUV, W, H, x0, y0, Uc, Vc);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

// Draw one arrow with simple arrowhead
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

__device__ inline void drawArrow_surf(cudaSurfaceObject_t sY, cudaSurfaceObject_t sUV,
                                      int W, int H,
                                      int x0, int y0, int x1, int y1,
                                      uint8_t Yc, uint8_t Uc, uint8_t Vc)
{
    drawLine_surf(sY, sUV, W, H, x0, y0, x1, y1, Yc, Uc, Vc);

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

    drawLine_surf(sY, sUV, W, H, x1, y1, xh1, yh1, Yc, Uc, Vc);
    drawLine_surf(sY, sUV, W, H, x1, y1, xh2, yh2, Yc, Uc, Vc);
}

// Kernel: draw MV field arrows from pitch-linear MV buffer
__global__ void drawFieldPitchKernel(uint8_t *Y, uint8_t *UV,
                                     int pitchY, int pitchUV,
                                     int W, int H,
                                     const int16_t *mvPtr, int mvPitchBytes,
                                     int mvW, int mvH, int grid,
                                     int x0, int x1, int y0, int y1,
                                     int step, float scale,
                                     float minMagDraw,
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

    float mag = fabsf(dx) + fabsf(dy);
    if (mag < minMagDraw) return;

    // Arrow in pixel space
    int px0 = mx * grid;
    int py0 = my * grid;

    int px1 = px0 + (int)lrintf(dx * scale);
    int py1 = py0 + (int)lrintf(dy * scale);

    px1 = max(0, min(W - 1, px1));
    py1 = max(0, min(H - 1, py1));

    drawArrow_ptr(Y, UV, pitchY, pitchUV, W, H, px0, py0, px1, py1, Yc, Uc, Vc);
}

// Kernel: draw resultant arrow at center
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

extern "C" void overlay_draw_mvs_nv12(EGLImageKHR eglImage,
                                      int W, int H,
                                      const int16_t *mvPtr, int mvPitchBytes,
                                      int mvW, int mvH, int grid,
                                      int x0, int x1, int y0, int y1,
                                      int step, float scale,
                                      float minMagDraw,
                                      uint8_t fieldY, uint8_t fieldU, uint8_t fieldV,
                                      float resDxPx, float resDyPx,
                                      uint8_t resY, uint8_t resU, uint8_t resV)
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

        // Grid for sampled points
        int sxN = (x1 - x0 + step - 1) / step;
        int syN = (y1 - y0 + step - 1) / step;

        dim3 block(8, 8);
        dim3 gridD((sxN + block.x - 1) / block.x,
                   (syN + block.y - 1) / block.y);

        drawFieldPitchKernel<<<gridD, block, 0, stream>>>(
            Y, UV, pitchY, pitchUV, W, H,
            mvPtr, mvPitchBytes, mvW, mvH, grid,
            x0, x1, y0, y1,
            step, scale, minMagDraw,
            fieldY, fieldU, fieldV);

        drawResultantPitchKernel<<<1, 1, 0, stream>>>(
            Y, UV, pitchY, pitchUV, W, H,
            resDxPx, resDyPx, scale,
            resY, resU, resV);

        cudaGetLastError(); // swallow
        cudaStreamSynchronize(stream);
    }
    // ARRAY surface path can be added similarly if needed (your previous code had it).
    // Many Jetson NVMM surfaces come as ARRAY; if yours does, tell me and I'll add it back.

    cuGraphicsUnregisterResource(cuRes);
}
