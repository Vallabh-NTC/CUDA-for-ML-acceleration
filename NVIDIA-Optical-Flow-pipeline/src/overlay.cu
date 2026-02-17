// overlay.cu
// (UNCHANGED LOGIC; shown here for completeness)

#include <cstdint>
#include <cstring>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>

#include "overlay.hpp"

#ifndef OVERLAY_HEAD_LEN
#define OVERLAY_HEAD_LEN 10
#endif

#ifndef OVERLAY_HEAD_W
#define OVERLAY_HEAD_W   7
#endif

#ifndef OVERLAY_RES_THICKNESS
#define OVERLAY_RES_THICKNESS 5
#endif

#ifndef OVERLAY_IMU_THICKNESS
#define OVERLAY_IMU_THICKNESS 3
#endif

#ifndef OVERLAY_RES_SCALE_MUL
#define OVERLAY_RES_SCALE_MUL 1.35f
#endif

__device__ __forceinline__ float s10_5_to_px(int16_t v) { return (float)v * (1.0f / 32.0f); }

__device__ __forceinline__ float fast_rsqrtf_safe(float x)
{
    return rsqrtf(fmaxf(x, 1e-12f));
}

__device__ __forceinline__ float hash01_u32(uint32_t x)
{
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return (x & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ void rotate2(float x, float y, float c, float s, float &ox, float &oy)
{
    ox = c * x - s * y;
    oy = s * x + c * y;
}

__device__ inline void putY_ptr(uint8_t *Y, int pitch, int W, int H, int x, int y, uint8_t v)
{
    if ((unsigned)x < (unsigned)W && (unsigned)y < (unsigned)H)
        Y[y * pitch + x] = v;
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

    const int HEAD_LEN = OVERLAY_HEAD_LEN;
    const int HEAD_W   = OVERLAY_HEAD_W;

    int ahx = (hx * HEAD_LEN) / len;
    int ahy = (hy * HEAD_LEN) / len;
    int apx = (px * HEAD_W)   / len;
    int apy = (py * HEAD_W)   / len;

    if (ahx == 0 && hx != 0) ahx = (hx > 0 ? 1 : -1);
    if (ahy == 0 && hy != 0) ahy = (hy > 0 ? 1 : -1);

    int xh1 = x1 - ahx + apx;
    int yh1 = y1 - ahy + apy;
    int xh2 = x1 - ahx - apx;
    int yh2 = y1 - ahy - apy;

    drawLine_ptr(Y, UV, pitchY, pitchUV, W, H, x1, y1, xh1, yh1, Yc, Uc, Vc);
    drawLine_ptr(Y, UV, pitchY, pitchUV, W, H, x1, y1, xh2, yh2, Yc, Uc, Vc);
}

__device__ inline void drawArrowThick_ptr(uint8_t *Y, uint8_t *UV,
                                          int pitchY, int pitchUV,
                                          int W, int H,
                                          int x0, int y0, int x1, int y1,
                                          uint8_t Yc, uint8_t Uc, uint8_t Vc,
                                          int thickness)
{
    thickness = max(1, thickness);
    int r = thickness / 2;

    for (int oy = -r; oy <= r; ++oy) {
        for (int ox = -r; ox <= r; ++ox) {
            drawArrow_ptr(Y, UV, pitchY, pitchUV, W, H,
                          x0 + ox, y0 + oy, x1 + ox, y1 + oy,
                          Yc, Uc, Vc);
        }
    }
}

__global__ void drawFieldPitchKernel(uint8_t *Y, uint8_t *UV,
                                     int pitchY, int pitchUV,
                                     int W, int H,
                                     const int16_t *mvPtr, int mvPitchBytes,
                                     int mvW, int mvH, int grid,
                                     int x0, int x1, int y0, int y1,
                                     int step, float scale,
                                     float minMagDraw,
                                     float resDxPx, float resDyPx,
                                     float forceDeg,
                                     float forceMinResMag,
                                     uint32_t frameTag,
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

    int16_t fx = row[mx * 2 + 0];
    int16_t fy = row[mx * 2 + 1];

    float dx = s10_5_to_px(fx);
    float dy = s10_5_to_px(fy);

    float magL1 = fabsf(dx) + fabsf(dy);
    if (magL1 < minMagDraw) return;

    float resMagL1 = fabsf(resDxPx) + fabsf(resDyPx);
    if (forceDeg > 0.0f && resMagL1 >= forceMinResMag)
    {
        float rx = resDxPx;
        float ry = resDyPx;
        float rinv = fast_rsqrtf_safe(rx*rx + ry*ry);
        rx *= rinv;
        ry *= rinv;

        float v2 = dx*dx + dy*dy;
        float vinv = fast_rsqrtf_safe(v2);
        float vx = dx * vinv;
        float vy = dy * vinv;

        float maxRad = forceDeg * 0.01745329252f;
        float cosTh  = cosf(maxRad);
        float dot    = vx*rx + vy*ry;

        if (dot < cosTh)
        {
            uint32_t h = (uint32_t)(mx * 73856093u) ^
                         (uint32_t)(my * 19349663u) ^
                         (uint32_t)(frameTag * 83492791u);
            float u = hash01_u32(h);
            float a = (u * 2.0f - 1.0f) * maxRad;
            float c = cosf(a), s = sinf(a);

            float fx2, fy2;
            rotate2(rx, ry, c, s, fx2, fy2);

            float vmag = sqrtf(v2);
            dx = fx2 * vmag;
            dy = fy2 * vmag;
        }
    }

    int px0 = mx * grid;
    int py0 = my * grid;

    int px1 = px0 + (int)lrintf(dx * scale);
    int py1 = py0 + (int)lrintf(dy * scale);

    px1 = max(0, min(W - 1, px1));
    py1 = max(0, min(H - 1, py1));

    drawArrow_ptr(Y, UV, pitchY, pitchUV, W, H, px0, py0, px1, py1, Yc, Uc, Vc);
}

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

    float s = scale * OVERLAY_RES_SCALE_MUL;

    int x1 = cx + (int)lrintf(resDxPx * s);
    int y1 = cy + (int)lrintf(resDyPx * s);

    x1 = max(0, min(W - 1, x1));
    y1 = max(0, min(H - 1, y1));

    drawArrowThick_ptr(Y, UV, pitchY, pitchUV, W, H,
                       cx, cy, x1, y1,
                       Yc, Uc, Vc,
                       OVERLAY_RES_THICKNESS);
}

__global__ void drawImuPitchKernel(uint8_t *Y, uint8_t *UV,
                                   int pitchY, int pitchUV,
                                   int W, int H,
                                   float imuDx, float imuDy,
                                   float imuScale,
                                   uint8_t Yc, uint8_t Uc, uint8_t Vc)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    if ((fabsf(imuDx) + fabsf(imuDy)) < 1e-6f) return;

    int cx = W / 2 + 14;
    int cy = H / 2 + 14;

    int x1 = cx + (int)lrintf(imuDx * imuScale);
    int y1 = cy + (int)lrintf(imuDy * imuScale);

    x1 = max(0, min(W - 1, x1));
    y1 = max(0, min(H - 1, y1));

    drawArrowThick_ptr(Y, UV, pitchY, pitchUV, W, H,
                       cx, cy, x1, y1,
                       Yc, Uc, Vc,
                       OVERLAY_IMU_THICKNESS);
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
                                      float imuDx, float imuDy,
                                      float imuScale,
                                      uint8_t imuY, uint8_t imuU, uint8_t imuV,
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
            resDxPx, resDyPx,
            forceDeg, forceMinResMag, frameTag,
            fieldY, fieldU, fieldV);

        drawResultantPitchKernel<<<1, 1, 0, stream>>>(
            Y, UV, pitchY, pitchUV, W, H,
            resDxPx, resDyPx, scale,
            resY, resU, resV);

        drawImuPitchKernel<<<1, 1, 0, stream>>>(
            Y, UV, pitchY, pitchUV, W, H,
            imuDx, imuDy, imuScale,
            imuY, imuU, imuV);

        cudaGetLastError();
        cudaStreamSynchronize(stream);
    }

    cuGraphicsUnregisterResource(cuRes);
}
