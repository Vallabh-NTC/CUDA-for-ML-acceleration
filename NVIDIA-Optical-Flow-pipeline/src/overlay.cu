// overlay.cu
// CUDA overlay on NV12 EGLImage (Jetson NVMM surface-array friendly)
// Draws RED arrows by writing both Y and UV planes.
// No printf, no device-side logging.

#include <cstdint>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>

#include "overlay.hpp"

// ----------------- Color (approx BT.709 red) -----------------
static constexpr uint8_t kRedY = 76;
static constexpr uint8_t kRedU = 85;
static constexpr uint8_t kRedV = 255;

// ----------------- Helpers (PITCH path) -----------------
__device__ inline void putY_ptr(uint8_t *Y, int pitch, int W, int H, int x, int y, uint8_t v)
{
    if ((unsigned)x < (unsigned)W && (unsigned)y < (unsigned)H)
        Y[y * pitch + x] = v;
}

__device__ inline void putUV_ptr(uint8_t *UV, int pitchUV, int W, int H, int x, int y, uint8_t U, uint8_t V)
{
    // NV12 UV is 4:2:0 (half res). UV interleaved.
    int uvx = x >> 1;      // /2
    int uvy = y >> 1;      // /2
    int uvW = W >> 1;
    int uvH = H >> 1;

    if ((unsigned)uvx < (unsigned)uvW && (unsigned)uvy < (unsigned)uvH) {
        uint8_t *p = &UV[uvy * pitchUV + uvx * 2];
        p[0] = U;
        p[1] = V;
    }
}

__device__ inline void drawLine_ptr(uint8_t *Y, uint8_t *UV,
                                    int pitchY, int pitchUV,
                                    int W, int H,
                                    int x0, int y0, int x1, int y1)
{
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;

    while (true) {
        putY_ptr(Y, pitchY, W, H, x0, y0, kRedY);
        putUV_ptr(UV, pitchUV, W, H, x0, y0, kRedU, kRedV);

        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

// ----------------- Helpers (ARRAY path via surfaces) -----------------
__device__ inline void putY_surf(cudaSurfaceObject_t sY, int W, int H, int x, int y, uint8_t v)
{
    if ((unsigned)x < (unsigned)W && (unsigned)y < (unsigned)H) {
        // 8-bit surface: x is byte offset
        surf2Dwrite(v, sY, x, y);
    }
}

__device__ inline void putUV_surf(cudaSurfaceObject_t sUV, int W, int H, int x, int y, uint8_t U, uint8_t V)
{
    int uvx = x >> 1;
    int uvy = y >> 1;
    int uvW = W >> 1;
    int uvH = H >> 1;

    if ((unsigned)uvx < (unsigned)uvW && (unsigned)uvy < (unsigned)uvH) {
        // UV plane: each "pixel" is 2 bytes (U,V). Write as ushort.
        uint16_t uv = (uint16_t)U | ((uint16_t)V << 8);
        surf2Dwrite(uv, sUV, uvx * 2, uvy); // x in bytes
    }
}

__device__ inline void drawLine_surf(cudaSurfaceObject_t sY, cudaSurfaceObject_t sUV,
                                     int W, int H,
                                     int x0, int y0, int x1, int y1)
{
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;

    while (true) {
        putY_surf(sY, W, H, x0, y0, kRedY);
        putUV_surf(sUV, W, H, x0, y0, kRedU, kRedV);

        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

// ----------------- Kernels -----------------
__global__ void overlayPitchKernel(uint8_t *Y, uint8_t *UV,
                                   int pitchY, int pitchUV,
                                   int W, int H,
                                   const Arrow *arrows, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Arrow a = arrows[i];
    drawLine_ptr(Y, UV, pitchY, pitchUV, W, H, a.x0, a.y0, a.x1, a.y1);

    // small arrowhead
    int hx = a.x1 - a.x0;
    int hy = a.y1 - a.y0;
    if (hx == 0 && hy == 0) return;

    int px = -hy;
    int py =  hx;

    int len = max(1, abs(hx) + abs(hy));
    int ahx = (hx * 6) / len;
    int ahy = (hy * 6) / len;
    int apx = (px * 4) / len;
    int apy = (py * 4) / len;

    int xh1 = a.x1 - ahx + apx;
    int yh1 = a.y1 - ahy + apy;
    int xh2 = a.x1 - ahx - apx;
    int yh2 = a.y1 - ahy - apy;

    drawLine_ptr(Y, UV, pitchY, pitchUV, W, H, a.x1, a.y1, xh1, yh1);
    drawLine_ptr(Y, UV, pitchY, pitchUV, W, H, a.x1, a.y1, xh2, yh2);
}

__global__ void overlaySurfKernel(cudaSurfaceObject_t sY, cudaSurfaceObject_t sUV,
                                  int W, int H,
                                  const Arrow *arrows, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Arrow a = arrows[i];
    drawLine_surf(sY, sUV, W, H, a.x0, a.y0, a.x1, a.y1);

    int hx = a.x1 - a.x0;
    int hy = a.y1 - a.y0;
    if (hx == 0 && hy == 0) return;

    int px = -hy;
    int py =  hx;

    int len = max(1, abs(hx) + abs(hy));
    int ahx = (hx * 6) / len;
    int ahy = (hy * 6) / len;
    int apx = (px * 4) / len;
    int apy = (py * 4) / len;

    int xh1 = a.x1 - ahx + apx;
    int yh1 = a.y1 - ahy + apy;
    int xh2 = a.x1 - ahx - apx;
    int yh2 = a.y1 - ahy - apy;

    drawLine_surf(sY, sUV, W, H, a.x1, a.y1, xh1, yh1);
    drawLine_surf(sY, sUV, W, H, a.x1, a.y1, xh2, yh2);
}

// ----------------- Host entry -----------------
extern "C" void overlay_draw_arrows_nv12(EGLImageKHR eglImage,
                                        int W, int H,
                                        const Arrow *arrows, int n,
                                        uint8_t /*unused*/)
{
    if (!eglImage || !arrows || n <= 0) return;

    static bool cuInitDone = false;
    if (!cuInitDone) {
        if (cuInit(0) != CUDA_SUCCESS) return;
        cuInitDone = true;
    }

    static cudaStream_t stream = nullptr;
    if (!stream) {
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
            return;
    }

    static Arrow *d_arrows = nullptr;
    static int d_cap = 0;

    if (n > d_cap) {
        if (d_arrows) cudaFree(d_arrows);
        int cap = 1;
        while (cap < n) cap <<= 1;
        d_cap = cap;
        if (cudaMalloc(&d_arrows, sizeof(Arrow) * d_cap) != cudaSuccess)
            return;
    }

    if (cudaMemcpyAsync(d_arrows, arrows, sizeof(Arrow) * n,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return;

    CUgraphicsResource cuRes = nullptr;
    CUeglFrame eglFrame;

    if (cuGraphicsEGLRegisterImage(&cuRes, (EGLImageKHR)eglImage,
                                   CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE) != CUDA_SUCCESS)
        return;

    if (cuGraphicsResourceGetMappedEglFrame(&eglFrame, cuRes, 0, 0) != CUDA_SUCCESS) {
        cuGraphicsUnregisterResource(cuRes);
        return;
    }

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    if (eglFrame.frameType == CU_EGL_FRAME_TYPE_PITCH) {
        uint8_t *Y  = (uint8_t*)eglFrame.frame.pPitch[0];
        uint8_t *UV = (uint8_t*)eglFrame.frame.pPitch[1];

        int pitchY  = (int)eglFrame.pitch;
        int pitchUV = (int)eglFrame.pitch;

        overlayPitchKernel<<<blocks, threads, 0, stream>>>(Y, UV, pitchY, pitchUV, W, H, d_arrows, n);
        cudaGetLastError(); // swallow
    } else if (eglFrame.frameType == CU_EGL_FRAME_TYPE_ARRAY) {
        CUarray arrY  = eglFrame.frame.pArray[0];
        CUarray arrUV = eglFrame.frame.pArray[1];

        cudaResourceDesc rdY, rdUV;
        std::memset(&rdY,  0, sizeof(rdY));
        std::memset(&rdUV, 0, sizeof(rdUV));
        rdY.resType = cudaResourceTypeArray;
        rdY.res.array.array = (cudaArray_t)arrY;
        rdUV.resType = cudaResourceTypeArray;
        rdUV.res.array.array = (cudaArray_t)arrUV;

        cudaSurfaceObject_t sY = 0, sUV = 0;
        if (cudaCreateSurfaceObject(&sY, &rdY) == cudaSuccess &&
            cudaCreateSurfaceObject(&sUV, &rdUV) == cudaSuccess) {

            overlaySurfKernel<<<blocks, threads, 0, stream>>>(sY, sUV, W, H, d_arrows, n);
            cudaGetLastError();
        }

        if (sY)  cudaDestroySurfaceObject(sY);
        if (sUV) cudaDestroySurfaceObject(sUV);
    }

    cudaStreamSynchronize(stream);
    cuGraphicsUnregisterResource(cuRes);
}
