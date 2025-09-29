#include "wire_lineedge.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace {

// clamp helpers
__device__ __forceinline__ int clampi(int v,int a,int b){
    return v<a ? a : (v>b ? b : v);
}
__device__ __forceinline__ float clampf(float v,float a,float b){
    return v<a ? a : (v>b ? b : v);
}

} // anon

namespace wire {

// ========================= overlays (unchanged) =========================

__global__ void k_overlay_mask_Y(
    const uint8_t* __restrict__ mask, int mPitch,
    uint8_t* __restrict__ Y, int pY,
    int W, int H, uint8_t y_on)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    if (mask[y * mPitch + x]) Y[y * pY + x] = y_on;
}

__global__ void k_overlay_mask_UV(
    const uint8_t* __restrict__ mask, int mPitch,
    uint8_t* __restrict__ UV, int pUV,
    int W, int H, uint8_t u_on, uint8_t v_on)
{
    int cu = blockIdx.x * blockDim.x + threadIdx.x;
    int cv = blockIdx.y * blockDim.y + threadIdx.y;
    int CW = (W + 1) >> 1, CH = (H + 1) >> 1;
    if (cu >= CW || cv >= CH) return;

    int x0 = cu * 2, y0 = cv * 2;
    uint8_t any = 0;
    if (y0 < H) {
        if (x0   < W) any |= mask[y0 * mPitch + x0];
        if (x0+1 < W) any |= mask[y0 * mPitch + x0+1];
    }
    if (y0+1 < H) {
        if (x0   < W) any |= mask[(y0+1) * mPitch + x0];
        if (x0+1 < W) any |= mask[(y0+1) * mPitch + x0+1];
    }
    if (any) {
        uint8_t* p = UV + cv * pUV + cu*2;
        p[0]=u_on; p[1]=v_on;
    }
}

void overlay_mask_nv12(
    uint8_t* dY, int W, int H, int pitchY,
    uint8_t* dUV, int pitchUV,
    const uint8_t* dMask, int maskPitch,
    uint8_t y_on, uint8_t u_on, uint8_t v_on,
    cudaStream_t stream)
{
    dim3 blkY(32,8), grdY((W+31)/32,(H+7)/8);
    k_overlay_mask_Y<<<grdY, blkY, 0, stream>>>(dMask, maskPitch, dY, pitchY, W, H, y_on);

    int CW=(W+1)>>1, CH=(H+1)>>1;
    dim3 blkUV(32,8), grdUV((CW+31)/32,(CH+7)/8);
    k_overlay_mask_UV<<<grdUV, blkUV, 0, stream>>>(dMask, maskPitch, dUV, pitchUV, W, H, u_on, v_on);
}

// kept for ABI, unused by radial path (no-op kernels)
__global__ void k_overlay_polylineY(uint8_t*,int,const int*,int,int,uint8_t) {}
__global__ void k_overlay_polylineUV(uint8_t*,int,const int*,int,int,uint8_t,uint8_t) {}

void overlay_polyline_nv12(
    uint8_t* dY, int pY, uint8_t* dUV, int pUV,
    int W, int H, const int* dTop,
    uint8_t y_on, uint8_t u_on, uint8_t v_on,
    cudaStream_t stream)
{
    (void)dY;(void)pY;(void)dUV;(void)pUV;(void)W;(void)H;(void)dTop;
    (void)y_on;(void)u_on;(void)v_on;(void)stream;
}

// =================== helpers: bilinear fetch (pitched) ===================

__device__ __forceinline__ float bilinear_Y(
    const uint8_t* Y, int pY, int W, int H,
    float xf, float yf)
{
    xf = clampf(xf, 0.0f, (float)(W-1));
    yf = clampf(yf, 0.0f, (float)(H-1));
    int x0 = (int)floorf(xf), y0 = (int)floorf(yf);
    int x1 = min(W-1, x0+1), y1 = min(H-1, y0+1);
    float ax = xf - x0, ay = yf - y0;

    int i00 = Y[y0 * pY + x0];
    int i10 = Y[y0 * pY + x1];
    int i01 = Y[y1 * pY + x0];
    int i11 = Y[y1 * pY + x1];

    float v0 = i00*(1.0f-ax) + i10*ax;
    float v1 = i01*(1.0f-ax) + i11*ax;
    return v0*(1.0f-ay) + v1*ay;
}

__device__ __forceinline__ void fetch_UV_nearest(
    const uint8_t* UV, int pUV, int W, int H,
    float xf, float yf, uint8_t& U, uint8_t& V)
{
    // NV12 chroma half-res; nearest per evitare smear
    float cxf = 0.5f * xf, cyf = 0.5f * yf;
    int CW = (W + 1) >> 1, CH = (H + 1) >> 1;
    int cu = clampi((int)floorf(cxf + 0.5f), 0, CW-1);
    int cv = clampi((int)floorf(cyf + 0.5f), 0, CH-1);
    const uint8_t* p = UV + cv * pUV + cu * 2;
    U = p[0];
    V = p[1];
}

// ============ radial (fisheye-aligned) disappearance =================

__global__ void k_disappear_mask_radial_YUV(
    uint8_t* __restrict__ Y, int pY,
    uint8_t* __restrict__ UV, int pUV,
    int W, int H,
    const uint8_t* __restrict__ mask, int pMask,
    float cx, float cy, float offIn, float offOut)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    if (mask[y * pMask + x] == 0) return; // process only inside mask

    // unit radial direction
    float vx = (float)x - cx;
    float vy = (float)y - cy;
    float len = sqrtf(vx*vx + vy*vy);
    if (len < 1e-6f) { vx = 0.0f; vy = 1.0f; len = 1.0f; }
    float nx = vx / len, ny = vy / len;

    // donor positions
    float xin = (float)x - offIn * nx;
    float yin = (float)y - offIn * ny;
    float xout= (float)x + offOut * nx;
    float yout= (float)y + offOut * ny;

    // --- Luma: bilinear blend
    float Yin = bilinear_Y(Y, pY, W, H, xin, yin);
    float Yout = bilinear_Y(Y, pY, W, H, xout, yout);
    float t = 0.5f; // proxy semplice
    float Ymix = (1.0f - t) * Yin + t * Yout;
    Y[y * pY + x] = (uint8_t)clampi((int)(Ymix + 0.5f), 0, 255);

    // --- Chroma: nearest donor
    float din = (x - xin)*(x - xin) + (y - yin)*(y - yin);
    float dout = (x - xout)*(x - xout) + (y - yout)*(y - yout);
    uint8_t U,V;
    if (din <= dout) fetch_UV_nearest(UV, pUV, W, H, xin, yin, U, V);
    else             fetch_UV_nearest(UV, pUV, W, H, xout, yout, U, V);

    // scrivi UV sulla cella che copre (x,y)
    int cu = x >> 1, cv = y >> 1;
    int CW = (W + 1) >> 1, CH = (H + 1) >> 1;
    if (cu < CW && cv < CH) {
        uint8_t* p = UV + cv * pUV + cu * 2;
        p[0] = U; p[1] = V;
    }
}

void disappear_mask_radial_nv12(
    uint8_t* dY, int pitchY,
    uint8_t* dUV, int pitchUV,
    int W, int H,
    const uint8_t* dMask, int maskPitch,
    float cx, float cy, float offIn, float offOut,
    cudaStream_t stream)
{
    dim3 blk(32,8), grd((W+31)/32, (H+7)/8);
    k_disappear_mask_radial_YUV<<<grd, blk, 0, stream>>>(
        dY, pitchY, dUV, pitchUV, W, H,
        dMask, maskPitch, cx, cy, offIn, offOut);
}

} // namespace wire
