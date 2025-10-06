#include "wire_lineremoval.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace {

// ---------- small helpers ----------
__device__ __forceinline__ int clampi(int v,int a,int b){
    return v<a ? a : (v>b ? b : v);
}
__device__ __forceinline__ float clampf(float v,float a,float b){
    return v<a ? a : (v>b ? b : v);
}

// bilinear fetch on pitched Y
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

// nearest on UV (NV12 4:2:0, pitched)
__device__ __forceinline__ void fetch_UV_nearest(
    const uint8_t* UV, int pUV, int W, int H,
    float xf, float yf, uint8_t& U, uint8_t& V)
{
    // map luma coords to chroma grid (half resolution)
    float cxf = 0.5f * xf;
    float cyf = 0.5f * yf;
    int CW = (W + 1) >> 1, CH = (H + 1) >> 1;
    int cu = clampi((int)floorf(cxf + 0.5f), 0, CW-1);
    int cv = clampi((int)floorf(cyf + 0.5f), 0, CH-1);
    const uint8_t* p = UV + cv * pUV + cu * 2;
    U = p[0];
    V = p[1];
}

} // anon

namespace wire {

// ================== two-donor blend (top & bottom dots) ==================
//
// For each pixel with mask!=0:
//   Y  = 0.5 * Y(x+dx,y+dy) + 0.5 * Y(x-dx,y-dy)  (bilinear)
//   UV = average( UV(x+dx,y+dy), UV(x-dx,y-dy) )  (nearest)
//
// You can tweak 'T_BLEND' below to bias toward one side if desired.
// ========================================================================

__global__ void k_apply_mask_blend2_Y(
    uint8_t* __restrict__ Y, int pY,
    const uint8_t* __restrict__ mask, int pMask,
    int W, int H, float dx, float dy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    if (mask[y * pMask + x] == 0) return;

    float xs1 = (float)x + dx;   // donor 1: "top" dot
    float ys1 = (float)y + dy;
    float xs2 = (float)x - dx;   // donor 2: "bottom" dot
    float ys2 = (float)y - dy;

    float Y1 = bilinear_Y(Y, pY, W, H, xs1, ys1);
    float Y2 = bilinear_Y(Y, pY, W, H, xs2, ys2);

    const float T_BLEND = 0.5f; // 0..1 (0.5 = equal blend)
    float Ym = (1.0f - T_BLEND) * Y1 + T_BLEND * Y2;

    int val = (int)(Ym + 0.5f);
    val = val < 0 ? 0 : (val > 255 ? 255 : val);
    Y[y * pY + x] = (uint8_t)val;
}

__global__ void k_apply_mask_blend2_UV(
    uint8_t* __restrict__ UV, int pUV,
    const uint8_t* __restrict__ mask, int pMask,
    int W, int H, float dx, float dy)
{
    int cu = blockIdx.x * blockDim.x + threadIdx.x;
    int cv = blockIdx.y * blockDim.y + threadIdx.y;
    int CW = (W + 1) >> 1, CH = (H + 1) >> 1;
    if (cu >= CW || cv >= CH) return;

    // If ANY covered luma pixel is masked, write this chroma sample
    int x0 = cu * 2, y0 = cv * 2;
    uint8_t any = 0;
    if (y0 < H) {
        if (x0   < W) any |= mask[y0 * pMask + x0];
        if (x0+1 < W) any |= mask[y0 * pMask + x0+1];
    }
    if (y0+1 < H) {
        if (x0   < W) any |= mask[(y0+1) * pMask + x0];
        if (x0+1 < W) any |= mask[(y0+1) * pMask + x0+1];
    }
    if (!any) return;

    float xc = (float)cu * 2.0f + 0.5f;
    float yc = (float)cv * 2.0f + 0.5f;

    uint8_t U1,V1, U2,V2;
    fetch_UV_nearest(UV, pUV, W, H, xc + dx, yc + dy, U1, V1);
    fetch_UV_nearest(UV, pUV, W, H, xc - dx, yc - dy, U2, V2);

    // integer average (round-to-nearest)
    uint8_t U = (uint8_t)(((int)U1 + (int)U2 + 1) >> 1);
    uint8_t V = (uint8_t)(((int)V1 + (int)V2 + 1) >> 1);

    uint8_t* p = UV + cv * pUV + cu * 2;
    p[0] = U; p[1] = V;
}

void apply_mask_shift_nv12(
    uint8_t* dY,  int pitchY,
    uint8_t* dUV, int pitchUV,
    int W, int H,
    const uint8_t* dMask, int maskPitch,
    float dx, float dy,
    cudaStream_t stream)
{
    dim3 blk(32,8), grd((W+31)/32, (H+7)/8);
    k_apply_mask_blend2_Y <<<grd, blk, 0, stream>>> (dY, pitchY, dMask, maskPitch, W, H, dx, dy);

    int CW=(W+1)>>1, CH=(H+1)>>1;
    dim3 blkUV(32,8), grdUV((CW+31)/32,(CH+7)/8);
    k_apply_mask_blend2_UV<<<grdUV, blkUV, 0, stream>>> (dUV, pitchUV, dMask, maskPitch, W, H, dx, dy);
}

} // namespace wire
