#include "wire_lineedge.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
__device__ __forceinline__ int clampi(int v,int a,int b){ return v<a?a : (v>b?b:v); }
} // anon

namespace wire {

// --------------------- band from mask --------------------
__global__ void k_top_bottom_from_mask(
    const uint8_t* __restrict__ mask, int mPitch,
    int W, int H,
    int* __restrict__ top,
    int* __restrict__ bot)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= W) return;

    int t = -1, b = -1;
    for (int y = 0; y < H; ++y) {
        if (mask[y * mPitch + x]) {
            if (t < 0) t = y;
            b = y;
        }
    }
    top[x] = t;
    bot[x] = b;
}

void top_bottom_from_mask(
    const uint8_t* dMask, int maskPitch,
    int W, int H,
    int* dTop, int* dBot,
    cudaStream_t stream)
{
    dim3 blk(256);
    dim3 grd((W + blk.x - 1)/blk.x);
    k_top_bottom_from_mask<<<grd, blk, 0, stream>>>(dMask, maskPitch, W, H, dTop, dBot);
}

// ------------------------ overlays (unchanged) -----------------------
__global__ void k_overlay_mask_Y(
    const uint8_t* __restrict__ mask, int mPitch,
    uint8_t* __restrict__ Y, int pY,
    int W, int H, uint8_t y_on)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    if (mask[y * mPitch + x]) {
        Y[y * pY + x] = y_on;
    }
}

__global__ void k_overlay_mask_UV(
    const uint8_t* __restrict__ mask, int mPitch,
    uint8_t* __restrict__ UV, int pUV,
    int W, int H, uint8_t u_on, uint8_t v_on)
{
    int cu = blockIdx.x * blockDim.x + threadIdx.x;
    int cv = blockIdx.y * blockDim.y + threadIdx.y;
    int CW = (W + 1) >> 1;
    int CH = (H + 1) >> 1;
    if (cu >= CW || cv >= CH) return;

    int x0 = cu * 2, y0 = cv * 2;
    uint8_t any = 0;
    if (y0 < H) {
        if (x0 < W)     any |= mask[y0 * mPitch + x0];
        if (x0+1 < W)   any |= mask[y0 * mPitch + x0+1];
    }
    if (y0+1 < H) {
        if (x0 < W)     any |= mask[(y0+1) * mPitch + x0];
        if (x0+1 < W)   any |= mask[(y0+1) * mPitch + x0+1];
    }
    if (any) {
        uint8_t* p = UV + cv * pUV + cu*2;
        p[0] = u_on; p[1] = v_on;
    }
}

void overlay_mask_nv12(
    uint8_t* dY,  int W, int H, int pitchY,
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

__global__ void k_overlay_polylineY(
    uint8_t* __restrict__ Y, int pY,
    const int* __restrict__ top,
    int W, int H, uint8_t y_on)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=W) return;
    int y = top[x];
    if (y>=0 && y<H) Y[y*pY+x]=y_on;
}

__global__ void k_overlay_polylineUV(
    uint8_t* __restrict__ UV, int pUV,
    const int* __restrict__ top,
    int W, int H, uint8_t u_on, uint8_t v_on)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x>=W) return;
    int y = top[x];
    if (y<0 || y>=H) return;
    int cu=x>>1, cv=y>>1;
    int CW=(W+1)>>1, CH=(H+1)>>1;
    if (cu>=CW || cv>=CH) return;
    uint8_t* p=UV+cv*pUV+cu*2; p[0]=u_on; p[1]=v_on;
}

void overlay_polyline_nv12(
    uint8_t* dY,  int pY,
    uint8_t* dUV, int pUV,
    int W, int H,
    const int* dTop,
    uint8_t y_on, uint8_t u_on, uint8_t v_on,
    cudaStream_t stream)
{
    dim3 blk(256), grd((W+255)/256);
    k_overlay_polylineY<<<grd,blk,0,stream>>>(dY,pY,dTop,W,H,y_on);
    k_overlay_polylineUV<<<grd,blk,0,stream>>>(dUV,pUV,dTop,W,H,u_on,v_on);
}

// Y (luma): interpolate between donor rows above/below across the band.
__global__ void k_disappear_Y(
    uint8_t* __restrict__ Y, int pY,
    int W, int H,
    const int* __restrict__ top,
    const int* __restrict__ bot,
    int offTop, int offBot)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= W) return;

    int yt = top[x], yb = bot[x];
    if (yt < 0 || yb < 0 || yt > yb) return;

    offTop = max(1, offTop);
    offBot = max(1, offBot);

    // donor rows (robust fallbacks)
    int srcTop = (yt - offTop >= 0) ? (yt - offTop)
               : (yb + offBot < H)  ? (yb + offBot)
                                    : max(0, yt - 1);

    int srcBot = (yb + offBot < H)  ? (yb + offBot)
               : (yt - offTop >= 0) ? (yt - offTop)
                                    : min(H - 1, yb + 1);

    const float yAbove = (float)Y[srcTop * pY + x];
    const float yBelow = (float)Y[srcBot * pY + x];

    const int bandH = yb - yt + 1;
    if (bandH <= 1) {
        // single row: mid of donors
        float yMid = 0.5f * (yAbove + yBelow);
        Y[yt * pY + x] = (uint8_t)(__saturatef(yMid / 255.f) * 255.0f);
        return;
    }

    const float denom = (float)(bandH - 1);
    for (int y = yt; y <= yb; ++y) {
        float t = (float)(y - yt) / denom;           // 0..1 top→bottom
        float yOut = (1.0f - t) * yAbove + t * yBelow;
        Y[y * pY + x] = (uint8_t)(__saturatef(yOut / 255.f) * 255.0f);
    }
}

// UV (chroma, NV12): copy from the nearest donor (no interpolation).
__global__ void k_disappear_UV(
    uint8_t* __restrict__ UV, int pUV,
    int W, int H,
    const int* __restrict__ top,
    const int* __restrict__ bot,
    int offTop, int offBot)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= W) return;

    int yt = top[x], yb = bot[x];
    if (yt < 0 || yb < 0 || yt > yb) return;

    const int CW = (W + 1) >> 1;
    const int CH = (H + 1) >> 1;
    int cu = x >> 1;
    if (cu >= CW) return;

    // chroma rows covered by the band
    int cvTop = max(0,        yt >> 1);
    int cvBot = min((H - 1) >> 1, yb >> 1);
    if (cvTop > cvBot) return;

    // luma offsets → chroma offsets
    int cvOffTop = max(1, offTop >> 1);
    int cvOffBot = max(1, offBot >> 1);

    // donor chroma rows (robust fallbacks)
    int srcTop = (cvTop - cvOffTop >= 0) ? (cvTop - cvOffTop)
               : (cvBot + cvOffBot < CH) ? (cvBot + cvOffBot)
                                         : max(0, cvTop - 1);

    int srcBot = (cvBot + cvOffBot < CH) ? (cvBot + cvOffBot)
               : (cvTop - cvOffTop >= 0) ? (cvTop - cvOffTop)
                                         : min(CH - 1, cvBot + 1);

    const uint8_t* pT = UV + srcTop * pUV + cu * 2;
    const uint8_t* pB = UV + srcBot * pUV + cu * 2;
    uint8_t Ut = pT[0], Vt = pT[1];
    uint8_t Ub = pB[0], Vb = pB[1];

    // For each chroma row inside the band, choose nearest donor and copy
    int bandRows = cvBot - cvTop + 1;
    int mid = cvTop + (bandRows - 1) / 2;  // nearest top up to mid, bottom after

    for (int cv = cvTop; cv <= cvBot; ++cv) {
        uint8_t* p = UV + cv * pUV + cu * 2;
        if (abs(cv - cvTop) <= abs(cvBot - cv)) {
            // nearer to top donor
            p[0] = Ut; p[1] = Vt;
        } else {
            // nearer to bottom donor
            p[0] = Ub; p[1] = Vb;
        }
    }
}

void disappear_band_nv12(
    uint8_t* dY,  int pitchY,
    uint8_t* dUV, int pitchUV,
    int W, int H,
    const int* dTop, const int* dBot,
    int offTop, int offBot,
    float, float,
    cudaStream_t stream)
{
    dim3 blk(256), grd((W+255)/256);
    k_disappear_Y <<<grd,blk,0,stream>>>(dY,pitchY,W,H,dTop,dBot,offTop,offBot);
    k_disappear_UV<<<grd,blk,0,stream>>>(dUV,pitchUV,W,H,dTop,dBot,offTop,offBot);
}

} // namespace wire
