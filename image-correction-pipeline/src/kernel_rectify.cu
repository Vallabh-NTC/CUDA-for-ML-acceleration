/**
 * @file kernel_rectify.cu
 * @brief CUDA implementation:
 *   - fisheye rectification (NV12 → NV12)
 *   - post-rectification center-crop/zoom (NV12 → NV12, same size)
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include "kernel_rectify.cuh"

namespace icp {

__device__ __forceinline__ uint8_t clamp_u8f(float v){
    v = v<0.f?0.f:(v>255.f?255.f:v);
    return (uint8_t)(v + 0.5f);
}
template<typename T> static inline T divUp(T a, T b){ return (a+b-1)/b; }

__device__ __forceinline__
float sampleY_bilinear(const uint8_t* y, int pitchY, int w, int h, float x, float yv)
{
    if (x < 0.f || yv < 0.f || x > (float)(w-1) || yv > (float)(h-1)) return 0.f;
    int x0 = (int)floorf(x), y0 = (int)floorf(yv);
    int x1 = x0 + 1 < w ? x0 + 1 : w - 1;
    int y1 = y0 + 1 < h ? y0 + 1 : h - 1;
    float dx = x - x0, dy = yv - y0;

    const uint8_t* p00 = y + y0 * pitchY + x0;
    const uint8_t* p10 = y + y0 * pitchY + x1;
    const uint8_t* p01 = y + y1 * pitchY + x0;
    const uint8_t* p11 = y + y1 * pitchY + x1;

    float v00 = (float)(*p00), v10 = (float)(*p10);
    float v01 = (float)(*p01), v11 = (float)(*p11);

    float v0 = v00 + dx*(v10 - v00);
    float v1 = v01 + dx*(v11 - v01);
    return v0 + dy*(v1 - v0);
}

__device__ __forceinline__
void sampleUV_bilinear(const uint8_t* uv, int pitchUV, int wUV, int hUV, float x, float y, float& U, float& V)
{
    float u = x * 0.5f, v = y * 0.5f;
    if (u < 0.f || v < 0.f || u > (float)(wUV-1) || v > (float)(hUV-1)) { U=128.f; V=128.f; return; }

    int x0 = (int)floorf(u), y0 = (int)floorf(v);
    int x1 = x0 + 1 < wUV ? x0 + 1 : wUV - 1;
    int y1 = y0 + 1 < hUV ? y0 + 1 : hUV - 1;
    float du = u - x0, dv = v - y0;

    const uint8_t* p00 = uv + y0 * pitchUV + (x0<<1);
    const uint8_t* p10 = uv + y0 * pitchUV + (x1<<1);
    const uint8_t* p01 = uv + y1 * pitchUV + (x0<<1);
    const uint8_t* p11 = uv + y1 * pitchUV + (x1<<1);

    float U00=p00[0], V00=p00[1];
    float U10=p10[0], V10=p10[1];
    float U01=p01[0], V01=p01[1];
    float U11=p11[0], V11=p11[1];

    float U0 = U00 + du*(U10 - U00);
    float U1 = U01 + du*(U11 - U01);
    float V0 = V00 + du*(V10 - V00);
    float V1 = V01 + du*(V11 - V01); // fix
    U = U0 + dv*(U1 - U0);
    V = V0 + dv*(V1 - V0);
}

/* -------------------- Rectification kernel -------------------- */

__global__ void rectifyNV12Kernel(
    const uint8_t* __restrict__ srcY,  int sW, int sH, int sPitchY,
    const uint8_t* __restrict__ srcUV,              int sPitchUV,
    uint8_t* __restrict__ dstY,        int dW, int dH, int dPitchY,
    uint8_t* __restrict__ dstUV,                     int dPitchUV,
    float cx_f, float cy_f, float r_f,
    float f_fish, float fx, float cx_rect, float cy_rect)
{
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if (x>=dW || y>=dH) return;

    // perspective → equidistant (no crop here)
    float xn = ( ((float)x - cx_rect) / fx );
    float yn = ( ((float)y - cy_rect) / fx );
    float zn = 1.f;
    float invn = rsqrtf(xn*xn + yn*yn + zn*zn);
    xn*=invn; yn*=invn; zn*=invn;

    float theta = acosf(zn);
    float phi   = atan2f(yn, xn);
    float r     = f_fish * theta;
    float sx    = cx_f + r * cosf(phi);
    float sy    = cy_f + r * sinf(phi);

    float dx = sx - cx_f, dy = sy - cy_f;
    float maxr = r_f + 1.0f;
    bool inside = (dx*dx + dy*dy) <= (maxr*maxr);

    float Y = 0.f, U=128.f, V=128.f;
    if (inside) {
        Y = sampleY_bilinear(srcY, sPitchY, sW, sH, sx, sy);
        sampleUV_bilinear(srcUV, sPitchUV, sW/2, sH/2, sx, sy, U, V);
    }

    dstY[y*dPitchY + x] = clamp_u8f(Y);
    if ((x%2==0) && (y%2==0)) {
        int uv = (y/2)*dPitchUV + x;
        dstUV[uv+0] = clamp_u8f(U);
        dstUV[uv+1] = clamp_u8f(V);
    }
}

void launch_rectify_nv12(
    const uint8_t* d_src_y,  int src_w, int src_h, int src_pitch_y,
    const uint8_t* d_src_uv,                   int src_pitch_uv,
    uint8_t* d_dst_y,        int dst_w, int dst_h, int dst_pitch_y,
    uint8_t* d_dst_uv,                       int dst_pitch_uv,
    float cx_f, float cy_f, float r_f,
    float f_fish, float fx, float cx_rect, float cy_rect,
    cudaStream_t stream)
{
    dim3 block(16,16);
    dim3 grid((unsigned)divUp(dst_w,(int)block.x),
              (unsigned)divUp(dst_h,(int)block.y));

    rectifyNV12Kernel<<<grid, block, 0, stream>>>(
        d_src_y, src_w, src_h, src_pitch_y,
        d_src_uv,       src_pitch_uv,
        d_dst_y, dst_w, dst_h, dst_pitch_y,
        d_dst_uv,       dst_pitch_uv,
        cx_f, cy_f, r_f, f_fish, fx, cx_rect, cy_rect);
}

/* -------------------- Post-rectification crop kernel -------------------- */

__global__ void cropCenterNV12Kernel(
    const uint8_t* __restrict__ srcY, int W, int H, int srcPitchY,
    const uint8_t* __restrict__ srcUV,          int srcPitchUV,
    uint8_t* __restrict__ dstY,                 int dstPitchY,
    uint8_t* __restrict__ dstUV,                int dstPitchUV,
    float crop_frac)
{
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if (x>=W || y>=H) return;

    // Zoom factor after cropping central region.
    float c = fminf(fmaxf(crop_frac, 0.0f), 0.45f);
    float s = 1.0f / (1.0f - 2.0f * c); // e.g. 0.2 → 1/0.6 ≈ 1.6667

    float cx = 0.5f * (float)W;
    float cy = 0.5f * (float)H;

    // Map output pixel to source coordinate inside rectified image
    float xs = cx + ((float)x - cx) / s;
    float ys = cy + ((float)y - cy) / s;

    float Y = sampleY_bilinear(srcY, srcPitchY, W, H, xs, ys);
    float U=128.f, V=128.f;
    sampleUV_bilinear(srcUV, srcPitchUV, W/2, H/2, xs, ys, U, V);

    dstY[y*dstPitchY + x] = clamp_u8f(Y);
    if ((x%2==0) && (y%2==0)) {
        int uv = (y/2)*dstPitchUV + x;
        dstUV[uv+0] = clamp_u8f(U);
        dstUV[uv+1] = clamp_u8f(V);
    }
}

void launch_crop_center_nv12(
    const uint8_t* d_src_y,  int W, int H, int src_pitch_y,
    const uint8_t* d_src_uv,             int src_pitch_uv,
    uint8_t* d_dst_y,                    int dst_pitch_y,
    uint8_t* d_dst_uv,                   int dst_pitch_uv,
    float crop_frac,
    cudaStream_t stream)
{
    dim3 block(16,16);
    dim3 grid((unsigned)divUp(W,(int)block.x),
              (unsigned)divUp(H,(int)block.y));

    cropCenterNV12Kernel<<<grid, block, 0, stream>>>(
        d_src_y, W, H, src_pitch_y,
        d_src_uv,     src_pitch_uv,
        d_dst_y,      dst_pitch_y,
        d_dst_uv,     dst_pitch_uv,
        crop_frac);
}

} // namespace icp
