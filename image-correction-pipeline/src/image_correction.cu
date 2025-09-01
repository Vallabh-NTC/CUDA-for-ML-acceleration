#include "image_correction.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s at %s:%d -> %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e)); \
} } while(0)

namespace icp {

// -------------------- device sampling --------------------
__device__ __forceinline__ void bilinearSampleRGBA(
    const uint8_t* src, int src_w, int src_h, int src_stride,
    float x, float y, uint8_t out[4])
{
    if (x < 0.f || y < 0.f || x > (float)(src_w-1) || y > (float)(src_h-1)) {
        out[0]=out[1]=out[2]=0; out[3]=255;
        return;
    }
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);
    float dx = x - (float)x0;
    float dy = y - (float)y0;

    const uint8_t* p00 = src + y0 * src_stride + 4 * x0;
    const uint8_t* p10 = src + y0 * src_stride + 4 * x1;
    const uint8_t* p01 = src + y1 * src_stride + 4 * x0;
    const uint8_t* p11 = src + y1 * src_stride + 4 * x1;

    #pragma unroll
    for (int c=0; c<4; ++c) {
        float v00 = (float)p00[c];
        float v10 = (float)p10[c];
        float v01 = (float)p01[c];
        float v11 = (float)p11[c];
        float v0  = v00 + dx * (v10 - v00);
        float v1  = v01 + dx * (v11 - v01);
        float v   = v0  + dy * (v1  - v0);
        int iv    = (int)lrintf(v);
        out[c]    = (uint8_t)max(0, min(255, iv));
    }
}

// -------------------- color helpers --------------------

// Clamp helper for float -> uint8 conversion.
__device__ __forceinline__ uint8_t clamp_u8(float v) {
    v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
    return (uint8_t)(v + 0.5f);
}

// Luma according to Rec.709 coefficients (perceptual).
__device__ __forceinline__ float luma709(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Apply contrast & brightness & white balance in 8-bit linear space.
// Then apply saturation around luma, and finally gamma on [0,1].
__device__ __forceinline__ void applyColorControls(
    float& r, float& g, float& b,
    float contrast, float brightness,
    float sat, float gammaInv,   // pass 1.0/gamma for speed
    float wb_r, float wb_g, float wb_b)
{
    // 1) Contrast/brightness (linear 8-bit domain)
    r = r * contrast + brightness;
    g = g * contrast + brightness;
    b = b * contrast + brightness;

    // 2) White balance (linear multipliers)
    r *= wb_r; g *= wb_g; b *= wb_b;

    // 3) Saturation via luma separation (still linear 8-bit domain)
    float y = luma709(r, g, b);
    r = y + sat * (r - y);
    g = y + sat * (g - y);
    b = y + sat * (b - y);

    // 4) Clamp to [0,255] before gamma
    r = r < 0.f ? 0.f : (r > 255.f ? 255.f : r);
    g = g < 0.f ? 0.f : (g > 255.f ? 255.f : g);
    b = b < 0.f ? 0.f : (b > 255.f ? 255.f : b);

    // 5) Gamma on normalized [0,1]; gammaInv = 1/gamma
    float rn = r * (1.0f/255.0f);
    float gn = g * (1.0f/255.0f);
    float bn = b * (1.0f/255.0f);

    rn = powf(fmaxf(rn, 0.0f), gammaInv);
    gn = powf(fmaxf(gn, 0.0f), gammaInv);
    bn = powf(fmaxf(bn, 0.0f), gammaInv);

    r = rn * 255.0f;
    g = gn * 255.0f;
    b = bn * 255.0f;
}

// -------------------- kernel --------------------
__global__ void rectifyKernel(
    const uint8_t* __restrict__ src, int src_w, int src_h, int src_stride,
    uint8_t* __restrict__ dst,       int dst_w, int dst_h, int dst_stride,
    // precomputed scalars
    float cx_f, float cy_f, float r_f,
    float f_fish,                // r_f / (FOV_fish/2)
    float fx, float cx_rect, float cy_rect,
    // color controls passed per launch ---
    float brightness, float contrast,
    float saturation, float gammaInv,
    float wb_r, float wb_g, float wb_b)
{
    int u = blockDim.x * blockIdx.x + threadIdx.x; // x in output
    int v = blockDim.y * blockIdx.y + threadIdx.y; // y in output
    if (u >= dst_w || v >= dst_h) return;

    // Perspective ray 
    float xn = ( (float)u - cx_rect ) / fx;
    float yn = ( (float)v - cy_rect ) / fx;
    float zn = 1.f;

    // normalize
    float invn = rsqrtf(xn*xn + yn*yn + zn*zn);
    xn *= invn; yn *= invn; zn *= invn;

    // Equidistant fisheye mapping
    float theta = acosf(zn);
    float phi   = atan2f(yn, xn);
    float r     = f_fish * theta;

    float sx = cx_f + r * cosf(phi);
    float sy = cy_f + r * sinf(phi);

    // mask by fisheye circle
    float dx = sx - cx_f;
    float dy = sy - cy_f;
    float maxr = r_f + 1.0f;
    float r2   = dx*dx + dy*dy;
    float maxr2 = maxr * maxr;

    uint8_t rgba[4];
    if (r2 <= maxr2) {
        bilinearSampleRGBA(src, src_w, src_h, src_stride, sx, sy, rgba);
    } else {
        rgba[0]=rgba[1]=rgba[2]=0; rgba[3]=255;
    }

    // Convert to float for processing in linear 8-bit domain
    float r8 = (float)rgba[0];
    float g8 = (float)rgba[1];
    float b8 = (float)rgba[2];

    // --- apply color controls ---
    applyColorControls(r8, g8, b8,
                       contrast, brightness,
                       saturation, gammaInv,
                       wb_r, wb_g, wb_b);

    // Write out (alpha preserved)
    uint8_t* out = dst + v * dst_stride + 4 * u;
    out[0] = clamp_u8(r8);
    out[1] = clamp_u8(g8);
    out[2] = clamp_u8(b8);
    out[3] = rgba[3];
}

// -------------------- host entry --------------------
void fisheye_rectify_rgba(
    const uint8_t* d_src_rgba, int src_w, int src_h, size_t src_stride,
    uint8_t* d_dst_rgba,       int dst_w, int dst_h, size_t dst_stride,
    const RectifyConfig& cfg,
    cudaStream_t stream)
{
    // Keep source aspect ratio for destination if not matching
    if (dst_w <= 0 || dst_h <= 0) return;

    // Precompute constants on host
    const float FOV_fish = cfg.fish_fov_deg * (float)M_PI / 180.f;
    const float f_fish   = cfg.r_f / (FOV_fish * 0.5f);

    const float fx       = (dst_w * 0.5f) / tanf(cfg.out_hfov_deg * (float)M_PI / 360.f);
    const float cx_rect  = dst_w * 0.5f;
    const float cy_rect  = dst_h * 0.5f;
    const float gammaInv = (cfg.gamma > 0.f) ? (1.0f / cfg.gamma) : 1.0f;


    dim3 block(16,16);
    dim3 grid( (dst_w + block.x - 1)/block.x,
               (dst_h + block.y - 1)/block.y );

    rectifyKernel<<<grid, block, 0, stream>>>(
        d_src_rgba, src_w, src_h, (int)src_stride,
        d_dst_rgba, dst_w, dst_h, (int)dst_stride,
        cfg.cx_f, cfg.cy_f, cfg.r_f,
        f_fish, fx, cx_rect, cy_rect,
        // --- color controls forwarded to the kernel ---
        cfg.brightness, cfg.contrast,
        cfg.saturation, gammaInv,
        cfg.wb_r, cfg.wb_g, cfg.wb_b
    );
}

} 
