// image-correction-pipeline/src/rectify_kernels.cu
#include "rectify_kernels.cuh"
#include <cmath>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ----------------------------- Device utilities ------------------------------
// All helpers are __device__ __forceinline__ to keep per-thread overhead minimal.

__device__ __forceinline__ float luma709(float r,float g,float b){
    // Perceptual luma (Rec.709 coefficients); used for saturation processing
    return 0.2126f*r + 0.7152f*g + 0.0722f*b;
}

__device__ __forceinline__ uint8_t clamp_u8(float v){
    // Clamp a float expected to be in [0..255] to uint8_t safely and cheaply.
    v = (v < 0.f) ? 0.f : ((v > 255.f) ? 255.f : v);
    return (uint8_t)(v + 0.5f); // round-to-nearest
}

__device__ __forceinline__ void applyColorControls(
    float& r,float& g,float& b,
    float con,float bri,float sat,float gammaInv,
    float wb_r,float wb_g,float wb_b)
{
    // 1) Linear contrast/brightness
    r = r*con + bri; g = g*con + bri; b = b*con + bri;

    // 2) Per-channel white balance multipliers (still linear 8-bit domain)
    r *= wb_r; g *= wb_g; b *= wb_b;

    // 3) Saturation relative to luma (keeps luminance stable)
    float y = luma709(r,g,b);
    r = y + sat*(r - y);
    g = y + sat*(g - y);
    b = y + sat*(b - y);

    // 4) Clamp before gamma to preserve numerical sanity
    r = (r < 0.f) ? 0.f : ((r > 255.f) ? 255.f : r);
    g = (g < 0.f) ? 0.f : ((g > 255.f) ? 255.f : g);
    b = (b < 0.f) ? 0.f : ((b > 255.f) ? 255.f : b);

    // 5) Gamma correction in normalized space, then back to 8-bit scale
    float rn = powf(fmaxf(r/255.f, 0.f), gammaInv);
    float gn = powf(fmaxf(g/255.f, 0.f), gammaInv);
    float bn = powf(fmaxf(b/255.f, 0.f), gammaInv);
    r = rn*255.f; g = gn*255.f; b = bn*255.f;
}

// Bilinear sampler over pitch-linear ABGR buffer.
// - src: base pointer to first pixel (row 0, col 0)
// - spitch: row stride in BYTES
// - (x,y): floating-point source coordinate in pixel space
// Returns 4 bytes in ABGR order.
__device__ __forceinline__ void bilinearABGR(
    const uint8_t* src,int sw,int sh,int spitch,
    float x,float y,uint8_t out[4])
{
    // Quick reject: outside source bounds → return opaque black
    if (x < 0.f || y < 0.f || x > (float)(sw-1) || y > (float)(sh-1)) {
        out[0]=255; out[1]=0; out[2]=0; out[3]=0; return;
    }

    // Integer neighbors
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    int x1 = (x0 + 1 < sw) ? (x0 + 1) : (sw - 1);
    int y1 = (y0 + 1 < sh) ? (y0 + 1) : (sh - 1);
    float dx = x - (float)x0, dy = y - (float)y0;

    const uint8_t* p00 = src + y0*spitch + 4*x0;
    const uint8_t* p10 = src + y0*spitch + 4*x1;
    const uint8_t* p01 = src + y1*spitch + 4*x0;
    const uint8_t* p11 = src + y1*spitch + 4*x1;

    // Interpolate each channel independently
    #pragma unroll
    for (int c=0; c<4; ++c){
        float v00 = (float)p00[c];
        float v10 = (float)p10[c];
        float v01 = (float)p01[c];
        float v11 = (float)p11[c];
        float v0  = v00 + dx*(v10 - v00);
        float v1  = v01 + dx*(v11 - v01);
        float v   = v0  + dy*(v1  - v0);
        int iv    = __float2int_rn(v);
        iv = (iv < 0) ? 0 : ((iv > 255) ? 255 : iv);
        out[c] = (uint8_t)iv;
    }
}

// ------------------------------- Main kernel ---------------------------------
// For each output pixel (u,v), compute its unit ray in perspective space,
// then map that ray back to equidistant fisheye pixel coordinates and sample
// the source image with bilinear filtering. Finally, apply the color pipeline.
__global__ void rectifyABGRKernel(
    const uint8_t* __restrict__ src,int sw,int sh,int spitch,
          uint8_t* __restrict__ dst,int dw,int dh,int dpitch,
    // Optics mapping constants
    float cx_f,float cy_f,float r_f,
    float f_fish,float fx,float cx_rect,float cy_rect,
    // Color controls (already preprocessed on host: gammaInv = 1/gamma)
    float bri,float con,float sat,float gammaInv,
    float wb_r,float wb_g,float wb_b)
{
    int u = blockIdx.x*blockDim.x + threadIdx.x; // output X
    int v = blockIdx.y*blockDim.y + threadIdx.y; // output Y
    if (u>=dw || v>=dh) return;

    // Perspective normalized ray (note 'fx' maps horizontal FOV)
    float xn = ((float)u - cx_rect) / fx;
    float yn = ((float)v - cy_rect) / fx;
    float zn = 1.f;

    // Normalize ray to unit length
    float invn = rsqrtf(xn*xn + yn*yn + zn*zn);
    xn *= invn; yn *= invn; zn *= invn;

    // Equidistant fisheye projection:
    // angle theta from optical axis maps linearly to radius r in the fisheye plane
    float theta = acosf(zn);
    float phi   = atan2f(yn, xn);
    float r     = f_fish * theta;

    // Source sampling coordinate in fisheye image
    float sx = cx_f + r * cosf(phi);
    float sy = cy_f + r * sinf(phi);

    // Mask out anything beyond the calibrated fisheye circle radius
    float dx = sx - cx_f, dy = sy - cy_f;
    float maxr = r_f + 1.f;
    float r2 = dx*dx + dy*dy, maxr2 = maxr*maxr;

    uint8_t abgr[4];
    if (r2 <= maxr2) bilinearABGR(src, sw, sh, spitch, sx, sy, abgr);
    else { abgr[0]=255; abgr[1]=abgr[2]=abgr[3]=0; } // opaque black

    // Convert to floats in linear 8-bit domain and apply color pipeline
    float r8 = (float)abgr[3]; // ABGR → R at index 3
    float g8 = (float)abgr[2]; // G at index 2
    float b8 = (float)abgr[1]; // B at index 1
    applyColorControls(r8,g8,b8, con,bri,sat,gammaInv, wb_r,wb_g,wb_b);

    // Store back in ABGR order
    uint8_t* o = dst + v*dpitch + 4*u;
    o[0] = 255;             // A
    o[1] = clamp_u8(b8);    // B
    o[2] = clamp_u8(g8);    // G
    o[3] = clamp_u8(r8);    // R
}

// ------------------------------- Host launcher -------------------------------
// Precomputes constants that depend on output geometry and config, then launches
// the CUDA kernel on the provided stream. d_src and d_dst may alias.
void launch_rectify_kernel(
    const uint8_t* d_src, int src_w, int src_h, int src_pitch,
    uint8_t* d_dst,       int dst_w, int dst_h, int dst_pitch,
    const RectifyConfig& cfg,
    cudaStream_t stream)
{
    if (dst_w<=0 || dst_h<=0) return;

    // Precompute mapping constants:
    // f_fish derives from the equidistant model: r = f_fish * theta
    const float FOV_fish = cfg.fish_fov_deg * (float)M_PI / 180.f;
    const float f_fish   = cfg.r_f / (FOV_fish * 0.5f);

    // Perspective intrinsics: fx from desired HFOV; principal point at image center
    const float fx      = (dst_w * 0.5f) / tanf(cfg.out_hfov_deg * (float)M_PI / 360.f);
    const float cx_rect = dst_w * 0.5f;
    const float cy_rect = dst_h * 0.5f;

    // Gamma is inverted once for performance; 1/gamma used in device code
    const float gammaInv = (cfg.gamma > 0.f) ? (1.f / cfg.gamma) : 1.f;

    dim3 block(16,16);
    dim3 grid((dst_w + block.x - 1)/block.x,
              (dst_h + block.y - 1)/block.y);

    rectifyABGRKernel<<<grid, block, 0, stream>>>(
        d_src, src_w, src_h, src_pitch,
        d_dst, dst_w, dst_h, dst_pitch,
        cfg.cx_f, cfg.cy_f, cfg.r_f,
        f_fish, fx, cx_rect, cy_rect,
        cfg.brightness, cfg.contrast, cfg.saturation, gammaInv,
        cfg.wb_r, cfg.wb_g, cfg.wb_b
    );
}
