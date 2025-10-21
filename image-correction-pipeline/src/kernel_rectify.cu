/**
 * @file kernel_rectify.cu
 * @brief CUDA implementation (optimized for Orin):
 *   - fisheye rectification (NV12 → NV12, equidistant r = f_fish * theta)
 *   - uses texture objects with linear filtering
 *   - separated Y (full-res) and UV (half-res) kernels for better utilization
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include "kernel_rectify.cuh"

namespace icp {

// ------------------------- helpers -------------------------

__device__ __forceinline__ uint8_t clamp_u8f(float v){
    v = v<0.f?0.f:(v>255.f?255.f:v);
    return (uint8_t)(v + 0.5f);
}
template<typename T> static inline T divUp(T a, T b){ return (a+b-1)/b; }

// -------------------- shared rectification math --------------------

struct RectifyParams {
    float cx_f, cy_f, r_f;      // fisheye circle (px)
    float f_fish;               // equidistant focal (px/rad)
    float fx;                   // rectified focal (px) – note: used for both x/y as in original
    float cx_rect, cy_rect;     // rectified principal point (px)
    int sW, sH;                 // source dims
    int dW, dH;                 // dest dims
};

// Map an output rectified pixel (x,y) → fisheye source coords (sx,sy).
__device__ __forceinline__
bool map_rect_to_fisheye(int x, int y, float& sx, float& sy, const RectifyParams& P)
{
    // Perspective → unit sphere (rectified camera model)
    // (stays faithful to your original math; we use fast intrinsics where available)
    float xn = ((float)x - P.cx_rect) / P.fx;
    float yn = ((float)y - P.cy_rect) / P.fx;
    float zn = 1.f;

    float invn = rsqrtf(xn*xn + yn*yn + zn*zn);
    xn*=invn; yn*=invn; zn*=invn;

    // angles
    float theta = acosf(zn);                 // __acosf is available too; keeping accuracy
    float phi   = atan2f(yn, xn);

    // Equidistant fisheye: r = f_fish * theta
    float r     = P.f_fish * theta;

    sx = P.cx_f + r * cosf(phi);
    sy = P.cy_f + r * sinf(phi);

    // inside fisheye circle (slightly padded like original)
    float dx = sx - P.cx_f, dy = sy - P.cy_f;
    float maxr = P.r_f + 1.0f;
    return (dx*dx + dy*dy) <= (maxr*maxr);
}

// -------------------- texture-backed kernels --------------------
//
// Textures are bound as:
//  - Y: 8-bit UNORM, linear filter, non-normalized coords → tex2D<float> ∈ [0,1]
//  - UV: 2×8-bit UNORM, linear filter, non-normalized coords → tex2D<float2> ∈ [0,1]
//
// This gives hardware bilinear + cache for resampling.
// ----------------------------------------------------------------

__global__ void rectifyNV12_Y_kernel(cudaTextureObject_t yTex,
                                     uint8_t* __restrict__ dstY, int dW, int dH, int dPitchY,
                                     RectifyParams P)
{
    const int x = blockDim.x*blockIdx.x + threadIdx.x;
    const int y = blockDim.y*blockIdx.y + threadIdx.y;
    if (x>=dW || y>=dH) return;

    float sx, sy;
    bool inside = map_rect_to_fisheye(x, y, sx, sy, P);

    float yNorm = 0.0f;
    if (inside && sx>=0.f && sy>=0.f && sx<=(float)(P.sW-1) && sy<=(float)(P.sH-1)) {
        // normalized [0,1]; linear filtered
        yNorm = tex2D<float>(yTex, sx, sy);
    }
    dstY[y*dPitchY + x] = clamp_u8f(yNorm * 255.0f);
}

__global__ void rectifyNV12_UV_kernel(cudaTextureObject_t uvTex,
                                      uint8_t* __restrict__ dstUV, int dW, int dH, int dPitchUV,
                                      RectifyParams P)
{
    // Launch over UV grid (half res)
    const int ux = blockDim.x*blockIdx.x + threadIdx.x; // [0 .. dW/2)
    const int uy = blockDim.y*blockIdx.y + threadIdx.y; // [0 .. dH/2)
    if (ux >= (dW>>1) || uy >= (dH>>1)) return;

    // Center of the 2×2 luma block in output space
    const float x_center = (float)(ux*2) + 0.5f;
    const float y_center = (float)(uy*2) + 0.5f;

    float sx, sy;
    bool inside = map_rect_to_fisheye((int)x_center, (int)y_center, sx, sy, P);

    float U = 0.5f; // 0.5 in normalized space == 128
    float V = 0.5f;

    if (inside) {
        // Sample UV texture in UV pixel space (half-res of Y)
        // Coordinates are non-normalized; UV plane is w/2 × h/2.
        const float u = sx * 0.5f;
        const float v = sy * 0.5f;

        // Clamp to valid range to avoid edge artifacts
        if (u>=0.f && v>=0.f && u <= (float)((P.sW>>1)-1) && v <= (float)((P.sH>>1)-1)) {
            float2 uvNorm = tex2D<float2>(uvTex, u, v); // each ∈ [0,1]
            U = uvNorm.x;
            V = uvNorm.y;
        }
    }

    const int out = uy*dPitchUV + (ux<<1);
    dstUV[out+0] = clamp_u8f(U * 255.0f);
    dstUV[out+1] = clamp_u8f(V * 255.0f);
}

// -------------------- texture setup helpers --------------------

static cudaTextureObject_t make_y_tex(const uint8_t* srcY, int pitchY, int w, int h)
{
    cudaResourceDesc res{};
    res.resType = cudaResourceTypePitch2D;
    res.res.pitch2D.devPtr = (void*)srcY;
    res.res.pitch2D.desc = cudaCreateChannelDesc<unsigned char>(); // 8-bit
    res.res.pitch2D.width  = w;         // elements
    res.res.pitch2D.height = h;         // rows
    res.res.pitch2D.pitchInBytes = pitchY;

    cudaTextureDesc tex{};
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode     = cudaFilterModeLinear;          // bilinear
    tex.readMode       = cudaReadModeNormalizedFloat;   // 8-bit → [0,1]
    tex.normalizedCoords = 0;                           // use pixel coords

    cudaTextureObject_t t = 0;
    cudaCreateTextureObject(&t, &res, &tex, nullptr);
    return t;
}

static cudaTextureObject_t make_uv_tex(const uint8_t* srcUV, int pitchUV, int wUV, int hUV)
{
    cudaResourceDesc res{};
    res.resType = cudaResourceTypePitch2D;
    // 2×8-bit interleaved (uchar2)
    cudaChannelFormatDesc ch{};
    ch.x = 8; ch.y = 8; ch.z = 0; ch.w = 0; ch.f = cudaChannelFormatKindUnsigned;
    res.res.pitch2D.devPtr = (void*)srcUV;
    res.res.pitch2D.desc = ch;
    res.res.pitch2D.width  = wUV;       // elements (uchar2 count)
    res.res.pitch2D.height = hUV;
    res.res.pitch2D.pitchInBytes = pitchUV;

    cudaTextureDesc tex{};
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode     = cudaFilterModeLinear;
    tex.readMode       = cudaReadModeNormalizedFloat; // → float2 in [0,1]
    tex.normalizedCoords = 0;

    cudaTextureObject_t t = 0;
    cudaCreateTextureObject(&t, &res, &tex, nullptr);
    return t;
}

// -------------------- public launch API --------------------

void launch_rectify_nv12(
    const uint8_t* d_src_y,  int src_w, int src_h, int src_pitch_y,
    const uint8_t* d_src_uv,                   int src_pitch_uv,
    uint8_t* d_dst_y,        int dst_w, int dst_h, int dst_pitch_y,
    uint8_t* d_dst_uv,                       int dst_pitch_uv,
    float cx_f, float cy_f, float r_f,
    float f_fish, float fx, float cx_rect, float cy_rect,
    cudaStream_t stream)
{
    // Build texture objects for the source planes
    cudaTextureObject_t yTex  = make_y_tex (d_src_y,  src_pitch_y,  src_w,      src_h);
    cudaTextureObject_t uvTex = make_uv_tex(d_src_uv, src_pitch_uv, src_w >> 1, src_h >> 1);

    RectifyParams P{};
    P.cx_f = cx_f; P.cy_f = cy_f; P.r_f = r_f;
    P.f_fish = f_fish;
    P.fx = fx; P.cx_rect = cx_rect; P.cy_rect = cy_rect;
    P.sW = src_w; P.sH = src_h;
    P.dW = dst_w; P.dH = dst_h;

    // --- Y pass (full resolution) ---
    dim3 blockY(32, 8);
    dim3 gridY((unsigned)divUp(dst_w, (int)blockY.x),
               (unsigned)divUp(dst_h, (int)blockY.y));

    rectifyNV12_Y_kernel<<<gridY, blockY, 0, stream>>>(yTex, d_dst_y, dst_w, dst_h, dst_pitch_y, P);

    // --- UV pass (half resolution) ---
    const int uvW = dst_w >> 1, uvH = dst_h >> 1;
    dim3 blockUV(32, 8);
    dim3 gridUV((unsigned)divUp(uvW, (int)blockUV.x),
                (unsigned)divUp(uvH, (int)blockUV.y));

    rectifyNV12_UV_kernel<<<gridUV, blockUV, 0, stream>>>(uvTex, d_dst_uv, dst_w, dst_h, dst_pitch_uv, P);

    // Destroy texture objects (scoped to this call)
    cudaDestroyTextureObject(yTex);
    cudaDestroyTextureObject(uvTex);
}

} // namespace icp
