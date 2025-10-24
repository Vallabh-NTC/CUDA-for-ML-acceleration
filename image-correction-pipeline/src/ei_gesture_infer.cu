#include "ei_gesture_infer.cuh"
#include <cstdio>
#include <algorithm>

namespace {

constexpr int OW = 96;
constexpr int OH = 96;

__device__ __forceinline__ float clamp01(float v) {
    return v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
}

/**
 * @brief NV12(Y) → 96x96, FIT_LONGEST + letterbox (padding=0 after normalization).
 *
 * Mapping:
 *  - Compute a single scale so that the **longest** source side matches the output side.
 *  - Center the scaled image inside the 96x96 canvas, filling the rest with 0.0.
 *  - Bilinear interpolation in source space.
 *  - Normalization:
 *      * tv_range: (Y - Ymin) / (Ymax - Ymin), clamped to [0,1]
 *      * full-range: Y / 255.0
 */
__global__ void nv12_y_to_norm_96_fit_longest(
    const uint8_t* __restrict__ y,
    int srcW, int srcH, int srcPitch,
    float* __restrict__ out, /* 1x1x96x96 */
    int Ymin, int Ymax, int use_tv_range)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= OW || oy >= OH) return;

    // Guard for invalid pitch
    if (srcPitch < srcW) {
        // Write padding to avoid UB; early out.
        out[oy * OW + ox] = 0.0f;
        return;
    }

    // Scale so that the LONGEST source side matches output
    const float sX = static_cast<float>(OW) / static_cast<float>(srcW);
    const float sY = static_cast<float>(OH) / static_cast<float>(srcH);
    const float scale = (sX > sY) ? sX : sY;     // max(sX, sY)

    // Drawn area in output space
    const float drawnW = scale * static_cast<float>(srcW);
    const float drawnH = scale * static_cast<float>(srcH);

    // Centered letterbox padding
    const float padX = 0.5f * (static_cast<float>(OW) - drawnW);
    const float padY = 0.5f * (static_cast<float>(OH) - drawnH);

    // Coordinates in the scaled image space (pixel centers)
    const float fx = (static_cast<float>(ox) + 0.5f) - padX;
    const float fy = (static_cast<float>(oy) + 0.5f) - padY;

    float norm_out = 0.0f; // letterbox padding value (post-normalization)

    if (fx >= 0.f && fy >= 0.f && fx < drawnW && fy < drawnH) {
        // Map back to source space
        const float sx = fx / scale;
        const float sy = fy / scale;

        int x0 = static_cast<int>(floorf(sx));
        int y0 = static_cast<int>(floorf(sy));
        float ax = sx - static_cast<float>(x0);
        float ay = sy - static_cast<float>(y0);

        // Clamp sampling points to valid range
        int x1 = min(x0 + 1, srcW - 1);
        int y1 = min(y0 + 1, srcH - 1);

        const uint8_t* row0 = y + y0 * srcPitch;
        const uint8_t* row1 = y + y1 * srcPitch;

        float v00 = static_cast<float>(row0[x0]);
        float v01 = static_cast<float>(row0[x1]);
        float v10 = static_cast<float>(row1[x0]);
        float v11 = static_cast<float>(row1[x1]);

        float v0 = v00 * (1.0f - ax) + v01 * ax;
        float v1 = v10 * (1.0f - ax) + v11 * ax;
        float Y   = v0  * (1.0f - ay) + v1  * ay;

        if (use_tv_range) {
            const float num = (Y - static_cast<float>(Ymin));
            const float den = static_cast<float>(Ymax - Ymin); // 219 for 16..235
            norm_out = clamp01(num / den);
        } else {
            norm_out = Y * (1.0f / 255.0f);
        }
    }

    // NCHW contiguous (1x1x96x96)
    out[oy * OW + ox] = norm_out;
}

__global__ void cast_f32_to_f16(const float* __restrict__ in, __half* __restrict__ out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

} // anon namespace

namespace ei {

bool ensure_stream_created(cudaStream_t& s) {
    if (s) return true;
#if CUDART_VERSION >= 11000
    // Non-blocking, default priority (you can choose higher priority if desired)
    cudaError_t e = cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, 0);
#else
    cudaError_t e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
#endif
    return (e == cudaSuccess);
}

void destroy_stream(cudaStream_t& s) {
    if (s) { cudaStreamDestroy(s); s = nullptr; }
}

bool enqueue_preprocess_to_trt_input(const uint8_t* dY,
                                     int srcW, int srcH, int srcPitch,
                                     void* dst, bool inIsFP16,
                                     bool tv_range,
                                     cudaStream_t stream)
{
    if (!dY || !dst || srcW <= 0 || srcH <= 0 || srcPitch <= 0) return false;

    // TV-range parameters
    const int Ymin = tv_range ? 16  : 0;
    const int Ymax = tv_range ? 235 : 255;
    const int use_tv = tv_range ? 1 : 0;

    dim3 blk(16,16);
    dim3 grd((OW + blk.x - 1)/blk.x, (OH + blk.y - 1)/blk.y);

    if (inIsFP16) {
        // Temporary FP32 workspace → FP16 cast
        float* tmp = nullptr;
        size_t bytes = sizeof(float) * OW * OH;

        // Try async alloc first, then fallback to cudaMalloc
        cudaError_t e = cudaMallocAsync(&tmp, bytes, stream);
        if (e != cudaSuccess) {
            e = cudaMalloc(&tmp, bytes);
            if (e != cudaSuccess) return false;
        }

        nv12_y_to_norm_96_fit_longest<<<grd, blk, 0, stream>>>(
            dY, srcW, srcH, srcPitch,
            tmp, Ymin, Ymax, use_tv);

        int n = OW * OH;
        int tpb = 256, b = (n + tpb - 1) / tpb;
        cast_f32_to_f16<<<b, tpb, 0, stream>>>(tmp, reinterpret_cast<__half*>(dst), n);

        // Free with the matching API
        cudaError_t efree = cudaFreeAsync(tmp, stream);
        if (efree == cudaErrorNotSupported) {
            cudaGetLastError(); // clear sticky error
            cudaFree(tmp);
        }
    } else {
        // Directly write FP32 into destination
        nv12_y_to_norm_96_fit_longest<<<grd, blk, 0, stream>>>(
            dY, srcW, srcH, srcPitch,
            reinterpret_cast<float*>(dst), Ymin, Ymax, use_tv);
    }

    // Optional: check for asynchronous launch errors (kept lightweight)
#if !defined(NDEBUG)
    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess) {
        fprintf(stderr, "[preproc] kernel launch error: %s\n", cudaGetErrorString(kerr));
        return false;
    }
#endif

    return true;
}

/**
 * @brief Simple block-level reduction for min/max/mean over a small buffer.
 *        This is intended for debug prints; not optimized for very large n.
 */
__global__ void k_stats_simple(const float* __restrict__ a, int n,
                               float* outMin, float* outMax, float* outMean)
{
    extern __shared__ float sdata[]; // [2*blockDim + blockDim] -> min, max, sum
    float* smin = sdata;
    float* smax = sdata + blockDim.x;
    float* ssum = sdata + 2 * blockDim.x;

    float tmin = 1e30f;
    float tmax = -1e30f;
    float tsum = 0.0f;

    // Strided loop over input
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = a[i];
        tmin = fminf(tmin, v);
        tmax = fmaxf(tmax, v);
        tsum += v;
    }

    smin[threadIdx.x] = tmin;
    smax[threadIdx.x] = tmax;
    ssum[threadIdx.x] = tsum;
    __syncthreads();

    // In-place reduction
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smin[threadIdx.x] = fminf(smin[threadIdx.x], smin[threadIdx.x + stride]);
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + stride]);
            ssum[threadIdx.x] += ssum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *outMin = smin[0];
        *outMax = smax[0];
        *outMean = ssum[0] / static_cast<float>(n);
    }
}

void launch_debug_stats_f32(const float* dIn, int n, cudaStream_t stream){
    if (!dIn || n <= 0) return;
    float *dmin=nullptr, *dmax=nullptr, *dmean=nullptr;
    cudaMallocAsync(&dmin,sizeof(float),stream);
    cudaMallocAsync(&dmax,sizeof(float),stream);
    cudaMallocAsync(&dmean,sizeof(float),stream);

    // One-block reduction is fine for small buffers (96*96 = 9216)
    const int tpb = 256;
    k_stats_simple<<<1, tpb, (3*tpb)*sizeof(float), stream>>>(dIn, n, dmin, dmax, dmean);

    float hmin=0.f, hmax=0.f, hmean=0.f;
    cudaMemcpyAsync(&hmin,dmin,sizeof(float),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(&hmax,dmax,sizeof(float),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(&hmean,dmean,sizeof(float),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
    fprintf(stderr,"[preproc] min=%.3f max=%.3f mean=%.3f\n",hmin,hmax,hmean);

    cudaFreeAsync(dmin,stream);
    cudaFreeAsync(dmax,stream);
    cudaFreeAsync(dmean,stream);
}

} // namespace ei
