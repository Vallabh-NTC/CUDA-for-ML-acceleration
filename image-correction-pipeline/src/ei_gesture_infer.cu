#include "ei_gesture_infer.cuh"
#include <algorithm>
#include <cstdio>

namespace {
constexpr int OW = 96;
constexpr int OH = 96;

__device__ __forceinline__ float clamp01(float v) {
    return v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
}

// Bilinear resize + normalize to [0,1].
// If tv_range=true:   norm = clamp((Y-16)/219, 0, 1)
// If tv_range=false:  norm = Y/255
__global__ void nv12_y_to_norm_96_bilinear(const uint8_t* __restrict__ y,
                                           int srcW, int srcH, int srcPitch,
                                           float* __restrict__ out /*1x1x96x96*/,
                                           int Ymin, int Ymax, int use_tv_range)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= OW || oy >= OH) return;

    // center-square crop from source, then scale to 96x96
    int side = min(srcW, srcH);
    int x0 = (srcW - side) / 2;
    int y0 = (srcH - side) / 2;

    float sx = (ox + 0.5f) * (float)side / (float)OW + (float)x0;
    float sy = (oy + 0.5f) * (float)side / (float)OH + (float)y0;

    int x = (int)floorf(sx), yv = (int)floorf(sy);
    float ax = sx - x, ay = sy - yv;

    int x1 = min(x+1, srcW-1);
    int y1 = min(yv+1, srcH-1);

    const uint8_t* row0 = y + yv * srcPitch;
    const uint8_t* row1 = y + y1 * srcPitch;

    float v00 = static_cast<float>(row0[x]);
    float v01 = static_cast<float>(row0[x1]);
    float v10 = static_cast<float>(row1[x]);
    float v11 = static_cast<float>(row1[x1]);

    float v0 = v00 * (1.0f - ax) + v01 * ax;
    float v1 = v10 * (1.0f - ax) + v11 * ax;
    float Y  = v0  * (1.0f - ay) + v1  * ay;

    float norm;
    if (use_tv_range) {
        // map 16..235 → 0..1, clamp outside
        float num = (Y - (float)Ymin);
        float den = (float)(Ymax - Ymin); // 219 for 16..235
        norm = clamp01(num / den);
    } else {
        norm = Y * (1.0f / 255.0f);
    }

    out[oy * OW + ox] = norm; // NCHW contiguous (1x1x96x96)
}

__global__ void cast_f32_to_f16(const float* __restrict__ in, __half* __restrict__ out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

} // anon

namespace ei {

bool ensure_stream_created(cudaStream_t& s) {
    if (s) return true;
#if CUDART_VERSION >= 11000
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
        float* tmp = nullptr;
        size_t bytes = sizeof(float) * OW * OH;
        if (cudaMallocAsync(&tmp, bytes, stream) != cudaSuccess) return false;

        nv12_y_to_norm_96_bilinear<<<grd, blk, 0, stream>>>(dY, srcW, srcH, srcPitch,
                                                            tmp, Ymin, Ymax, use_tv);

        int n = OW * OH;
        int tpb = 256, b = (n + tpb - 1)/tpb;
        cast_f32_to_f16<<<b, tpb, 0, stream>>>(tmp, reinterpret_cast<__half*>(dst), n);

        cudaFreeAsync(tmp, stream);
    } else {
        nv12_y_to_norm_96_bilinear<<<grd, blk, 0, stream>>>(
            dY, srcW, srcH, srcPitch,
            reinterpret_cast<float*>(dst), Ymin, Ymax, use_tv);
    }
    return true;
}

// (Optional) tiny debug kernel – prints basic stats once synchronized
__global__ void k_stats(const float* __restrict__ a, int n, float* outMin, float* outMax, float* outMean){
    __shared__ float smin, smax, ssum;
    if (threadIdx.x==0){ smin=1e9f; smax=-1e9f; ssum=0.f; }
    __syncthreads();
    for (int i=threadIdx.x;i<n;i+=blockDim.x){
        float v=a[i];
        atomicMin((int*)&smin, __float_as_int(fminf(smin, v)));
        atomicMax((int*)&smax, __float_as_int(fmaxf(smax, v)));
        atomicAdd(&ssum, v);
    }
    __syncthreads();
    if (threadIdx.x==0){
        *outMin = smin;
        *outMax = smax;
        *outMean= ssum / n;
    }
}

void launch_debug_stats_f32(const float* dIn, int n, cudaStream_t stream){
    float *dmin=nullptr, *dmax=nullptr, *dmean=nullptr;
    cudaMallocAsync(&dmin,sizeof(float),stream);
    cudaMallocAsync(&dmax,sizeof(float),stream);
    cudaMallocAsync(&dmean,sizeof(float),stream);
    k_stats<<<1,256,0,stream>>>(dIn,n,dmin,dmax,dmean);
    float hmin,hmax,hmean;
    cudaMemcpyAsync(&hmin,dmin,sizeof(float),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(&hmax,dmax,sizeof(float),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(&hmean,dmean,sizeof(float),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
    fprintf(stderr,"[preproc] min=%.3f max=%.3f mean=%.3f\n",hmin,hmax,hmean);
    cudaFreeAsync(dmin,stream); cudaFreeAsync(dmax,stream); cudaFreeAsync(dmean,stream);
}

} // namespace ei
