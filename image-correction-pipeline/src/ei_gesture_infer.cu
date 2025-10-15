#include "ei_gesture_infer.cuh"
#include <algorithm>

namespace {
constexpr int OW = 96;
constexpr int OH = 96;

__global__ void nv12_y_to_f32_96_bilinear(const uint8_t* __restrict__ y,
                                          int srcW, int srcH, int srcPitch,
                                          float* __restrict__ out /*1x1x96x96*/)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= OW || oy >= OH) return;

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

    float v00 = row0[x]  * (1.0f/255.0f);
    float v01 = row0[x1] * (1.0f/255.0f);
    float v10 = row1[x]  * (1.0f/255.0f);
    float v11 = row1[x1] * (1.0f/255.0f);

    float v0 = v00 * (1.0f - ax) + v01 * ax;
    float v1 = v10 * (1.0f - ax) + v11 * ax;
    float v  = v0  * (1.0f - ay) + v1  * ay;

    out[oy * OW + ox] = v; // NCHW contiguous (1x1x96x96)
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
                                     cudaStream_t stream)
{
    if (!dY || !dst || srcW <= 0 || srcH <= 0 || srcPitch <= 0) return false;
    dim3 blk(16,16);
    dim3 grd((OW + blk.x - 1)/blk.x, (OH + blk.y - 1)/blk.y);

    if (inIsFP16) {
        float* tmp = nullptr;
        size_t bytes = sizeof(float) * OW * OH;
        if (cudaMallocAsync(&tmp, bytes, stream) != cudaSuccess) return false;

        nv12_y_to_f32_96_bilinear<<<grd, blk, 0, stream>>>(dY, srcW, srcH, srcPitch, tmp);

        int n = OW * OH;
        int tpb = 256, b = (n + tpb - 1)/tpb;
        cast_f32_to_f16<<<b, tpb, 0, stream>>>(tmp, reinterpret_cast<__half*>(dst), n);

        cudaFreeAsync(tmp, stream);
    } else {
        nv12_y_to_f32_96_bilinear<<<grd, blk, 0, stream>>>(dY, srcW, srcH, srcPitch,
                                                           reinterpret_cast<float*>(dst));
    }
    return true;
}

} // namespace ei
