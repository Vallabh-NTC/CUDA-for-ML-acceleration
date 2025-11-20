#include "nv12_to_rgb_chw.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace preprocess {

template<typename T>
__device__ inline T clamp01(T v) {
    return v < (T)0 ? (T)0 : (v > (T)1 ? (T)1 : v);
}

// Convert NV12 ROI -> normalized RGB tensor in CHW
template<typename T>
__global__ void nv12_to_chw_kernel(const uint8_t* __restrict__ dY,
                                   const uint8_t* __restrict__ dUV,
                                   int W, int H, int pitch,
                                   int roiX, int roiY, int roiW, int roiH,
                                   T* __restrict__ out)  // out = 3×roiH×roiW
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= roiW || y >= roiH) return;

    const int srcX = roiX + x;
    const int srcY = roiY + y;
    const uint8_t Y = dY[srcY * pitch + srcX];
    const int uv_index = (srcY / 2) * pitch + (srcX & ~1);
    const float U = (float)dUV[uv_index]     - 128.0f;
    const float V = (float)dUV[uv_index + 1] - 128.0f;

    float r = Y + 1.402f * V;
    float g = Y - 0.344136f * U - 0.714136f * V;
    float b = Y + 1.772f * U;

    r = clamp01(r / 255.0f);
    g = clamp01(g / 255.0f);
    b = clamp01(b / 255.0f);

    const int planeSize = roiW * roiH;
    const int idx = y * roiW + x;
    if constexpr (std::is_same<T, __half>::value) {
        out[idx] = __float2half(r);
        out[idx + planeSize] = __float2half(g);
        out[idx + 2 * planeSize] = __float2half(b);
    } else {
        out[idx] = r;
        out[idx + planeSize] = g;
        out[idx + 2 * planeSize] = b;
    }
}

void launch_nv12_to_chw(const uint8_t* dY, const uint8_t* dUV,
                        int W, int H, int pitch,
                        int roiX, int roiY, int roiW, int roiH,
                        void* dTensor, bool fp16, cudaStream_t stream)
{
    dim3 block(32,16);
    dim3 grid((roiW + block.x - 1)/block.x,
              (roiH + block.y - 1)/block.y);
    if (fp16)
        nv12_to_chw_kernel<<<grid,block,0,stream>>>(dY,dUV,W,H,pitch,roiX,roiY,roiW,roiH,(__half*)dTensor);
    else
        nv12_to_chw_kernel<<<grid,block,0,stream>>>(dY,dUV,W,H,pitch,roiX,roiY,roiW,roiH,(float*)dTensor);
}

} // namespace preprocess
