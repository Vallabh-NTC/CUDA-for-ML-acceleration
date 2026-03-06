// nv12_to_rgb_fp32.cu — NV12 → RGB float32 [1,3,H,W] normalized [0,1]
#include "nv12_to_rgb_fp16.hpp"
#include <cuda_runtime.h>
#include <cstdint>

__global__ void nv12_to_rgb_fp32_kernel(
    const uint8_t *__restrict__ d_y,
    const uint8_t *__restrict__ d_uv,
    int   pitchY,
    int   pitchUV,
    int   W,
    int   H,
    float *__restrict__ d_out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const float Y  = static_cast<float>(d_y[y * pitchY + x]);
    const int uvRow = y >> 1;
    const int uvCol = (x & ~1);
    const float U  = static_cast<float>(d_uv[uvRow * pitchUV + uvCol]);
    const float V  = static_cast<float>(d_uv[uvRow * pitchUV + uvCol + 1]);

    const float cb = U - 128.0f;
    const float cr = V - 128.0f;

    float r = fminf(fmaxf(Y               + 1.402000f * cr, 0.0f), 255.0f) * (1.0f/255.0f);
    float g = fminf(fmaxf(Y - 0.344136f * cb - 0.714136f * cr, 0.0f), 255.0f) * (1.0f/255.0f);
    float b = fminf(fmaxf(Y + 1.772000f * cb, 0.0f), 255.0f) * (1.0f/255.0f);

    const int planeSize = H * W;
    const int idx       = y * W + x;
    d_out[0 * planeSize + idx] = r;
    d_out[1 * planeSize + idx] = g;
    d_out[2 * planeSize + idx] = b;
}

void nv12_to_rgb_fp16(
    const uint8_t *d_y,
    const uint8_t *d_uv,
    int            pitchY,
    int            pitchUV,
    int            W,
    int            H,
    float         *d_out,
    cudaStream_t   stream)
{
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);
    nv12_to_rgb_fp32_kernel<<<grid, block, 0, stream>>>(
        d_y, d_uv, pitchY, pitchUV, W, H, d_out);
}