// src/mv_reduce.cu
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstring>

#include "mv_reduce.hpp"

// OFA MV are S10.5 -> px = v/32
__device__ __forceinline__ float s10_5_to_px(int16_t v) { return (float)v * (1.0f/32.0f); }
__device__ __forceinline__ float hypot2(float x, float y) { return sqrtf(x*x + y*y); }

struct MVPureAcc
{
    int   count;
    float sum_dx;
    float sum_dy;
    float sum_dx2;
    float sum_dy2;
};

__global__ void mv_pure_accum_kernel(const int16_t *mvPtr, MVPureParams p, MVPureAcc *acc)
{
    int roiW = p.x1 - p.x0;
    int roiH = p.y1 - p.y0;

    int sxN = (roiW + p.step - 1) / p.step;
    int syN = (roiH + p.step - 1) / p.step;
    int N   = sxN * syN;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int sy = tid / sxN;
    int sx = tid - sy * sxN;

    int mx = p.x0 + sx * p.step;
    int my = p.y0 + sy * p.step;
    if (mx >= p.x1 || my >= p.y1) return;

    const uint8_t *rowB = (const uint8_t*)mvPtr + (size_t)my * (size_t)p.mvPitchBytes;
    const int16_t *row  = (const int16_t*)rowB;

    int16_t fx = row[mx*2 + 0];
    int16_t fy = row[mx*2 + 1];

    float dx = s10_5_to_px(fx);
    float dy = s10_5_to_px(fy);

    atomicAdd(&acc->count,  1);
    atomicAdd(&acc->sum_dx, dx);
    atomicAdd(&acc->sum_dy, dy);
    atomicAdd(&acc->sum_dx2, dx*dx);
    atomicAdd(&acc->sum_dy2, dy*dy);
}

__global__ void mv_pure_finalize_kernel(const MVPureAcc *acc, MVPureParams p, MVPureOut *out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    MVPureOut o{};
    o.count = acc->count;

    if (o.count > 0 && p.dtSec > 0.0f && p.pxPerMeter > 0.0f) {
        float invn = 1.0f / (float)o.count;

        // Means
        o.mean_dx = acc->sum_dx * invn;
        o.mean_dy = acc->sum_dy * invn;

        // Population variance = E[x^2] - (E[x])^2
        float ex2_dx = acc->sum_dx2 * invn;
        float ex2_dy = acc->sum_dy2 * invn;

        float var_dx = ex2_dx - o.mean_dx * o.mean_dx;
        float var_dy = ex2_dy - o.mean_dy * o.mean_dy;

        // Numerical safety
        var_dx = fmaxf(var_dx, 0.0f);
        var_dy = fmaxf(var_dy, 0.0f);

        o.std_dx = sqrtf(var_dx);
        o.std_dy = sqrtf(var_dy);
        o.std_mag = hypot2(o.std_dx, o.std_dy);

        // Resultant + speed
        o.res_mag = hypot2(o.mean_dx, o.mean_dy);
        o.speed_mps = (o.res_mag / p.pxPerMeter) / p.dtSec;
    } else {
        o.mean_dx = o.mean_dy = 0.0f;
        o.std_dx  = o.std_dy  = 0.0f;
        o.std_mag = 0.0f;
        o.res_mag = 0.0f;
        o.speed_mps = 0.0f;
    }

    *out = o;
}

extern "C" void mv_reduce_pure_cuda(const int16_t *mvPtr,
                                   const MVPureParams *params,
                                   MVPureOut *d_out)
{
    if (!mvPtr || !params || !d_out) return;

    MVPureAcc *d_acc = nullptr;
    cudaMalloc(&d_acc, sizeof(MVPureAcc));
    cudaMemset(d_acc, 0, sizeof(MVPureAcc));

    MVPureParams p = *params;

    int roiW = p.x1 - p.x0;
    int roiH = p.y1 - p.y0;
    int sxN  = (roiW + p.step - 1) / p.step;
    int syN  = (roiH + p.step - 1) / p.step;
    int N    = sxN * syN;

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    mv_pure_accum_kernel<<<blocks, threads>>>(mvPtr, p, d_acc);
    mv_pure_finalize_kernel<<<1, 1>>>(d_acc, p, d_out);

    cudaFree(d_acc);
}
