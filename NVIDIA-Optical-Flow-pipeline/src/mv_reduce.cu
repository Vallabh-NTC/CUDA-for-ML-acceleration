// mv_reduce.cu
// GPU-only MV reduction + gating + adaptive EMA.
// Reads pitch-linear int16 MV field: interleaved dx,dy in S10.5.

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include "mv_reduce.hpp"

__device__ __forceinline__ float s10_5_to_px(int16_t v) { return (float)v * (1.0f/32.0f); }
__device__ __forceinline__ float hypotf2(float x, float y) { return sqrtf(x*x + y*y); }
__device__ __forceinline__ float clampf(float v, float lo, float hi) { return fminf(hi, fmaxf(lo, v)); }

// atomicMax for non-negative float
__device__ inline void atomicMaxFloat(float *addr, float val)
{
    int *addr_i = (int*)addr;
    int old = *addr_i, assumed;
    do {
        assumed = old;
        float oldf = __int_as_float(assumed);
        if (oldf >= val) break;
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    } while (assumed != old);
}

__global__ void mv_reduce_kernel(const int16_t *mvPtr, MVParams p, MVReduceOut *out)
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
    float mag = hypotf2(dx, dy);

    atomicMaxFloat(&out->max_mag, mag);

    if (mag < p.minMag || mag > p.maxMag) return;

    atomicAdd(&out->count, 1);
    atomicAdd(&out->sum_dx, dx);
    atomicAdd(&out->sum_dy, dy);
    atomicAdd(&out->sum_mag, mag);
    atomicAdd(&out->sum_mag2, mag*mag);
}

__global__ void mv_finalize_kernel(DevEmaState *st, MVParams p, const MVReduceOut *red, float *outSpeed)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float out = st->lastOut;

    // Keep last direction unless we compute a new one.
    float dir_dx = st->lastResDx;
    float dir_dy = st->lastResDy;

    int n = red->count;
    if (n > 0 && p.dtSec > 0.0f) {
        float invn = 1.0f / (float)n;

        float mean_dx  = red->sum_dx  * invn;
        float mean_dy  = red->sum_dy  * invn;
        float mean_mag = red->sum_mag * invn;

        float res_mag  = hypotf2(mean_dx, mean_dy);

        float mean_mag2 = red->sum_mag2 * invn;
        float var_mag   = fmaxf(0.0f, mean_mag2 - mean_mag*mean_mag);
        float std_mag   = sqrtf(var_mag);

        float coherence = res_mag / (mean_mag + 1e-6f);
        coherence = clampf(coherence, 0.0f, 1.0f);

        bool tailEnable = (red->max_mag >= p.tailMinAbs);
        bool tailSpike  = tailEnable && (red->max_mag > mean_mag * p.tailRatio);

        bool lowSamples   = (n < p.minSamples);
        bool lowCoherence = (coherence < p.cohMin);
        bool highStd      = (std_mag > p.stdMax);

        bool badFrame = tailSpike || lowSamples || lowCoherence || highStd;

        float speed_raw = (res_mag / p.pxPerMeter) / p.dtSec;

        if (badFrame) {
            out = st->haveEma ? st->speedEma : st->lastOut;
        } else {
            dir_dx = mean_dx;
            dir_dy = mean_dy;

            bool excellent = (coherence >= p.cohExcellent) && (std_mag <= p.stdExcellentMax);
            float alpha = excellent ? p.alphaHi : p.alphaLo;
            alpha = clampf(alpha, 0.0f, 1.0f);

            if (!st->haveEma) {
                st->speedEma = speed_raw;
                st->haveEma = 1;
            } else {
                st->speedEma = (1.0f - alpha) * st->speedEma + alpha * speed_raw;
            }
            out = st->speedEma;
        }
    }

    st->lastOut   = out;
    st->lastResDx = dir_dx;
    st->lastResDy = dir_dy;
    *outSpeed = out;
}

extern "C" void mv_reduce_gating_ema_cuda(const int16_t *mvPtr,
                                         DevEmaState   *devState,
                                         const MVParams *params,
                                         float         *d_outSpeed,
                                         MVReduceOut   *d_outDiag)
{
    if (!mvPtr || !devState || !params || !d_outSpeed) return;

    MVReduceOut *d_red = d_outDiag;
    bool temp = false;
    if (!d_red) {
        cudaMalloc(&d_red, sizeof(MVReduceOut));
        temp = true;
    }

    cudaMemset(d_red, 0, sizeof(MVReduceOut));

    MVParams p = *params;

    int roiW = p.x1 - p.x0;
    int roiH = p.y1 - p.y0;
    int sxN  = (roiW + p.step - 1) / p.step;
    int syN  = (roiH + p.step - 1) / p.step;
    int N    = sxN * syN;

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    mv_reduce_kernel<<<blocks, threads>>>(mvPtr, p, d_red);
    mv_finalize_kernel<<<1, 1>>>(devState, p, d_red, d_outSpeed);

    if (temp) cudaFree(d_red);
}
