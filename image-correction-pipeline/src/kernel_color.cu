/**
 * @file kernel_color.cu
 * @brief CUDA implementations for NV12 color grading, histogram statistics, and tone-mapping.
 *
 * This file contains the GPU kernels that power the color pipeline:
 *
 *  - Histogram + mean U/V computation for auto-exposure and auto white balance.
 *  - In-place color grading:
 *      * LUT-based exposure/gamma mapping (filmic curve).
 *      * Contrast/brightness adjustments.
 *      * Saturation rolloff for highlights (to avoid cartoon look).
 *      * White balance corrections on UV plane.
 *  - Local tone-mapping (CLAHE-lite) to improve contrast in shadows/highlights.
 *
 * All kernels assume NV12 format: Y plane full-res, UV plane interleaved half-res.
 *
 * These are building blocks called by the nvivafilter plugin.
 */
 
#include "kernel_color.cuh"
#include <cuda_runtime.h>
#include <math.h>

namespace {

__device__ __forceinline__ float clampf(float v, float a, float b) {
    return v < a ? a : (v > b ? b : v);
}
__device__ __forceinline__ float smoothstep(float a, float b, float x) {
    float t = clampf((x - a) / (b - a + 1e-6f), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}
__device__ __forceinline__ int imin(int a,int b){ return a<b?a:b; }

// ==============================
// Stats (istogramma Y + medie U/V)
// ==============================
__global__ void k_stats_nv12(const uint8_t* __restrict__ Y,  int pitchY,
                             const uint8_t* __restrict__ UV, int pitchUV,
                             int W, int H,
                             int rx, int ry, int rw, int rh,
                             int step,
                             unsigned int* __restrict__ gHist,
                             double* __restrict__ gSumU,
                             double* __restrict__ gSumV,
                             unsigned int* __restrict__ gCount)
{
    __shared__ unsigned int shist[256];
    if (threadIdx.x < 256) shist[threadIdx.x] = 0u;
    __syncthreads();

    double localU = 0.0, localV = 0.0;
    unsigned int localC = 0;

    int xs = (rw + step - 1) / step;
    int ys = (rh + step - 1) / step;
    int total = xs * ys;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nts = gridDim.x * blockDim.x;

    for (int idx = tid; idx < total; idx += nts) {
        int sy = idx / xs;
        int sx = idx % xs;

        int x = rx + sx * step;
        int y = ry + sy * step;
        if (x >= W) x = W - 1;
        if (y >= H) y = H - 1;

        unsigned int yv = Y[y * pitchY + x];
        atomicAdd(&shist[yv], 1u);

        int uvx = (x & ~1);
        int uvy = (y >> 1);
        int uvidx = uvy * pitchUV + uvx;

        float U = UV[uvidx + 0];
        float V = UV[uvidx + 1];

        localU += (double)U;
        localV += (double)V;
        localC += 1u;
    }

    __syncthreads();
    if (threadIdx.x < 256) atomicAdd(&gHist[threadIdx.x], shist[threadIdx.x]);
    if (localC) {
        atomicAdd(gSumU, localU);
        atomicAdd(gSumV, localV);
        atomicAdd(gCount, localC);
    }
}

// ==============================
// Temporal denoise (NV12)
// ==============================
__global__ void k_tdenoise_y(uint8_t* curY, uint8_t* prevY,
                             int pitch, int ppitch, int W, int H,
                             float alpha, int thr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    uint8_t* cptr = curY  + y * pitch  + x;
    uint8_t* pptr = prevY + y * ppitch + x;

    int c = *cptr;
    int p = *pptr;
    int d = abs(c - p);
    float out = (d < thr) ? ((1.f - alpha) * c + alpha * p) : (float)c;

    *pptr = (uint8_t)c;
    *cptr = (uint8_t)(out + 0.5f);
}

__global__ void k_tdenoise_uv(uint8_t* curUV, uint8_t* prevUV,
                              int pitch, int ppitch, int W, int H2,
                              float alpha, int thr)
{
    // Operiamo per "coppie" UV (2 byte) -> griglia su W/2 x H/2
    int px = blockIdx.x * blockDim.x + threadIdx.x; // coppia
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pairsW = (W + 1) >> 1;
    if (px >= pairsW || py >= H2) return;

    int x = px << 1;
    int off  = py * pitch  + x;
    int offp = py * ppitch + x;

    uint8_t Uc = curUV[off + 0],  Vc = curUV[off + 1];
    uint8_t Up = prevUV[offp + 0], Vp = prevUV[offp + 1];

    int dU = abs((int)Uc - (int)Up);
    int dV = abs((int)Vc - (int)Vp);

    float Uo = (dU < thr) ? ((1.f - alpha) * Uc + alpha * Up) : (float)Uc;
    float Vo = (dV < thr) ? ((1.f - alpha) * Vc + alpha * Vp) : (float)Vc;

    prevUV[offp + 0] = Uc;  prevUV[offp + 1] = Vc;
    curUV[off  + 0] = (uint8_t)(Uo + 0.5f);
    curUV[off  + 1] = (uint8_t)(Vo + 0.5f);
}

// ==============================
// Color grading NV12 in-place
// ==============================
__global__ void k_color_nv12(uint8_t* __restrict__ Y,  uint8_t* __restrict__ UV,
                             int pitchY, int pitchUV,
                             int W, int H,
                             const uint8_t* __restrict__ lutY,
                             float contrast, float addY,
                             float sat_base, float gamma,
                             float wb_r, float /*wb_g*/, float wb_b,
                             float sat_hi_start, float sat_hi_end, float sat_hi_min)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    // Y
    uint8_t y0 = Y[y * pitchY + x];
    float y1 = lutY[y0] * (1.f/255.f);
    y1 = powf(y1, 1.f / fmaxf(1e-6f, gamma));
    float y_lin = y1 * 255.f;
    y_lin = (y_lin - 128.f) * contrast + 128.f + addY;
    y_lin = clampf(y_lin, 0.f, 255.f);
    Y[y * pitchY + x] = (uint8_t)(y_lin + 0.5f);

    // UV (per 2x2)
    if ((x & 1) == 0 && (y & 1) == 0) {
        int x1 = imin(x+1, W-1);
        int y1r= imin(y+1, H-1);
        float yA = (float)Y[y  * pitchY + x ];
        float yB = (float)Y[y  * pitchY + x1];
        float yC = (float)Y[y1r* pitchY + x ];
        float yD = (float)Y[y1r* pitchY + x1];
        float yAvg = 0.25f * (yA + yB + yC + yD);
        float yNorm = yAvg * (1.f/255.f);

        float t = smoothstep(sat_hi_start, sat_hi_end, yNorm);
        float sat_eff = sat_base * ((1.f - t) + t * sat_hi_min);

        int uvIdx = (y >> 1) * pitchUV + x;
        float U = (float)UV[uvIdx + 0];
        float V = (float)UV[uvIdx + 1];

        float uC = U - 128.f;
        float vC = V - 128.f;

        float kU = 1.f + 0.50f * (wb_b - 1.f);
        float kV = 1.f + 0.50f * (wb_r - 1.f);

        uC *= sat_eff * kU;
        vC *= sat_eff * kV;

        float Uo = clampf(128.f + uC, 0.f, 255.f);
        float Vo = clampf(128.f + vC, 0.f, 255.f);

        UV[uvIdx + 0] = (uint8_t)(Uo + 0.5f);
        UV[uvIdx + 1] = (uint8_t)(Vo + 0.5f);
    }
}

// ==============================
// Local tone map (box mean + detail boost con rolloff alte luci)
// ==============================
__global__ void k_ltm(uint8_t* Y, int pitch, int W, int H,
                      int r, float amount, float hs, float he)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int x0 = max(0, x - r), x1 = min(W-1, x + r);
    int y0 = max(0, y - r), y1 = min(H-1, y + r);

    float sum = 0.f; int cnt = 0;
    for (int yy=y0; yy<=y1; ++yy) {
        const uint8_t* row = Y + yy * pitch;
        for (int xx=x0; xx<=x1; ++xx) {
            sum += (float)row[xx];
        }
    }
    cnt = (x1 - x0 + 1) * (y1 - y0 + 1);
    float mean = sum / fmaxf(1, cnt);

    float yv = (float)Y[y * pitch + x];
    float detail = yv - mean;

    float w = 1.f - smoothstep(hs, he, yv * (1.f/255.f));
    float out = yv + amount * detail * w;

    out = clampf(out, 0.f, 255.f);
    Y[y * pitch + x] = (uint8_t)(out + 0.5f);
}

} // anon namespace

// ==============================
// Host wrappers
// ==============================
namespace icp {

void compute_stats_nv12(const uint8_t* dY,  int pitchY,
                        const uint8_t* dUV, int pitchUV,
                        int W, int H,
                        int roi_x, int roi_y, int roi_w, int roi_h,
                        int step,
                        uint32_t hist[256], float* meanU, float* meanV,
                        cudaStream_t stream)
{
    unsigned int *dHist = nullptr, *dCount = nullptr;
    double *dSumU = nullptr, *dSumV = nullptr;
    cudaMalloc(&dHist,  256 * sizeof(unsigned int));
    cudaMalloc(&dCount, sizeof(unsigned int));
    cudaMalloc(&dSumU,  sizeof(double));
    cudaMalloc(&dSumV,  sizeof(double));

    cudaMemsetAsync(dHist,  0, 256 * sizeof(unsigned int), stream);
    cudaMemsetAsync(dCount, 0, sizeof(unsigned int),       stream);
    cudaMemsetAsync(dSumU,  0, sizeof(double),             stream);
    cudaMemsetAsync(dSumV,  0, sizeof(double),             stream);

    dim3 block(256);
    int xs = (roi_w + step - 1) / step;
    int ys = (roi_h + step - 1) / step;
    int total = xs * ys;
    dim3 grid(max(1, min(1024, (total + block.x - 1) / block.x)));

    k_stats_nv12<<<grid, block, 0, stream>>>(
        dY, pitchY, dUV, pitchUV, W, H,
        roi_x, roi_y, roi_w, roi_h, step,
        dHist, dSumU, dSumV, dCount);

    unsigned int hHist[256]; unsigned int hCount=0;
    double hSumU=0.0, hSumV=0.0;

    cudaMemcpyAsync(hHist,  dHist,  256*sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&hCount,dCount, sizeof(unsigned int),     cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&hSumU, dSumU,  sizeof(double),           cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&hSumV, dSumV,  sizeof(double),           cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    for (int i=0;i<256;++i) hist[i] = hHist[i];
    if (hCount > 0) {
        *meanU = (float)(hSumU / (double)hCount);
        *meanV = (float)(hSumV / (double)hCount);
    } else {
        *meanU = 128.f; *meanV = 128.f;
    }

    cudaFree(dHist); cudaFree(dCount); cudaFree(dSumU); cudaFree(dSumV);
}

void launch_temporal_denoise_nv12(
    uint8_t* dY,  uint8_t* dUV,
    int pitchY,   int pitchUV,
    int W, int H,
    uint8_t* prevY, uint8_t* prevUV, int ppY, int ppUV,
    float alphaY, float alphaUV,
    int thrY, int thrUV,
    cudaStream_t stream)
{
    dim3 block(16,16);
    dim3 gridY((W + block.x - 1)/block.x,
               (H + block.y - 1)/block.y);
    dim3 gridUV(((W+1)/2 + block.x - 1)/block.x,
                ((H/2)    + block.y - 1)/block.y);

    k_tdenoise_y <<<gridY,  block, 0, stream>>> (dY,  prevY,  pitchY, ppY,  W, H,   alphaY,  thrY);
    k_tdenoise_uv<<<gridUV, block, 0, stream>>> (dUV, prevUV, pitchUV,ppUV, W, H/2, alphaUV, thrUV);
}

void launch_color_grade_nv12_inplace(uint8_t* dY, uint8_t* dUV,
                                     int pitchY, int pitchUV,
                                     int W, int H,
                                     const uint8_t lutY[256],
                                     float contrast, float addY,
                                     float sat_base, float gamma,
                                     float wb_r, float wb_g, float wb_b,
                                     float sat_hi_start, float sat_hi_end, float sat_hi_min,
                                     cudaStream_t stream)
{
    dim3 block2(16,16);
    dim3 grid2((W + block2.x - 1)/block2.x,
               (H + block2.y - 1)/block2.y);

    uint8_t* dLUT = nullptr;
    cudaMalloc(&dLUT, 256);
    cudaMemcpyAsync(dLUT, lutY, 256, cudaMemcpyHostToDevice, stream);

    k_color_nv12<<<grid2, block2, 0, stream>>>(
        dY, dUV, pitchY, pitchUV, W, H,
        dLUT,
        contrast, addY,
        sat_base, gamma,
        wb_r, wb_g, wb_b,
        sat_hi_start, sat_hi_end, sat_hi_min);

    cudaFree(dLUT);
}

void launch_local_tonemap_nv12(uint8_t* dY, int pitchY, int W, int H,
                               int radius, float amount, float hi_start, float hi_end,
                               cudaStream_t stream)
{
    if (radius <= 0 || amount == 0.f) return;

    dim3 block(16,16);
    dim3 grid((W + block.x - 1)/block.x,
              (H + block.y - 1)/block.y);
    k_ltm<<<grid, block, 0, stream>>>(dY, pitchY, W, H, radius, amount, hi_start, hi_end);
}

} // namespace icp
