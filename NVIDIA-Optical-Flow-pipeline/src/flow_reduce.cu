// flow_reduce.cu
// Computes mean u,v over the asphalt ROI using parallel reduction on GPU.
// Uses atomicAdd into shared accumulators — no CPU transfer needed.

#include "flow_reduce.hpp"
#include <cuda_runtime.h>
#include <cstdio>

// GPU-side accumulators (persistent across kernel call via d_result)
__global__ void flow_reduce_kernel(
    const float *__restrict__ d_flow,
    int W, int H,
    int roi_x0, int roi_x1,
    int roi_y0, int roi_y1,
    int step,
    float *d_sum_u,
    float *d_sum_v,
    int   *d_count)
{
    const int sx = blockIdx.x * blockDim.x + threadIdx.x;
    const int sy = blockIdx.y * blockDim.y + threadIdx.y;

    const int px = roi_x0 + sx * step;
    const int py = roi_y0 + sy * step;

    if (px >= roi_x1 || py >= roi_y1) return;
    if (px < 0 || py < 0 || px >= W || py >= H) return;

    const int planeSize = H * W;
    const int idx       = py * W + px;

    const float u = d_flow[0 * planeSize + idx];
    const float v = d_flow[1 * planeSize + idx];

    // Skip NaN/Inf
    if (!isfinite(u) || !isfinite(v)) return;

    atomicAdd(d_sum_u, u);
    atomicAdd(d_sum_v, v);
    atomicAdd(d_count, 1);
}

__global__ void flow_finalize_kernel(
    const float *d_sum_u,
    const float *d_sum_v,
    const int   *d_count,
    FlowResult  *d_result)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int n = *d_count;
    if (n > 0) {
        d_result->mean_u = *d_sum_u / (float)n;
        d_result->mean_v = *d_sum_v / (float)n;
    } else {
        d_result->mean_u = 0.0f;
        d_result->mean_v = 0.0f;
    }
}

// Persistent GPU scratch buffers (allocated once)
static float *d_sum_u = nullptr;
static float *d_sum_v = nullptr;
static int   *d_count = nullptr;

static void ensure_scratch()
{
    if (d_sum_u) return;
    cudaMalloc(&d_sum_u, sizeof(float));
    cudaMalloc(&d_sum_v, sizeof(float));
    cudaMalloc(&d_count, sizeof(int));
}

void flow_reduce(
    const float  *d_flow,
    int           W,
    int           H,
    int           roi_x0,
    int           roi_x1,
    int           roi_y0,
    int           roi_y1,
    int           step,
    FlowResult   *d_result,
    cudaStream_t  stream)
{
    ensure_scratch();

    // Reset accumulators
    cudaMemsetAsync(d_sum_u, 0, sizeof(float), stream);
    cudaMemsetAsync(d_sum_v, 0, sizeof(float), stream);
    cudaMemsetAsync(d_count, 0, sizeof(int),   stream);

    const int nx = (roi_x1 - roi_x0 + step - 1) / step;
    const int ny = (roi_y1 - roi_y0 + step - 1) / step;

    dim3 block(8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);

    flow_reduce_kernel<<<grid, block, 0, stream>>>(
        d_flow, W, H,
        roi_x0, roi_x1, roi_y0, roi_y1,
        step,
        d_sum_u, d_sum_v, d_count);

    flow_finalize_kernel<<<1, 1, 0, stream>>>(
        d_sum_u, d_sum_v, d_count, d_result);
}