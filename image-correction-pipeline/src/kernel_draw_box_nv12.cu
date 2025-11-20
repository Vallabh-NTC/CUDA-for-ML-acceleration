#include <cstdio>
#include <cuda_runtime.h>
#include "kernel_draw_box_nv12.cuh"

namespace draw {

__global__ void draw_box_nv12_kernel(
    uint8_t* dY, uint8_t* dUV,
    int W, int H, int pitch,
    int x0, int y0, int w, int h,
    uint8_t Yval, uint8_t Uval, uint8_t Vval)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const int t = 8; // border thickness
    const bool on_border =
        (x >= x0 && x < x0 + w &&
         y >= y0 && y < y0 + h &&
         (x < x0 + t || x >= x0 + w - t ||
          y < y0 + t || y >= y0 + h - t));

    if (on_border)
    {
        // --- write luma (always safe) ---
        dY[y * pitch + x] = Yval;

        // --- write chroma (one pair per 2×2 block) ---
        if ((x % 2 == 0) && (y % 2 == 0))
        {
            // Use plain byte stores — no typecasting, no alignment assumptions
            const int uv_row = y / 2;
            const int uv_col = x;
            const int uv_idx = uv_row * pitch + uv_col;
            dUV[uv_idx + 0] = Uval;
            dUV[uv_idx + 1] = Vval;
        }
    }
}

void launch_draw_box_nv12(
    uint8_t* dY, uint8_t* dUV,
    int W, int H, int pitch,
    int x0, int y0, int w, int h,
    cudaStream_t stream)
{
    dim3 block(32, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    // Choose bright red (easy to see)
    const uint8_t Y = 76, U = 84, V = 255;

    draw_box_nv12_kernel<<<grid, block, 0, stream>>>(
        dY, dUV, W, H, pitch,
        x0, y0, w, h,
        Y, U, V);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[draw-box] kernel launch error: %s\n", cudaGetErrorString(err));
    else
        fprintf(stderr, "[draw-box] kernel launched successfully.\n");
}

} // namespace draw
