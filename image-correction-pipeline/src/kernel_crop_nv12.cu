#include "kernel_crop_nv12.cuh"
#include <cstdio>

namespace crop {

__global__ void crop_nv12_kernel(uint8_t* dY, uint8_t* dUV,
                                 int W, int H, int pitch,
                                 int roiX, int roiY, int roiW, int roiH)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const bool inside = (x >= roiX && x < roiX + roiW &&
                         y >= roiY && y < roiY + roiH);

    if (!inside) {
        dY[y * pitch + x] = 0;
        if ((y % 2 == 0) && (x % 2 == 0)) {
            const int uv_idx = (y / 2) * pitch + x;
            dUV[uv_idx]     = 128;
            dUV[uv_idx + 1] = 128;
        }
    }
}

void launch_crop_nv12(uint8_t* dY, uint8_t* dUV,
                      int W, int H, int pitch,
                      int roiX, int roiY, int roiW, int roiH,
                      cudaStream_t stream)
{
    dim3 block(32, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    //fprintf(stderr, "[crop][debug] ROI(%d,%d,%d,%d) pitch=%d → black fill top=%d rows, left=%d cols\n",
    //        roiX, roiY, roiW, roiH, pitch, roiY, roiX);

    crop_nv12_kernel<<<grid, block, 0, stream>>>(dY, dUV, W, H, pitch, roiX, roiY, roiW, roiH);
}

} // namespace crop
