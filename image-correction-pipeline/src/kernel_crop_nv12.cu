#include "kernel_crop_nv12.cuh"
#include <cstdio>

namespace crop {

__global__ void crop_nv12_kernel(const uint8_t* srcY, const uint8_t* srcUV,
                                 int srcW, int srcH, int srcPitch,
                                 int roiX, int roiY, int roiW, int roiH,
                                 uint8_t* dstY, uint8_t* dstUV,
                                 int dstPitchY, int dstPitchUV)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Copy Y plane
    if (x < roiW && y < roiH)
        dstY[y * dstPitchY + x] = srcY[(roiY + y) * srcPitch + (roiX + x)];

    // Copy UV plane (interleaved, every 2x2 block)
    int uvX = x & ~1;
    int uvY = y >> 1;
    if (uvX < roiW && uvY < roiH / 2) {
        const uint8_t* srcPtr = srcUV + (roiY / 2 + uvY) * srcPitch + roiX + uvX;
        uint8_t* dstPtr       = dstUV + uvY * dstPitchUV + uvX;
        dstPtr[0] = srcPtr[0];
        dstPtr[1] = srcPtr[1];
    }
}

void launch_crop_nv12(const uint8_t* srcY, const uint8_t* srcUV,
                      int srcW, int srcH, int srcPitch,
                      int roiX, int roiY, int roiW, int roiH,
                      uint8_t* dstY, uint8_t* dstUV,
                      int dstPitchY, int dstPitchUV,
                      cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid((roiW + block.x - 1) / block.x,
              (roiH + block.y - 1) / block.y);

    crop_nv12_kernel<<<grid, block, 0, stream>>>(
        srcY, srcUV, srcW, srcH, srcPitch,
        roiX, roiY, roiW, roiH,
        dstY, dstUV, dstPitchY, dstPitchUV);
}

} // namespace crop
