#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace crop {
void launch_crop_nv12(const uint8_t* srcY, const uint8_t* srcUV,
                      int srcW, int srcH, int srcPitch,
                      int roiX, int roiY, int roiW, int roiH,
                      uint8_t* dstY, uint8_t* dstUV,
                      int dstPitchY, int dstPitchUV,
                      cudaStream_t stream);
}

