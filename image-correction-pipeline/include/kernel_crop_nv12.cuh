#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace crop {

// Launches a simple black-fill crop on NV12
void launch_crop_nv12(uint8_t* dY, uint8_t* dUV,
                      int W, int H, int pitch,
                      int roiX, int roiY, int roiW, int roiH,
                      cudaStream_t stream);

} // namespace crop
