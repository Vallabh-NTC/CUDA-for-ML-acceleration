#pragma once
#include <cstdint> 
#include <cuda_runtime.h>

namespace preprocess {
void launch_nv12_to_chw(const uint8_t* dY, const uint8_t* dUV,
                        int W, int H, int pitch,
                        int roiX, int roiY, int roiW, int roiH,
                        void* dTensor, bool fp16, cudaStream_t stream);
}
