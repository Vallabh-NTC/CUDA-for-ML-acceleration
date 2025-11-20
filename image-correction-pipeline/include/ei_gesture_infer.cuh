// ei_gesture_infer.cuh
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace ei {

// Preprocess: NV12 (full frame + ROI) → tensor CHW 1x3x244x244 FP32
void preprocess_nv12_roi_to_chw_fp32(
    const uint8_t* dY,
    const uint8_t* dUV,
    int srcW, int srcH, int srcPitch,
    int roiX, int roiY, int roiW, int roiH,
    float* dstCHW,           // 1x3x244x244, fp32
    cudaStream_t stream
);

} // namespace ei
