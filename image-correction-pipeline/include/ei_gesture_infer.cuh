#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace ei {

/**
 * Preprocess NV12 frame (with a given ROI) to a CHW FP32 tensor suitable
 * for a hand pose network (e.g. ResNet18-based from trt_pose_hand).
 *
 * Input:
 *   - dY / dUV: NV12 planes on device (Y full-res, UV interleaved 4:2:0)
 *   - srcW, srcH, srcPitch: source frame dimensions and pitch (bytes)
 *   - roiX, roiY, roiW, roiH: region of interest in the source frame
 *
 * Output:
 *   - dstCHW: device pointer to FP32 tensor with shape (1,3,H,W) in CHW layout
 *             (here H = W = 244, see implementation details).
 *   - stream: CUDA stream used to enqueue the preprocessing kernel
 *
 * The kernel:
 *   - samples NV12 with bilinear interpolation
 *   - converts YUV → RGB (BT.601 approx)
 *   - scales 0..255 → 0..1
 *   - applies torchvision-style mean/std normalization
 *   - writes R, G, B into CHW layout.
 */
void preprocess_nv12_roi_to_chw_fp32(
    const uint8_t* dY,
    const uint8_t* dUV,
    int srcW,
    int srcH,
    int srcPitch,
    int roiX,
    int roiY,
    int roiW,
    int roiH,
    float* dstCHW,           // 1 x 3 x 244 x 244, FP32
    cudaStream_t stream
);

} // namespace ei
