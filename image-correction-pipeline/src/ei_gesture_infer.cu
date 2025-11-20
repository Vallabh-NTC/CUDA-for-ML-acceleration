#include "ei_gesture_infer.cuh"
#include <cuda_fp16.h>

namespace ei {

// NOTE: these must match the input resolution expected by your network.
// For the original trt_pose_hand ResNet18 models, the default is often 224x224,
// but here we use 244x244 because that is what the rest of your pipeline expects.
// If you rebuild the engine with a different input size, update these values.
constexpr int GESTURE_W = 244;
constexpr int GESTURE_H = 244;

// torchvision-style mean/std for RGB in range [0,1]
__constant__ float kMean[3] = {0.485f, 0.456f, 0.406f};
__constant__ float kStd[3]  = {0.229f, 0.224f, 0.225f};

__device__ inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// -----------------------------------------------------------------------------
// NV12 sampling helpers
// -----------------------------------------------------------------------------

// Bilinear sampling on the Y plane
__device__ float sampleY_bilinear(const uint8_t* dY, int pitch,
                                  int W, int H, float x, float y) {
    x = clampf(x, 0.0f, (float)(W - 1));
    y = clampf(y, 0.0f, (float)(H - 1));

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = x0 + 1 < W ? x0 + 1 : x0;
    int y1 = y0 + 1 < H ? y0 + 1 : y0;

    float dx = x - (float)x0;
    float dy = y - (float)y0;

    const uint8_t* row0 = dY + y0 * pitch;
    const uint8_t* row1 = dY + y1 * pitch;

    float v00 = (float)row0[x0];
    float v01 = (float)row0[x1];
    float v10 = (float)row1[x0];
    float v11 = (float)row1[x1];

    float v0 = v00 + dx * (v01 - v00);
    float v1 = v10 + dx * (v11 - v10);
    return v0 + dy * (v1 - v0);
}

// Load U,V from NV12 (UV interleaved, 4:2:0) with bilinear interpolation.
// U and V are subsampled 2x2 → we operate in half-resolution coordinates.
__device__ void loadUV_nv12(const uint8_t* dUV, int pitch,
                            int W, int H, float x, float y,
                            float& U, float& V) {
    // Map full-res (x,y) into UV half-res space
    float ux = clampf(x * 0.5f, 0.0f, (float)(W/2 - 1));
    float uy = clampf(y * 0.5f, 0.0f, (float)(H/2 - 1));

    int x0 = (int)floorf(ux);
    int y0 = (int)floorf(uy);
    int x1 = x0 + 1 < (W/2) ? x0 + 1 : x0;
    int y1 = y0 + 1 < (H/2) ? y0 + 1 : y0;

    float dx = ux - (float)x0;
    float dy = uy - (float)uy;

    const uint8_t* row0 = dUV + y0 * pitch;
    const uint8_t* row1 = dUV + y1 * pitch;

    // In NV12: UVUVUV... (U at even indices, V at odd indices)
    auto sampleUV = [&](const uint8_t* row, int xx, float& u, float& v) {
        int idx = xx * 2;
        u = (float)row[idx + 0];
        v = (float)row[idx + 1];
    };

    float u00, v00, u01, v01, u10, v10, u11, v11;
    sampleUV(row0, x0, u00, v00);
    sampleUV(row0, x1, u01, v01);
    sampleUV(row1, x0, u10, v10);
    sampleUV(row1, x1, u11, v11);

    float u0 = u00 + dx * (u01 - u00);
    float u1 = u10 + dx * (u11 - u10);
    float v0 = v00 + dx * (v01 - v00);
    float v1 = v10 + dx * (v11 - v10);

    U = u0 + dy * (u1 - u0);
    V = v0 + dy * (v1 - v0);
}

// YUV (BT.601 approx) → RGB, output clamped to [0,255]
__device__ void yuv_to_rgb(float Y, float U, float V,
                           float& R, float& G, float& B) {
    float C = Y - 16.0f;
    float D = U - 128.0f;
    float E = V - 128.0f;

    float r = 1.164f * C + 1.596f * E;
    float g = 1.164f * C - 0.392f * D - 0.813f * E;
    float b = 1.164f * C + 2.017f * D;

    R = clampf(r, 0.0f, 255.0f);
    G = clampf(g, 0.0f, 255.0f);
    B = clampf(b, 0.0f, 255.0f);
}

// -----------------------------------------------------------------------------
// Main preprocessing kernel
// -----------------------------------------------------------------------------

__global__ void k_preprocess_nv12_roi_to_chw_fp32(
    const uint8_t* __restrict__ dY,
    const uint8_t* __restrict__ dUV,
    int srcW,
    int srcH,
    int srcPitch,
    int roiX,
    int roiY,
    int roiW,
    int roiH,
    float* __restrict__ dst
) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= GESTURE_W || oy >= GESTURE_H) return;

    // Map output pixel (ox,oy) into ROI coordinates in the source image.
    // We sample at pixel centers: (ox + 0.5f) / GESTURE_W.
    float sx = roiX + ((ox + 0.5f) / (float)GESTURE_W) * roiW;
    float sy = roiY + ((oy + 0.5f) / (float)GESTURE_H) * roiH;

    float Y = sampleY_bilinear(dY, srcPitch, srcW, srcH, sx, sy);
    float U, V;
    loadUV_nv12(dUV, srcPitch, srcW, srcH, sx, sy, U, V);

    float R, G, B;
    yuv_to_rgb(Y, U, V, R, G, B);

    // 0..255 → 0..1
    R *= (1.0f / 255.0f);
    G *= (1.0f / 255.0f);
    B *= (1.0f / 255.0f);

    // torchvision-style normalization
    R = (R - kMean[0]) / kStd[0];
    G = (G - kMean[1]) / kStd[1];
    B = (B - kMean[2]) / kStd[2];

    int idx = oy * GESTURE_W + ox;
    int planeSize = GESTURE_W * GESTURE_H;

    // CHW layout: [0]=R, [1]=G, [2]=B
    dst[0 * planeSize + idx] = R;
    dst[1 * planeSize + idx] = G;
    dst[2 * planeSize + idx] = B;
}

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
    float* dstCHW,
    cudaStream_t stream
) {
    if (roiW <= 0 || roiH <= 0) {
        // Invalid ROI, nothing to do
        return;
    }

    dim3 block(16, 16);
    dim3 grid(
        (GESTURE_W + block.x - 1) / block.x,
        (GESTURE_H + block.y - 1) / block.y
    );

    k_preprocess_nv12_roi_to_chw_fp32<<<grid, block, 0, stream>>>(
        dY, dUV,
        srcW, srcH, srcPitch,
        roiX, roiY, roiW, roiH,
        dstCHW
    );
}

} // namespace ei
