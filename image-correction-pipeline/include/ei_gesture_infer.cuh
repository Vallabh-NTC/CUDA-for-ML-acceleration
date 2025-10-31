#pragma once
/**
 * @file ei_gesture_infer.cuh
 * @brief NV12 (Y plane) → 96x96 grayscale tensor.
 *
 * Resize policy: **fit-longest** + center letterbox (padding).
 * Normalization: [0, 1]
 *  - If tv_range=true, map Y 16..235 → 0..1 (clamped).
 *  - If tv_range=false, map Y 0..255 → 0..1.
 * Output layout: NCHW 1x1x96x96; type FP16 if inIsFP16, else FP32.
 *
 * Notes:
 *  - The letterbox padding value is 0.0 after normalization.
 *  - Designed to run on the given CUDA stream.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ei {

/**
 * @brief Enqueue preprocess from NV12 Y-plane into TensorRT input buffer.
 *
 * @param dY       Device pointer to NV12 Y plane (uint8).
 * @param srcW     Source image width in pixels.
 * @param srcH     Source image height in pixels.
 * @param srcPitch Source image line pitch in bytes (>= srcW).
 * @param dst      Destination device buffer (TRT input). FP16 if inIsFP16, otherwise FP32.
 * @param inIsFP16 Whether the TRT input binding expects FP16.
 * @param tv_range If true, normalize 16..235 → 0..1; else 0..255 → 0..1.
 * @param stream   CUDA stream to use.
 * @return false on invalid args or allocation failure; true if the kernels were launched.
 */
bool enqueue_preprocess_to_trt_input(const uint8_t* dY,
                                     int srcW, int srcH, int srcPitch,
                                     void* dst, bool inIsFP16,
                                     bool tv_range,
                                     cudaStream_t stream);

/**
 * @brief Optionally create a non-blocking CUDA stream if null.
 */
bool ensure_stream_created(cudaStream_t& s);

/**
 * @brief Destroy a CUDA stream created earlier (no-op if null).
 */
void destroy_stream(cudaStream_t& s);

/**
 * @brief Optional debug helper: print min/max/mean of a device FP32 buffer of size n.
 *        (Synchronizes the stream before printing.)
 */
void launch_debug_stats_f32(const float* dIn, int n, cudaStream_t stream);

} // namespace ei
