#pragma once
/**
 * @file ei_gesture_infer.cuh
 * @brief NV12(Y) → 96x96 grayscale tensor (fit-shortest-axis), TV/full range aware.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ei {

// Reads NV12 Y plane (dY,pitch,W,H) and writes a 96x96 grayscale tensor:
// - Resize policy: "fit shortest axis" + center-crop
// - Normalization: [0,1]
//     * if tv_range=true, map Y 16..235 → 0..1 (clamped).
//     * if tv_range=false, map Y 0..255 → 0..1.
// - Output layout: NCHW 1x1x96x96; type FP16 if inIsFP16, else FP32.
// Runs on 'stream'. Returns false on invalid args.
bool enqueue_preprocess_to_trt_input(const uint8_t* dY,
                                     int srcW, int srcH, int srcPitch,
                                     void* dst, bool inIsFP16,
                                     bool tv_range,
                                     cudaStream_t stream);

// Optional helpers for a dedicated non-blocking stream
bool ensure_stream_created(cudaStream_t& s);
void destroy_stream(cudaStream_t& s);

// Optional debug helper: print min/max/mean of a device FP32 buffer of size n
void launch_debug_stats_f32(const float* dIn, int n, cudaStream_t stream);

} // namespace ei
