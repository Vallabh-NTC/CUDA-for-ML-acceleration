#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ei {

// Enqueue: reads NV12 Y plane (dY,pitch,W,H), center-crops to square,
// resizes to 96x96, normalizes to [0,1], and writes to 'dst' (NCHW 1x1x96x96).
// If inIsFP16=true, writes FP16; otherwise FP32.
// Runs entirely on 'stream'. Returns false if args invalid.
bool enqueue_preprocess_to_trt_input(const uint8_t* dY,
                                     int srcW, int srcH, int srcPitch,
                                     void* dst, bool inIsFP16,
                                     cudaStream_t stream);

// Helpers for a dedicated non-blocking stream (optional)
bool ensure_stream_created(cudaStream_t& s);
void destroy_stream(cudaStream_t& s);

} // namespace ei