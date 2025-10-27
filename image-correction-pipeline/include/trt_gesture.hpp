#pragma once
/**
 * @file trt_gesture.hpp
 * @brief Minimal TensorRT wrapper for a single .engine used for gesture recognition.
 *
 * Key points:
 * - Loads a serialized engine from disk.
 * - Assumes one input and one output binding by default (dynamic shapes allowed).
 * - Input is NCHW 1x1x96x96 (FP16 or FP32). We write the preprocessed tensor into device input buffer.
 * - Output is a 3-class head [START, STOP, OK] (float logits). If your model differs,
 *   you can still use top1(), or adjust try_get_start_stop_ok().
 * - Designed for **non-blocking** inference: preprocessing happens on the "video" stream,
 *   TRT enqueue + D2H happen on a **dedicated TRT stream**. No stream sync on the video path.
 *
 * Typical usage (see nvivafilter):
 *   1) engine.load_from_file(path, video_stream);
 *   2) preprocess on video_stream → record event;
 *   3) on trt_stream: wait for event → enqueueV2 → async D2H → record done-event;
 *   4) periodically poll try_get_start_stop_ok(...) to log results without blocking video.
 */

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace trt {

// ----------------------------------------------------------------------------
// Lightweight TRT logger
// ----------------------------------------------------------------------------
struct Logger final : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) fprintf(stderr, "[trt] %s\n", msg);
    }
};

// ----------------------------------------------------------------------------
// Engine wrapper (1 input, 1 output)
// ----------------------------------------------------------------------------
struct Engine {
    // TRT objects
    std::unique_ptr<nvinfer1::IRuntime>            runtime;
    std::unique_ptr<nvinfer1::ICudaEngine>         engine;
    std::unique_ptr<nvinfer1::IExecutionContext>   context;

    // Binding indices
    int inIdx  = -1;
    int outIdx = -1;

    // IO properties
    bool   inputIsFP16 = false;
    size_t inBytes  = 0;
    size_t outBytes = 0;

    // Device buffers
    void* dIn  = nullptr;
    void* dOut = nullptr;

    // Host output:
    //  - pinned buffer for async D2H
    //  - std::vector<float> as a convenient view (copied from pinned when "ready")
    void*             hostOutPinned = nullptr;
    std::vector<float> hostOut;

    // Optional event to mark "inference+copy finished"
    cudaEvent_t ev_trt_done = nullptr;

    // ---- API ----

    // Load serialized engine, allocate device+host buffers, detect types/shapes.
    // The video_stream is only used to satisfy potential shape setup that requires a stream;
    // it can be nullptr. Returns false on failure.
    bool load_from_file(const std::string& path, cudaStream_t video_stream);

    // Prepare/ensure shapes (for explicit batch engines with -1 dims).
    // Sets binding dims for input to 1x1x96x96 when needed. Returns false on failure.
    bool ensure_binding_dims_96x96();

    // Top-1 softmax result from the last **committed** hostOut (non-blocking consumers should
    // call try_commit_host_output() first). Returns class index or -1.
    int  top1(float* probOut = nullptr) const;

    // For 3-class head convenience: get logits & softmax in [START, STOP, OK]
    // topology. Returns false if output size < 3 or data not yet committed.
    bool get_start_stop_ok(float& start_logit, float& stop_logit, float& ok_logit,
                           float& p_start, float& p_stop, float& p_ok, int& top_class) const;

    // Commit the pinned output into hostOut **if** ev_trt_done has fired.
    // Returns true if new data was copied to hostOut; false if not ready.
    bool try_commit_host_output();

    // Destroy all resources.
    void destroy();
};

} // namespace trt
