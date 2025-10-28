#pragma once
/**
 * @file trt_gesture.hpp
 * @brief Minimal TensorRT wrapper for a single .engine used for gesture recognition,
 *        instrumented for debugging.
 *
 * Two-class head (no "OK"):
 *  - Classes: START and STOP (indices configurable via env).
 *  - Output size: 2 values. If they already look like probabilities ([0,1], sum≈1),
 *    we do NOT apply softmax again; otherwise we softmax once.
 *
 * Env override:
 *   TRT_LABEL_MAP="start=0,stop=1"
 *
 * Input:
 *  - One input binding 1x1x96x96 (FP16 or FP32).
 * Output:
 *  - One output binding with 2 elements (FP16 or FP32).
 */

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace trt {

struct Logger final : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) fprintf(stderr, "[trt] %s\n", msg);
    }
};

struct Engine {
    // TRT objects
    std::unique_ptr<nvinfer1::IRuntime>            runtime;
    std::unique_ptr<nvinfer1::ICudaEngine>         engine;
    std::unique_ptr<nvinfer1::IExecutionContext>   context;

    // Bindings
    int inIdx  = -1;
    int outIdx = -1;

    // IO properties
    bool   inputIsFP16   = false;
    bool   outputIsFP16  = false;
    size_t inBytes       = 0;
    size_t outElems      = 0;    // number of floats in output (expect 2)
    size_t outBytesDev   = 0;    // device bytes (depends on dtype)

    // Device buffers
    void* dIn  = nullptr;
    void* dOut = nullptr;

    // Host buffers
    // - hostOutPinnedRaw matches device dtype (fp16 or fp32) and is filled by async D2H.
    // - hostOut is a float view we commit into for logging & math.
    void*              hostOutPinnedRaw = nullptr;
    std::vector<float> hostOut;

    // Event set by the caller when D2H has been enqueued; polled via try_commit_host_output()
    cudaEvent_t ev_trt_done = nullptr;

    // Label mapping (default: start=0, stop=1)
    int idx_start = 0;
    int idx_stop  = 1;

    // API
    bool load_from_file(const std::string& path, cudaStream_t video_stream /*unused*/);
    bool ensure_binding_dims_96x96();

    // Commit new output if ev_trt_done has fired (convert fp16→fp32 if needed).
    bool try_commit_host_output();

    // Top-1 index (returns argmax; if probOut!=nullptr returns the winning prob).
    int  top1(float* probOut = nullptr) const;

    // 2-class helper in logical order (START/STOP via idx_* mapping).
    bool get_start_stop(float& start_score, float& stop_score,
                        float& p_start, float& p_stop, int& top_class) const;

    // Mapping helpers
    void set_label_map(int start_idx, int stop_idx);
    bool load_label_map_from_env();

    // Cleanup
    void destroy();
};

} // namespace trt
