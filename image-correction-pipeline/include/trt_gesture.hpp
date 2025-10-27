#pragma once
/**
 * @file trt_gesture.hpp
 * @brief Minimal TensorRT wrapper for a single .engine used for gesture recognition,
 *        instrumented for debugging.
 *
 * Key points:
 * - One input, one output binding (dynamic shapes allowed).
 * - Input 1x1x96x96 (FP16 or FP32).
 * - Output 3 values, default label order matches Edge Impulse:
 *      ok=0, start=1, stop=2   (override via env TRT_LABEL_MAP="start=1,stop=2,ok=0")
 * - Robust dtype handling: output can be FP32 or FP16; we convert to float on host.
 * - Probability-head detection: if outputs already look like probabilities, we do NOT
 *   apply softmax again; else we softmax logits exactly once.
 * - Exposes a "done" event so the caller can fence the async D2H.
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
    size_t outElems      = 0;    // number of floats in output
    size_t outBytesDev   = 0;    // device bytes (depends on dtype)

    // Device buffers
    void* dIn  = nullptr;
    void* dOut = nullptr;

    // Host buffers
    // - hostOutPinnedRaw matches device dtype (fp16 or fp32) and is filled by async D2H.
    // - hostOut is a float view we commit into for logging & math.
    void*              hostOutPinnedRaw = nullptr;
    std::vector<float> hostOut;

    // Optional event: caller will record this after async D2H to signal completion.
    cudaEvent_t ev_trt_done = nullptr;

    // Label mapping (default = Edge Impulse: ok=0, start=1, stop=2)
    int idx_ok    = 0;
    int idx_start = 1;
    int idx_stop  = 2;

    // API
    bool load_from_file(const std::string& path, cudaStream_t video_stream /*unused*/);
    bool ensure_binding_dims_96x96();

    // Commit new output if ev_trt_done has fired (convert fp16→fp32 if needed).
    bool try_commit_host_output();

    // Top-1 index (returns argmax; if probOut!=nullptr returns the winning prob).
    int  top1(float* probOut = nullptr) const;

    // 3-class helper in logical order (START/STOP/OK via idx_* mapping).
    bool get_start_stop_ok(float& start_score, float& stop_score, float& ok_score,
                           float& p_start, float& p_stop, float& p_ok, int& top_class) const;

    // Mapping helpers
    void set_label_map(int start_idx, int stop_idx, int ok_idx);
    bool load_label_map_from_env();

    // Cleanup
    void destroy();
};

} // namespace trt
