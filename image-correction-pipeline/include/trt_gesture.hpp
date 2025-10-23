#pragma once
/**
 * @file trt_gesture.hpp
 * @brief Minimal TensorRT wrapper to run a single .engine for gesture recognition.
 *
 * - Loads a serialized engine from disk.
 * - Assumes one input and one output binding by default.
 * - Input is NCHW 1x1x96x96 (FP16 or FP32), we write into device input buffer.
 * - Output is a 2-class head [start, stop] (float logits). If your model differs,
 *   adjust get_start_stop() accordingly.
 */

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

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

    // Host copy of the output (float)
    std::vector<float> hostOut;

    // API
    bool load_from_file(const std::string& path, cudaStream_t stream);
    bool infer_from_nv12_y(const uint8_t* dY, int W, int H, int pitch, cudaStream_t stream);
    int  top1(float* probOut = nullptr) const;

    // Convenience for 2-class head: returns logits & softmax for [start, stop]
    bool get_start_stop(float& start_logit, float& stop_logit,
                        float& p_start, float& p_stop, int& top_class) const;

    void destroy();
};

} // namespace trt
