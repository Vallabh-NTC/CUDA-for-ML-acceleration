#pragma once

#include <vector>
#include <string>
#include <mutex>
#include <cstdint>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <NvInfer.h>

namespace trt {

// Simple 2D keypoint in heatmap coordinates
struct Keypoint2D {
    float u;     // x coordinate in heatmap space
    float v;     // y coordinate in heatmap space
    float conf;  // confidence value (heatmap peak)
};

struct Engine {
    // --- raw TensorRT objects ---
    nvinfer1::IRuntime*          runtime  = nullptr;
    nvinfer1::ICudaEngine*       engine   = nullptr;
    nvinfer1::IExecutionContext* context  = nullptr;

    // --- binding info ---
    int inIdx      = -1;   // index of the primary input binding
    int outIdx     = -1;   // index of the primary output binding (heatmap)
    int nbBindings = 0;    // total number of bindings in the engine

    bool inputIsFP16  = false;
    bool outputIsFP16 = false;

    // --- primary output buffer (for outIdx only) ---
    void*  dOut             = nullptr;   // device buffer for outIdx
    void*  hostOutPinnedRaw = nullptr;   // pinned host buffer (FP16 or FP32)
    size_t outElems         = 0;         // number of scalar elements in outIdx tensor

    // Device buffers for ALL output bindings (including outIdx).
    // Inputs will have nullptr entries here and are provided externally.
    std::vector<void*> devBindings;      // size = nbBindings

    // Host-side float buffer for the primary output (outIdx), always converted to float.
    // For pose models this is typically the heatmap tensor (C x H x W).
    std::vector<float> hostOut;

    // --- primary output tensor shape (for pose heatmaps) ---
    // We cache the Dims of outIdx and its interpreted C/H/W.
    nvinfer1::Dims outDims{};
    int outC = 0;   // number of channels (keypoints)
    int outH = 0;   // heatmap height
    int outW = 0;   // heatmap width

    // Legacy indices used by the old start/stop gesture classifier logic.
    // They can still be used if you interpret hostOut as class logits.
    int idx_start = 0;  // default: 0, override via env GESTURE_IDX_START
    int idx_stop  = 1;  // default: 1, override via env GESTURE_IDX_STOP

    // Event signaled when the last D2H memcpy for outIdx is queued/completed.
    cudaEvent_t ev_trt_done = nullptr;

    // Internal state / synchronization
    std::mutex mtx;
    bool hasPending   = false;   // reserved, not strictly used in current code
    bool hasCommitted = false;   // true when hostOut contains a fresh result

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    // Load a serialized TensorRT engine from file, create runtime/engine/context
    // and allocate device/host buffers for ALL output bindings.
    bool load_from_file(const char* path, cudaStream_t videoStreamForDebug);

    // Release all resources (runtime, engine, context, buffers, events).
    // Should be called from your destroy_instance() path.
    void destroy();

    // Check ev_trt_done and, if ready, copy/convert hostOutPinnedRaw -> hostOut (float).
    // Returns true if a new result is committed and ready to be consumed.
    bool try_commit_host_output();

    // ------------------------------------------------------------------
    // Legacy "classifier-style" API (start/stop from logits)
    // ------------------------------------------------------------------

    // Interpret hostOut as a logit vector over classes and compute:
    //  - sLogit / tLogit: raw logits for start/stop
    //  - pStart / pStop: softmax probabilities
    //  - top: index of the most probable class
    bool get_start_stop(float& sLogit,
                        float& tLogit,
                        float& pStart,
                        float& pStop,
                        int&   top);

    // Return top-1 class index and, optionally, its raw value in *probOut.
    // Note: "probOut" is not a softmax probability, just the raw logit.
    int top1(float* probOut = nullptr) const;

    // ------------------------------------------------------------------
    // New pose-oriented API (heatmaps -> keypoints)
    // ------------------------------------------------------------------

    // Get the shape of the primary output tensor (assumed to be C x H x W).
    // Returns false if the shape is not valid or not initialized yet.
    bool get_heatmap_shape(int& C, int& H, int& W) const;

    // Decode one keypoint per channel by taking the argmax over each heatmap.
    // The result is a vector of Keypoint2D in heatmap coordinates.
    // Returns false if there is no committed output available.
    bool decode_argmax_keypoints(std::vector<Keypoint2D>& kpts);
};

} // namespace trt
