#ifndef NV_TENSORRT_MAJOR
#define NV_TENSORRT_MAJOR (TENSORRT_VERSION / 1000)
#endif

#include "trt_gesture.hpp"
#include "ei_gesture_infer.cuh"   // Only needed by the caller (nvivafilter) for preprocessing

#include <fstream>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstring>

using namespace nvinfer1;

namespace trt {

static Logger gLogger;

static std::vector<char> read_all(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    return std::vector<char>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

bool Engine::load_from_file(const std::string& path, cudaStream_t /*video_stream*/) {
    destroy();

    auto blob = read_all(path);
    if (blob.empty()) {
        fprintf(stderr, "[trt] failed to read %s\n", path.c_str());
        return false;
    }

    runtime.reset(createInferRuntime(gLogger));
    if (!runtime) {
        fprintf(stderr, "[trt] createInferRuntime failed\n");
        return false;
    }

#if TENSORRT_VERSION >= 8000
    engine.reset(runtime->deserializeCudaEngine(blob.data(), blob.size()));
#else
    engine.reset(runtime->deserializeCudaEngine(blob.data(), blob.size(), nullptr));
#endif
    if (!engine) {
        fprintf(stderr, "[trt] deserializeCudaEngine failed\n");
        return false;
    }

    context.reset(engine->createExecutionContext());
    if (!context) {
        fprintf(stderr, "[trt] createExecutionContext failed\n");
        return false;
    }

    // Discover 1 input and 1 output binding
    const int nb = engine->getNbBindings();
    for (int i = 0; i < nb; ++i) {
        if (engine->bindingIsInput(i)) inIdx = i;
        else                           outIdx = i;
    }
    if (inIdx < 0 || outIdx < 0) {
        fprintf(stderr, "[trt] could not find input/output bindings\n");
        return false;
    }

    // Input type and tentative shape
    auto inType = engine->getBindingDataType(inIdx);
    inputIsFP16 = (inType == DataType::kHALF);

    // For explicit batch and dynamic shapes, set the input dims to 1x1x96x96 if needed.
    if (!ensure_binding_dims_96x96()) {
        fprintf(stderr, "[trt] ensure_binding_dims_96x96 failed\n");
        return false;
    }

    // Compute buffer sizes
    auto volume = [](Dims d)->size_t {
        size_t v = 1;
        for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
        return v;
    };

    auto inDims  = context->getBindingDimensions(inIdx);
    auto outDims = context->getBindingDimensions(outIdx);

    size_t inElems  = volume(inDims);
    size_t outElems = volume(outDims);

    inBytes  = inElems  * (inputIsFP16 ? sizeof(__half) : sizeof(float));
    outBytes = outElems * sizeof(float); // we read as float

    // Allocate device buffers
    if (cudaMalloc(&dIn,  inBytes)  != cudaSuccess) { fprintf(stderr, "[trt] cudaMalloc dIn failed\n");  return false; }
    if (cudaMalloc(&dOut, outBytes) != cudaSuccess) { fprintf(stderr, "[trt] cudaMalloc dOut failed\n"); return false; }

    // Allocate pinned host buffer and a std::vector view
    if (cudaMallocHost(&hostOutPinned, outBytes) != cudaSuccess) {
        fprintf(stderr, "[trt] cudaMallocHost(hostOutPinned) failed\n");
        return false;
    }
    hostOut.resize(outElems);

    // Create an event (disabled timing) to signal "done"
    cudaEventCreateWithFlags(&ev_trt_done, cudaEventDisableTiming);

    fprintf(stderr, "[trt] engine loaded: inIdx=%d(%zuB) outIdx=%d(%zuB) FP16in=%d\n",
            inIdx, inBytes, outIdx, outBytes, (int)inputIsFP16);
    return true;
}

bool Engine::ensure_binding_dims_96x96() {
#if NV_TENSORRT_MAJOR >= 7
    const bool explicitBatch = !engine->hasImplicitBatchDimension();
#else
    const bool explicitBatch = true;
#endif

    if (!explicitBatch) {
        // Implicit batch engines already have fixed dims.
        return true;
    }

    nvinfer1::Dims inDims = context->getBindingDimensions(inIdx);
    bool needSet = false;
    for (int i = 0; i < inDims.nbDims; ++i) {
        if (inDims.d[i] == -1) { needSet = true; break; }
    }
    if (!needSet) return true;

    nvinfer1::Dims wanted{};
    wanted.nbDims = inDims.nbDims;
    if (inDims.nbDims == 4) {            // N, C, H, W
        wanted.d[0] = 1;
        wanted.d[1] = 1;
        wanted.d[2] = 96;
        wanted.d[3] = 96;
    } else if (inDims.nbDims == 3) {     // C, H, W
        wanted.d[0] = 1;
        wanted.d[1] = 96;
        wanted.d[2] = 96;
    } else {
        // Generic fallback: put 1,96,96 on any -1 dims in order
        for (int i=0;i<wanted.nbDims;++i)
            wanted.d[i] = (inDims.d[i] == -1 ? (i==0?1:(i==1?96:96)) : inDims.d[i]);
    }

    if (!context->setBindingDimensions(inIdx, wanted)) {
        fprintf(stderr, "[trt] setBindingDimensions failed (nbDims=%d)\n", wanted.nbDims);
        return false;
    }
#if NV_TENSORRT_MAJOR >= 7
    if (engine->getNbOptimizationProfiles() > 0) {
        context->setOptimizationProfile(0);
    }
#endif
    return true;
}

int Engine::top1(float* probOut) const {
    if (hostOut.empty()) return -1;

    // Numerically-stable softmax
    float maxv = *std::max_element(hostOut.begin(), hostOut.end());
    double sum = 0.0;
    std::vector<float> probs(hostOut.size());
    for (size_t i = 0; i < hostOut.size(); ++i) {
        probs[i] = std::exp(hostOut[i] - maxv);
        sum += probs[i];
    }
    int arg = -1;
    float best = -1.f;
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] = (float)(probs[i] / sum);
        if (probs[i] > best) { best = probs[i]; arg = (int)i; }
    }
    if (probOut) *probOut = best;
    return arg;
}

bool Engine::get_start_stop_ok(float& start_logit, float& stop_logit, float& ok_logit,
                               float& p_start, float& p_stop, float& p_ok, int& top_class) const
{
    if (hostOut.size() < 3) return false;

    start_logit = hostOut[0];
    stop_logit  = hostOut[1];
    ok_logit    = hostOut[2];

    // 3-class softmax
    float m = std::max(start_logit, std::max(stop_logit, ok_logit));
    float e0 = std::exp(start_logit - m);
    float e1 = std::exp(stop_logit  - m);
    float e2 = std::exp(ok_logit    - m);
    float den = e0 + e1 + e2;

    p_start = e0 / den;
    p_stop  = e1 / den;
    p_ok    = e2 / den;

    // top-class by probability
    if (p_start >= p_stop && p_start >= p_ok)      top_class = 0;
    else if (p_stop >= p_start && p_stop >= p_ok)  top_class = 1;
    else                                           top_class = 2;

    return true;
}

bool Engine::try_commit_host_output() {
    if (!ev_trt_done || !hostOutPinned || hostOut.empty()) return false;
    if (cudaEventQuery(ev_trt_done) != cudaSuccess) return false;

    std::memcpy(hostOut.data(), hostOutPinned, outBytes);
    return true;
}

void Engine::destroy() {
    if (dIn)  { cudaFree(dIn);  dIn  = nullptr; }
    if (dOut) { cudaFree(dOut); dOut = nullptr; }
    if (hostOutPinned) { cudaFreeHost(hostOutPinned); hostOutPinned = nullptr; }
    hostOut.clear();

    if (ev_trt_done) { cudaEventDestroy(ev_trt_done); ev_trt_done = nullptr; }

    context.reset();
    engine.reset();
    runtime.reset();
}

} // namespace trt
