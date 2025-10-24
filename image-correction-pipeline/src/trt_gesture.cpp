#ifndef NV_TENSORRT_MAJOR
#define NV_TENSORRT_MAJOR (TENSORRT_VERSION / 1000)
#endif
#include "trt_gesture.hpp"
#include "ei_gesture_infer.cuh"     // your GPU preprocess (NV12 Y -> 1x1x96x96)

#include <fstream>
#include <iterator>
#include <algorithm>    // <-- for std::max_element
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

bool Engine::load_from_file(const std::string& path, cudaStream_t /*stream*/) {
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

    // Discover one input and one output binding
    const int nb = engine->getNbBindings();
    for (int i = 0; i < nb; ++i) {
        if (engine->bindingIsInput(i)) inIdx = i;
        else outIdx = i;
    }
    if (inIdx < 0 || outIdx < 0) {
        fprintf(stderr, "[trt] could not find input/output bindings\n");
        return false;
    }

    // Input type and shape
    auto inType = engine->getBindingDataType(inIdx);
    inputIsFP16 = (inType == DataType::kHALF);

    auto inDims  = context->getBindingDimensions(inIdx);
    auto outDims = context->getBindingDimensions(outIdx);

    auto volume = [](Dims d)->size_t {
        size_t v = 1;
        for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
        return v;
    };

    size_t inElems  = volume(inDims);
    size_t outElems = volume(outDims);

    inBytes  = inElems  * (inputIsFP16 ? sizeof(__half) : sizeof(float));
    outBytes = outElems * sizeof(float); // read as float

    if (cudaMalloc(&dIn,  inBytes)  != cudaSuccess) { fprintf(stderr, "[trt] cudaMalloc dIn failed\n");  return false; }
    if (cudaMalloc(&dOut, outBytes) != cudaSuccess) { fprintf(stderr, "[trt] cudaMalloc dOut failed\n"); return false; }

    hostOut.resize(outElems);

    fprintf(stderr, "[trt] engine loaded: inIdx=%d(%zuB) outIdx=%d(%zuB) FP16in=%d\n",
            inIdx, inBytes, outIdx, outBytes, (int)inputIsFP16);
    return true;
}

bool Engine::infer_from_nv12_y(const uint8_t* dY, int W, int H, int pitch, cudaStream_t stream) {
    if (!context || !dIn || !dOut) return false;

#if NV_TENSORRT_MAJOR >= 7
    const bool explicitBatch = !engine->hasImplicitBatchDimension();
#else
    const bool explicitBatch = true;
#endif

    if (explicitBatch) {
        nvinfer1::Dims inDims = context->getBindingDimensions(inIdx);
        bool needSet = false;
        for (int i = 0; i < inDims.nbDims; ++i) {
            if (inDims.d[i] == -1) { needSet = true; break; }
        }
        if (needSet) {
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
                for (int i=0;i<wanted.nbDims;++i)
                    wanted.d[i] = (inDims.d[i] == -1 ? (i==0?1:(i==1?96:96)) : inDims.d[i]);
            }
            if (!context->setBindingDimensions(inIdx, wanted)) {
                fprintf(stderr, "[trt] setBindingDimensions failed (nbDims=%d)\n", wanted.nbDims);
                return false;
            }
        }
#if NV_TENSORRT_MAJOR >= 7
        if (engine->getNbOptimizationProfiles() > 0) {
            context->setOptimizationProfile(0);
        }
#endif
    }

    // Pass tv_range=true for NV12 (16..235)
    const bool tv_range = true;
    if (!ei::enqueue_preprocess_to_trt_input(dY, W, H, pitch, dIn, inputIsFP16, tv_range, stream)) {
        fprintf(stderr, "[trt] preprocess enqueue failed\n");
        return false;
    }
    // Optional debug (uncomment if input is FP32):
    // if (!inputIsFP16) ei::launch_debug_stats_f32(reinterpret_cast<const float*>(dIn), 96*96, stream);

    void* bindings[2];
    bindings[inIdx]  = dIn;
    bindings[outIdx] = dOut;

    if (!context->enqueueV2(bindings, stream, nullptr)) {
        fprintf(stderr, "[trt] enqueueV2 failed\n");
        return false;
    }

    if (cudaMemcpyAsync(hostOut.data(), dOut, outBytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        fprintf(stderr, "[trt] D2H failed\n"); return false;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
    return true;
}


int Engine::top1(float* probOut) const {
    if (hostOut.empty()) return -1;

    // Softmax (numerically stable) for a generic-length vector
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

bool Engine::get_start_stop(float& start_logit, float& stop_logit,
                            float& p_start, float& p_stop, int& top_class) const
{
    if (hostOut.size() < 2) return false;

    start_logit = hostOut[0];
    stop_logit  = hostOut[1];

    // Softmax over 2 logits
    const float m  = (start_logit > stop_logit) ? start_logit : stop_logit;
    const float e0 = std::exp(start_logit - m);
    const float e1 = std::exp(stop_logit  - m);
    const float den = e0 + e1;
    p_start = e0 / den;
    p_stop  = e1 / den;

    top_class = (p_stop > p_start) ? 1 : 0;
    return true;
}

void Engine::destroy() {
    if (dIn)  { cudaFree(dIn);  dIn  = nullptr; }
    if (dOut) { cudaFree(dOut); dOut = nullptr; }
    hostOut.clear();
    context.reset();
    engine.reset();
    runtime.reset();
}

} // namespace trt
