#ifndef NV_TENSORRT_MAJOR
#define NV_TENSORRT_MAJOR (TENSORRT_VERSION / 1000)
#endif

#include "trt_gesture.hpp"

#include <fstream>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdlib>   // getenv, atoi
#include <cstdio>

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

    // Bindings
    const int nb = engine->getNbBindings();
    for (int i = 0; i < nb; ++i) {
        if (engine->bindingIsInput(i)) inIdx = i;
        else                           outIdx = i;
    }
    if (inIdx < 0 || outIdx < 0) {
        fprintf(stderr, "[trt] could not find input/output bindings\n");
        return false;
    }

    // Types
    auto inType  = engine->getBindingDataType(inIdx);
    auto outType = engine->getBindingDataType(outIdx);
    inputIsFP16  = (inType  == DataType::kHALF);
    outputIsFP16 = (outType == DataType::kHALF);
    fprintf(stderr, "[trt] dtypes: in=%d out=%d  (0=float32, 1=float16)\n", (int)inType, (int)outType);

    if (!ensure_binding_dims_96x96()) {
        fprintf(stderr, "[trt] ensure_binding_dims_96x96 failed\n");
        return false;
    }

    // Sizes
    auto volume = [](Dims d)->size_t {
        size_t v = 1;
        for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
        return v;
    };
    auto inDims  = context->getBindingDimensions(inIdx);
    auto outDims = context->getBindingDimensions(outIdx);

    const size_t inElems = volume(inDims);
    outElems             = volume(outDims);

    if (outElems < 2) {
        fprintf(stderr, "[trt] expected 2 outputs (START/STOP), got %zu\n", outElems);
        return false;
    }

    inBytes     = inElems  * (inputIsFP16  ? sizeof(__half) : sizeof(float));
    outBytesDev = outElems * (outputIsFP16 ? sizeof(__half) : sizeof(float));

    // Device buffers
    if (cudaMalloc(&dIn,  inBytes)     != cudaSuccess) { fprintf(stderr, "[trt] cudaMalloc dIn failed\n");  return false; }
    if (cudaMalloc(&dOut, outBytesDev) != cudaSuccess) { fprintf(stderr, "[trt] cudaMalloc dOut failed\n"); return false; }

    // Host buffers
    if (cudaMallocHost(&hostOutPinnedRaw, outBytesDev) != cudaSuccess) {
        fprintf(stderr, "[trt] cudaMallocHost(hostOutPinnedRaw) failed\n"); return false;
    }
    hostOut.resize(outElems, 0.0f);

    // Event
    cudaEventCreateWithFlags(&ev_trt_done, cudaEventDisableTiming);

    // Label map (default 2-class: start=0, stop=1; allow env override)
    set_label_map(/*start*/0, /*stop*/1);
    load_label_map_from_env();
    fprintf(stderr, "[trt] label map: start=%d stop=%d\n", idx_start, idx_stop);

    fprintf(stderr, "[trt] engine loaded: inIdx=%d(%zuB) outIdx=%d(outElems=%zu, devBytes=%zu) FP16in=%d FP16out=%d\n",
            inIdx, inBytes, outIdx, outElems, outBytesDev, (int)inputIsFP16, (int)outputIsFP16);
    return true;
}

bool Engine::ensure_binding_dims_96x96() {
#if NV_TENSORRT_MAJOR >= 7
    const bool explicitBatch = !engine->hasImplicitBatchDimension();
#else
    const bool explicitBatch = true;
#endif
    if (!explicitBatch) return true;

    Dims inDims = context->getBindingDimensions(inIdx);
    bool needSet = false;
    for (int i = 0; i < inDims.nbDims; ++i) {
        if (inDims.d[i] == -1) { needSet = true; break; }
    }
    if (!needSet) return true;

    Dims wanted{}; wanted.nbDims = inDims.nbDims;
    if (inDims.nbDims == 4) { wanted.d[0]=1; wanted.d[1]=1; wanted.d[2]=96; wanted.d[3]=96; }
    else if (inDims.nbDims == 3) { wanted.d[0]=1; wanted.d[1]=96; wanted.d[2]=96; }
    else {
        for (int i=0;i<wanted.nbDims;++i)
            wanted.d[i] = (inDims.d[i] == -1 ? (i==0?1:(i==1?96:96)) : inDims.d[i]);
    }
    if (!context->setBindingDimensions(inIdx, wanted)) {
        fprintf(stderr, "[trt] setBindingDimensions failed (nbDims=%d)\n", wanted.nbDims);
        return false;
    }
#if NV_TENSORRT_MAJOR >= 7
    if (engine->getNbOptimizationProfiles() > 0) context->setOptimizationProfile(0);
#endif
    return true;
}

bool Engine::try_commit_host_output() {
    if (!ev_trt_done || !hostOutPinnedRaw || hostOut.empty()) return false;
    if (cudaEventQuery(ev_trt_done) != cudaSuccess) return false;

    if (outputIsFP16) {
        const __half* src = reinterpret_cast<const __half*>(hostOutPinnedRaw);
        for (size_t i=0; i<outElems; ++i) hostOut[i] = __half2float(src[i]);
    } else {
        std::memcpy(hostOut.data(), hostOutPinnedRaw, outElems * sizeof(float));
    }
    return true;
}

// Heuristic: looks like probabilities if all in [0,1] and sum ~ 1
static inline bool is_prob_like(const float* v, size_t n) {
    if (n == 0) return false;
    double sum = 0.0;
    for (size_t i=0;i<n;++i) {
        if (v[i] < -0.01f || v[i] > 1.01f) return false;
        sum += v[i];
    }
    return (sum > 0.95 && sum < 1.05);
}

int Engine::top1(float* probOut) const {
    if (hostOut.empty()) return -1;

    // If it looks like probabilities, just argmax.
    if (is_prob_like(hostOut.data(), hostOut.size())) {
        int arg = 0; float best = hostOut[0];
        for (size_t i=1;i<hostOut.size();++i) {
            if (hostOut[i] > best) { best = hostOut[i]; arg = (int)i; }
        }
        if (probOut) *probOut = best;
        return arg;
    }

    // Otherwise logits → softmax once.
    float maxv = *std::max_element(hostOut.begin(), hostOut.end());
    double den = 0.0;
    std::vector<float> probs(hostOut.size());
    for (size_t i=0;i<hostOut.size();++i) {
        probs[i] = std::exp(hostOut[i]-maxv);
        den += probs[i];
    }
    int arg = 0; float best = (float)(probs[0]/den);
    for (size_t i=1;i<probs.size();++i) {
        float p = (float)(probs[i]/den);
        if (p > best) { best = p; arg = (int)i; }
    }
    if (probOut) *probOut = best;
    return arg;
}

bool Engine::get_start_stop(float& start_score, float& stop_score,
                            float& p_start, float& p_stop, int& top_class) const
{
    const int n = (int)hostOut.size();
    if (n <= idx_start || n <= idx_stop) return false;

    // Engine output in mapped order
    start_score = hostOut[idx_start];
    stop_score  = hostOut[idx_stop];

    if (is_prob_like(hostOut.data(), hostOut.size())) {
        // Already probabilities.
        p_start = start_score;
        p_stop  = stop_score;
    } else {
        // Softmax across the two mapped indices only.
        const float m  = std::max(start_score, stop_score);
        const float eS = std::exp(start_score - m);
        const float eT = std::exp(stop_score  - m);
        const float den = eS + eT;
        p_start = eS / den;
        p_stop  = eT / den;
    }

    top_class = (p_start >= p_stop) ? 0 : 1; // 0=START, 1=STOP
    return true;
}

void Engine::set_label_map(int start_idx, int stop_idx) {
    idx_start = start_idx; idx_stop = stop_idx;
}

static bool parse_kv_index(const char* env, const char* key, int& out) {
    const char* p = std::strstr(env, key);
    if (!p) return false;
    p += std::strlen(key);
    if (*p != '=') return false;
    ++p;
    out = std::atoi(p);
    return true;
}

bool Engine::load_label_map_from_env() {
    const char* env = std::getenv("TRT_LABEL_MAP"); // e.g. "start=0,stop=1"
    if (!env) return false;
    int s = idx_start, t = idx_stop;
    parse_kv_index(env, "start", s);
    parse_kv_index(env, "stop",  t);
    idx_start = s; idx_stop = t;
    fprintf(stderr, "[trt] label map (env): start=%d stop=%d\n", idx_start, idx_stop);
    return true;
}

void Engine::destroy() {
    if (dIn)  { cudaFree(dIn);  dIn  = nullptr; }
    if (dOut) { cudaFree(dOut); dOut = nullptr; }
    if (hostOutPinnedRaw) { cudaFreeHost(hostOutPinnedRaw); hostOutPinnedRaw=nullptr; }
    hostOut.clear();
    if (ev_trt_done) { cudaEventDestroy(ev_trt_done); ev_trt_done = nullptr; }
    context.reset(); engine.reset(); runtime.reset();
}

} // namespace trt
