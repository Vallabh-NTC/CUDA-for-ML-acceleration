#include "trt_gesture.hpp"

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace nvinfer1;

namespace {

// Minimal TensorRT logger
class TrtLogger : public ILogger {
public:
    Severity minSeverity = Severity::kWARNING;

    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= minSeverity) {
            const char* s = nullptr;
            switch (severity) {
                case Severity::kINTERNAL_ERROR: s = "INTERNAL_ERROR"; break;
                case Severity::kERROR:          s = "ERROR";          break;
                case Severity::kWARNING:        s = "WARNING";        break;
                case Severity::kINFO:           s = "INFO";           break;
                case Severity::kVERBOSE:        s = "VERBOSE";        break;
                default:                        s = "UNKNOWN";        break;
            }
            std::cerr << "[TRT][" << s << "] " << msg << std::endl;
        }
    }
};

TrtLogger gLogger;

// Compute the total number of elements in a TensorRT Dims
static size_t volume(const Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        v *= static_cast<size_t>(d.d[i]);
    }
    return v;
}

// Read integer from environment with default value fallback
static int env_int(const char* name, int defVal) {
    const char* v = std::getenv(name);
    if (!v || !*v) return defVal;
    try {
        return std::stoi(v);
    } catch (...) {
        return defVal;
    }
}

// Interpret a Dims as CHW (or NCHW) and extract C, H, W.
// This is a heuristic that works for common layouts.
static void dims_to_CHW(const Dims& d, int& C, int& H, int& W) {
    C = H = W = 0;

    if (d.nbDims == 4) {
        // Typical NCHW: [N, C, H, W]
        C = d.d[1];
        H = d.d[2];
        W = d.d[3];
    } else if (d.nbDims == 3) {
        // Typical CHW: [C, H, W]
        C = d.d[0];
        H = d.d[1];
        W = d.d[2];
    } else {
        // Fallback: try to interpret as 1D
        if (d.nbDims == 1) {
            C = d.d[0];
            H = 1;
            W = 1;
        }
    }
}

} // anonymous namespace

namespace trt {

bool Engine::load_from_file(const char* path, cudaStream_t /*videoStreamForDebug*/) {
    if (!path || !*path) {
        std::cerr << "[trt_gesture] empty engine path\n";
        return false;
    }

    // Avoid double load if already initialized
    if (engine) {
        std::cerr << "[trt_gesture] engine already loaded, skipping\n";
        return true;
    }

    // Read serialized engine from file
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "[trt_gesture] failed to open engine: " << path << "\n";
        return false;
    }
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!f.read(buffer.data(), size)) {
        std::cerr << "[trt_gesture] failed to read engine: " << path << "\n";
        return false;
    }

    // Create runtime and deserialize engine
    runtime = createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "[trt_gesture] createInferRuntime failed\n";
        return false;
    }

    engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size());
    if (!engine) {
        std::cerr << "[trt_gesture] deserializeCudaEngine failed\n";
        return false;
    }

    context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "[trt_gesture] createExecutionContext failed\n";
        return false;
    }

    // Number of total bindings (inputs + outputs)
    nbBindings = engine->getNbBindings();
    if (nbBindings < 2) {
        std::cerr << "[trt_gesture] nbBindings=" << nbBindings
                  << " (expected at least 2: 1 input + 1 output)\n";
        return false;
    }

    devBindings.clear();
    devBindings.resize(nbBindings, nullptr);

    inIdx       = -1;
    outIdx      = -1;
    inputIsFP16 = false;
    outputIsFP16 = false;
    outElems     = 0;
    dOut         = nullptr;

    outDims = Dims{};
    outC = outH = outW = 0;

    // Iterate over all bindings, find input and outputs and allocate device buffers
    for (int i = 0; i < nbBindings; ++i) {
        bool isInput = engine->bindingIsInput(i);
        DataType dt  = engine->getBindingDataType(i);
        Dims dims    = engine->getBindingDimensions(i);
        size_t elems = volume(dims);

        if (isInput) {
            // First input is considered the primary one
            if (inIdx < 0) {
                inIdx = i;
                inputIsFP16 = (dt == DataType::kHALF);
            } else {
                // Engine with multiple inputs: log a warning and ignore the rest.
                std::cerr << "[trt_gesture] WARNING: multiple input bindings, "
                             "only the first will be used (inIdx="
                          << inIdx << ")\n";
            }
            continue; // Input memory is provided externally (no alloc here).
        }

        // Output binding → allocate device buffer
        size_t elemBytes = (dt == DataType::kHALF) ? sizeof(__half) : sizeof(float);
        size_t bytes     = elems * elemBytes;

        void* devPtr = nullptr;
        if (cudaMalloc(&devPtr, bytes) != cudaSuccess) {
            std::cerr << "[trt_gesture] cudaMalloc(devBindings[" << i
                      << "], bytes=" << bytes << ") failed\n";
            return false;
        }
        devBindings[i] = devPtr;

        // The first output is considered the "primary" one (e.g. heatmaps)
        if (outIdx < 0) {
            outIdx       = i;
            outputIsFP16 = (dt == DataType::kHALF);
            outElems     = elems;
            dOut         = devPtr;

            outDims = dims;
            dims_to_CHW(outDims, outC, outH, outW);

            if (outElems == 0 || outC <= 0 || outH <= 0 || outW <= 0) {
                std::cerr << "[trt_gesture] primary output tensor has invalid shape "
                          << "(elems=" << outElems
                          << " C=" << outC
                          << " H=" << outH
                          << " W=" << outW
                          << ")\n";
                return false;
            }

            size_t outBytes = outElems * (outputIsFP16 ? sizeof(__half) : sizeof(float));

            if (cudaMallocHost(&hostOutPinnedRaw, outBytes) != cudaSuccess) {
                std::cerr << "[trt_gesture] cudaMallocHost(hostOutPinnedRaw) failed\n";
                return false;
            }

            hostOut.resize(outElems, 0.0f);

            if (cudaEventCreateWithFlags(&ev_trt_done, cudaEventDisableTiming) != cudaSuccess) {
                std::cerr << "[trt_gesture] cudaEventCreate(ev_trt_done) failed\n";
                return false;
            }
        }
    }

    if (inIdx < 0 || outIdx < 0) {
        std::cerr << "[trt_gesture] failed to find valid input/output bindings (inIdx="
                  << inIdx << " outIdx=" << outIdx << ")\n";
        return false;
    }

    // Legacy indices for "start/stop" classifier mode (still available if needed).
    idx_start = env_int("GESTURE_IDX_START", 0);
    idx_stop  = env_int("GESTURE_IDX_STOP", 1);

    std::cerr << "[trt_gesture] loaded engine from " << path
              << " | nbBindings=" << nbBindings
              << " inIdx=" << inIdx
              << " outIdx=" << outIdx
              << " outElems=" << outElems
              << " inFP16=" << (inputIsFP16 ? 1 : 0)
              << " outFP16=" << (outputIsFP16 ? 1 : 0)
              << " outC=" << outC
              << " outH=" << outH
              << " outW=" << outW
              << " idx_start=" << idx_start
              << " idx_stop="  << idx_stop
              << std::endl;

    hasPending   = false;
    hasCommitted = false;
    return true;
}

void Engine::destroy() {
    // Free all device buffers for outputs
    for (void* p : devBindings) {
        if (p) cudaFree(p);
    }
    devBindings.clear();

    dOut = nullptr;

    if (hostOutPinnedRaw) {
        cudaFreeHost(hostOutPinnedRaw);
        hostOutPinnedRaw = nullptr;
    }
    if (ev_trt_done) {
        cudaEventDestroy(ev_trt_done);
        ev_trt_done = nullptr;
    }

    if (context) {
        context->destroy();
        context = nullptr;
    }
    if (engine) {
        engine->destroy();
        engine = nullptr;
    }
    if (runtime) {
        runtime->destroy();
        runtime = nullptr;
    }

    hostOut.clear();
    outElems     = 0;
    inIdx        = -1;
    outIdx       = -1;
    nbBindings   = 0;
    inputIsFP16  = false;
    outputIsFP16 = false;
    hasPending   = false;
    hasCommitted = false;

    outDims = Dims{};
    outC = outH = outW = 0;
}

// This is called from gpu_process() after cudaMemcpyAsync(dOut -> hostOutPinnedRaw)
// is enqueued and ev_trt_done is recorded on the TRT stream.
// Here we check whether the event has completed and, if so, copy/convert from
// hostOutPinnedRaw into hostOut (float).
bool Engine::try_commit_host_output() {
    if (!ev_trt_done || !hostOutPinnedRaw || outElems == 0) {
        return false;
    }

    cudaError_t q = cudaEventQuery(ev_trt_done);
    if (q == cudaErrorNotReady) {
        return false;
    }
    if (q != cudaSuccess) {
        static bool s_logged_once = false;
        if (!s_logged_once) {
            std::cerr << "[trt_gesture] cudaEventQuery(ev_trt_done) error: "
                      << (int)q << " (" << cudaGetErrorString(q) << ")\n";
            s_logged_once = true;
        }
        return false;
    }

    std::lock_guard<std::mutex> lk(mtx);

    // Always convert to float into hostOut
    if (outputIsFP16) {
        const __half* src = reinterpret_cast<const __half*>(hostOutPinnedRaw);
        hostOut.resize(outElems);
        for (size_t i = 0; i < outElems; ++i) {
            hostOut[i] = __half2float(src[i]);
        }
    } else {
        const float* src = reinterpret_cast<const float*>(hostOutPinnedRaw);
        hostOut.assign(src, src + outElems);
    }

    hasPending   = false;
    hasCommitted = true;
    return true;
}

// ----------------------------------------------------------------------
// Legacy classifier-style helpers (still available if needed).
// ----------------------------------------------------------------------

bool Engine::get_start_stop(float& sLogit,
                            float& tLogit,
                            float& pStart,
                            float& pStop,
                            int&   top) {
    std::lock_guard<std::mutex> lk(mtx);

    if (!hasCommitted || hostOut.empty() || outElems < 2) {
        return false;
    }

    if (idx_start < 0 || idx_start >= static_cast<int>(outElems) ||
        idx_stop  < 0 || idx_stop  >= static_cast<int>(outElems)) {
        return false;
    }

    // Raw logits
    sLogit = hostOut[idx_start];
    tLogit = hostOut[idx_stop];

    // Softmax over the entire vector
    float maxv = hostOut[0];
    for (size_t i = 1; i < outElems; ++i) {
        if (hostOut[i] > maxv) maxv = hostOut[i];
    }

    float sum = 0.0f;
    int   bestIdx = 0;
    float bestVal = -1e30f;

    for (size_t i = 0; i < outElems; ++i) {
        float e = std::exp(hostOut[i] - maxv);
        sum += e;
        if (e > bestVal) {
            bestVal = e;
            bestIdx = static_cast<int>(i);
        }
    }

    if (sum <= 0.f) {
        pStart = 0.f;
        pStop  = 0.f;
        top    = -1;
        return true;
    }

    pStart = std::exp(hostOut[idx_start] - maxv) / sum;
    pStop  = std::exp(hostOut[idx_stop]  - maxv) / sum;
    top    = bestIdx;

    return true;
}

int Engine::top1(float* probOut) const {
    if (!hasCommitted || hostOut.empty()) {
        if (probOut) *probOut = 0.0f;
        return -1;
    }

    int   bestIdx = 0;
    float bestVal = hostOut[0];

    for (size_t i = 1; i < hostOut.size(); ++i) {
        if (hostOut[i] > bestVal) {
            bestVal = hostOut[i];
            bestIdx = static_cast<int>(i);
        }
    }

    if (probOut) {
        // Note: this is the raw logit/value, not a softmax probability.
        *probOut = bestVal;
    }
    return bestIdx;
}

// ----------------------------------------------------------------------
// New pose-oriented helpers (heatmaps -> keypoints).
// ----------------------------------------------------------------------

bool Engine::get_heatmap_shape(int& C, int& H, int& W) const {
    if (!outElems || outC <= 0 || outH <= 0 || outW <= 0) {
        C = H = W = 0;
        return false;
    }
    C = outC;
    H = outH;
    W = outW;
    return true;
}

// Decode one keypoint per channel by taking argmax over each heatmap plane.
// This is a simple and robust approach for single-hand pose when you don't
// need multi-person parsing or PAFs.
bool Engine::decode_argmax_keypoints(std::vector<Keypoint2D>& kpts) {
    std::lock_guard<std::mutex> lk(mtx);

    if (!hasCommitted || hostOut.empty() || outC <= 0 || outH <= 0 || outW <= 0) {
        return false;
    }

    const int planeSize = outH * outW;
    if (static_cast<int>(hostOut.size()) < outC * planeSize) {
        std::cerr << "[trt_gesture] decode_argmax_keypoints: hostOut size mismatch "
                  << "(hostOut.size=" << hostOut.size()
                  << " vs C*H*W=" << (outC * planeSize) << ")\n";
        return false;
    }

    kpts.resize(outC);

    for (int c = 0; c < outC; ++c) {
        float maxv = -1e30f;
        int maxx = 0, maxy = 0;

        const int base = c * planeSize;

        for (int y = 0; y < outH; ++y) {
            const int rowBase = base + y * outW;
            for (int x = 0; x < outW; ++x) {
                float v = hostOut[rowBase + x];
                if (v > maxv) {
                    maxv = v;
                    maxx = x;
                    maxy = y;
                }
            }
        }

        kpts[c].u    = static_cast<float>(maxx);
        kpts[c].v    = static_cast<float>(maxy);
        kpts[c].conf = maxv;  // raw heatmap peak value
    }

    return true;
}

} // namespace trt
