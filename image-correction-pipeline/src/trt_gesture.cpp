#include "trt_gesture.hpp"

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace nvinfer1;

namespace {

// Logger minimo per TensorRT
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

// helper: volume di un Dims
static size_t volume(const Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        v *= static_cast<size_t>(d.d[i]);
    }
    return v;
}

// helper: leggi intero da env con default
static int env_int(const char* name, int defVal) {
    const char* v = std::getenv(name);
    if (!v || !*v) return defVal;
    try {
        return std::stoi(v);
    } catch (...) {
        return defVal;
    }
}

} // anonymous namespace

namespace trt {

bool Engine::load_from_file(const char* path, cudaStream_t /*videoStreamForDebug*/) {
    if (!path || !*path) {
        std::cerr << "[trt_gesture] empty engine path\n";
        return false;
    }

    // evita doppio load
    if (engine) {
        std::cerr << "[trt_gesture] engine already loaded, skipping\n";
        return true;
    }

    // leggi file in memoria
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

    // crea runtime + engine
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

    // numero totale di binding
    nbBindings = engine->getNbBindings();
    if (nbBindings < 2) {
        std::cerr << "[trt_gesture] nbBindings=" << nbBindings
                  << " (expected at least 2: 1 input + 1 output)\n";
        return false;
    }

    devBindings.clear();
    devBindings.resize(nbBindings, nullptr);

    inIdx      = -1;
    outIdx     = -1;
    inputIsFP16  = false;
    outputIsFP16 = false;
    outElems     = 0;
    dOut         = nullptr;

    // loop su tutti i binding: troviamo input/output e allochiamo buffer per TUTTI gli output
    for (int i = 0; i < nbBindings; ++i) {
        bool isInput = engine->bindingIsInput(i);
        DataType dt  = engine->getBindingDataType(i);
        Dims dims    = engine->getBindingDimensions(i);
        size_t elems = volume(dims);

        if (isInput) {
            if (inIdx < 0) {
                inIdx = i;
                inputIsFP16 = (dt == DataType::kHALF);
            } else {
                // modello con più input: per ora non supportiamo, ma logghiamo
                std::cerr << "[trt_gesture] WARNING: multiple input bindings, only the first will be used (idx="
                          << inIdx << ")\n";
            }
            continue; // niente alloc: l'input viene da fuori (slot.dTensor)
        }

        // è un OUTPUT binding → alloc device buffer
        size_t elemBytes = (dt == DataType::kHALF) ? sizeof(__half) : sizeof(float);
        size_t bytes     = elems * elemBytes;

        void* devPtr = nullptr;
        if (cudaMalloc(&devPtr, bytes) != cudaSuccess) {
            std::cerr << "[trt_gesture] cudaMalloc(devBindings[" << i
                      << "], bytes=" << bytes << ") failed\n";
            return false;
        }
        devBindings[i] = devPtr;

        // il PRIMO output lo consideriamo "principale" (da copiare a hostOut)
        if (outIdx < 0) {
            outIdx       = i;
            outputIsFP16 = (dt == DataType::kHALF);
            outElems     = elems;
            dOut         = devPtr;

            if (outElems == 0) {
                std::cerr << "[trt_gesture] primary output tensor has zero elements\n";
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

    // indici classi start/stop da env (default 0,1)
    idx_start = env_int("GESTURE_IDX_START", 0);
    idx_stop  = env_int("GESTURE_IDX_STOP", 1);

    std::cerr << "[trt_gesture] loaded engine from " << path
              << " | nbBindings=" << nbBindings
              << " inIdx=" << inIdx
              << " outIdx=" << outIdx
              << " outElems=" << outElems
              << " inFP16=" << (inputIsFP16 ? 1 : 0)
              << " outFP16=" << (outputIsFP16 ? 1 : 0)
              << " idx_start=" << idx_start
              << " idx_stop="  << idx_stop
              << std::endl;

    hasPending   = false;
    hasCommitted = false;
    return true;
}

void Engine::destroy() {
    // libera tutti i buffer device degli output
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
}

// viene chiamata dal tuo gpu_process dopo che il cudaMemcpyAsync è stato
// lanciato e l'evento ev_trt_done registrato sul trt_stream.
// Qui controlliamo se l'evento è completato, e se sì, copiamo/convertiamo
// hostOutPinnedRaw -> hostOut (float).
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

    // converte sempre in float (hostOut)
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

    // logit grezzi
    sLogit = hostOut[idx_start];
    tLogit = hostOut[idx_stop];

    // softmax su tutto il vettore
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
        *probOut = bestVal;  // se lo vuoi come "probabilità" puoi poi applicare softmax fuori
    }
    return bestIdx;
}

} // namespace trt
