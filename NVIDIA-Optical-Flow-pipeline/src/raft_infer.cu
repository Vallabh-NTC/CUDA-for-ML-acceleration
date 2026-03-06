// raft_infer.cu — TensorRT 8.6, float32 I/O
#include "raft_infer.hpp"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <fstream>
#include <vector>
#include <string>

namespace {
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity sev, const char *msg) noexcept override {
        if (sev <= Severity::kWARNING)
            std::fprintf(stderr, "[TRT] %s\n", msg);
    }
};
static TRTLogger g_logger;

static std::vector<char> load_file(const std::string &p) {
    std::ifstream f(p, std::ios::binary|std::ios::ate);
    if (!f.is_open()) return {};
    auto sz = f.tellg(); f.seekg(0);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    return buf;
}
} // namespace

bool RaftInfer::init(const std::string &engine_path, int W, int H)
{
    m_W = W; m_H = H;

    auto data = load_file(engine_path);
    if (data.empty()) {
        std::fprintf(stderr, "[raft_infer] Cannot load: %s\n", engine_path.c_str());
        return false;
    }

    initLibNvInferPlugins(&g_logger, "");

    auto *runtime = nvinfer1::createInferRuntime(g_logger);
    if (!runtime) { std::fprintf(stderr, "[raft_infer] createInferRuntime failed\n"); return false; }
    m_runtime = runtime;

    auto *engine = runtime->deserializeCudaEngine(data.data(), data.size());
    if (!engine) { std::fprintf(stderr, "[raft_infer] deserializeCudaEngine failed\n"); return false; }
    m_engine = engine;

    const int nb = engine->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char *name = engine->getIOTensorName(i);
        auto dims  = engine->getTensorShape(name);
        auto dtype = engine->getTensorDataType(name);
        auto mode  = engine->getTensorIOMode(name);
        std::fprintf(stderr, "[raft_infer] [%d] %-8s dtype=%d shape=[", i, name, (int)dtype);
        for (int d = 0; d < dims.nbDims; ++d)
            std::fprintf(stderr, "%d%s", (int)dims.d[d], d<dims.nbDims-1?",":"");
        std::fprintf(stderr, "] %s\n",
            mode==nvinfer1::TensorIOMode::kINPUT?"INPUT":"OUTPUT");
    }

    auto *ctx = engine->createExecutionContext();
    if (!ctx) { std::fprintf(stderr, "[raft_infer] createExecutionContext failed\n"); return false; }
    m_context = ctx;

    m_ready = true;
    std::fprintf(stderr, "[raft_infer] Ready — float32 I/O [1,3,%d,%d]\n", H, W);
    return true;
}

bool RaftInfer::infer(const float *d_frame_prev,
                      const float *d_frame_curr,
                      float       *d_flow_out,
                      cudaStream_t stream)
{
    if (!m_ready) return false;
    auto *ctx    = reinterpret_cast<nvinfer1::IExecutionContext*>(m_context);
    auto *engine = reinterpret_cast<nvinfer1::ICudaEngine*>(m_engine);

    const int nb = engine->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char *name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        void *ptr = nullptr;
        if (mode == nvinfer1::TensorIOMode::kINPUT)
            ptr = (std::string(name) == "frame1")
                ? const_cast<float*>(d_frame_prev)
                : const_cast<float*>(d_frame_curr);
        else
            ptr = d_flow_out;

        if (!ctx->setTensorAddress(name, ptr)) {
            std::fprintf(stderr, "[raft_infer] setTensorAddress failed: %s\n", name);
            return false;
        }
    }

    if (!ctx->enqueueV3(stream)) {
        std::fprintf(stderr, "[raft_infer] enqueueV3 failed\n");
        return false;
    }
    return true;
}

void RaftInfer::destroy()
{
    if (m_context) { delete reinterpret_cast<nvinfer1::IExecutionContext*>(m_context); m_context=nullptr; }
    if (m_engine)  { delete reinterpret_cast<nvinfer1::ICudaEngine*>(m_engine);        m_engine=nullptr;  }
    if (m_runtime) { delete reinterpret_cast<nvinfer1::IRuntime*>(m_runtime);          m_runtime=nullptr; }
    m_ready = false;
}