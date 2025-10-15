/**
 * nvivafilter_imagecorrection.cpp  (gesture TRT add-on, JSON-config, no OpenCV)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <deque>
#include <cmath>
#include <dlfcn.h>
#include <cctype>
#include <atomic>
#include <thread>
#include <fstream>
#include <sstream>
#include <regex>

#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "nvivafilter_customer_api.hpp"
#include "rectify_config.hpp"
#include "kernel_rectify.cuh"
#include "color_ops.cuh"
#include "runtime_controls.hpp"
#include "wire_lineremoval.cuh"

// TRT + preprocess
#include "NvInfer.h"
#include "ei_gesture_infer.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
constexpr float M_PI_F = static_cast<float>(M_PI);

// ===== CUDA primary context =====
namespace {
    static std::once_flag g_ctx_once;
    static CUcontext g_primary_ctx = nullptr;
    static CUdevice  g_device      = 0;
    static void retain_primary_context_once() {
        cuInit(0); cuDeviceGet(&g_device, 0);
        CUresult r = cuDevicePrimaryCtxRetain(&g_primary_ctx, g_device);
        if (r != CUDA_SUCCESS) { fprintf(stderr,"[ic] primary ctx retain failed\n"); g_primary_ctx=nullptr; }
        else { cuCtxSetCurrent(g_primary_ctx); fprintf(stderr,"[ic] Using shared primary CUDA context\n"); }
    }
    static void release_primary_context() {
        if (g_primary_ctx) { cuDevicePrimaryCtxRelease(g_device); g_primary_ctx=nullptr; }
    }
}

// ===== Misc helpers =====
static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return (char)std::tolower(c); });
    return s;
}
static std::string section_from_loaded_name()
{
    Dl_info info{};
    if (dladdr((void*)&init, &info) && info.dli_fname) {
        std::string path(info.dli_fname);
        std::string low  = to_lower(path);
        std::string base = path.substr(path.find_last_of('/') + 1);
        std::string base_low = to_lower(base);
        if (base_low.find("_cam0") != std::string::npos) return "cam0";
        if (base_low.find("_cam1") != std::string::npos) return "cam1";
        if (base_low.find("_cam2") != std::string::npos) return "cam2";
        if (low.find("/cam0/") != std::string::npos) return "cam0";
        if (low.find("/cam1/") != std::string::npos) return "cam1";
        if (low.find("/cam2/") != std::string::npos) return "cam2";
    }
    fprintf(stderr, "[ic] WARNING: cannot detect cam section; defaulting to cam0\n");
    return "cam0";
}
static int section_to_index(const std::string& sec) {
    if (sec == "cam0") return 0;
    if (sec == "cam1") return 1;
    if (sec == "cam2") return 2;
    return -1;
}
static std::string mask_raw_path_for(int cam_idx)  { return "/dev/shm/wire_mask_cam" + std::to_string(cam_idx) + ".raw"; }
static std::string mask_meta_path_for(int cam_idx) { return "/dev/shm/wire_mask_cam" + std::to_string(cam_idx) + ".meta"; }

// ===== Tiny JSON helpers (regex-based, tolerant) =====
static bool read_text(const std::string& path, std::string& out) {
    std::ifstream f(path); if (!f) return false; std::ostringstream ss; ss << f.rdbuf(); out = ss.str(); return true;
}
static bool json_extract_object(const std::string& s, const std::string& key, std::string& outObj) {
    std::regex rk("\"" + key + "\"\\s*:\\s*\\{");
    std::smatch m;
    if (!std::regex_search(s, m, rk)) return false;
    size_t start = m.position() + m.length() - 1; // at '{'
    int depth = 0;
    for (size_t i = start; i < s.size(); ++i) {
        if (s[i] == '{') { if (depth++ == 0) start = i; }
        else if (s[i] == '}') { if (--depth == 0) { outObj = s.substr(start, i - start + 1); return true; } }
    }
    return false;
}
template<typename T>
static bool json_get_number(const std::string& s, const char* key, T& out) {
    std::regex re(std::string("\"") + key + R"xx("\s*:\s*([-+]?\d+(\.\d+)?([eE][-+]?\d+)?))xx");
    std::smatch m; if (std::regex_search(s, m, re)) { out = static_cast<T>(std::stod(m[1])); return true; }
    return false;
}
static bool json_get_bool(const std::string& s, const char* key, bool& out) {
    std::regex re(std::string("\"") + key + R"xx("\s*:\s*(true|false))xx");
    std::smatch m; if (std::regex_search(s, m, re)) { out = (m[1] == "true"); return true; }
    return false;
}
static std::string json_get_string(const std::string& s, const char* key, const std::string& def = "") {
    // custom raw-string delimiter to avoid early termination
    std::regex re(std::string("\"") + key + R"xx("\s*:\s*"(.*?)")xx");
    std::smatch m;
    if (std::regex_search(s, m, re)) return m[1].str();
    return def;
}

// ===== MQTT helper =====
static std::string ic_shell_quote(const std::string& s){
    std::string o; o.reserve(s.size()+8); o.push_back('\'');
    for(char c: s){ if(c=='\'') o += "'\"'\"'"; else o.push_back(c); } o.push_back('\''); return o;
}
static void ic_mqtt_publish(const std::string& host, int port, const std::string& topic, const std::string& payload,
                            const std::string& user="", const std::string& pass="") {
    if (host.empty()) return;
    const char* bin = "/usr/bin/mosquitto_pub";
    std::string cmd = std::string(bin) + " -h " + ic_shell_quote(host)
                    + " -p " + std::to_string(port)
                    + " -t " + ic_shell_quote(topic)
                    + " -m " + ic_shell_quote(payload);
    if (!user.empty()) cmd += " -u " + ic_shell_quote(user);
    if (!pass.empty()) cmd += " -P " + ic_shell_quote(pass);
    int rc = std::system(cmd.c_str());
    fprintf(stderr, "[MQTT] rc=%d topic=%s payload=%s\n", rc, topic.c_str(), payload.c_str());
}

// ===== TRT Logger =====
struct ICLogger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
    }
};
static ICLogger g_trt_logger;

// ===== Per-instance state =====
struct ICPState {
    CUcontext     ctx   = nullptr;
    cudaStream_t  stream= nullptr;

    // Scratch NV12
    CUdeviceptr sY = 0, sUV = 0;
    size_t      pY = 0, pUV = 0;
    int         sW = 0, sH = 0;

    // Geometry + color controls (existing)
    icp::RectifyConfig cfg{};
    icp::RuntimeControls controls{"/home/jetson_ntc/config.json", "cam0"}; // JSON path, section camX

    float crop_frac = 0.20f;

    // Mask (wire removal)
    struct DeviceMask { CUdeviceptr dMask=0; size_t pitch=0; int W=0,H=0; float dx=0,dy=0; bool valid=false; int cam_index=-1; } mask{};
    bool mask_checked_once = false;

    // Gesture (TRT) — non-blocking
    struct {
        bool enabled=false;
        // config
        std::string engine_path;
        float start_th=0.80f, stop_th=0.80f; int streak_need=3;
        std::string host, topic="jetson/stream/cmd", user, pass; int port=1883;

        // runtime
        nvinfer1::IRuntime* runtime=nullptr;
        nvinfer1::ICudaEngine* engine=nullptr;
        nvinfer1::IExecutionContext* ctx=nullptr;
        int inIdx=-1,outIdx=-1; nvinfer1::DataType inType=nvinfer1::DataType::kFLOAT, outType=nvinfer1::DataType::kFLOAT;
        int64_t nInput=0,nOutput=0;
        void* dIn=nullptr; void* dOut=nullptr;

        cudaStream_t stream=nullptr;
        cudaEvent_t  evt_frame_ready=nullptr, evt_done=nullptr;
        std::atomic<bool> pending{false};

        std::vector<float>  hOutF32;
        std::vector<__half> hOutF16;

        int start_streak=0, stop_streak=0;
        enum class RecState { IDLE, RECORDING } rec = RecState::IDLE;

        std::thread worker; std::atomic<bool> stop_worker{false};
    } trt;
};

// Global instance tracking
static std::mutex              g_instances_mtx;
static std::vector<ICPState*>  g_instances;

static std::mutex              g_hint_mtx;
static std::deque<std::string> g_next_section_hints;

extern "C" void ic_bind_next_instance_to(const char* section) {
    if (!section || !*section) return;
    std::string s = to_lower(section);
    if (s!="cam0" && s!="cam1" && s!="cam2") return;
    std::lock_guard<std::mutex> lk(g_hint_mtx);
    g_next_section_hints.push_back(std::move(s));
}
extern "C" void ic_clear_instance_hints() { std::lock_guard<std::mutex> lk(g_hint_mtx); g_next_section_hints.clear(); }

// ===== Scratch alloc =====
static void ensure_scratch(ICPState* st, int W, int H)
{
    if (st->sY && st->sUV && st->sW==W && st->sH==H) return;
    if (st->sY)  { cuMemFree(st->sY);  st->sY  = 0; }
    if (st->sUV) { cuMemFree(st->sUV); st->sUV = 0; }
    cuMemAllocPitch(&st->sY,  &st->pY,  (size_t)W, (size_t)H,    4);
    cuMemAllocPitch(&st->sUV, &st->pUV, (size_t)W, (size_t)(H/2),4);
    st->sW = W; st->sH = H;
}
static void copy_to_scratch_async(ICPState* st,
                                  const uint8_t* dY,const uint8_t* dUV,
                                  size_t pitchY,size_t pitchUV,
                                  int W,int H)
{
    CUDA_MEMCPY2D c{}; c.srcMemoryType=CU_MEMORYTYPE_DEVICE; c.dstMemoryType=CU_MEMORYTYPE_DEVICE;
    c.srcDevice=(CUdeviceptr)dY; c.srcPitch=pitchY; c.dstDevice=st->sY; c.dstPitch=st->pY;
    c.WidthInBytes=(size_t)W; c.Height=(size_t)H;
    CUDA_MEMCPY2D c2{}; c2.srcMemoryType=CU_MEMORYTYPE_DEVICE; c2.dstMemoryType=CU_MEMORYTYPE_DEVICE;
    c2.srcDevice=(CUdeviceptr)dUV; c2.srcPitch=pitchUV; c2.dstDevice=st->sUV; c2.dstPitch=st->pUV;
    c2.WidthInBytes=(size_t)W; c2.Height=(size_t)(H/2);
    CUstream drs = reinterpret_cast<CUstream>(st->stream);
    cuMemcpy2DAsync(&c,  drs);
    cuMemcpy2DAsync(&c2, drs);
}

// ===== Mask loader =====
static bool load_mask_from_cpu_once(ICPState* st)
{
    const int my_idx = section_to_index(st->controls.section());
    if (my_idx < 0) return false;
    const std::string meta = mask_meta_path_for(my_idx);
    const std::string raw  = mask_raw_path_for(my_idx);

    FILE* fm = fopen(meta.c_str(), "r");
    if (!fm) { fprintf(stderr, "[ic] mask meta not found: %s\n", meta.c_str()); return false; }
    int W=0,H=0; float dx=0,dy=0;
    if (fscanf(fm, "%d %d %f %f", &W,&H,&dx,&dy) != 4 || W<=0 || H<=0) { fclose(fm); return false; }
    fclose(fm);

    std::vector<uint8_t> hostMask((size_t)W*(size_t)H);
    FILE* fr = fopen(raw.c_str(), "rb");
    if (!fr) { fprintf(stderr, "[ic] mask raw not found: %s\n", raw.c_str()); return false; }
    size_t rd = fread(hostMask.data(),1,hostMask.size(),fr); fclose(fr);
    if (rd != hostMask.size()) { fprintf(stderr, "[ic] mask size mismatch\n"); return false; }

    if (!st->mask.dMask || st->mask.W!=W || st->mask.H!=H) {
        if (st->mask.dMask) { cuMemFree(st->mask.dMask); st->mask.dMask=0; }
        CUdeviceptr dptr=0; size_t pitch=0;
        if (cuMemAllocPitch(&dptr,&pitch,(size_t)W,(size_t)H,4)!=CUDA_SUCCESS) return false;
        st->mask.dMask=dptr; st->mask.pitch=pitch; st->mask.W=W; st->mask.H=H;
    }
    CUDA_MEMCPY2D c{};
    c.srcMemoryType=CU_MEMORYTYPE_HOST; c.srcHost=hostMask.data(); c.srcPitch=(size_t)W;
    c.dstMemoryType=CU_MEMORYTYPE_DEVICE; c.dstDevice=st->mask.dMask; c.dstPitch=st->mask.pitch;
    c.WidthInBytes=(size_t)W; c.Height=(size_t)H;
    cuMemcpy2D(&c);

    st->mask.dx=dx; st->mask.dy=dy; st->mask.cam_index=my_idx; st->mask.valid=true;
    fprintf(stderr,"[ic] mask loaded for %s (W=%d H=%d)\n", st->controls.section().c_str(), W,H);
    return true;
}

// ===== Read gesture config from JSON (within camX) =====
static void load_gesture_from_json(ICPState* st, const std::string& json_full)
{
    // select camX object
    std::string camObj = json_full;
    std::string camKey = st->controls.section(); // "cam0"|"cam1"|"cam2"
    std::string sub;
    if (json_extract_object(json_full, camKey, sub)) camObj = sub;

    // gesture object
    std::string gestureObj;
    if (!json_extract_object(camObj, "gesture", gestureObj)) {
        st->trt.enabled = false;
        return;
    }
    bool enable=false; (void)json_get_bool(gestureObj, "enable", enable);
    st->trt.enabled = enable;
    st->trt.engine_path = json_get_string(gestureObj, "engine", "/home/jetson_ntc/gesture.engine");
    json_get_number(gestureObj, "start_th", st->trt.start_th);
    json_get_number(gestureObj, "stop_th",  st->trt.stop_th);
    int streak=st->trt.streak_need; if (json_get_number(gestureObj, "streak", streak)) st->trt.streak_need = std::max(1, streak);

    // nested mqtt
    std::string mqttObj;
    if (json_extract_object(gestureObj, "mqtt", mqttObj)) {
        st->trt.host  = json_get_string(mqttObj, "host", "");
        int port = st->trt.port; if (json_get_number(mqttObj, "port", port)) st->trt.port = port;
        st->trt.topic = json_get_string(mqttObj, "topic", st->trt.topic);
        st->trt.user  = json_get_string(mqttObj, "user",  "");
        st->trt.pass  = json_get_string(mqttObj, "pass",  "");
    }
}

// ===== TRT init / worker =====
static bool trt_init(ICPState* st)
{
    if (!st->trt.enabled) return false;

    if (!ei::ensure_stream_created(st->trt.stream)) { fprintf(stderr,"[TRT] stream create failed\n"); return false; }
    cudaEventCreateWithFlags(&st->trt.evt_frame_ready, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&st->trt.evt_done,        cudaEventDisableTiming);

    // load engine
    std::ifstream f(st->trt.engine_path, std::ios::binary);
    if (!f) { fprintf(stderr,"[TRT] cannot open engine %s\n", st->trt.engine_path.c_str()); return false; }
    f.seekg(0,std::ios::end); size_t sz=(size_t)f.tellg(); f.seekg(0); std::vector<char> blob(sz); f.read(blob.data(),sz);

    st->trt.runtime = nvinfer1::createInferRuntime(g_trt_logger);
    if (!st->trt.runtime) { fprintf(stderr,"[TRT] createInferRuntime failed\n"); return false; }
    st->trt.engine = st->trt.runtime->deserializeCudaEngine(blob.data(), blob.size());
    if (!st->trt.engine){ fprintf(stderr,"[TRT] deserialize failed\n"); return false; }
    st->trt.ctx = st->trt.engine->createExecutionContext();
    if (!st->trt.ctx)   { fprintf(stderr,"[TRT] createExecutionContext failed\n"); return false; }

    int nb=st->trt.engine->getNbBindings();
    for (int i=0;i<nb;i++){ if (st->trt.engine->bindingIsInput(i)) st->trt.inIdx=i; else st->trt.outIdx=i; }
    if (st->trt.inIdx<0 || st->trt.outIdx<0){ fprintf(stderr,"[TRT] bad bindings\n"); return false; }

    nvinfer1::Dims inD = st->trt.engine->getBindingDimensions(st->trt.inIdx);
    if (inD.nbDims==0 || inD.d[0]==-1){
        nvinfer1::Dims4 d(1,1,96,96);
        if (!st->trt.ctx->setBindingDimensions(st->trt.inIdx,d)){ fprintf(stderr,"[TRT] set dims failed\n"); return false; }
        inD = st->trt.ctx->getBindingDimensions(st->trt.inIdx);
    }
    st->trt.inType  = st->trt.engine->getBindingDataType(st->trt.inIdx);
    st->trt.outType = st->trt.engine->getBindingDataType(st->trt.outIdx);

    st->trt.nInput=1; for(int i=0;i<inD.nbDims;i++) st->trt.nInput*=inD.d[i];
    nvinfer1::Dims outD = st->trt.ctx->getBindingDimensions(st->trt.outIdx);
    st->trt.nOutput=1; for(int i=0;i<outD.nbDims;i++) st->trt.nOutput*=outD.d[i];

    size_t inBytes  = (st->trt.inType==nvinfer1::DataType::kHALF? sizeof(__half):sizeof(float)) * st->trt.nInput;
    size_t outBytes = (st->trt.outType==nvinfer1::DataType::kHALF?sizeof(__half):sizeof(float)) * st->trt.nOutput;
    cudaMalloc(&st->trt.dIn,  inBytes);
    cudaMalloc(&st->trt.dOut, outBytes);
    if (st->trt.outType==nvinfer1::DataType::kHALF) st->trt.hOutF16.resize(st->trt.nOutput);
    else                                             st->trt.hOutF32.resize(st->trt.nOutput);

    fprintf(stderr,"[TRT] ready (%s) in=%ld out=%ld types[%d,%d] for %s\n",
            st->trt.engine_path.c_str(), (long)st->trt.nInput, (long)st->trt.nOutput,
            (int)st->trt.inType, (int)st->trt.outType, st->controls.section().c_str());
    return true;
}

static void worker_loop(ICPState* st)
{
    while (!st->trt.stop_worker.load(std::memory_order_relaxed)) {
        if (!st->trt.pending.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        cudaError_t q = cudaEventQuery(st->trt.evt_done);
        if (q == cudaSuccess) {
            std::vector<float> scores((size_t)st->trt.nOutput);
            if (st->trt.outType==nvinfer1::DataType::kHALF)
                for (int i=0;i<st->trt.nOutput;i++) scores[i]=__half2float(st->trt.hOutF16[i]);
            else
                std::copy(st->trt.hOutF32.begin(), st->trt.hOutF32.end(), scores.begin());

            float maxv=*std::max_element(scores.begin(), scores.end());
            float sum=0.f; for (auto& s: scores){ s=expf(s-maxv); sum+=s; }
            for (auto& s: scores) s/=sum;

            int best=int(std::max_element(scores.begin(), scores.end())-scores.begin());
            float p=scores[best];
            const char* lab = (best==0?"start":(best==1?"stop":"cls"));

            if (strcmp(lab,"start")==0 && p>=st->trt.start_th){ st->trt.start_streak++; st->trt.stop_streak=0; }
            else if (strcmp(lab,"stop")==0 && p>=st->trt.stop_th){ st->trt.stop_streak++; st->trt.start_streak=0; }
            else { st->trt.start_streak=0; st->trt.stop_streak=0; }

            bool do_pub=false; std::string payload;
            if (st->trt.rec==decltype(st->trt.rec)::IDLE && st->trt.start_streak>=st->trt.streak_need){
                st->trt.rec=decltype(st->trt.rec)::RECORDING; st->trt.start_streak=0;
                payload=R"({ "value": { "recording": "start" } })"; do_pub=true;
                fprintf(stderr,"[TRIGGER] recording START (p=%.3f)\n", p);
            } else if (st->trt.rec==decltype(st->trt.rec)::RECORDING && st->trt.stop_streak>=st->trt.streak_need){
                st->trt.rec=decltype(st->trt.rec)::IDLE; st->trt.stop_streak=0;
                payload=R"({ "value": { "recording": "stop" } })"; do_pub=true;
                fprintf(stderr,"[TRIGGER] recording STOP  (p=%.3f)\n", p);
            }
            if (do_pub) ic_mqtt_publish(st->trt.host, st->trt.port, st->trt.topic, payload, st->trt.user, st->trt.pass);

            st->trt.pending.store(false, std::memory_order_relaxed);
        } else if (q != cudaErrorNotReady) {
            (void)cudaGetLastError();
            st->trt.pending.store(false, std::memory_order_relaxed);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

// ===== Instance lifecycle =====
static ICPState* create_instance()
{
    std::call_once(g_ctx_once, retain_primary_context_once);
    auto* st = new ICPState();
    st->ctx = g_primary_ctx; cuCtxSetCurrent(st->ctx);

#if CUDART_VERSION >= 11000
    cudaStreamCreateWithPriority(&st->stream, cudaStreamNonBlocking, 0);
#else
    cudaStreamCreateWithFlags(&st->stream, cudaStreamNonBlocking);
#endif

    // Bind section
    std::string sec;
    { std::lock_guard<std::mutex> lk(g_hint_mtx);
      if (!g_next_section_hints.empty()){ sec=std::move(g_next_section_hints.front()); g_next_section_hints.pop_front();}
    }
    if (sec.empty()) sec = section_from_loaded_name();
    st->controls.set_section(sec);

    // Load JSON once for gesture config
    std::string json_full;
    read_text("/home/jetson_ntc/config.json", json_full);
    load_gesture_from_json(st, json_full);

    // Init TRT (optional)
    if (st->trt.enabled) {
        if (trt_init(st)) { st->trt.stop_worker.store(false); st->trt.worker = std::thread(worker_loop, st); }
        else st->trt.enabled=false;
    }

    fprintf(stderr,"[ic] Instance %s (gesture %s)\n", sec.c_str(), st->trt.enabled?"ENABLED":"disabled");

    { std::lock_guard<std::mutex> lk(g_instances_mtx); g_instances.push_back(st); }
    return st;
}

static void destroy_instance(ICPState* st)
{
    if (!st) return; cuCtxSetCurrent(st->ctx);

    st->trt.stop_worker.store(true);
    if (st->trt.worker.joinable()) st->trt.worker.join();

    if (st->sY)  { cuMemFree(st->sY);  st->sY  = 0; }
    if (st->sUV) { cuMemFree(st->sUV); st->sUV = 0; }
    if (st->mask.dMask) { cuMemFree(st->mask.dMask); st->mask.dMask=0; }
    st->mask = {};

    if (st->trt.evt_frame_ready){ cudaEventDestroy(st->trt.evt_frame_ready); st->trt.evt_frame_ready=nullptr; }
    if (st->trt.evt_done){        cudaEventDestroy(st->trt.evt_done);        st->trt.evt_done=nullptr; }
    ei::destroy_stream(st->trt.stream);

    if (st->trt.dIn){ cudaFree(st->trt.dIn); st->trt.dIn=nullptr; }
    if (st->trt.dOut){ cudaFree(st->trt.dOut); st->trt.dOut=nullptr; }
    if (st->trt.ctx){ st->trt.ctx->destroy(); st->trt.ctx=nullptr; }
    if (st->trt.engine){ st->trt.engine->destroy(); st->trt.engine=nullptr; }
    if (st->trt.runtime){ st->trt.runtime->destroy(); st->trt.runtime=nullptr; }

    if (st->stream) { cudaStreamDestroy(st->stream); st->stream=nullptr; }
    st->ctx=nullptr; delete st;
}

// ===== nvivafilter hooks =====
static void pre_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*, ColorFormat*, unsigned int, void **){}
static void post_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*, ColorFormat*, unsigned int, void **){}

static void gpu_process(EGLImageKHR image, void **userPtr)
{
    ICPState* st = static_cast<ICPState*>(*userPtr);
    if (!st) { st = create_instance(); *userPtr = st; }
    cuCtxSetCurrent(st->ctx);

    if (!st->mask_checked_once) { st->mask_checked_once = true; (void)load_mask_from_cpu_once(st); }

    // Map EGLImage → CUDA
    CUgraphicsResource res=nullptr;
    if(cuGraphicsEGLRegisterImage(&res,image,CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)!=CUDA_SUCCESS) return;
    CUeglFrame f{};
    if(cuGraphicsResourceGetMappedEglFrame(&f,res,0,0)!=CUDA_SUCCESS){ cuGraphicsUnregisterResource(res); return; }
    if(f.frameType!=CU_EGL_FRAME_TYPE_PITCH || f.planeCount<2){ cuGraphicsUnregisterResource(res); return; }

    uint8_t* dY  = static_cast<uint8_t*>(f.frame.pPitch[0]);
    uint8_t* dUV = static_cast<uint8_t*>(f.frame.pPitch[1]);
    const int W = (int)f.width, H=(int)f.height, pitch=(int)f.pitch;

    ensure_scratch(st, W, H);

    // Rectify
    icp::RectifyConfig cfg = st->cfg;
    const float FOV_fish = cfg.fish_fov_deg * (float)M_PI_F / 180.f;
    const float f_fish   = cfg.r_f / (FOV_fish * 0.5f);
    const float fx       = (W * 0.5f) / tanf(cfg.out_hfov_deg * (float)M_PI_F / 360.f);
    const float cx_rect  = W * 0.5f, cy_rect = H * 0.5f;

    copy_to_scratch_async(st, dY, dUV, pitch, pitch, W, H);
    icp::launch_rectify_nv12(
        (const uint8_t*)(uintptr_t)st->sY,  W,H,(int)st->pY,
        (const uint8_t*)(uintptr_t)st->sUV, (int)st->pUV,
        dY, W,H,pitch, dUV, pitch,
        cfg.cx_f, cfg.cy_f, cfg.r_f, f_fish, fx, cx_rect, cy_rect, st->stream);

    // Crop/zoom
    copy_to_scratch_async(st, dY, dUV, pitch, pitch, W, H);
    icp::launch_crop_center_nv12(
        (const uint8_t*)(uintptr_t)st->sY,  W,H,(int)st->pY,
        (const uint8_t*)(uintptr_t)st->sUV, (int)st->pUV,
        dY, pitch, dUV, pitch, st->crop_frac, st->stream);

    // Wire removal (if mask matches this cam)
    if (st->mask.valid && st->mask.W==W && st->mask.H==H) {
        if (section_to_index(st->controls.section()) == st->mask.cam_index) {
            wire::apply_mask_shift_nv12(dY, pitch, dUV, pitch, W, H,
                                        (const uint8_t*)(uintptr_t)st->mask.dMask, (int)st->mask.pitch,
                                        st->mask.dx, st->mask.dy, st->stream);
        }
    }

    // Tone/color
    icp::ColorParams cp = st->controls.current();
    icp::launch_tone_saturation_nv12(dY, W,H,pitch, dUV,pitch, cp, st->stream);

    // ===== Non-blocking gesture enqueue =====
    if (st->trt.enabled && st->trt.ctx && !st->trt.pending.load(std::memory_order_relaxed)) {
        cudaEventRecord(st->trt.evt_frame_ready, st->stream);
        cudaStreamWaitEvent(st->trt.stream, st->trt.evt_frame_ready, 0);

        bool inIsFP16 = (st->trt.inType == nvinfer1::DataType::kHALF);
        (void)ei::enqueue_preprocess_to_trt_input(dY, W, H, pitch, st->trt.dIn, inIsFP16, st->trt.stream);

        void* bindings[2]; bindings[st->trt.inIdx]=st->trt.dIn; bindings[st->trt.outIdx]=st->trt.dOut;
        if (!st->trt.ctx->enqueueV2(bindings, st->trt.stream, nullptr)) {
            fprintf(stderr,"[TRT] enqueueV2 failed\n");
        } else {
            if (st->trt.outType==nvinfer1::DataType::kHALF)
                cudaMemcpyAsync(st->trt.hOutF16.data(), st->trt.dOut, sizeof(__half)*st->trt.nOutput, cudaMemcpyDeviceToHost, st->trt.stream);
            else
                cudaMemcpyAsync(st->trt.hOutF32.data(), st->trt.dOut, sizeof(float)*st->trt.nOutput, cudaMemcpyDeviceToHost, st->trt.stream);
            cudaEventRecord(st->trt.evt_done, st->trt.stream);
            st->trt.pending.store(true, std::memory_order_relaxed);
        }
    }
    // Do NOT wait for gesture stream
    cudaStreamSynchronize(st->stream);
    cuGraphicsUnregisterResource(res);
}

// ===== init / deinit =====
extern "C" void init(CustomerFunction* f)
{
    if(!f) return;
    f->fPreProcess  = pre_process;
    f->fGPUProcess  = gpu_process;
    f->fPostProcess = post_process;
    std::call_once(g_ctx_once, retain_primary_context_once);
    fprintf(stderr,"[ic] imagecorrection initialized (gesture add-on via JSON)\n");
}

extern "C" void deinit(void)
{
    std::vector<ICPState*> to_free;
    { std::lock_guard<std::mutex> lk(g_instances_mtx); to_free.swap(g_instances); }
    for (auto* st : to_free) destroy_instance(st);
    release_primary_context();
    ic_clear_instance_hints();
}

// Optional host-query
extern "C" int ic_has_instance_for(const char* section) {
    if (!section || !*section) return 0;
    std::lock_guard<std::mutex> lk(g_instances_mtx);
    for (auto* st : g_instances)
        if (st && st->controls.section() == section) return 1;
    return 0;
}
