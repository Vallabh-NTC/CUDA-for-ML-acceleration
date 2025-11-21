/**
 * @file nvivafilter_imagecorrection.cpp
 * (decoupled TRT stream + device FIFO + hand pose keypoints overlay + peace gesture)
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
#include <chrono>
#include <unistd.h>   // getpid(), unlink()

#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>

#include "nvivafilter_customer_api.hpp"
#include "rectify_config.hpp"
#include "kernel_rectify.cuh"
#include "color_ops.cuh"
#include "runtime_controls.hpp"
#include "wire_lineremoval.cuh"

#include "trt_gesture.hpp"
#include "ei_gesture_infer.cuh"

// Debug ROI box on NV12
#include "kernel_draw_box_nv12.cuh"
// Hand keypoints + skeleton on NV12
#include "kernel_draw_hand_points_nv12.cuh"
// Gesture classification (peace ✌️)
#include "gesture_classify.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
constexpr float M_PI_F = static_cast<float>(M_PI);

// Path of the hand pose TensorRT engine
static const char* kGestureEnginePath =
    "/usr/local/lib/nvivafilter/models/hand_pose_resnet18_fp32.engine";

// ============================================================================
// Global CUDA primary context (shared across instances)
// ============================================================================
namespace {
    static std::once_flag g_ctx_once;
    static CUcontext g_primary_ctx = nullptr;
    static CUdevice  g_device      = 0;

    static void retain_primary_context_once() {
        cuInit(0);
        cuDeviceGet(&g_device, 0);
        CUresult r = cuDevicePrimaryCtxRetain(&g_primary_ctx, g_device);
        if (r != CUDA_SUCCESS) {
            fprintf(stderr, "[ic] Failed to retain primary CUDA context (%d)\n", (int)r);
            g_primary_ctx = nullptr;
        } else {
            cuCtxSetCurrent(g_primary_ctx);
            fprintf(stderr, "[ic] Using shared primary CUDA context\n");
        }
    }

    static void release_primary_context() {
        if (g_primary_ctx) {
            cuDevicePrimaryCtxRelease(g_device);
            g_primary_ctx = nullptr;
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================
static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return (char)std::tolower(c); });
    return s;
}

static float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

static std::string section_from_loaded_name(){
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
    fprintf(stderr, "[ic] WARNING: cannot detect cam section from .so name/path, defaulting to cam0\n");
    return "cam0";
}

static int section_to_index(const std::string& sec) {
    if (sec == "cam0") return 0;
    if (sec == "cam1") return 1;
    if (sec == "cam2") return 2;
    return -1;
}

static std::string mask_raw_path_for(int cam_idx) {
    return "/dev/shm/wire_mask_cam" + std::to_string(cam_idx) + ".raw";
}
static std::string mask_meta_path_for(int cam_idx) {
    return "/dev/shm/wire_mask_cam" + std::to_string(cam_idx) + ".meta";
}

// ============================================================================
// MQTT helpers (kept for compatibility, but not used for peace gesture)
// ============================================================================
struct MqttCfg {
    std::string host;
    int port=1883;
    std::string topic="jetson/stream/cmd";
    std::string user;
    std::string pass;
    bool valid=false;
};

static std::string shell_quote(const std::string& s){
    std::string o;
    o.reserve(s.size()+8);
    o.push_back('\'');
    for(char c: s){
        if(c=='\'') o += "'\"'\"'";
        else o.push_back(c);
    }
    o.push_back('\'');
    return o;
}

static void parse_mqtt_env(MqttCfg& out){
    const char* url = std::getenv("MQTT_URL");
    if (url && *url) {
        std::string u(url);
        auto pos = u.find("://");
        std::string rest = (pos==std::string::npos)? u : u.substr(pos+3);
        auto colon = rest.find(':');
        if (colon!=std::string::npos){
            out.host = rest.substr(0, colon);
            try { out.port = std::stoi(rest.substr(colon+1)); } catch(...) {}
        } else {
            out.host = rest;
        }
    }
    if (const char* h = std::getenv("MQTT_HOST"); h && *h) out.host = h;
    if (const char* p = std::getenv("MQTT_PORT"); p && *p){
        try{ out.port = std::stoi(p);}catch(...){} }
    if (const char* t = std::getenv("MQTT_TOPIC"); t && *t) out.topic = t;
    if (const char* u = std::getenv("MQTT_USER");  u && *u) out.user  = u;
    if (const char* pw= std::getenv("MQTT_PASS");  pw&& *pw) out.pass  = pw;
    out.valid = !out.host.empty();
}

static void mqtt_publish(const MqttCfg& c, const std::string& payload){
    if (!c.valid) return;
    const char* bin = "/usr/bin/mosquitto_pub";
    std::string cmd = std::string(bin) + " -h " + shell_quote(c.host)
                    + " -p " + std::to_string(c.port)
                    + " -t " + shell_quote(c.topic)
                    + " -m " + shell_quote(payload);
    if (!c.user.empty()) cmd += " -u " + shell_quote(c.user);
    if (!c.pass.empty()) cmd += " -P " + shell_quote(c.pass);
    int rc = std::system(cmd.c_str());
    fprintf(stderr, "[MQTT] rc=%d topic=%s payload=%s\n", rc, c.topic.c_str(), payload.c_str());
}

// ============================================================================
// Audio TTS helper (kept for compatibility, not used here)
// ============================================================================
static void speak_async(const std::string& phrase) {
    const char* pico_bin = std::getenv("PICOWAVE_BIN");
    const char* play_bin = std::getenv("PAPLAY_BIN");
    const std::string pico = (pico_bin && *pico_bin) ? pico_bin : "/usr/bin/pico2wave";
    const std::string play = (play_bin && *play_bin) ? play_bin : "/usr/bin/paplay";
    const char* lang_env = std::getenv("PICOWAVE_LANG");
    const std::string lang = (lang_env && *lang_env) ? lang_env : "en-US";
    if (const char* disa = std::getenv("GESTURE_AUDIO_DISABLE"); disa && *disa=='1') {
        fprintf(stderr, "[AUDIO] disabled by GESTURE_AUDIO_DISABLE=1 (phrase=%s)\n", phrase.c_str());
        return;
    }
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    const std::string tmpwav = "/dev/shm/tts_" + std::to_string(getpid()) + "_" + std::to_string(ns) + ".wav";
    std::string cmd = shell_quote(pico) + " -l " + shell_quote(lang)
                    + " -w " + shell_quote(tmpwav) + " " + shell_quote(phrase)
                    + " && " + shell_quote(play) + " " + shell_quote(tmpwav)
                    + " ; rm -f " + shell_quote(tmpwav);
    std::thread([cmd=std::move(cmd), phrase](){
        int rc = std::system(cmd.c_str());
        fprintf(stderr, "[AUDIO] rc=%d phrase=%s\n", rc, phrase.c_str());
    }).detach();
}

// ============================================================================
// UI indicator helper — atomic JSON to /dev/shm (legacy, used at startup)
// ============================================================================
static void emit_ui_indicator_json(int cam, const char* indicator) {
    if (cam != 1) return; // Only cam1 uses this UI
    const char* final_path = "/dev/shm/ui_cam1.json";
    const char* tmp_path   = "/dev/shm/ui_cam1.tmp";

    char buf[128];
    int n = snprintf(buf, sizeof(buf), "{ \"cam\": %d, \"indicator\": \"%s\" }\n",
                     cam, indicator ? indicator : "off");
    if (n <= 0 || n >= (int)sizeof(buf)) {
        fprintf(stderr, "[UI] snprintf failed for indicator='%s'\n",
                indicator ? indicator : "(null)");
        return;
    }

    FILE* f = fopen(tmp_path, "wb");
    if (!f) {
        fprintf(stderr, "[UI] fopen(tmp) failed: %s\n", tmp_path);
        return;
    }
    size_t wr = fwrite(buf, 1, (size_t)n, f);
    if (wr != (size_t)n) {
        fprintf(stderr, "[UI] fwrite short (%zu/%d) to %s\n", wr, n, tmp_path);
        fclose(f);
        unlink(tmp_path);
        return;
    }
    fclose(f);
    if (rename(tmp_path, final_path) != 0) {
        fprintf(stderr, "[UI] rename(%s -> %s) failed\n", tmp_path, final_path);
        unlink(tmp_path);
        return;
    }
    fprintf(stderr, "[UI] wrote %s: %s", final_path, buf);
}

// ============================================================================
// Per-instance state
// ============================================================================
struct ICPState {
    CUcontext     ctx   = nullptr;
    cudaStream_t  video_stream = nullptr;   // main video stream

    // Separate TRT stream for inference
    cudaStream_t  trt_stream   = nullptr;

    // Scratch (NV12) for rectification
    CUdeviceptr sY = 0, sUV = 0; size_t pY = 0, pUV = 0; int sW = 0, sH = 0;

    // Geometry (fisheye rectification)
    icp::RectifyConfig cfg{};

    // Runtime controls (hot-reload)
    icp::RuntimeControls controls{"/home/moviemaker/config.json", "cam0"};

    float crop_frac = 0.0f;

    struct DeviceMask {
        CUdeviceptr dMask=0;
        size_t pitch=0;
        int W=0,H=0;
        float dx=0.f,dy=0.f;
        bool valid=false;
        int cam_index=-1;
    } mask{};
    bool mask_checked_once = false;

    // Gesture NN (TensorRT), only used on cam1
    trt::Engine gesture;
    bool gesture_loaded_once = false;

    // bytes of the input tensor (1x3x244x244 fp32/fp16)
    size_t gesture_input_bytes = 0;

    // Device FIFO of preprocessed tensors
    static constexpr int kSlots = 4; // small ring to keep latency bounded
    enum class SlotState : int { FREE=0, READY=1, INFLIGHT=2 };
    struct Slot {
        void*      dTensor = nullptr;   // FP16 or FP32 matching engine input
        cudaEvent_t ev_ready = nullptr; // signaled on video_stream when tensor is written
        cudaEvent_t ev_done  = nullptr; // signaled on trt_stream when TRT+D2H queued
        std::atomic<int> state{(int)SlotState::FREE};
    } slots[kSlots];
    int prod_idx = 0; // producer writes here

    // MQTT + FSM kept for compatibility but no longer used for gesture logic
    MqttCfg mqtt{}; bool mqtt_checked_once = false;
    struct {
        int   hold_start_target_ms = 800;
        int   hold_stop_target_ms  = 800;
        int   cycle_reset_ms       = 5000;
        float start_prob_thresh    = 0.75f;
        float stop_prob_thresh     = 0.80f;
        enum class Phase { SEEK_START, SEEK_STOP } phase = Phase::SEEK_START;
        std::chrono::steady_clock::time_point t_last = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point t_cycle_start = std::chrono::steady_clock::now();
        int hold_start_ms = 0; int hold_stop_ms = 0; bool recording = false;
        void reset_phase_to_start(){
            phase=Phase::SEEK_START;
            hold_start_ms=hold_stop_ms=0;
            t_cycle_start=std::chrono::steady_clock::now();
        }
    } fsm;
};

static std::mutex              g_instances_mtx;
static std::vector<ICPState*>  g_instances;
static std::mutex              g_hint_mtx;
static std::deque<std::string> g_next_section_hints;

extern "C" void ic_bind_next_instance_to(const char* section){
    if(!section||!*section) return;
    std::string s=to_lower(section);
    if (s!="cam0"&&s!="cam1"&&s!="cam2") return;
    std::lock_guard<std::mutex> lk(g_hint_mtx);
    g_next_section_hints.push_back(std::move(s));
}
extern "C" void ic_clear_instance_hints(){
    std::lock_guard<std::mutex> lk(g_hint_mtx);
    g_next_section_hints.clear();
}
extern "C" int ic_has_instance_for(const char* section){
    if(!section||!*section) return 0;
    std::lock_guard<std::mutex> lk(g_instances_mtx);
    for(auto* st:g_instances){
        if(!st) continue;
        if(st->controls.section()==section) return 1;
    }
    return 0;
}

// ----------------------------------------------------------------------------
// Alloc helpers
// ----------------------------------------------------------------------------
static void ensure_scratch(ICPState* st, int W, int H){
    if (st->sY && st->sUV && st->sW==W && st->sH==H) return;
    if (st->sY)  { cuMemFree(st->sY);  st->sY  = 0; }
    if (st->sUV) { cuMemFree(st->sUV); st->sUV = 0; }
    cuMemAllocPitch(&st->sY,  &st->pY,  (size_t)W, (size_t)H,    4);
    cuMemAllocPitch(&st->sUV, &st->pUV, (size_t)W, (size_t)(H/2),4);
    st->sW=W; st->sH=H;
}

static void copy_to_scratch_async(ICPState* st,
                                  const uint8_t* dY,const uint8_t* dUV,
                                  size_t pitchY,size_t pitchUV,
                                  int W,int H)
{
    CUDA_MEMCPY2D c{};
    c.srcMemoryType=CU_MEMORYTYPE_DEVICE;
    c.dstMemoryType=CU_MEMORYTYPE_DEVICE;
    c.srcDevice=(CUdeviceptr)dY;  c.srcPitch=pitchY;
    c.dstDevice=st->sY;           c.dstPitch=st->pY;
    c.WidthInBytes=(size_t)W;     c.Height=(size_t)H;

    CUDA_MEMCPY2D c2{};
    c2.srcMemoryType=CU_MEMORYTYPE_DEVICE;
    c2.dstMemoryType=CU_MEMORYTYPE_DEVICE;
    c2.srcDevice=(CUdeviceptr)dUV; c2.srcPitch=pitchUV;
    c2.dstDevice=st->sUV;          c2.dstPitch=st->pUV;
    c2.WidthInBytes=(size_t)W;     c2.Height=(size_t)(H/2);

    CUstream s = reinterpret_cast<CUstream>(st->video_stream);
    cuMemcpy2DAsync(&c,  s);
    cuMemcpy2DAsync(&c2, s);
}

// ----------------------------------------------------------------------------
// Mask loader
// ----------------------------------------------------------------------------
static bool load_mask_from_cpu_once(ICPState* st){
    const std::string sec = st->controls.section();
    const int my_idx = section_to_index(sec);
    if (my_idx < 0) {
        fprintf(stderr, "[ic] mask: unknown section '%s'\n", sec.c_str());
        return false;
    }

    const std::string meta = mask_meta_path_for(my_idx);
    const std::string raw  = mask_raw_path_for(my_idx);

    FILE* fm = fopen(meta.c_str(), "r");
    if (!fm) {
        fprintf(stderr, "[ic] mask: meta not found for %s (skipping)\n", sec.c_str());
        return false;
    }

    int W=0,H=0; float dx=0.f,dy=0.f;
    int n = fscanf(fm, "%d %d %f %f", &W, &H, &dx, &dy);
    fclose(fm);
    if (n!=4 || W<=0 || H<=0) {
        fprintf(stderr, "[ic] mask: bad meta format for %s (skipping)\n", sec.c_str());
        return false;
    }

    const size_t bytes = (size_t)W*(size_t)H;
    std::vector<uint8_t> hostMask(bytes);

    FILE* fr = fopen(raw.c_str(), "rb");
    if (!fr) {
        fprintf(stderr, "[ic] mask: raw not found for %s (skipping)\n", sec.c_str());
        return false;
    }
    size_t rd=fread(hostMask.data(),1,bytes,fr);
    fclose(fr);
    if (rd!=bytes){
        fprintf(stderr,"[ic] mask: raw size mismatch for %s (%zu vs %zu)\n",
                sec.c_str(), rd, bytes);
        return false;
    }

    if (!st->mask.dMask || st->mask.W!=W || st->mask.H!=H){
        if (st->mask.dMask) { cuMemFree(st->mask.dMask); st->mask.dMask=0; }
        CUdeviceptr dptr=0; size_t pitch=0;
        CUresult rc = cuMemAllocPitch(&dptr,&pitch,(size_t)W,(size_t)H,4);
        if (rc!=CUDA_SUCCESS){
            fprintf(stderr, "[ic] mask: cuMemAllocPitch failed (%d)\n", (int)rc);
            return false;
        }
        st->mask.dMask=dptr; st->mask.pitch=pitch; st->mask.W=W; st->mask.H=H;
    }

    CUDA_MEMCPY2D c{};
    c.srcMemoryType=CU_MEMORYTYPE_HOST;
    c.srcHost=hostMask.data();
    c.srcPitch=(size_t)W;
    c.dstMemoryType=CU_MEMORYTYPE_DEVICE;
    c.dstDevice=st->mask.dMask;
    c.dstPitch=st->mask.pitch;
    c.WidthInBytes=(size_t)W;
    c.Height=(size_t)H;
    cuMemcpy2D(&c);

    st->mask.dx=dx;
    st->mask.dy=dy;
    st->mask.cam_index=my_idx;
    st->mask.valid=true;

    fprintf(stderr, "[ic] mask: loaded for %s (W=%d H=%d, dx=%.2f dy=%.2f)\n",
            sec.c_str(), W,H,dx,dy);
    return true;
}

// ----------------------------------------------------------------------------
// Gesture helper: lazy-load engine only for cam1 + allocate slots/streams
// ----------------------------------------------------------------------------
static void ensure_gesture_loaded_for_cam1(ICPState* st){
    if (st->gesture_loaded_once) return;
    st->gesture_loaded_once = true;

    if (st->controls.section() != std::string("cam1")) {
        fprintf(stderr, "[ic] gesture: skipped (section is '%s', only cam1 runs it)\n",
                st->controls.section().c_str());
        return;
    }

    if (!st->gesture.load_from_file(kGestureEnginePath, st->video_stream)) {
        fprintf(stderr, "[ic] gesture: load failed (path=%s)\n", kGestureEnginePath);
        return;
    }

    // Create trt_stream with lower priority than video_stream
    int leastPri=0, greatestPri=0;
    cudaDeviceGetStreamPriorityRange(&leastPri,&greatestPri);
#if CUDART_VERSION >= 11000
    cudaStreamCreateWithPriority(&st->trt_stream, cudaStreamNonBlocking,
                                 (leastPri+greatestPri)/2);
#else
    cudaStreamCreateWithFlags(&st->trt_stream, cudaStreamNonBlocking);
#endif

    // Hardcoded: input 1x3x244x244 (matches preprocess and engine)
    const bool fp16 = st->gesture.inputIsFP16;
    const int  C = 3;
    const int  H = 244;
    const int  W = 244;

    size_t elemBytes   = fp16 ? sizeof(__half) : sizeof(float);
    size_t tensorBytes = (size_t)C * H * W * elemBytes;

    st->gesture_input_bytes = tensorBytes;

    for (int i=0; i<ICPState::kSlots; ++i) {
        cudaMalloc(&st->slots[i].dTensor, tensorBytes);
        cudaEventCreateWithFlags(&st->slots[i].ev_ready, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&st->slots[i].ev_done,  cudaEventDisableTiming);
        st->slots[i].state.store((int)ICPState::SlotState::FREE);
    }

    fprintf(stderr,
        "[ic] gesture: input shape 1x%dx%dx%d bytes/elem=%zu tensorBytes=%zu (fp16=%d)\n",
        C, H, W, elemBytes, tensorBytes, fp16?1:0);

    fprintf(stderr,
        "[ic] gesture: ready (path=%s) with %d slots, dtype=%s (input 1x%dx%dx%d, %zu bytes)\n",
        kGestureEnginePath,
        ICPState::kSlots,
        fp16 ? "fp16" : "fp32",
        C, H, W,
        tensorBytes);
}

// ----------------------------------------------------------------------------
// FSM helper: init MQTT + durations from env (legacy)
// ----------------------------------------------------------------------------
static void ensure_mqtt_and_fsm_config(ICPState* st){
    if (!st->mqtt_checked_once){
        st->mqtt_checked_once = true;
        parse_mqtt_env(st->mqtt);
        if (const char* v = std::getenv("GESTURE_HOLD_START_MS"); v && *v)
            st->fsm.hold_start_target_ms = std::max(100, atoi(v));
        if (const char* v = std::getenv("GESTURE_HOLD_STOP_MS"); v && *v)
            st->fsm.hold_stop_target_ms  = std::max(100, atoi(v));
        if (const char* v = std::getenv("GESTURE_CYCLE_RESET_MS"); v && *v)
            st->fsm.cycle_reset_ms       = std::max(1000, atoi(v));
        if (const char* v = std::getenv("GESTURE_START_P"); v && *v)
            st->fsm.start_prob_thresh = clamp01(strtof(v, nullptr));
        if (const char* v = std::getenv("GESTURE_STOP_P");  v && *v)
            st->fsm.stop_prob_thresh  = clamp01(strtof(v, nullptr));
        fprintf(stderr,
            "[FSM] (legacy) hold_start=%dms hold_stop=%dms cycle_reset=%dms "
            "p_thresh(start=%.2f stop=%.2f) mqtt=%s:%d topic=%s\n",
            st->fsm.hold_start_target_ms,
            st->fsm.hold_stop_target_ms,
            st->fsm.cycle_reset_ms,
            st->fsm.start_prob_thresh,
            st->fsm.stop_prob_thresh,
            st->mqtt.host.c_str(), st->mqtt.port, st->mqtt.topic.c_str());
    }
}

// ----------------------------------------------------------------------------
// Instance allocation / destruction
// ----------------------------------------------------------------------------
static ICPState* create_instance(){
    std::call_once(g_ctx_once, retain_primary_context_once);
    auto* st = new ICPState();
    st->ctx = g_primary_ctx;
    cuCtxSetCurrent(st->ctx);
#if CUDART_VERSION >= 11000
    cudaStreamCreateWithPriority(&st->video_stream, cudaStreamNonBlocking, /*high*/ 0);
#else
    cudaStreamCreateWithFlags(&st->video_stream, cudaStreamNonBlocking);
#endif
    std::string sec;
    {
        std::lock_guard<std::mutex> lk(g_hint_mtx);
        if (!g_next_section_hints.empty()){
            sec = std::move(g_next_section_hints.front());
            g_next_section_hints.pop_front();
        }
    }
    if (sec.empty()) sec = section_from_loaded_name();
    st->controls.set_section(sec);
    fprintf(stderr, "[ic] Instance bound to section '%s'\n", sec.c_str());

    // Seed UI for cam1 as "yellow" (legacy indicator)
    if (sec == "cam1") {
        emit_ui_indicator_json(1, "yellow");
    }

    {
        std::lock_guard<std::mutex> lk(g_instances_mtx);
        g_instances.push_back(st);
    }
    return st;
}

static void destroy_instance(ICPState* st){
    if (!st) return;
    cuCtxSetCurrent(st->ctx);
    if (st->sY)  { cuMemFree(st->sY);  st->sY  = 0; }
    if (st->sUV) { cuMemFree(st->sUV); st->sUV = 0; }
    if (st->mask.dMask) { cuMemFree(st->mask.dMask); st->mask.dMask = 0; }
    for (int i=0;i<ICPState::kSlots;++i){
        if (st->slots[i].dTensor) cudaFree(st->slots[i].dTensor);
        if (st->slots[i].ev_ready) cudaEventDestroy(st->slots[i].ev_ready);
        if (st->slots[i].ev_done)  cudaEventDestroy(st->slots[i].ev_done);
    }
    st->gesture.destroy();
    if (st->trt_stream)   { cudaStreamDestroy(st->trt_stream);   st->trt_stream=nullptr; }
    if (st->video_stream) { cudaStreamDestroy(st->video_stream); st->video_stream=nullptr; }
    st->ctx=nullptr;
    delete st;
}

// ----------------------------------------------------------------------------
// nvivafilter hooks
// ----------------------------------------------------------------------------
static void pre_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*, ColorFormat*, unsigned int, void **){}
static void post_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*, ColorFormat*, unsigned int, void **){}

// Helper: advance to a FREE slot, dropping oldest if needed
static int acquire_free_slot(ICPState* st){
    for (int tries=0; tries<ICPState::kSlots; ++tries){
        int idx = st->prod_idx;
        auto s = (ICPState::SlotState) st->slots[idx].state.load(std::memory_order_relaxed);
        if (s == ICPState::SlotState::FREE) return idx;
        if (s == ICPState::SlotState::INFLIGHT &&
            cudaEventQuery(st->slots[idx].ev_done) == cudaSuccess){
            st->slots[idx].state.store((int)ICPState::SlotState::FREE,
                                       std::memory_order_relaxed);
            return idx;
        }
        st->prod_idx = (st->prod_idx + 1) % ICPState::kSlots;
    }
    return st->prod_idx;
}

// Launch TRT inference for READY slots; use all bindings (input+outputs)
static void kick_trt_for_ready_slots(ICPState* st){
    if (!st->gesture.engine || !st->trt_stream) return;

    // Find one READY slot to send into TRT
    for (int i = 0; i < ICPState::kSlots; ++i) {
        auto& slot = st->slots[i];
        auto s = (ICPState::SlotState)slot.state.load(std::memory_order_acquire);
        if (s != ICPState::SlotState::READY) continue;

        // Wait for preprocessing on video_stream
        cudaStreamWaitEvent(st->trt_stream, slot.ev_ready, 0);

        if (st->gesture.nbBindings <= 0) {
            fprintf(stderr, "[gesture] nbBindings invalid (%d)\n", st->gesture.nbBindings);
            slot.state.store((int)ICPState::SlotState::FREE, std::memory_order_release);
            continue;
        }

        // Fill all bindings: input from slot.dTensor, outputs from devBindings
        std::vector<void*> bindings(st->gesture.nbBindings, nullptr);
        for (int b = 0; b < st->gesture.nbBindings; ++b) {
            if (b == st->gesture.inIdx) {
                bindings[b] = slot.dTensor; // input tensor (ROI preprocessed)
            } else {
                if (b >= 0 && b < (int)st->gesture.devBindings.size()) {
                    bindings[b] = st->gesture.devBindings[b];
                }
            }
        }

        // Sanity check for primary bindings
        if (!bindings[st->gesture.inIdx] || !bindings[st->gesture.outIdx]) {
            fprintf(stderr, "[gesture] bindings for inIdx(%d) or outIdx(%d) are null\n",
                    st->gesture.inIdx, st->gesture.outIdx);
            slot.state.store((int)ICPState::SlotState::FREE, std::memory_order_release);
            continue;
        }

        (void)cudaGetLastError();

        if (!st->gesture.context->enqueueV2(bindings.data(), st->trt_stream, nullptr)) {
            fprintf(stderr, "[gesture] enqueueV2 failed\n");
            slot.state.store((int)ICPState::SlotState::FREE, std::memory_order_release);
            continue;
        }

        const size_t dbytes = st->gesture.outElems *
                              (st->gesture.outputIsFP16 ? sizeof(__half) : sizeof(float));
        cudaError_t mrc = cudaMemcpyAsync(
            st->gesture.hostOutPinnedRaw,
            st->gesture.dOut,
            dbytes,
            cudaMemcpyDeviceToHost,
            st->trt_stream
        );
        if (mrc != cudaSuccess) {
            fprintf(stderr, "[gesture] cudaMemcpyAsync D2H failed: %d (%s)\n",
                    (int)mrc, cudaGetErrorString(mrc));
            slot.state.store((int)ICPState::SlotState::FREE, std::memory_order_release);
            continue;
        }

        cudaEventRecord(slot.ev_done, st->trt_stream);
        cudaEventRecord(st->gesture.ev_trt_done, st->trt_stream);
        slot.state.store((int)ICPState::SlotState::INFLIGHT, std::memory_order_release);
        break; // process one per frame for simplicity
    }

    // Try to free INFLIGHT slots that have completed
    for (int i = 0; i < ICPState::kSlots; ++i) {
        auto& slot = st->slots[i];
        if ((ICPState::SlotState)slot.state.load(std::memory_order_relaxed)
                == ICPState::SlotState::INFLIGHT) {
            cudaError_t q = cudaEventQuery(slot.ev_done);
            if (q == cudaSuccess) {
                slot.state.store((int)ICPState::SlotState::FREE,
                                 std::memory_order_release);
            } else if (q != cudaErrorNotReady && q != cudaSuccess) {
                fprintf(stderr, "[gesture] cudaEventQuery(slot.ev_done) error: %d (%s)\n",
                        (int)q, cudaGetErrorString(q));
            }
        }
    }
}

static void gpu_process(EGLImageKHR image, void **userPtr){
    ICPState* st = static_cast<ICPState*>(*userPtr);
    if (!st) { st = create_instance(); *userPtr = st; }
    cuCtxSetCurrent(st->ctx);

    if (!st->mask_checked_once) {
        st->mask_checked_once = true;
        (void)load_mask_from_cpu_once(st);
    }

    // Map EGLImage → CUDA (expect NV12, pitched)
    CUgraphicsResource res=nullptr;
    if(cuGraphicsEGLRegisterImage(&res,image,CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)!=CUDA_SUCCESS) return;
    CUeglFrame f{};
    if(cuGraphicsResourceGetMappedEglFrame(&f,res,0,0)!=CUDA_SUCCESS){
        cuGraphicsUnregisterResource(res);
        return;
    }
    if(f.frameType!=CU_EGL_FRAME_TYPE_PITCH || f.planeCount<2){
        cuGraphicsUnregisterResource(res);
        return;
    }

    uint8_t* dY  = static_cast<uint8_t*>(f.frame.pPitch[0]);
    uint8_t* dUV = static_cast<uint8_t*>(f.frame.pPitch[1]);
    const int W = (int)f.width;
    const int H = (int)f.height;
    const int pitch = (int)f.pitch;

    // Debug: log the first few frames to validate geometry
    static int dbg_frame = 0;
    if (dbg_frame < 20) {
        fprintf(stderr,
                "[gpu_process][%s] frame=%d size=%dx%d pitch=%d ptrY=%p ptrUV=%p\n",
                st->controls.section().c_str(),
                dbg_frame, W, H, pitch, (void*)dY, (void*)dUV);
        dbg_frame++;
    }

    ensure_scratch(st, W, H);

    // -------------------- Rectification geometry --------------------
    icp::RectifyConfig cfg = st->cfg;
    constexpr float REF_W_1080 = 1920.f, REF_H_1080 = 1080.f;
    const float sx_1080 = (float)W / REF_W_1080;
    const float sy_1080 = (float)H / REF_H_1080;
    const float cx_f = cfg.cx_f * sx_1080;
    const float cy_f = cfg.cy_f * sy_1080;
    const float r_f  = cfg.r_f  * 0.5f * (sx_1080 + sy_1080);

    constexpr float K_PX_PER_DEG_CENTER = 32.38f;
    const float f_fish = K_PX_PER_DEG_CENTER * (180.0f / (float)M_PI_F);
    const float fx = (W * 0.5f) / tanf(cfg.out_hfov_deg * (float)M_PI_F / 360.f);
    const float cx_rect = W * 0.5f;
    const float cy_rect = H * 0.5f;

    // 1) Rectification
    copy_to_scratch_async(st, dY, dUV, pitch, pitch, W, H);
    icp::launch_rectify_nv12(
        (const uint8_t*)(uintptr_t)st->sY,  W,H,(int)st->pY,
        (const uint8_t*)(uintptr_t)st->sUV,(int)st->pUV,
        dY, W,H,pitch,
        dUV, pitch,
        cx_f, cy_f, r_f,
        f_fish, fx, cx_rect, cy_rect,
        st->video_stream);

    // 2) Wire removal mask (if available)
    if (st->mask.valid && st->mask.W==W && st->mask.H==H){
        const int my_idx = section_to_index(st->controls.section());
        if (my_idx == st->mask.cam_index){
            wire::apply_mask_shift_nv12(
                dY, pitch, dUV, pitch, W, H,
                (const uint8_t*)(uintptr_t)st->mask.dMask, (int)st->mask.pitch,
                st->mask.dx, st->mask.dy, st->video_stream);
        }
    }

    // AI on/off gate
    const bool ai_on = st->controls.ai_enabled();

    // -------------------- ROI shared by AI + debug box --------------------
    // ROI in normalized coordinates, tuned to focus on the driver's hand.

    const float roi_w_frac = 0.42f;  // ~42% of frame width
    const float roi_h_frac = 0.58f;  // ~58% of frame height

    const float roi_cx_frac = 0.38f; // x center (0=left, 1=right)
    const float roi_cy_frac = 0.65f; // y center (0=top, 1=bottom)

    int roiW = static_cast<int>(roi_w_frac * W);
    int roiH = static_cast<int>(roi_h_frac * H);

    int cx = static_cast<int>(roi_cx_frac * W);
    int cy = static_cast<int>(roi_cy_frac * H);

    int roiX = cx - roiW / 2;
    int roiY = cy - roiH / 2;

    // Clamp ROI to frame boundaries
    if (roiX < 0) roiX = 0;
    if (roiY < 0) roiY = 0;
    if (roiX + roiW > W) roiW = W - roiX;
    if (roiY + roiH > H) roiH = H - roiY;

    // 2.x) Gesture pipeline setup (cam1 only) — using pose heatmaps
    if (ai_on) {
        ensure_gesture_loaded_for_cam1(st);
        if (st->gesture.engine && st->trt_stream && st->gesture_input_bytes > 0) {

            if (st->gesture.inputIsFP16) {
                static bool s_once = false;
                if (!s_once) {
                    fprintf(stderr,
                        "[AI-ERROR][%s] Engine input is FP16, but preprocess generates FP32. "
                        "You need either a FP32 engine or a FP32->FP16 conversion path.\n",
                        st->controls.section().c_str());
                    s_once = true;
                }
            } else {
                // Acquire a free slot in the device FIFO
                int slot_idx = acquire_free_slot(st);
                auto &slot = st->slots[slot_idx];

                slot.state.store((int)ICPState::SlotState::FREE, std::memory_order_relaxed);

                // Preprocess NV12 ROI → CHW FP32 (1x3x244x244) into dTensor
                ei::preprocess_nv12_roi_to_chw_fp32(
                    dY, dUV,
                    W, H, pitch,
                    roiX, roiY, roiW, roiH,
                    reinterpret_cast<float*>(slot.dTensor),
                    st->video_stream
                );

                // Signal that the slot is READY for TRT
                cudaEventRecord(slot.ev_ready, st->video_stream);
                slot.state.store((int)ICPState::SlotState::READY, std::memory_order_release);
                st->prod_idx = (slot_idx + 1) % ICPState::kSlots;

                // Trigger TRT on READY slots (async on trt_stream)
                kick_trt_for_ready_slots(st);
            }
        }
    }

    // 3) Tone + color (hot-reload)
    icp::ColorParams cp = st->controls.current();
    icp::launch_tone_saturation_nv12(dY, W, H, pitch, dUV, pitch, cp, st->video_stream);

    // 4) Debug: draw ROI box on cam1 to visualize the region used for hand pose
    {
        if (st->controls.section() == std::string("cam1")) {
            draw::launch_draw_box_nv12(
                dY, dUV,
                W, H, pitch,
                roiX, roiY, roiW, roiH,
                st->video_stream
            );
        }
    }

    // Fence VIDEO work (rectify, wire removal, tone, ROI box)
    cudaStreamSynchronize(st->video_stream);

    // ---- Hand pose visualization: decode heatmaps and draw keypoints ----
    if (ai_on && st->gesture.engine && st->gesture.try_commit_host_output()) {
        int C = 0, hmH = 0, hmW = 0;
        if (!st->gesture.get_heatmap_shape(C, hmH, hmW)) {
            fprintf(stderr, "[gesture][%s] invalid heatmap shape\n",
                    st->controls.section().c_str());
        } else {
            std::vector<trt::Keypoint2D> kpts;
            if (st->gesture.decode_argmax_keypoints(kpts)) {
                // Map keypoints from heatmap space → ROI → image space
                std::vector<draw::Point2D> pts;
                pts.reserve(kpts.size());

                for (int i = 0; i < C; ++i) {
                    const auto& kp = kpts[i];

                    // Convert (u,v) in heatmap to normalized [0,1] within the ROI
                    float nx = (kp.u + 0.5f) / (float)hmW;
                    float ny = (kp.v + 0.5f) / (float)hmH;

                    // Clamp normalized coordinates
                    if (nx < 0.0f) nx = 0.0f;
                    if (nx > 1.0f) nx = 1.0f;
                    if (ny < 0.0f) ny = 0.0f;
                    if (ny > 1.0f) ny = 1.0f;

                    int x_img = roiX + (int)(nx * roiW);
                    int y_img = roiY + (int)(ny * roiH);

                    draw::Point2D p;
                    p.x    = x_img;
                    p.y    = y_img;
                    p.conf = 1.0f; // you can wire confidence if you extend Keypoint2D
                    pts.push_back(p);
                }

                // --- Gesture classification: PEACE (✌️) vs NULL ---
                gesture::Result gres = gesture::classify_peace(pts);
                const float kPeaceScoreThresh = 0.8f;

                bool isPeace =
                    (gres.type == gesture::Type::PEACE && gres.score >= kPeaceScoreThresh);

                if (isPeace) {
                    fprintf(stderr, "[GESTURE] PEACE (score=%.2f)\n", gres.score);
                } else {
                    fprintf(stderr, "[GESTURE] NULL (score=%.2f)\n", gres.score);
                }

                if (!pts.empty() && st->controls.section() == std::string("cam1")) {
                    // Draw keypoints + simple skeleton (dots + lines) on NV12
                    // PEACE → green; otherwise white
                    draw::launch_draw_hand_points_nv12(
                        dY, dUV,
                        W, H, pitch,
                        pts.data(), (int)pts.size(),
                        isPeace,
                        st->video_stream
                    );
                    // Synchronize to make sure overlay is visible before unmapping
                    cudaStreamSynchronize(st->video_stream);
                }
            }
        }
    }

    cuGraphicsUnregisterResource(res);
}

// ----------------------------------------------------------------------------
// init / deinit
// ----------------------------------------------------------------------------
extern "C" void init(CustomerFunction* f){
    if(!f) return;
    f->fPreProcess=pre_process;
    f->fGPUProcess=gpu_process;
    f->fPostProcess=post_process;
    std::call_once(g_ctx_once, retain_primary_context_once);
    fprintf(stderr,
        "[ic] imagecorrection initialized (hand pose overlay + peace gesture on cam1; "
        "decoupled TRT stream + ROI preprocess)\n");
}

extern "C" void deinit(void){
    std::vector<ICPState*> to_free;
    {
        std::lock_guard<std::mutex> lk(g_instances_mtx);
        to_free.swap(g_instances);
    }
    for (auto* st : to_free) destroy_instance(st);
    release_primary_context();
    ic_clear_instance_hints();
}
