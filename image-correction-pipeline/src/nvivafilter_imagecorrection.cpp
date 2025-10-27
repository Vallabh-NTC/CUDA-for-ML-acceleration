/**
 * @file nvivafilter_imagecorrection.cpp
 * @brief GStreamer `nvivafilter` plugin: fisheye rectification → (optional) wire inpaint
 *        → gesture inference (cam1 only) → tone/color.
 *
 * Notes (debug build):
 *  - NV12 in/out (pitched), CUDA primary context shared across instances.
 *  - Runtime color controls hot-reload from JSON section "cam0|cam1|cam2".
 *  - Per-instance wire mask loaded once from /dev/shm/wire_mask_camN.{raw,meta}.
 *  - Gesture inference (TensorRT) runs ONLY for the cam1-bound instance.
 *  - Robust TRT output dtype handling (FP16/FP32) + env-driven debugging.
 *
 * Env toggles:
 *   TRT_TV_RANGE=0/1     : 0 = full-range 0..255 → 0..1 (default, like Edge Impulse)
 *                          1 = TV range 16..235 → 0..1
 *   TRT_LABEL_MAP="start=1,stop=2,ok=0"  : override label indices without rebuild
 *   TRT_DUMP_EI_RAW=1    : print EI-order raw vector (ok,start,stop) per frame
 *
 * MQTT / FSM (questa integrazione):
 *   MQTT_URL=tcp://HOST:PORT   (oppure MQTT_HOST / MQTT_PORT)
 *   MQTT_TOPIC=jetson/stream/cmd
 *   MQTT_USER=...  MQTT_PASS=...
 *   GESTURE_TH_START=0.80  GESTURE_TH_STOP=0.80  GESTURE_TH_OK=0.80
 *   GESTURE_HOLD_MS=1200   (durata richiesta per ciascun passo)
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
constexpr float M_PI_F = static_cast<float>(M_PI);

// Path to the TensorRT engine
static const char* kGestureEnginePath = "/home/moviemaker/gesture_recog.engine";

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

// Determine "cam0|cam1|cam2" from the loaded .so name (or a parent folder)
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
// MQTT helpers (shell out to mosquitto_pub, config via env)
// ============================================================================
struct MqttCfg {
    std::string host;
    int         port   = 1883;
    std::string topic  = "jetson/stream/cmd";
    std::string user;
    std::string pass;
    bool        valid  = false;
};

static std::string shell_quote(const std::string& s){
    std::string o; o.reserve(s.size()+8); o.push_back('\'');
    for(char c: s){ if(c=='\'') o += "'\"'\"'"; else o.push_back(c); } o.push_back('\''); return o;
}

static void parse_mqtt_env(MqttCfg& out){
    const char* url = std::getenv("MQTT_URL");
    if (url && *url) {
        // tcp://host:port
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
    if (const char* p = std::getenv("MQTT_PORT"); p && *p){ try{ out.port = std::stoi(p);}catch(...){} }
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
// Per-instance state
// ============================================================================
struct ICPState {
    CUcontext     ctx   = nullptr;
    cudaStream_t  stream= nullptr;

    // Scratch (NV12) for rectification
    CUdeviceptr sY = 0, sUV = 0;
    size_t      pY = 0, pUV = 0;
    int         sW = 0, sH = 0;

    // Geometry (fisheye rectification)
    icp::RectifyConfig cfg{};

    // Runtime controls (hot-reload)
    icp::RuntimeControls controls{"/home/moviemaker/config.json", "cam0"};

    // Post-rectification center crop/zoom (declared in headers, not implemented here)
    float crop_frac = 0.0f;

    // Per-instance, device-resident mask loaded once from CPU
    struct DeviceMask {
        CUdeviceptr dMask = 0;   // 8-bit mask on device
        size_t      pitch = 0;   // bytes per row for mask
        int         W = 0, H = 0;
        float       dx = 0.f, dy = 0.f;
        bool        valid = false;
        int         cam_index = -1; // 0/1/2 corresponding to the files used
    } mask{};
    bool mask_checked_once = false;

    // Gesture NN (TensorRT), only used on cam1
    trt::Engine gesture;
    bool gesture_loaded_once = false;

    // MQTT + FSM
    MqttCfg mqtt{};
    bool mqtt_checked_once = false;

    struct {
        // config
        float th_start = 0.80f;
        float th_stop  = 0.80f;
        float th_ok    = 0.80f;
        int   hold_ms  = 1200;

        // state machine
        enum class State { IDLE, AWAIT_OK_FOR_START, RECORDING, AWAIT_OK_FOR_STOP } st = State::IDLE;
        std::chrono::steady_clock::time_point t_last = std::chrono::steady_clock::now();

        int hold_start_ms = 0;
        int hold_stop_ms  = 0;
        int hold_ok_ms    = 0;
        int ok_after_gate_ms = 0; // timer while waiting the second step

        // helpers
        void reset_holds(){ hold_start_ms=hold_stop_ms=hold_ok_ms=0; }
        void tick_delta_ms(int dms, float ps, float pt, float pok){
            // Update per-class holds
            if (ps >= th_start) hold_start_ms += dms; else hold_start_ms = 0;
            if (pt >= th_stop ) hold_stop_ms  += dms; else hold_stop_ms  = 0;
            if (pok>= th_ok   ) hold_ok_ms    += dms; else hold_ok_ms    = 0;

            // Gate timer for the second step
            ok_after_gate_ms = (st==State::AWAIT_OK_FOR_START || st==State::AWAIT_OK_FOR_STOP) ? (ok_after_gate_ms + dms) : 0;
        }
    } fsm;
};

static std::mutex              g_instances_mtx;
static std::vector<ICPState*>  g_instances;

// Per-instance section hints (consumed by create_instance)
static std::mutex              g_hint_mtx;
static std::deque<std::string> g_next_section_hints;

// ---- OPTIONAL host hint API -------------------------------------------------
extern "C" void ic_bind_next_instance_to(const char* section /* "cam0|cam1|cam2" */) {
    if (!section || !*section) return;
    std::string s = to_lower(section);
    if (s != "cam0" && s != "cam1" && s != "cam2") return;
    std::lock_guard<std::mutex> lk(g_hint_mtx);
    g_next_section_hints.push_back(std::move(s));
}

extern "C" void ic_clear_instance_hints() {
    std::lock_guard<std::mutex> lk(g_hint_mtx);
    g_next_section_hints.clear();
}

extern "C" int ic_has_instance_for(const char* section /* "cam0|cam1|cam2" */) {
    if (!section || !*section) return 0;
    std::lock_guard<std::mutex> lk(g_instances_mtx);
    for (auto* st : g_instances) {
        if (!st) continue;
        if (st->controls.section() == section) return 1;
    }
    return 0;
}
// -----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// Alloc helpers
// ----------------------------------------------------------------------------
static void ensure_scratch(ICPState* st, int W, int H)
{
    if (st->sY && st->sUV && st->sW == W && st->sH == H) return;

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

// ----------------------------------------------------------------------------
// Mask loader: read once from CPU file(s) and upload to device
// ----------------------------------------------------------------------------
static bool load_mask_from_cpu_once(ICPState* st)
{
    const std::string sec = st->controls.section();
    const int my_idx = section_to_index(sec);
    if (my_idx < 0) {
        fprintf(stderr, "[ic] mask: unknown section '%s'\n", sec.c_str());
        return false;
    }

    const std::string meta = mask_meta_path_for(my_idx);
    const std::string raw  = mask_raw_path_for(my_idx);

    // Read META: "W H dx dy"
    FILE* fm = fopen(meta.c_str(), "r");
    if (!fm) {
        fprintf(stderr, "[ic] mask: meta not found for %s (skipping)\n", sec.c_str());
        return false;
    }
    int W=0, H=0; float dx=0.f, dy=0.f;
    int n = fscanf(fm, "%d %d %f %f", &W, &H, &dx, &dy);
    fclose(fm);
    if (n != 4 || W<=0 || H<=0) {
        fprintf(stderr, "[ic] mask: bad meta format for %s (skipping)\n", sec.c_str());
        return false;
    }

    // Read RAW: W*H bytes
    const size_t bytes = (size_t)W * (size_t)H;
    std::vector<uint8_t> hostMask(bytes);
    FILE* fr = fopen(raw.c_str(), "rb");
    if (!fr) {
        fprintf(stderr, "[ic] mask: raw not found for %s (skipping)\n", sec.c_str());
        return false;
    }
    size_t rd = fread(hostMask.data(), 1, bytes, fr);
    fclose(fr);
    if (rd != bytes) {
        fprintf(stderr, "[ic] mask: raw size mismatch for %s (%zu vs %zu)\n",
                sec.c_str(), rd, bytes);
        return false;
    }

    // Allocate device mask (pitched) if needed
    if (!st->mask.dMask || st->mask.W != W || st->mask.H != H) {
        if (st->mask.dMask) { cuMemFree(st->mask.dMask); st->mask.dMask = 0; }
        CUdeviceptr dptr = 0; size_t pitch = 0;
        CUresult rc = cuMemAllocPitch(&dptr, &pitch, (size_t)W, (size_t)H, 4);
        if (rc != CUDA_SUCCESS) {
            fprintf(stderr, "[ic] mask: cuMemAllocPitch failed (%d)\n", (int)rc);
            return false;
        }
        st->mask.dMask = dptr;
        st->mask.pitch = pitch;
        st->mask.W = W; st->mask.H = H;
    }

    // Upload H2D
    CUDA_MEMCPY2D c{};
    c.srcMemoryType = CU_MEMORYTYPE_HOST;   c.srcHost   = hostMask.data(); c.srcPitch = (size_t)W;
    c.dstMemoryType = CU_MEMORYTYPE_DEVICE; c.dstDevice = st->mask.dMask;  c.dstPitch = st->mask.pitch;
    c.WidthInBytes  = (size_t)W; c.Height = (size_t)H;
    cuMemcpy2D(&c);

    // Store meta and mark valid
    st->mask.dx = dx; st->mask.dy = dy;
    st->mask.cam_index = my_idx;
    st->mask.valid = true;

    fprintf(stderr, "[ic] mask: loaded for %s (W=%d H=%d, dx=%.2f dy=%.2f)\n",
            sec.c_str(), W, H, dx, dy);
    return true;
}

// ----------------------------------------------------------------------------
// Gesture helper: lazy-load engine only for cam1
// ----------------------------------------------------------------------------
static void ensure_gesture_loaded_for_cam1(ICPState* st)
{
    if (st->gesture_loaded_once) return;
    st->gesture_loaded_once = true;

    if (st->controls.section() != std::string("cam1")) {
        fprintf(stderr, "[ic] gesture: skipped (section is '%s', only cam1 runs it)\n",
                st->controls.section().c_str());
        return;
    }

    if (!st->gesture.load_from_file(kGestureEnginePath, st->stream)) {
        fprintf(stderr, "[ic] gesture: load failed (path=%s)\n", kGestureEnginePath);
    } else {
        fprintf(stderr, "[ic] gesture: ready (path=%s)\n", kGestureEnginePath);
    }
}

// ----------------------------------------------------------------------------
// FSM helper: init MQTT + thresholds from env
// ----------------------------------------------------------------------------
static void ensure_mqtt_and_fsm_config(ICPState* st){
    if (!st->mqtt_checked_once){
        st->mqtt_checked_once = true;
        parse_mqtt_env(st->mqtt);

        if (const char* v = std::getenv("GESTURE_TH_START"); v && *v) st->fsm.th_start = std::max(0.f, std::min(1.f, (float)atof(v)));
        if (const char* v = std::getenv("GESTURE_TH_STOP" ); v && *v) st->fsm.th_stop  = std::max(0.f, std::min(1.f, (float)atof(v)));
        if (const char* v = std::getenv("GESTURE_TH_OK"   ); v && *v) st->fsm.th_ok    = std::max(0.f, std::min(1.f, (float)atof(v)));
        if (const char* v = std::getenv("GESTURE_HOLD_MS" ); v && *v) st->fsm.hold_ms  = std::max(100, atoi(v));

        fprintf(stderr, "[FSM] th_start=%.2f th_stop=%.2f th_ok=%.2f hold_ms=%d mqtt=%s:%d topic=%s\n",
            st->fsm.th_start, st->fsm.th_stop, st->fsm.th_ok, st->fsm.hold_ms,
            st->mqtt.host.c_str(), st->mqtt.port, st->mqtt.topic.c_str());
    }
}

// ----------------------------------------------------------------------------
// Instance allocation / destruction
// ----------------------------------------------------------------------------
static ICPState* create_instance()
{
    std::call_once(g_ctx_once, retain_primary_context_once);

    auto* st = new ICPState();
    st->ctx = g_primary_ctx;
    cuCtxSetCurrent(st->ctx);

#if CUDART_VERSION >= 11000
    cudaStreamCreateWithPriority(&st->stream, cudaStreamNonBlocking, 0);
#else
    cudaStreamCreateWithFlags(&st->stream, cudaStreamNonBlocking);
#endif

    // Consume next hint if present; otherwise fallback to SONAME/path
    std::string sec;
    {
        std::lock_guard<std::mutex> lk(g_hint_mtx);
        if (!g_next_section_hints.empty()) {
            sec = std::move(g_next_section_hints.front());
            g_next_section_hints.pop_front();
        }
    }
    if (sec.empty())
        sec = section_from_loaded_name();

    st->controls.set_section(sec);
    fprintf(stderr, "[ic] Instance bound to section '%s'\n", sec.c_str());

    {
        std::lock_guard<std::mutex> lk(g_instances_mtx);
        g_instances.push_back(st);
    }
    return st;
}

static void destroy_instance(ICPState* st)
{
    if (!st) return;
    cuCtxSetCurrent(st->ctx);

    if (st->sY)  { cuMemFree(st->sY);  st->sY  = 0; }
    if (st->sUV) { cuMemFree(st->sUV); st->sUV = 0; }

    if (st->mask.dMask) { cuMemFree(st->mask.dMask); st->mask.dMask = 0; }
    st->mask = {};

    st->gesture.destroy();

    if (st->stream) { cudaStreamDestroy(st->stream); st->stream = nullptr; }
    st->ctx = nullptr;
    delete st;
}

// ----------------------------------------------------------------------------
// nvivafilter hooks
// ----------------------------------------------------------------------------
static void pre_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*, ColorFormat*, unsigned int, void **){}
static void post_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*, ColorFormat*, unsigned int, void **){}

static void gpu_process(EGLImageKHR image, void **userPtr)
{
    // Lazily create per-instance state
    ICPState* st = static_cast<ICPState*>(*userPtr);
    if (!st) {
        st = create_instance();
        *userPtr = st;
    }
    cuCtxSetCurrent(st->ctx);

    // Load the per-instance mask once (if present on CPU)
    if (!st->mask_checked_once) {
        st->mask_checked_once = true;
        (void)load_mask_from_cpu_once(st); // soft-fail: simply no mask applied
    }

    // Map EGLImage → CUDA (expect NV12, pitched)
    CUgraphicsResource res=nullptr;
    if(cuGraphicsEGLRegisterImage(&res,image,CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)!=CUDA_SUCCESS) return;
    CUeglFrame f{};
    if(cuGraphicsResourceGetMappedEglFrame(&f,res,0,0)!=CUDA_SUCCESS){ cuGraphicsUnregisterResource(res); return; }
    if(f.frameType!=CU_EGL_FRAME_TYPE_PITCH || f.planeCount<2){ cuGraphicsUnregisterResource(res); return; }

    uint8_t* dY  = static_cast<uint8_t*>(f.frame.pPitch[0]);
    uint8_t* dUV = static_cast<uint8_t*>(f.frame.pPitch[1]);
    const int W = (int)f.width, H=(int)f.height, pitch=(int)f.pitch;

    // Scratch for rectification textures
    ensure_scratch(st, W, H);

    // -------------------- Rectification geometry --------------------
    icp::RectifyConfig cfg = st->cfg;

    // Scale fisheye circle from 1080p reference → runtime
    constexpr float REF_W_1080 = 1920.f;
    constexpr float REF_H_1080 = 1080.f;
    const float sx_1080 = (float)W / REF_W_1080;
    const float sy_1080 = (float)H / REF_H_1080;

    const float cx_f = cfg.cx_f * sx_1080;
    const float cy_f = cfg.cy_f * sy_1080;
    const float r_f  = cfg.r_f  * 0.5f * (sx_1080 + sy_1080);

    // Equidistant focal (px/rad) using 4K px/deg center value
    constexpr float K_PX_PER_DEG_CENTER = 32.38f;
    const float f_fish = K_PX_PER_DEG_CENTER * (180.0f / (float)M_PI_F);

    // Perspective intrinsics for the rectified target HFOV
    const float fx      = (W * 0.5f) / tanf(cfg.out_hfov_deg * (float)M_PI_F / 360.f);
    const float cx_rect = W * 0.5f;
    const float cy_rect = H * 0.5f;

    // 1) Rectification (src → dst in-place)
    copy_to_scratch_async(st, dY, dUV, pitch, pitch, W, H);
    icp::launch_rectify_nv12(
        (const uint8_t*)(uintptr_t)st->sY,  W,H,(int)st->pY,
        (const uint8_t*)(uintptr_t)st->sUV, (int)st->pUV,
        dY, W,H,pitch,
        dUV, pitch,
        cx_f, cy_f, r_f,
        f_fish, fx, cx_rect, cy_rect,
        st->stream);

    // 2) Wire removal (two-donor inpaint) — only if mask is valid and belongs to this instance
    if (st->mask.valid && st->mask.W == W && st->mask.H == H) {
        const int my_idx = section_to_index(st->controls.section());
        if (my_idx == st->mask.cam_index) {
            wire::apply_mask_shift_nv12(
                dY,  pitch,
                dUV, pitch,
                W, H,
                (const uint8_t*)(uintptr_t)st->mask.dMask, (int)st->mask.pitch,
                st->mask.dx, st->mask.dy,
                st->stream);
        }
    }

    // 2.8) Gesture inference (ONLY for cam1)
    ensure_gesture_loaded_for_cam1(st);
    if (st->gesture.engine) { // engine loaded only when section == cam1
        // Read TV/full-range from env (default full-range like Edge Impulse)
        bool tv_range = false;
        if (const char* e = std::getenv("TRT_TV_RANGE")) {
            tv_range = (*e == '1');
        }

        // Preprocess NV12 Y → NCHW 1x1x96x96 into TRT input buffer
        if (!ei::enqueue_preprocess_to_trt_input(dY, W, H, pitch,
                                                 st->gesture.dIn,
                                                 st->gesture.inputIsFP16,
                                                 tv_range,
                                                 st->stream)) {
            fprintf(stderr, "[gesture] preprocess enqueue failed\n");
        } else {
            void* bindings[2];
            bindings[st->gesture.inIdx]  = st->gesture.dIn;
            bindings[st->gesture.outIdx] = st->gesture.dOut;

            if (!st->gesture.context->enqueueV2(bindings, st->stream, nullptr)) {
                fprintf(stderr, "[gesture] enqueueV2 failed\n");
            } else {
                // Async D2H into pinned buffer that matches device dtype
                const size_t dbytes = st->gesture.outElems *
                                      (st->gesture.outputIsFP16 ? sizeof(__half) : sizeof(float));
                cudaMemcpyAsync(st->gesture.hostOutPinnedRaw, st->gesture.dOut,
                                dbytes, cudaMemcpyDeviceToHost, st->stream);
                cudaEventRecord(st->gesture.ev_trt_done, st->stream);
            }
        }
    }

    // 3) Tone + color (hot-reload)
    icp::ColorParams cp = st->controls.current();
    icp::launch_tone_saturation_nv12(
        dY, W, H, pitch,
        dUV,     pitch,
        cp,
        st->stream);

    // Fence the per-frame GPU work
    cudaStreamSynchronize(st->stream);

    // ---- Emit gesture logs + drive FSM if a fresh result is available ----
    if (st->gesture.engine && st->gesture.try_commit_host_output()) {
        ensure_mqtt_and_fsm_config(st);

        // (A) EI-order raw dump (true tensor order: ok=0, start=1, stop=2)
        if (const char* dumpEI = std::getenv("TRT_DUMP_EI_RAW"); dumpEI && *dumpEI=='1') {
            const float v_ok    = st->gesture.hostOut[0];
            const float v_start = st->gesture.hostOut[1];
            const float v_stop  = st->gesture.hostOut[2];
            fprintf(stderr, "[gesture][%s] EI-order raw: ok=%.3f start=%.3f stop=%.3f | sum=%.3f\n",
                    st->controls.section().c_str(), v_ok, v_start, v_stop, v_ok+v_start+v_stop);
        }

        // (B) Pretty print using logical mapping (START/STOP/OK)
        float sL=0.f, tL=0.f, okL=0.f, ps=0.f, pt=0.f, pok=0.f; int top=-1;
        if (st->gesture.get_start_stop_ok(sL, tL, okL, ps, pt, pok, top)) {
            const char* top_lbl = (top==0) ? "START" : (top==1 ? "STOP" : "OK");
            fprintf(stderr,
                "[gesture][%s] logits/scores: start=%.3f stop=%.3f ok=%.3f | p: start=%.3f stop=%.3f ok=%.3f | top=%s\n",
                st->controls.section().c_str(), sL, tL, okL, ps, pt, pok, top_lbl);

            // --------- FSM update ----------
            using clk = std::chrono::steady_clock;
            auto now = clk::now();
            int dms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(now - st->fsm.t_last).count();
            if (dms < 0 || dms > 1000) dms = 0; // guard
            st->fsm.t_last = now;

            st->fsm.tick_delta_ms(dms, ps, pt, pok);

            auto &fsm = st->fsm;

            switch (fsm.st) {
                case decltype(fsm.st)::IDLE:
                    // Need START ≥ hold_ms, then OK ≥ hold_ms
                    if (fsm.hold_start_ms >= fsm.hold_ms) {
                        fsm.st = decltype(fsm.st)::AWAIT_OK_FOR_START;
                        fsm.hold_ok_ms = 0;
                        fsm.ok_after_gate_ms = 0;
                        fprintf(stderr, "[FSM] START qualified; awaiting OK...\n");
                    }
                    break;

                case decltype(fsm.st)::AWAIT_OK_FOR_START:
                    // OK must be held ≥ hold_ms after START was qualified
                    if (fsm.hold_ok_ms >= fsm.hold_ms) {
                        // publish START
                        mqtt_publish(st->mqtt, R"({"value":{"recording":"start"}})");
                        fprintf(stderr, "[TRIGGER] recording START\n");
                        fsm.st = decltype(fsm.st)::RECORDING;
                        fsm.reset_holds();
                    } else {
                        // optional timeout to reset if nothing happens for a while
                        if (fsm.ok_after_gate_ms > 4000) {
                            fprintf(stderr, "[FSM] timeout waiting OK after START; reset\n");
                            fsm.st = decltype(fsm.st)::IDLE;
                            fsm.reset_holds();
                        }
                    }
                    break;

                case decltype(fsm.st)::RECORDING:
                    // Need STOP ≥ hold_ms, then OK ≥ hold_ms
                    if (fsm.hold_stop_ms >= fsm.hold_ms) {
                        fsm.st = decltype(fsm.st)::AWAIT_OK_FOR_STOP;
                        fsm.hold_ok_ms = 0;
                        fsm.ok_after_gate_ms = 0;
                        fprintf(stderr, "[FSM] STOP qualified; awaiting OK...\n");
                    }
                    break;

                case decltype(fsm.st)::AWAIT_OK_FOR_STOP:
                    if (fsm.hold_ok_ms >= fsm.hold_ms) {
                        // publish STOP
                        mqtt_publish(st->mqtt, R"({"value":{"recording":"stop"}})");
                        fprintf(stderr, "[TRIGGER] recording STOP\n");
                        fsm.st = decltype(fsm.st)::IDLE;
                        fsm.reset_holds();
                    } else {
                        if (fsm.ok_after_gate_ms > 4000) {
                            fprintf(stderr, "[FSM] timeout waiting OK after STOP; back to RECORDING\n");
                            fsm.st = decltype(fsm.st)::RECORDING;
                            fsm.hold_stop_ms = 0;
                            fsm.hold_ok_ms   = 0;
                        }
                    }
                    break;
            }

        } else {
            float p=0.f; int cls = st->gesture.top1(&p);
            fprintf(stderr, "[gesture][%s] top1=%d prob=%.3f\n",
                    st->controls.section().c_str(), cls, p);
        }
    }

    cuGraphicsUnregisterResource(res);
}

// ----------------------------------------------------------------------------
// init / deinit
// ----------------------------------------------------------------------------
extern "C" void init(CustomerFunction* f)
{
    if(!f) return;
    f->fPreProcess  = pre_process;
    f->fGPUProcess  = gpu_process;
    f->fPostProcess = post_process;

    // Ensure CUDA primary context retained
    std::call_once(g_ctx_once, retain_primary_context_once);

    fprintf(stderr, "[ic] imagecorrection initialized (gesture runs only on cam1)\n");
}

extern "C" void deinit(void)
{
    std::vector<ICPState*> to_free;
    {
        std::lock_guard<std::mutex> lk(g_instances_mtx);
        to_free.swap(g_instances);
    }
    for (auto* st : to_free) destroy_instance(st);
    release_primary_context();

    // Clear any residual instance-binding hints
    ic_clear_instance_hints();
}
