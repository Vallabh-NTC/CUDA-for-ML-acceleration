/**
 * @file nvivafilter_imagecorrection.cpp
 * @brief GStreamer `nvivafilter` plugin: fisheye rectification → crop → (optional) wire removal inpaint → tone/color.
 *
 * Robust control:
 *   - Set WIRE_ENABLE=1 to enable wire removal + authoring server in this process.
 *   - Set WIRE_ENABLE=0 (default) to disable both (no port bind, no inpaint).
 *   - Set WIRE_PORT=NNNN to choose TCP port (default 5555).
 *
 * Typical usage:
 *   # Camera you want to author/remove wire on:
 *   WIRE_ENABLE=1 WIRE_PORT=5555 gst-launch-1.0 ... nvivafilter customer-lib-name=/usr/local/lib/nvivafilter/libnvivafilter_imagecorrection.so ...
 *
 *   # Other cameras (no wire removal, no bind conflicts):
 *   WIRE_ENABLE=0 gst-launch-1.0 ... nvivafilter customer-lib-name=/usr/local/lib/nvivafilter/libnvivafilter_imagecorrection.so ...
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
#include <dlfcn.h>     // dladdr
#include <cctype>      // std::tolower
#include <atomic>

#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>

#include "nvivafilter_customer_api.hpp"
#include "rectify_config.hpp"
#include "kernel_rectify.cuh"
#include "color_ops.cuh"
#include "runtime_controls.hpp"

// Authoring server (snapshot + device upload).
#include "wire_author_server.cuh"

// Two-donor NV12 inpaint (mask, dx, dy)
#include "wire_lineremoval.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
constexpr float M_PI_F = static_cast<float>(M_PI);

// ============================================================================
// Global CUDA primary context (shared across instances)
// ============================================================================
namespace {
    static std::once_flag g_ctx_once;
    static CUcontext g_primary_ctx = nullptr;
    static CUdevice  g_device      = 0;

    static std::atomic<bool> g_wire_enabled{false};  // process-wide enable
    static uint16_t          g_wire_port = 5555;     // process-wide port (if enabled)

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

// Determine "cam0|cam1|cam2" from the loaded .so name
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

static bool env_flag_true(const char* v) {
    if (!v) return false;
    std::string s = to_lower(v);
    return (s=="1" || s=="true" || s=="yes" || s=="on" || s=="y");
}

// ============================================================================
// Per-instance state
// ============================================================================
struct ICPState {
    CUcontext     ctx   = nullptr;
    cudaStream_t  stream= nullptr;

    // Scratch (NV12)
    CUdeviceptr sY = 0, sUV = 0;
    size_t      pY = 0, pUV = 0;
    int         sW = 0, sH = 0;

    // Geometry (fisheye rectification)
    icp::RectifyConfig cfg{};

    // Runtime controls (hot-reload) bound to section decided at create_instance()
    icp::RuntimeControls controls{"/home/jetson_ntc/config.json", "cam0"};

    // Post-rectification center crop/zoom
    float crop_frac = 0.20f;

    // Track last applied mask version (optional, for logging)
    uint32_t last_mask_version = 0;
};

static std::mutex              g_instances_mtx;
static std::vector<ICPState*>  g_instances;

// --- NEW: per-instance section hints (queue consumed by create_instance) ----
static std::mutex              g_hint_mtx;
static std::deque<std::string> g_next_section_hints;

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

    // NEW: consume next hint if present; otherwise fallback to SONAME/path
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

    // Map EGLImage → CUDA
    CUgraphicsResource res=nullptr;
    if(cuGraphicsEGLRegisterImage(&res,image,CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)!=CUDA_SUCCESS) return;
    CUeglFrame f{};
    if(cuGraphicsResourceGetMappedEglFrame(&f,res,0,0)!=CUDA_SUCCESS){ cuGraphicsUnregisterResource(res); return; }
    if(f.frameType!=CU_EGL_FRAME_TYPE_PITCH || f.planeCount<2){ cuGraphicsUnregisterResource(res); return; }

    uint8_t* dY  = static_cast<uint8_t*>(f.frame.pPitch[0]);
    uint8_t* dUV = static_cast<uint8_t*>(f.frame.pPitch[1]);
    const int W = (int)f.width, H=(int)f.height, pitch=(int)f.pitch;

    // Scratch
    ensure_scratch(st, W, H);

    // Geometry
    icp::RectifyConfig cfg = st->cfg;

    const float FOV_fish = cfg.fish_fov_deg * (float)M_PI_F / 180.f;
    const float f_fish   = cfg.r_f / (FOV_fish * 0.5f);
    const float fx       = (W * 0.5f) / tanf(cfg.out_hfov_deg * (float)M_PI_F / 360.f);
    const float cx_rect  = W * 0.5f;
    const float cy_rect  = H * 0.5f;

    // 1) Rectification
    copy_to_scratch_async(st, dY, dUV, pitch, pitch, W, H);
    icp::launch_rectify_nv12(
        (const uint8_t*)(uintptr_t)st->sY,  W,H,(int)st->pY,
        (const uint8_t*)(uintptr_t)st->sUV, (int)st->pUV,
        dY, W,H,pitch,
        dUV, pitch,
        cfg.cx_f, cfg.cy_f, cfg.r_f,
        f_fish, fx, cx_rect, cy_rect,
        st->stream);

    // 2) Center crop/zoom
    copy_to_scratch_async(st, dY, dUV, pitch, pitch, W, H);
    icp::launch_crop_center_nv12(
        (const uint8_t*)(uintptr_t)st->sY,  W,H,(int)st->pY,
        (const uint8_t*)(uintptr_t)st->sUV, (int)st->pUV,
        dY,                                   pitch,
        dUV,                                  pitch,
        st->crop_frac,                        st->stream);

    // 2.5) Wire removal (two-donor inpaint) — only if WIRE_ENABLE=1 AND a mask is active.
    if (g_wire_enabled.load(std::memory_order_relaxed)) {
        CUdeviceptr dMask = 0;
        int         maskPitch = 0;
        float       dx = 0.f, dy = 0.f;
        uint32_t    ver = 0;
        if (wire_author_get_active_mask(&dMask, &maskPitch, &dx, &dy, &ver)) {
            wire::apply_mask_shift_nv12(
                dY,  pitch,
                dUV, pitch,
                W, H,
                (const uint8_t*)(uintptr_t)dMask, maskPitch,
                dx, dy,
                st->stream);
            if (ver != st->last_mask_version) {
                fprintf(stderr, "[ic] applied device mask (version=%u, dx=%.2f, dy=%.2f)\n", ver, dx, dy);
                st->last_mask_version = ver;
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

    // 4) Optional snapshot (only if authoring is enabled; otherwise skip)
    if (g_wire_enabled.load(std::memory_order_relaxed)) {
        wire_author_snapshot_nv12_if_needed(
            dY,  pitch,
            dUV, pitch,
            W, H, st->stream);
    }

    cudaStreamSynchronize(st->stream);
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

    // Read process-wide enable flag and port from environment.
    // Default: disabled (no bind, no inpaint).
    bool enable = env_flag_true(std::getenv("WIRE_ENABLE"));
    g_wire_enabled.store(enable, std::memory_order_relaxed);

    uint16_t port = 5555;
    if (const char* p = std::getenv("WIRE_PORT")) {
        int v = atoi(p); if (v>0 && v<65536) port = (uint16_t)v;
    }
    g_wire_port = port;

    if (enable) {
        wire_author_start(g_primary_ctx /* may be null if retention failed */, g_wire_port);
        fprintf(stderr, "[ic] wire author server ENABLED (port %u)\n", g_wire_port);
    } else {
        fprintf(stderr, "[ic] wire author server DISABLED (WIRE_ENABLE=0 or unset)\n");
    }
}

extern "C" void deinit(void)
{
    if (g_wire_enabled.load(std::memory_order_relaxed)) {
        wire_author_stop();
        fprintf(stderr, "[ic] wire author server stopped\n");
    }

    std::vector<ICPState*> to_free;
    {
        std::lock_guard<std::mutex> lk(g_instances_mtx);
        to_free.swap(g_instances);
    }
    for (auto* st : to_free) destroy_instance(st);
    release_primary_context();

    // (opzionale) pulizia degli hint residui
    ic_clear_instance_hints();
}

// ============================================================================
// OPTIONAL HOST-QUERY API (used by the host app to fence instance binding)
// ============================================================================
extern "C" int ic_has_instance_for(const char* section /* "cam0|cam1|cam2" */) {
    if (!section || !*section) return 0;
    std::lock_guard<std::mutex> lk(g_instances_mtx);
    for (auto* st : g_instances) {
        if (!st) continue;
        if (st->controls.section() == section) return 1; // found a bound instance
    }
    return 0; // not found yet
}


