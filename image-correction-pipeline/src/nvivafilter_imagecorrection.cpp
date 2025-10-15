/**
 * @file nvivafilter_imagecorrection.cpp
 * @brief GStreamer `nvivafilter` plugin: fisheye rectification → crop → (optional) wire removal inpaint → tone/color.
 *
 * This version **removes** any built-in authoring/network server. The mask is
 * produced by an external process and saved on CPU as:
 *   - /dev/shm/wire_mask_camN.raw  (W*H bytes, 8-bit)
 *   - /dev/shm/wire_mask_camN.meta (single line: "W H dx dy")
 *
 * At the first frame of each instance, the .so loads the mask **once** for the
 * instance's section (cam0|cam1|cam2). At runtime the mask is applied **only**
 * if:
 *   - a valid device mask was loaded,
 *   - the current frame resolution matches the mask resolution,
 *   - the instance section index (0/1/2) equals the mask's cam index N.
 *
 * Examples:
 *   instance = cam0, mask files are "cam0" → apply
 *   instance = cam0, mask files are "cam2" → do NOT apply
 *
 * Typical pipeline:
 *   gst-launch-1.0 ... nvivafilter customer-lib-name=/usr/local/lib/nvivafilter/libnvivafilter_imagecorrection_cam0.so ...
 *   gst-launch-1.0 ... nvivafilter customer-lib-name=/usr/local/lib/nvivafilter/libnvivafilter_imagecorrection_cam1.so ...
 *   gst-launch-1.0 ... nvivafilter customer-lib-name=/usr/local/lib/nvivafilter/libnvivafilter_imagecorrection_cam2.so ...
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

// Wire removal kernel (two-donor NV12 inpaint)
#include "wire_lineremoval.cuh"

// NEW: Profiling tools
#include <nvtx3/nvToolsExt.h>
#include <cuda_profiler_api.h>


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

    // --- NEW: per-instance, device-resident mask loaded once from CPU ---
    struct DeviceMask {
        CUdeviceptr dMask = 0;   // 8-bit mask on device
        size_t      pitch = 0;   // bytes per row for mask
        int         W = 0, H = 0;
        float       dx = 0.f, dy = 0.f;
        bool        valid = false;
        int         cam_index = -1; // 0/1/2 corresponding to the files used
    } mask{};
    bool mask_checked_once = false; // ensures we attempt loading only once

    // Track last-applied mask version (kept for parity with older logs, unused)
    uint32_t last_mask_version = 0;
};

static std::mutex              g_instances_mtx;
static std::vector<ICPState*>  g_instances;

// --- per-instance section hints (queue consumed by create_instance) ----
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
        // Not an error: absence of mask simply means "do not apply"
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
    st->mask = {}; // reset

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
    nvtxRangePushA("gpu_process");
    
    // Lazily create per-instance state
    nvtxRangePushA("lazy_create_instance");
    ICPState* st = static_cast<ICPState*>(*userPtr);
    if (!st) {
        st = create_instance();
        *userPtr = st;
    }
    cuCtxSetCurrent(st->ctx);
    nvtxRangePop(); // lazy_create_instance

    // Load the per-instance mask once (if present on CPU)
    if (!st->mask_checked_once) {
        nvtxRangePush("load_mask_from_cpu_once");
        st->mask_checked_once = true;
        (void)load_mask_from_cpu_once(st); // soft-fail: simply no mask applied
        nvtxRangePop(); // load_mask_from_cpu_once
    }

    // Map EGLImage → CUDA (expect NV12, pitched)
    nvtxRangePushA("MapEGLImage");
    CUgraphicsResource res=nullptr;
    //if(cuGraphicsEGLRegisterImage(&res,image,CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)!=CUDA_SUCCESS) return;
    if (cuGraphicsEGLRegisterImage(&res, image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE) != CUDA_SUCCESS) {
        nvtxRangePop(); // MapEGLImage
        nvtxRangePop(); // gpu_process
        return;
    }
    CUeglFrame f{};
    //if(cuGraphicsResourceGetMappedEglFrame(&f,res,0,0)!=CUDA_SUCCESS){ cuGraphicsUnregisterResource(res); return; }
    if (cuGraphicsResourceGetMappedEglFrame(&f, res, 0, 0) != CUDA_SUCCESS) {
        cuGraphicsUnregisterResource(res);
        nvtxRangePop(); // MapEGLImage
        nvtxRangePop(); // gpu_process
        return;
    }
    //if(f.frameType!=CU_EGL_FRAME_TYPE_PITCH || f.planeCount<2){ cuGraphicsUnregisterResource(res); return; }
    if (f.frameType != CU_EGL_FRAME_TYPE_PITCH || f.planeCount < 2) {
        cuGraphicsUnregisterResource(res);
        nvtxRangePop(); // MapEGLImage
        nvtxRangePop(); // gpu_process
        return;
    }
    nvtxRangePop(); // MapEGLImage

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
    nvtxRangePushA("RectifyNV12");
    copy_to_scratch_async(st, dY, dUV, pitch, pitch, W, H);
    icp::launch_rectify_nv12(
        (const uint8_t*)(uintptr_t)st->sY,  W,H,(int)st->pY,
        (const uint8_t*)(uintptr_t)st->sUV, (int)st->pUV,
        dY, W,H,pitch,
        dUV, pitch,
        cfg.cx_f, cfg.cy_f, cfg.r_f,
        f_fish, fx, cx_rect, cy_rect,
        st->stream);
    nvtxRangePop(); // RectifyNV12

    // 2) Center crop/zoom
    nvtxRangePushA("CenterCropZoom");
    copy_to_scratch_async(st, dY, dUV, pitch, pitch, W, H);
    icp::launch_crop_center_nv12(
        (const uint8_t*)(uintptr_t)st->sY,  W,H,(int)st->pY,
        (const uint8_t*)(uintptr_t)st->sUV, (int)st->pUV,
        dY,                                   pitch,
        dUV,                                  pitch,
        st->crop_frac,                        st->stream);
    nvtxRangePop(); // CenterCropZoom

    // 2.5) Wire removal (two-donor inpaint) — only if mask is valid and belongs to this instance
    if (st->mask.valid && st->mask.W == W && st->mask.H == H) {
        nvtxRangePushA("WireRemoval");
        const int my_idx = section_to_index(st->controls.section());
        if (my_idx == st->mask.cam_index) {
            wire::apply_mask_shift_nv12(
                dY,  pitch,
                dUV, pitch,
                W, H,
                (const uint8_t*)(uintptr_t)st->mask.dMask, (int)st->mask.pitch,
                st->mask.dx, st->mask.dy,
                st->stream);
            // fprintf(stderr, "[ic] mask applied to %s\n", st->controls.section().c_str());
        } else {
            // Mask was loaded for a different cam index → do nothing
            // fprintf(stderr, "[ic] mask not for this instance (%s)\n", st->controls.section().c_str());
        }
        nvtxRangePop(); // WireRemoval
    }

    // 3) Tone + color (hot-reload)
    nvtxRangePushA("ToneColor");
    icp::ColorParams cp = st->controls.current();
    icp::launch_tone_saturation_nv12(
        dY, W, H, pitch,
        dUV,     pitch,
        cp,
        st->stream);
    nvtxRangePop(); // ToneColor

    cudaStreamSynchronize(st->stream);
    cuGraphicsUnregisterResource(res);

    nvtxRangePop(); // gpu_process
}

// ----------------------------------------------------------------------------
// init / deinit
// ----------------------------------------------------------------------------
extern "C" void init(CustomerFunction* f)
{
    if(!f) return;

    cudaProfilerStart();
    nvtxRangePush("ic_init");

    f->fPreProcess  = pre_process;
    f->fGPUProcess  = gpu_process;
    f->fPostProcess = post_process;

    // Ensure CUDA primary context retained
    std::call_once(g_ctx_once, retain_primary_context_once);

    // No authoring server here; mask is loaded once from CPU when the first frame arrives.
    fprintf(stderr, "[ic] imagecorrection initialized (no built-in authoring server)\n");
    nvtxRangePop(); // ic_init
}

extern "C" void deinit(void)
{
    nvtxRangePush("ic_deinit");

    std::vector<ICPState*> to_free;
    {
        std::lock_guard<std::mutex> lk(g_instances_mtx);
        to_free.swap(g_instances);
    }
    for (auto* st : to_free) destroy_instance(st);
    release_primary_context();

    // Clear any residual instance-binding hints
    ic_clear_instance_hints();

    nvtxRangePop(); // ic_deinit
    cudaProfilerStop();
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
