/**
 * @file nvivafilter_imagecorrection.cpp
 * @brief GStreamer `nvivafilter` plugin: fisheye rectification → crop → tone/color.
 *
 * Binding deterministico della sezione (cam0/cam1/cam2) in base al NOME della .so caricata:
 *   libnvivafilter_imagecorrection_cam0.so → "cam0"
 *   libnvivafilter_imagecorrection_cam1.so → "cam1"
 *   libnvivafilter_imagecorrection_cam2.so → "cam2"
 *
 * Così possiamo usare UNA sola .so reale + 3 symlink, senza env/hint/round-robin.
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

#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>

#include "nvivafilter_customer_api.hpp"
#include "rectify_config.hpp"
#include "kernel_rectify.cuh"
#include "color_ops.cuh"
#include "runtime_controls.hpp"

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

// Determina "cam0|cam1|cam2" dal NOME con cui è stata caricata la .so
static std::string section_from_loaded_name()
{
    Dl_info info{};
    if (dladdr((void*)&init, &info) && info.dli_fname) {
        std::string path(info.dli_fname);
        std::string low  = to_lower(path);
        // 1) prima prova dal basename (gestisce anche eventuali suffissi _camX)
        std::string base = path.substr(path.find_last_of('/') + 1);
        std::string base_low = to_lower(base);
        if (base_low.find("_cam0") != std::string::npos) return "cam0";
        if (base_low.find("_cam1") != std::string::npos) return "cam1";
        if (base_low.find("_cam2") != std::string::npos) return "cam2";
        // 2) poi guarda tutto il PATH (cartelle cam0/cam1/cam2)
        if (low.find("/cam0/") != std::string::npos) return "cam0";
        if (low.find("/cam1/") != std::string::npos) return "cam1";
        if (low.find("/cam2/") != std::string::npos) return "cam2";
    }
    fprintf(stderr, "[ic] WARNING: cannot detect cam section from .so name/path, defaulting to cam0\n");
    return "cam0";
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

    // Runtime controls (hot-reload) su sezione decisa a create_instance()
    icp::RuntimeControls controls{"/home/jetson_ntc/config.json", "cam0"};

    // Post-rectification center crop/zoom
    float crop_frac = 0.20f;
};

static std::mutex              g_instances_mtx;
static std::vector<ICPState*>  g_instances;

// ----------------------------------------------------------------------------
// Alloc helpers
// ----------------------------------------------------------------------------
static void ensure_scratch(ICPState* st, int W, int H)
{
    if (st->sY && st->sUV && st->sW == W && st->sH == H) return;

    if (st->sY)  { cuMemFree(st->sY);  st->sY  = 0; }
    if (st->sUV) { cuMemFree(st->sUV); st->sUV = 0; }

    // pitched alloc per NV12
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

    // === Bind deterministico in base al nome della .so (symlink _camX) ===
    {
        std::string sec = section_from_loaded_name();
        st->controls.set_section(sec);
        fprintf(stderr, "[ic] Instance bound to section (soname) '%s'\n", sec.c_str());
    }

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

    // 2) Crop/zoom centrale
    copy_to_scratch_async(st, dY, dUV, pitch, pitch, W, H);
    icp::launch_crop_center_nv12(
        (const uint8_t*)(uintptr_t)st->sY,  W,H,(int)st->pY,
        (const uint8_t*)(uintptr_t)st->sUV, (int)st->pUV,
        dY,                                   pitch,
        dUV,                                  pitch,
        st->crop_frac,                        st->stream);

    // 3) Tone + color (hot-reload)
    icp::ColorParams cp = st->controls.current();
    icp::launch_tone_saturation_nv12(
        dY, W, H, pitch,
        dUV,     pitch,
        cp,
        st->stream);

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
}
