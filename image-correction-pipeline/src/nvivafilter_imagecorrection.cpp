/**
 * @file nvivafilter_imagecorrection.cpp
 * @brief GStreamer `nvivafilter` plugin: fisheye rectification + image enhancement.
 *
 * Pipeline:
 *   1. nvivafilter gives us an EGLImageKHR (zero-copy GPU buffer).
 *   2. We map it to CUDA memory via CUDA/EGL interop.
 *   3. Copy input frame into scratch buffer to avoid overwrite.
 *   4. Apply fisheye rectification (NV12 → NV12).
 *   5. Apply adaptive exposure control (highlight guard).
 *   6. Apply enhancement pipeline:
 *        - gamma correction
 *        - local tone mapping
 *        - sharpening
 *        - saturation adjustment
 *   7. Apply highlight rolloff (mainly for sky regions).
 *   8. Unmap EGL resource and return to GStreamer.
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
#include <math.h>

#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>

#include "nvivafilter_customer_api.hpp"
#include "rectify_config.hpp"
#include "kernel_rectify.cuh"
#include "kernel_enhance.cuh"

constexpr float M_PI_F = 3.14159265358979323846f;

// ============================================================================
// Global CUDA primary context
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
// Per-instance state
// ============================================================================
struct ICPState {
    CUcontext     ctx   = nullptr;
    cudaStream_t  stream= nullptr;
    CUdeviceptr sY=0, sUV=0; size_t pY=0, pUV=0; int sW=0,sH=0;
    icp::RectifyConfig cfg{};
    float ae_gain = 1.0f;
};

static std::mutex              g_instances_mtx;
static std::vector<ICPState*>  g_instances;

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
// nvivafilter hooks
// ----------------------------------------------------------------------------
static void pre_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*, ColorFormat*, unsigned int, void **){}
static void post_process(void **, unsigned int*, unsigned int*, unsigned int*, unsigned int*, ColorFormat*, unsigned int, void **){}

static void gpu_process(EGLImageKHR image, void **userPtr)
{
    ICPState* st = static_cast<ICPState*>(*userPtr);
    if (!st) {
        st = create_instance();
        *userPtr = st;
    }
    cuCtxSetCurrent(st->ctx);

    CUgraphicsResource res=nullptr;
    if(cuGraphicsEGLRegisterImage(&res,image,CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)!=CUDA_SUCCESS) return;
    CUeglFrame f{};
    if(cuGraphicsResourceGetMappedEglFrame(&f,res,0,0)!=CUDA_SUCCESS){ cuGraphicsUnregisterResource(res); return; }
    if(f.frameType!=CU_EGL_FRAME_TYPE_PITCH || f.planeCount<2){ cuGraphicsUnregisterResource(res); return; }

    uint8_t* dY  = static_cast<uint8_t*>(f.frame.pPitch[0]);
    uint8_t* dUV = static_cast<uint8_t*>(f.frame.pPitch[1]);
    const int W = (int)f.width, H=(int)f.height, pitch=(int)f.pitch;

    icp::RectifyConfig cfg = st->cfg;

    ensure_scratch(st, W, H);
    copy_to_scratch_async(st, dY, dUV, pitch, pitch, W, H);

    const float FOV_fish = cfg.fish_fov_deg * (float)M_PI / 180.f;
    const float f_fish   = cfg.r_f / (FOV_fish * 0.5f);
    const float fx       = (W * 0.5f) / tanf(cfg.out_hfov_deg * (float)M_PI / 360.f);
    const float cx_rect  = W * 0.5f;
    const float cy_rect  = H * 0.5f;

    icp::launch_rectify_nv12(
        (const uint8_t*)(uintptr_t)st->sY,  W,H,(int)st->pY,
        (const uint8_t*)(uintptr_t)st->sUV, (int)st->pUV,
        dY, W,H,pitch,
        dUV, pitch,
        cfg.cx_f, cfg.cy_f, cfg.r_f,
        f_fish, fx, cx_rect, cy_rect,
        st->stream);

    icp::launch_highlight_guard(dY, W, H, pitch, &st->ae_gain, st->stream);

    // Enhancement
    icp::EnhanceParams ep;
    ep.gamma          = 0.98f;
    ep.local_tm       = 0.18f;
    ep.tm_white       = 1.10f;
    ep.sharpen_amount = 0.18f;
    ep.sharpen_clip   = 8.0f;
    ep.saturation     = 1.05f;

    icp::launch_enhance_nv12(dY, W, H, pitch, dUV, pitch, ep, st->stream);

    // Rolloff top-weighted
    icp::launch_highlight_rolloff_top(dY, W, H, pitch, 0.80f, 0.95f, 6.0f, st->stream);

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
