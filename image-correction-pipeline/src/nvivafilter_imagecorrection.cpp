/**
 * @file nvivafilter_imagecorrection.cpp
 * @brief GStreamer nvivafilter plugin – low‑latency version (single CUDA primary context).
 *
 * What changed vs your original file (high‑level):
 * ------------------------------------------------
 * 1) **Single CUDA primary context shared by all instances**
 *    - Removes per‑camera cuCtxCreate/cuCtxDestroy.
 *    - Avoids context switching and lets streams overlap.
 *
 * 2) **Stream creation only** per instance
 *    - Optional priority via env `ICP_STREAM_PRIORITY` (higher negative = higher prio).
 *
 * 3) **No functional/kernel changes**
 *    - We intentionally skip the smaller kernel details you mentioned.
 *    - The rest of the pipeline logic is kept intact.
 *
 * Notes:
 * ------
 * - This file does not change your kernels; it just removes context thrashing.
 * - Further latency wins will require (later) removing per‑frame cudaMalloc/Free
 *   inside `compute_stats_nv12` and `launch_color_grade_nv12_inplace`, and
 *   reworking LTM – kept out for now as requested.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>

#include "nvivafilter_customer_api.hpp"
#include "rectify_config.hpp"
#include "kernel_rectify.cuh"
#include "kernel_color.cuh"
#include "runtime_controls.hpp"

// ============================================================================
// Helpers
// ============================================================================
static inline float clampf(float v, float a, float b){ return v<a?a:(v>b?b:v); }
static inline int   i_min(int a,int b){ return a<b?a:b; }
static inline int   i_max(int a,int b){ return a>b?a:b; }

// ============================================================================
// Global CUDA primary context (shared across all instances)
// ============================================================================
namespace {
    static std::once_flag g_ctx_once;
    static CUcontext g_primary_ctx = nullptr;
    static CUdevice  g_device      = 0;

    // Retain the device primary context once for the whole process.
    // Using the primary context avoids expensive context switching when
    // multiple cameras/pipelines are active.
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

    // Release primary context at deinit (only when no instances remain).
    static void release_primary_context() {
        if (g_primary_ctx) {
            cuDevicePrimaryCtxRelease(g_device);
            g_primary_ctx = nullptr;
        }
    }
}

// ============================================================================
// Per‑instance state (one per pipeline / camera)
// ============================================================================
struct ICPState {
    // The shared primary CUDA context (not owned – do not destroy here)
    CUcontext     ctx   = nullptr;

    // One non‑blocking CUDA stream per instance
    cudaStream_t  stream= nullptr;

    // Scratch for rectify
    CUdeviceptr sY = 0, sUV = 0; size_t pY = 0, pUV = 0; int sW = 0, sH = 0;

    // Previous frame for temporal denoise
    CUdeviceptr pvY = 0, pvUV = 0; size_t ppY = 0, ppUV = 0; int pW = 0, pH = 0;

    // Stages (kept as in original)
    bool stage_rectify = true;
    bool stage_color   = true;

    // Config + live controls
    icp::RectifyConfig           cfg{};
    std::unique_ptr<RuntimeControls> rc;

    // AE/AWB state
    bool  auto_inited = false;
    float k_exp  = 1.0f;
    float gamma  = 1.0f;
    float wb_r   = 1.0f;
    float wb_b   = 1.0f;
};

// Keep track of all created instances so we can destroy them in deinit()
static std::mutex              g_instances_mtx;
static std::vector<ICPState*>  g_instances;

// ----------------------------------------------------------------------------
// Instance allocation / destruction
// ----------------------------------------------------------------------------
static int read_stream_priority_from_env() {
    // Optional env var to bias scheduling between instances (negative is higher prio).
    // e.g. ICP_STREAM_PRIORITY=-1
    if (const char* s = std::getenv("ICP_STREAM_PRIORITY")) {
        return std::atoi(s);
    }
    return 0; // default priority
}

static ICPState* create_instance()
{
    // Ensure primary context exists once
    std::call_once(g_ctx_once, retain_primary_context_once);

    auto* st = new ICPState();

    // Use the shared primary context
    st->ctx = g_primary_ctx;
    cuCtxSetCurrent(st->ctx);

    // Create one non‑blocking stream for this instance (optional priority)
    int prio = read_stream_priority_from_env();
#if CUDART_VERSION >= 11000
    cudaStreamCreateWithPriority(&st->stream, cudaStreamNonBlocking, prio);
#else
    (void)prio;
    cudaStreamCreateWithFlags(&st->stream, cudaStreamNonBlocking);
#endif

    // Stage flags (per instance) – same behavior as before via env.
    if (const char* s = getenv("ICP_STAGE")) {
        std::string v(s);
        std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c){ return std::tolower(c); });
        if (v == "rectify") { st->stage_rectify = true;  st->stage_color = false; }
        if (v == "color")   { st->stage_rectify = false; st->stage_color = true;  }
        if (v == "both")    { st->stage_rectify = true;  st->stage_color = true;  }
    }

    // Runtime controls (per instance)
    if (const char* p = getenv("ICP_CONTROLS")) {
        try {
            st->rc = std::make_unique<RuntimeControls>(std::string(p), true);
            fprintf(stderr,"[ic] RuntimeControls watching: %s\n", p);
        } catch (...) {
            st->rc.reset();
            fprintf(stderr,"[ic] RuntimeControls disabled\n");
        }
    } else {
        fprintf(stderr,"[ic] RuntimeControls disabled (set ICP_CONTROLS to enable)\n");
    }

    // Register instance
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
    if (st->pvY) { cuMemFree(st->pvY); st->pvY = 0; }
    if (st->pvUV){ cuMemFree(st->pvUV);st->pvUV= 0; }

    if (st->stream) { cudaStreamDestroy(st->stream); st->stream = nullptr; }

    // Do NOT destroy the shared primary context here.
    st->ctx = nullptr;

    delete st;
}

// ----------------------------------------------------------------------------
// Per‑instance alloc helpers
// ----------------------------------------------------------------------------
static void ensure_scratch(ICPState* st, int W, int H)
{
    if (st->sY && st->sUV && st->sW == W && st->sH == H) return;

    if (st->sY)  { cuMemFree(st->sY);  st->sY  = 0; }
    if (st->sUV) { cuMemFree(st->sUV); st->sUV = 0; }

    cuMemAllocPitch(&st->sY,  &st->pY,  (size_t)W, (size_t)H,    4);
    cuMemAllocPitch(&st->sUV, &st->pUV, (size_t)W, (size_t)(H/2),4);
    st->sW = W; st->sH = H;

    fprintf(stderr,"[ic] [%p] scratch allocated %dx%d pitchY=%zu pitchUV=%zu\n",
            (void*)st, W, H, st->pY, st->pUV);
}

static void ensure_prev(ICPState* st, int W, int H)
{
    if (st->pvY && st->pvUV && st->pW == W && st->pH == H) return;

    if (st->pvY)  { cuMemFree(st->pvY);  st->pvY  = 0; }
    if (st->pvUV) { cuMemFree(st->pvUV); st->pvUV = 0; }

    cuMemAllocPitch(&st->pvY,  &st->ppY,  (size_t)W, (size_t)H,    4);
    cuMemAllocPitch(&st->pvUV, &st->ppUV, (size_t)W, (size_t)(H/2),4);

    // init: Y=0, UV=128
    cuMemsetD8(st->pvY,  0,   st->ppY  * (size_t)H);
    cuMemsetD8(st->pvUV, 128, st->ppUV * (size_t)(H/2));

    st->pW = W; st->pH = H;
}

// Async copy input → scratch on the instance stream
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
    // Ensure instance
    ICPState* st = static_cast<ICPState*>(*userPtr);
    if (!st) {
        st = create_instance();
        *userPtr = st;
        fprintf(stderr,"[ic] init() instance %p (rectify=%d, color=%d)\n",
                (void*)st, st->stage_rectify?1:0, st->stage_color?1:0);
    }
    cuCtxSetCurrent(st->ctx);

    // Map EGL image
    CUgraphicsResource res=nullptr;
    if(cuGraphicsEGLRegisterImage(&res,image,CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)!=CUDA_SUCCESS){
        fprintf(stderr,"[ic] cuGraphicsEGLRegisterImage failed\n"); return;
    }
    CUeglFrame f{};
    if(cuGraphicsResourceGetMappedEglFrame(&f,res,0,0)!=CUDA_SUCCESS){
        fprintf(stderr,"[ic] GetMappedEglFrame failed\n"); cuGraphicsUnregisterResource(res); return;
    }
    if(f.frameType!=CU_EGL_FRAME_TYPE_PITCH || f.planeCount<2){
        fprintf(stderr,"[ic] Unexpected frameType=%d planeCount=%d\n",f.frameType,f.planeCount);
        cuGraphicsUnregisterResource(res); return;
    }

    uint8_t* dY  = static_cast<uint8_t*>(f.frame.pPitch[0]);
    uint8_t* dUV = static_cast<uint8_t*>(f.frame.pPitch[1]);
    const int W = (int)f.width, H=(int)f.height, pitch=(int)f.pitch;

    // Live config snapshot
    icp::RectifyConfig cfg = st->rc ? st->rc->get() : st->cfg;

    // 1) Rectification (GPU→GPU) via per‑instance scratch
    if (st->stage_rectify) {
        ensure_scratch(st, W, H);
        copy_to_scratch_async(st, dY, dUV, pitch, pitch, W, H);

        const float FOV_fish = cfg.fish_fov_deg * (float)M_PI / 180.f;
        const float f_fish   = cfg.r_f / (FOV_fish * 0.5f);
        const float fx       = (W * 0.5f) / std::tan(cfg.out_hfov_deg * (float)M_PI / 360.f);
        const float cx_rect  = W * 0.5f;
        const float cy_rect  = H * 0.5f;

        icp::launch_rectify_nv12(
            (const uint8_t*)(uintptr_t)st->sY,  W,H,(int)st->pY,
            (const uint8_t*)(uintptr_t)st->sUV,     (int)st->pUV,
            dY, W,H,pitch,
            dUV,     pitch,
            cfg.cx_f, cfg.cy_f, cfg.r_f,
            f_fish, fx, cx_rect, cy_rect,
            st->stream);
    }

    // 2) Temporal denoise (before stats/AE)
    ensure_prev(st, W, H);
    icp::launch_temporal_denoise_nv12(
        dY, dUV, pitch, pitch, W, H,
        (uint8_t*)(uintptr_t)st->pvY, (uint8_t*)(uintptr_t)st->pvUV,
        (int)st->ppY, (int)st->ppUV,
        /*alphaY*/0.35f, /*alphaUV*/0.25f,
        /*thrY*/6, /*thrUV*/8,
        st->stream);

    // 3) Stats (central ROI)
    int side = (int)std::lround((double)i_min(W,H) * (cfg.auto_roi_pct * 0.01));
    side = i_max(8, i_min(side, i_min(W,H)));
    const int rx = i_max(0, (W - side)/2);
    const int ry = i_max(0, (H - side)/2);
    const int rW = i_min(W, side);
    const int rH = i_min(H, side);

    uint32_t hist[256] = {};
    float mU=128.f, mV=128.f;
    icp::compute_stats_nv12(dY, pitch, dUV, pitch, W, H,
                            rx, ry, rW, rH,
                            std::max(1, cfg.auto_hist_step),
                            hist, &mU, &mV, st->stream);

    auto pct_bin = [&](float pct)->int {
        unsigned long long tot=0ULL; for(int i=0;i<256;++i) tot+=hist[i];
        if(!tot) return 0;
        unsigned long long thr = (unsigned long long)std::llround((pct/100.0)*(double)tot);
        unsigned long long acc=0ULL;
        for(int i=0;i<256;++i){ acc+=hist[i]; if(acc>=thr) return i; }
        return 255;
    };
    unsigned long long tot=0, clipCnt=0;
    for(int i=0;i<256;++i) tot+=hist[i];
    for(int i=250;i<=255;++i) clipCnt+=hist[i];
    float clipRatio = tot ? (float)clipCnt/(float)tot : 0.0f;

    const int p50  = pct_bin(50.0f);
    const int p995 = pct_bin(99.5f);
    const int p999 = pct_bin(99.9f);

    // 4) AE anti‑flicker + highlight‑protect (per instance)
    const float HI_CAP_Y  = 230.0f;
    const float HI_WEIGHT = 0.85f;
    const float CLIP_MAX  = 0.015f;

    float k_mid = clampf(cfg.target_Y / std::max(1, p50), 0.55f, 1.45f);
    float k_hi  = (p995 > 0) ? (HI_CAP_Y / (float)p995) : 1.0f;
    if (clipRatio > CLIP_MAX) k_hi = std::min(k_hi, 0.90f);

    float k_target = std::min(k_mid, std::pow(k_hi, HI_WEIGHT));

    // anti‑flicker: slower on brightening, faster on darkening
    const float STEP_UP   = std::max(0.02f, std::min(0.08f, cfg.auto_ae_step*0.7f));
    const float STEP_DOWN = std::max(0.05f, std::min(0.12f, cfg.auto_ae_step*1.3f));

    if (!st->auto_inited) st->k_exp = k_target;
    else {
        float ratio = k_target / st->k_exp;
        float step  = (ratio>1.f) ? STEP_UP : STEP_DOWN;
        ratio = clampf(ratio, 1.f - step, 1.f + step);
        st->k_exp *= ratio;
    }

    float gamma_tgt = 1.0f;
    if (p50 < 90)                               gamma_tgt = cfg.auto_gamma_min;
    else if (p999 > 245 || clipRatio > CLIP_MAX) gamma_tgt = cfg.auto_gamma_max;
    if (!st->auto_inited) st->gamma = gamma_tgt;
    else                   st->gamma = 0.95f*st->gamma + 0.05f*gamma_tgt;
    float gammaFinal = cfg.gamma * st->gamma;

    // 5) Filmic LUT (white at 99.9°) – host side for now
    auto filmicU2 = [](float x)->float {
        const float A=0.22f, B=0.30f, C=0.10f, D=0.20f, E=0.01f, F=0.30f;
        float num = x*(A*x + C*B) + D*E;
        float den = x*(A*x + B)   + D*F;
        return (num/den) - E/F;
    };
    float Lwhite = std::max(0.55f, std::min(1.2f, (float)p999/255.f));
    float fw = filmicU2(Lwhite); if (fw < 1e-6f) fw = 1.f;

    uint8_t lut[256];
    for (int i=0;i<256;++i){
        float n = (i/255.f) * st->k_exp;
        n = n<0.f?0.f:(n>3.0f?3.0f:n);
        float y = filmicU2(n) / fw;
        y = y<0.f?0.f:(y>1.f?1.f:y);
        lut[i] = (uint8_t)std::lround(y * 255.f);
    }

    // 6) AWB (per instance)
    float wb_r = cfg.wb_r, wb_g = cfg.wb_g, wb_b = cfg.wb_b;
    if (cfg.auto_wb) {
        float u = mU - 128.f, v = mV - 128.f, Ymid=110.f;
        float Rm = Ymid + 1.5748f*v;
        float Gm = Ymid - 0.1873f*u - 0.4681f*v;
        float Bm = Ymid + 1.8556f*u;
        float wr  = clampf((Gm+1e-3f)/(Rm+1e-3f), 1.f - cfg.auto_wb_clamp, 1.f + cfg.auto_wb_clamp);
        float wbb = clampf((Gm+1e-3f)/(Bm+1e-3f), 1.f - cfg.auto_wb_clamp, 1.f + cfg.auto_wb_clamp);
        if (!st->auto_inited) { st->wb_r=wr; st->wb_b=wbb; }
        else { st->wb_r = 0.95f*st->wb_r + 0.05f*wr; st->wb_b = 0.95f*st->wb_b + 0.05f*wbb; }
        wb_r *= st->wb_r; wb_b *= st->wb_b;
    }

    // 7) Global grading (in‑place)
    icp::launch_color_grade_nv12_inplace(
        dY, dUV, pitch, pitch, W, H,
        lut,
        cfg.contrast, cfg.brightness,
        cfg.saturation, gammaFinal,
        wb_r, wb_g, wb_b,
        /*sat rolloff*/0.70f, 0.97f, 0.25f,
        st->stream);

    // 8) Local tone‑mapping “CLAHE‑lite” on Y
    icp::launch_local_tonemap_nv12(
        dY, pitch, W, H,
        /*radius*/4,
        /*amount*/0.55f,
        /*hi_start*/0.70f, /*hi_end*/0.97f,
        st->stream);

    st->auto_inited = true;

    // Stream‑local sync to ensure frame is ready for downstream without stalling other instances
    cudaStreamSynchronize(st->stream);

    cuGraphicsUnregisterResource(res);
}

// ----------------------------------------------------------------------------
// init / deinit
// ----------------------------------------------------------------------------
extern "C" void init(CustomerFunction* f)
{
    fprintf(stderr,"[ic] init() loaded (rectify + color + denoise + LTM) [single‑context] \n");
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

    // Release the shared primary context now that all instances are gone
    release_primary_context();

    fprintf(stderr,"[ic] deinit()\n");
}
